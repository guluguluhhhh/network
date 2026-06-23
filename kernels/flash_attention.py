"""
Varlen FlashAttention V2 with GQA.

Unified kernel: handles any mix of prefill and decode sequences in a single launch.
Uses cu_seqlens (CSR format) to identify sequence boundaries.

Input layout:
  Q: [total_q_tokens, q_head, dim]   - all sequences' Q concatenated
  K: [total_kv_tokens, kv_head, dim] - all sequences' KV concatenated
  V: [total_kv_tokens, kv_head, dim]
  cu_seqlens_q: [num_seqs+1] int32   - CSR offsets for Q
  cu_seqlens_k: [num_seqs+1] int32   - CSR offsets for KV
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_varlen_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    tile_seq_ids_ptr,
    total_q_len,
    stride_q_seq, stride_q_head,
    stride_k_seq, stride_k_head,
    stride_v_seq, stride_v_head,
    stride_o_seq, stride_o_head,
    softmax_scale: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    Q_HEAD: tl.constexpr,
    KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Grid: (cdiv(total_q_len, BLOCK_M), Q_HEAD).

    Each program handles BLOCK_M Q rows from one head.
    tile_seq_ids[pid_m] tells which sequence this tile belongs to (precomputed on host).
    """
    pid_m = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // (Q_HEAD // KV_HEAD)

    # Global Q row range for this tile
    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    qm_mask = offs_m < total_q_len

    # Sequence lookup: O(1), precomputed on host
    seq_idx = tl.load(tile_seq_ids_ptr + pid_m).to(tl.int32)
    seq_q_start = tl.load(cu_seqlens_q_ptr + seq_idx).to(tl.int32)
    seq_q_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1).to(tl.int32)

    # Get this sequence's KV range
    seq_k_start = tl.load(cu_seqlens_k_ptr + seq_idx).to(tl.int32)
    seq_k_end = tl.load(cu_seqlens_k_ptr + seq_idx + 1).to(tl.int32)
    kv_seq_len = seq_k_end - seq_k_start
    q_seq_len = seq_q_end - seq_q_start

    # For causal: Q position in sequence = global_idx - seq_q_start
    # Offset so Q aligns to end of KV: local_q + (kv_len - q_len)
    causal_offset = kv_seq_len - q_seq_len

    # ── Load Q tile [BLOCK_M, HEAD_DIM] ──
    q_base = q_ptr + q_head_idx * stride_q_head
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_seq + offs_d[None, :],
        mask=qm_mask[:, None], other=0.0,
    )

    # ── Online softmax state ──
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ── KV base pointers (offset to this sequence's KV start) ──
    k_base = k_ptr + kv_head_idx * stride_k_head + seq_k_start * stride_k_seq
    v_base = v_ptr + kv_head_idx * stride_v_head + seq_k_start * stride_v_seq

    # ── Loop over this sequence's KV blocks ──
    n_blocks = tl.cdiv(kv_seq_len, BLOCK_N)
    for n_start in range(0, n_blocks * BLOCK_N, BLOCK_N):
        kn_mask = offs_n < (kv_seq_len - n_start)

        # Load K [BLOCK_N, HEAD_DIM]
        k = tl.load(
            k_base + (n_start + offs_n[:, None]) * stride_k_seq + offs_d[None, :],
            mask=kn_mask[:, None], other=0.0,
        )
        # S = Q @ K^T [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * softmax_scale
        s = tl.where(kn_mask[None, :], s, float("-inf"))

        # Causal mask: local_q_pos + causal_offset >= local_kv_pos
        if IS_CAUSAL:
            local_q = offs_m - seq_q_start  # [BLOCK_M] local Q position
            local_kv = n_start + offs_n     # [BLOCK_N] local KV position
            causal_mask = (local_q + causal_offset)[:, None] >= local_kv[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V [BLOCK_N, HEAD_DIM]
        v = tl.load(
            v_base + (n_start + offs_n[:, None]) * stride_v_seq + offs_d[None, :],
            mask=kn_mask[:, None], other=0.0,
        )
        acc += tl.dot(p.to(COMPUTE_DTYPE), v)
        m_i = m_new

    # ── Final renormalization ──
    acc = acc / l_i[:, None]

    # ── Write output ──
    o_base = o_ptr + q_head_idx * stride_o_head
    tl.store(
        o_base + offs_m[:, None] * stride_o_seq + offs_d[None, :],
        acc.to(q.dtype), mask=qm_mask[:, None],
    )


def invoke_flash_attention(
    q: torch.Tensor,            # [total_q, q_head, dim]
    k: torch.Tensor,            # [total_kv, kv_head, dim]
    v: torch.Tensor,            # [total_kv, kv_head, dim]
    cu_seqlens_q: torch.Tensor, # [num_seqs+1] int32
    cu_seqlens_k: torch.Tensor, # [num_seqs+1] int32
    is_causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """
    Varlen FlashAttention V2 with GQA.

    Handles any mix of prefill/decode in one launch via cu_seqlens.

    Returns: [total_q, q_head * dim] contiguous.
    """
    total_q_len, q_head, head_dim = q.shape
    _, kv_head, _ = k.shape
    num_seqs = cu_seqlens_q.shape[0] - 1

    scale = softmax_scale if softmax_scale is not None else (head_dim ** -0.5)
    compute_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16

    o = torch.empty(total_q_len, q_head, head_dim, dtype=q.dtype, device=q.device)

    BLOCK_M = 64
    BLOCK_N = 64
    num_tiles = triton.cdiv(total_q_len, BLOCK_M)

    # Precompute tile->seq mapping on host (O(1) lookup in kernel)
    tile_seq_ids = torch.searchsorted(
        cu_seqlens_q[1:], torch.arange(num_tiles, device=q.device) * BLOCK_M, right=True
    ).to(torch.int32)

    assert head_dim <= 128 and head_dim % 16 == 0
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    grid = (num_tiles, q_head)

    _flash_attn_varlen_kernel[grid](
        q, k, v, o,
        cu_seqlens_q, cu_seqlens_k,
        tile_seq_ids,
        total_q_len,
        stride_q_seq=q.stride(0), stride_q_head=q.stride(1),
        stride_k_seq=k.stride(0), stride_k_head=k.stride(1),
        stride_v_seq=v.stride(0), stride_v_head=v.stride(1),
        stride_o_seq=o.stride(0), stride_o_head=o.stride(1),
        softmax_scale=scale,
        COMPUTE_DTYPE=compute_dtype,
        Q_HEAD=q_head,
        KV_HEAD=kv_head,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
    )

    return o.reshape(total_q_len, q_head * head_dim)
