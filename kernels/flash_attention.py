"""
Varlen FlashAttention V2 with GQA - Zero-Copy KV Cache.

Reads KV directly from cache tensor [num_seqs, max_seq_len, kv_head, dim].
No concatenation needed. Each sequence's KV is at a fixed stride offset.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_varlen_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
    cu_seqlens_q_ptr, ctx_lens_ptr, tile_seq_ids_ptr,
    total_q_len,
    stride_q_seq, stride_q_head,
    stride_kc_seq, stride_kc_pos, stride_kc_head,  # k_cache: [num_seqs, max_len, kv_head, dim]
    stride_vc_seq, stride_vc_pos, stride_vc_head,
    stride_o_seq, stride_o_head,
    softmax_scale: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    Q_HEAD: tl.constexpr,
    KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Grid: (num_q_tiles, Q_HEAD).
    Reads KV directly from per-sequence cache (no cat).
    """
    pid_m = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // (Q_HEAD // KV_HEAD)

    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    qm_mask = offs_m < total_q_len

    # Sequence lookup: O(1)
    seq_idx = tl.load(tile_seq_ids_ptr + pid_m).to(tl.int32)
    seq_q_start = tl.load(cu_seqlens_q_ptr + seq_idx).to(tl.int32)
    seq_q_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1).to(tl.int32)

    # KV length for this sequence
    kv_seq_len = tl.load(ctx_lens_ptr + seq_idx).to(tl.int32)
    q_seq_len = seq_q_end - seq_q_start

    # Load Q tile [BLOCK_M, HEAD_DIM]
    q_base = q_ptr + q_head_idx * stride_q_head
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_seq + offs_d[None, :],
        mask=qm_mask[:, None], other=0.0,
    )

    # Online softmax state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # KV base: directly in cache at this sequence's slot
    k_base = k_cache_ptr + seq_idx * stride_kc_seq + kv_head_idx * stride_kc_head
    v_base = v_cache_ptr + seq_idx * stride_vc_seq + kv_head_idx * stride_vc_head

    # Loop over this sequence's KV
    n_blocks = tl.cdiv(kv_seq_len, BLOCK_N)
    for n_start in range(0, n_blocks * BLOCK_N, BLOCK_N):
        kn_mask = offs_n < (kv_seq_len - n_start)

        k = tl.load(
            k_base + (n_start + offs_n[:, None]) * stride_kc_pos + offs_d[None, :],
            mask=kn_mask[:, None], other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * softmax_scale
        s = tl.where(kn_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_base + (n_start + offs_n[:, None]) * stride_vc_pos + offs_d[None, :],
            mask=kn_mask[:, None], other=0.0,
        )
        acc += tl.dot(p.to(COMPUTE_DTYPE), v)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_base = o_ptr + q_head_idx * stride_o_head
    tl.store(
        o_base + offs_m[:, None] * stride_o_seq + offs_d[None, :],
        acc.to(q.dtype), mask=qm_mask[:, None],
    )


def invoke_flash_attention(
    q: torch.Tensor,            # [total_q, q_head, dim]
    k_cache: torch.Tensor,      # [num_seqs, max_seq_len, kv_head, dim]
    v_cache: torch.Tensor,      # [num_seqs, max_seq_len, kv_head, dim]
    cu_seqlens_q: torch.Tensor, # [num_seqs+1] int32
    ctx_lens: torch.Tensor,     # [num_seqs] int32 - actual KV length per seq
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """
    Varlen FlashAttention - reads KV directly from cache (zero-copy).

    Args:
        k_cache/v_cache: [num_seqs, max_seq_len, kv_head, dim] - cache views, no cat needed.
        ctx_lens: [num_seqs] actual filled KV length per sequence.

    Returns: [total_q, q_head * dim]
    """
    total_q_len, q_head, head_dim = q.shape
    num_seqs = cu_seqlens_q.shape[0] - 1
    kv_head = k_cache.shape[2]

    scale = softmax_scale if softmax_scale is not None else (head_dim ** -0.5)
    compute_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16

    o = torch.empty(total_q_len, q_head, head_dim, dtype=q.dtype, device=q.device)

    BLOCK_M = 64
    BLOCK_N = 64
    num_tiles = triton.cdiv(total_q_len, BLOCK_M)

    # Precompute tile->seq mapping
    tile_seq_ids = torch.searchsorted(
        cu_seqlens_q[1:], torch.arange(num_tiles, device=q.device) * BLOCK_M, right=True
    ).to(torch.int32)

    assert head_dim <= 128 and head_dim % 16 == 0
    assert q.is_contiguous()

    grid = (num_tiles, q_head)

    _flash_attn_varlen_kernel[grid](
        q, k_cache, v_cache, o,
        cu_seqlens_q, ctx_lens, tile_seq_ids,
        total_q_len,
        stride_q_seq=q.stride(0), stride_q_head=q.stride(1),
        stride_kc_seq=k_cache.stride(0), stride_kc_pos=k_cache.stride(1), stride_kc_head=k_cache.stride(2),
        stride_vc_seq=v_cache.stride(0), stride_vc_pos=v_cache.stride(1), stride_vc_head=v_cache.stride(2),
        stride_o_seq=o.stride(0), stride_o_head=o.stride(1),
        softmax_scale=scale,
        COMPUTE_DTYPE=compute_dtype,
        Q_HEAD=q_head,
        KV_HEAD=kv_head,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return o.reshape(total_q_len, q_head * head_dim)
