"""
Kernel 2: FlashAttention V2 with GQA, seq-first layout.

Accepts contiguous seq-first tensors [seq, heads, dim] from Kernel 1,
computes multi-head attention with online-softmax (no materialised S),
and outputs [seq, embed_dim] contiguous, ready for o_proj.

GQA mapping: kv_head = q_head // kv_ratio, merged into kernel loop.
No causal mask (is_causal=False), no dropout.

Ref: Dao et al., "FlashAttention-2: Faster Attention with Better
Parallelism and Work Partitioning", 2023.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len,
    stride_q_seq, stride_q_head,   # seq-first: [S, Hq, D]
    stride_k_seq, stride_k_head,   # [S, Hk, D]
    stride_v_seq, stride_v_head,   # [S, Hk, D]
    stride_o_seq, stride_o_head,   # [S, Hq, D]
    softmax_scale: tl.constexpr,    # 1/sqrt(head_dim)
    COMPUTE_DTYPE: tl.constexpr,
    Q_HEAD: tl.constexpr,
    KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Grid: (triton.cdiv(seq_len, BLOCK_M), Q_HEAD).

    Each program processes BLOCK_M query rows from one Q head against
    all KV blocks (BLOCK_N rows each), using the appropriate KV head
    via GQA (q_head -> kv_head = q_head // (Q_HEAD // KV_HEAD)).
    """
    pid_m = tl.program_id(0)        # which Q-tile along seq dim
    q_head_idx = tl.program_id(1)   # which Q head
    kv_head_idx = q_head_idx // (Q_HEAD // KV_HEAD)

    # Ranges
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M] Q row indices
    offs_n = tl.arange(0, BLOCK_N)                      # [BLOCK_N] KV row indices
    offs_d = tl.arange(0, HEAD_DIM)                     # [HEAD_DIM] feature dim
    qm_mask = offs_m < seq_len

    # ── Load Q tile [BLOCK_M, HEAD_DIM] ──
    q_base = q_ptr + q_head_idx * stride_q_head
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_seq + offs_d[None, :],
        mask=qm_mask[:, None], other=0.0,
    )  # [BLOCK_M, HEAD_DIM]

    # ── Online softmax state ──
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ── Pointer bases for this kv_head ──
    k_base = k_ptr + kv_head_idx * stride_k_head
    v_base = v_ptr + kv_head_idx * stride_v_head

    # ── Loop over KV blocks ──
    n_blocks = tl.cdiv(seq_len, BLOCK_N)
    for n_start in range(0, n_blocks * BLOCK_N, BLOCK_N):
        kn_mask = offs_n < (seq_len - n_start)

        # Load K [BLOCK_N, HEAD_DIM]
        k = tl.load(
            k_base + (n_start + offs_n[:, None]) * stride_k_seq + offs_d[None, :],
            mask=kn_mask[:, None], other=0.0,
        )
        # Compute S = Q @ K^T  [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * softmax_scale
        s = tl.where(kn_mask[None, :], s, float("-inf"))

        # Online softmax update
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

    # ── Final renormalisation ──
    acc = acc / l_i[:, None]

    # ── Write output [BLOCK_M, HEAD_DIM] ──
    o_base = o_ptr + q_head_idx * stride_o_head
    tl.store(
        o_base + offs_m[:, None] * stride_o_seq + offs_d[None, :],
        acc.to(q.dtype), mask=qm_mask[:, None],
    )


def invoke_flash_attention(
    q: torch.Tensor,   # [seq, q_head, dim]  seq-first contiguous
    k: torch.Tensor,   # [seq, kv_head, dim]
    v: torch.Tensor,   # [seq, kv_head, dim]
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """
    FlashAttention V2 with GQA, seq-first layout.

    Returns out: [seq, q_head * dim] contiguous, seq-first, ready for o_proj.
    """
    seq_len, q_head, head_dim = q.shape
    _, kv_head, _ = k.shape
    assert head_dim == v.shape[2] and kv_head == k.shape[1]

    scale = softmax_scale if softmax_scale is not None else (head_dim ** -0.5)
    compute_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16

    o = torch.empty(seq_len, q_head, head_dim, dtype=q.dtype, device=q.device)

    BLOCK_M = 64
    BLOCK_N = 64

    assert head_dim <= 128 and head_dim % 16 == 0, "head_dim for dot compatibility"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    grid = (triton.cdiv(seq_len, BLOCK_M), q_head)

    _flash_attn_kernel[grid](
        q, k, v, o,
        seq_len,
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
    )

    # Flatten: [seq, q_head, dim] → [seq, q_head * dim] = [seq, embed_dim]
    return o.reshape(seq_len, q_head * head_dim)
