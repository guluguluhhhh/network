"""
Kernel 1: Attention Data Preparation.

Reads Q/K/V directly from the fused qkv GEMM output (non-contiguous views),
applies RMSNorm + partial RoPE to Q and K, copies V straight through,
and writes everything to contiguous seq-first output tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_data_prep_kernel(
    qkv_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    kv_write_offsets_ptr,
    gamma_q_ptr, gamma_k_ptr,
    cos_ptr, sin_ptr,
    seq_len,
    eps,
    stride_qkv_seq,
    stride_cos_seq,
    stride_qo_seq, stride_qo_head,
    stride_kc_head,   # cache K/V head stride (shared for K and V)
    Q_HEAD: tl.constexpr,
    KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    Q_OFFSET: tl.constexpr,
    K_OFFSET: tl.constexpr,
    V_OFFSET: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    ROPE_OFF: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_h   = tl.program_id(1)
    if pid_seq >= seq_len:
        return

    offs_d = tl.arange(0, HEAD_DIM)
    rp      = tl.arange(0, HALF_ROPE)

    is_q = pid_h < Q_HEAD
    is_k = (not is_q) and (pid_h < Q_HEAD + KV_HEAD)
    # else is_v

    row_base = qkv_ptr + pid_seq * stride_qkv_seq
    if is_q:
        src = row_base + Q_OFFSET + pid_h * HEAD_DIM
    elif is_k:
        src = row_base + K_OFFSET + (pid_h - Q_HEAD) * HEAD_DIM
    else:
        src = row_base + V_OFFSET + (pid_h - Q_HEAD - KV_HEAD) * HEAD_DIM

    if is_q:
        dst = q_out_ptr + pid_seq * stride_qo_seq + pid_h * stride_qo_head
    elif is_k:
        # Write directly to cache: base + per-token offset + head offset
        write_off = tl.load(kv_write_offsets_ptr + pid_seq).to(tl.int64)
        dst = k_cache_ptr + write_off + (pid_h - Q_HEAD) * stride_kc_head
    else:
        write_off = tl.load(kv_write_offsets_ptr + pid_seq).to(tl.int64)
        dst = v_cache_ptr + write_off + (pid_h - Q_HEAD - KV_HEAD) * stride_kc_head

    x_all = tl.load(src + offs_d).to(tl.float32)

    if is_q or is_k:
        gamma_ptr = gamma_q_ptr if is_q else gamma_k_ptr
        cos_base  = cos_ptr + pid_seq * stride_cos_seq
        sin_base  = sin_ptr + pid_seq * stride_cos_seq

        g_all = tl.load(gamma_ptr + offs_d).to(tl.float32)
        g_x1  = tl.load(gamma_ptr + ROPE_OFF + rp).to(tl.float32)
        g_x2  = tl.load(gamma_ptr + ROPE_OFF + HALF_ROPE + rp).to(tl.float32)
        c = tl.load(cos_base + rp).to(tl.float32)
        s = tl.load(sin_base + rp).to(tl.float32)

        rms_inv = tl.rsqrt(tl.sum(x_all * x_all) / HEAD_DIM + eps)
        pref = x_all * rms_inv * g_all
        x1 = tl.load(src + ROPE_OFF + rp).to(tl.float32) * rms_inv * g_x1
        x2 = tl.load(src + ROPE_OFF + HALF_ROPE + rp).to(tl.float32) * rms_inv * g_x2

        tl.store(dst + offs_d, pref.to(COMPUTE_DTYPE), mask=offs_d < ROPE_OFF)
        tl.store(dst + ROPE_OFF + rp,             (x1 * c - x2 * s).to(COMPUTE_DTYPE))
        tl.store(dst + ROPE_OFF + HALF_ROPE + rp, (x2 * c + x1 * s).to(COMPUTE_DTYPE))
    else:
        tl.store(dst + offs_d, x_all.to(COMPUTE_DTYPE))


def invoke_attn_data_prep(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    q_head: int,
    kv_head: int,
    head_dim: int,
    k_cache: torch.Tensor,           # [num_seqs, max_seq_len, kv_head, head_dim]
    v_cache: torch.Tensor,           # same shape
    kv_write_offsets: torch.Tensor,  # [num_tokens] int64 - per-token byte offset into cache
    eps: float = 1e-6,
) -> torch.Tensor:
    """Returns Q only. K/V are written directly to k_cache/v_cache at given offsets."""
    seq_len = qkv.shape[0]
    qkv_dim = qkv.shape[1]
    dev, dt = qkv.device, qkv.dtype
    rope_dim  = cos.shape[1]
    half_rope = rope_dim // 2
    rope_off  = head_dim - rope_dim
    compute_dt = tl.float16 if dt == torch.float16 else tl.bfloat16

    q_out = torch.empty(seq_len, q_head, head_dim, dtype=dt, device=dev)

    assert qkv.is_contiguous()
    assert q_head + 2 * kv_head == qkv_dim // head_dim

    # Cache stride: position dim → bytes; head dim → bytes
    # k_cache layout: [num_seqs, max_seq_len, kv_head, head_dim]
    # stride_kc_head = stride along kv_head dim (in elements, Triton handles bytes)
    stride_kc_head = k_cache.stride(2)

    grid = (seq_len, q_head + 2 * kv_head)
    _attn_data_prep_kernel[grid](
        qkv, q_out, k_cache, v_cache,
        kv_write_offsets,
        gamma_q, gamma_k, cos, sin,
        seq_len, eps,
        stride_qkv_seq=qkv.stride(0),
        stride_cos_seq=cos.stride(0),
        stride_qo_seq=q_out.stride(0), stride_qo_head=q_out.stride(1),
        stride_kc_head=stride_kc_head,
        Q_HEAD=q_head,
        KV_HEAD=kv_head,
        HEAD_DIM=head_dim,
        QKV_DIM=qkv_dim,
        Q_OFFSET=0,
        K_OFFSET=q_head * head_dim,
        V_OFFSET=q_head * head_dim + kv_head * head_dim,
        ROPE_DIM=rope_dim,
        ROPE_OFF=rope_off,
        HALF_ROPE=half_rope,
        COMPUTE_DTYPE=compute_dt,
    )
    return q_out
