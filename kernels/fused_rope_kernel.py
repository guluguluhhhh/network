"""
Fused RoPE (Rotary Position Embedding) Triton kernel.

Applies partial RoPE to Q and K tensors in a single kernel launch,
eliminating ~10 fragmented PyTorch kernels (slice, cat, mul, add).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rope_kernel(
    # Q tensor: [seq_len, num_heads_q, head_dim]
    q_ptr,
    # K tensor: [seq_len, num_heads_k, head_dim]
    k_ptr,
    # Precomputed cos/sin: [seq_len, rope_dim]
    cos_ptr,
    sin_ptr,
    # Dimensions
    seq_len,
    num_heads_q: tl.constexpr,
    num_heads_k: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    # Strides for Q [seq_len, num_heads_q, head_dim]
    stride_q_seq,
    stride_q_head,
    # Strides for K [seq_len, num_heads_k, head_dim]
    stride_k_seq,
    stride_k_head,
    # Stride for cos/sin [seq_len, rope_dim]
    stride_cos_seq,
    # constexpr
    HALF_ROPE: tl.constexpr,  # rope_dim // 2
):
    """
    Each program handles one (position, head) pair.
    Grid: (seq_len, num_heads_q + num_heads_k)

    For each position p and head h:
      - Loads the last rope_dim elements of x[p, h, :]
      - Splits into x1 (first half) and x2 (second half)
      - Computes: out1 = x1 * cos - x2 * sin
                  out2 = x2 * cos + x1 * sin
      - Writes back in-place (prefix dimensions untouched)
    """
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)

    if pid_seq >= seq_len:
        return

    # Determine if this is a Q head or K head
    is_q = pid_head < num_heads_q
    head_idx = pid_head if is_q else (pid_head - num_heads_q)

    # Base pointer for this (seq, head)
    if is_q:
        base = q_ptr + pid_seq * stride_q_seq + head_idx * stride_q_head
    else:
        base = k_ptr + pid_seq * stride_k_seq + head_idx * stride_k_head

    # Offset to the rope suffix (last rope_dim elements)
    rope_offset = head_dim - rope_dim

    # Load cos/sin for this position
    cos_base = cos_ptr + pid_seq * stride_cos_seq
    sin_base = sin_ptr + pid_seq * stride_cos_seq

    # Process rope_dim elements in one shot (rope_dim=32, fits in registers)
    offs = tl.arange(0, HALF_ROPE)

    # Load x1 (first half of suffix) and x2 (second half of suffix)
    x1 = tl.load(base + rope_offset + offs)
    x2 = tl.load(base + rope_offset + HALF_ROPE + offs)

    # Load cos/sin (they are duplicated: cos[i] == cos[i + HALF_ROPE])
    c = tl.load(cos_base + offs)
    s = tl.load(sin_base + offs)

    # Rotary transform
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s

    # Store back in-place
    tl.store(base + rope_offset + offs, out1)
    tl.store(base + rope_offset + HALF_ROPE + offs, out2)


def invoke_fused_rope(
    q: torch.Tensor,          # [seq_len, num_heads_q, head_dim] (modified in-place)
    k: torch.Tensor,          # [seq_len, num_heads_k, head_dim] (modified in-place)
    cos: torch.Tensor,        # [seq_len, rope_dim]
    sin: torch.Tensor,        # [seq_len, rope_dim]
) -> None:
    """
    Apply partial RoPE to Q and K in-place using a single Triton kernel launch.

    Rotates only the last rope_dim dimensions of each head; prefix is untouched.
    """
    seq_len, num_heads_q, head_dim = q.shape
    _, num_heads_k, _ = k.shape
    rope_dim = cos.shape[1]

    assert rope_dim % 2 == 0
    assert q.is_contiguous() and k.is_contiguous()

    grid = (seq_len, num_heads_q + num_heads_k)

    fused_rope_kernel[grid](
        q, k, cos, sin,
        seq_len,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        rope_dim=rope_dim,
        stride_q_seq=q.stride(0),
        stride_q_head=q.stride(1),
        stride_k_seq=k.stride(0),
        stride_k_head=k.stride(1),
        stride_cos_seq=cos.stride(0),
        HALF_ROPE=rope_dim // 2,
    )
