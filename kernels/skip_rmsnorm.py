"""
SkipRMSNorm: fused residual-add + RMSNorm in one Triton kernel.

Replaces:  x = x + residual  (elementwise add, 1 kernel)
           out = rms_norm(x)  (RMSNorm, 1 kernel)
with a single launch that computes both and updates x in-place.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _skip_rmsnorm_kernel(
    x_ptr,          # [seq_len, N]  updated in-place
    residual_ptr,   # [seq_len, N]
    out_ptr,        # [seq_len, N]  norm(x + residual)
    gamma_ptr,      # [N]
    N: tl.constexpr,
    stride: tl.constexpr,   # N (contiguous rows)
    eps: tl.constexpr,
):
    """Grid: (seq_len,).  One program per token row."""
    row = tl.program_id(0)
    offs = tl.arange(0, N)

    xv = tl.load(x_ptr + row * stride + offs).to(tl.float32)
    rv = tl.load(residual_ptr + row * stride + offs).to(tl.float32)
    ss = xv + rv                      # residual sum

    var = tl.sum(ss * ss, axis=0) / N
    rms = tl.sqrt(var + eps)
    g   = tl.load(gamma_ptr + offs).to(tl.float32)
    out = ss / rms * g

    tl.store(x_ptr + row * stride + offs, ss.to(tl.float16))
    tl.store(out_ptr + row * stride + offs, out.to(tl.float16))


def invoke_skip_rmsnorm(
    x: torch.Tensor,          # [seq_len, N]  modified in-place
    residual: torch.Tensor,   # [seq_len, N]
    gamma: torch.Tensor,      # [N]
    eps: float = 1e-6,
) -> torch.Tensor:
    """Returns norm(x + residual).  x is updated to x + residual in-place."""
    seq_len, N = x.shape
    assert x.is_contiguous() and residual.is_contiguous()
    out = torch.empty(seq_len, N, dtype=x.dtype, device=x.device)

    grid = (seq_len,)
    _skip_rmsnorm_kernel[grid](
        x, residual, out, gamma,
        N=N, stride=x.stride(0), eps=eps,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone RMSNorm (no residual add)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _rmsnorm_kernel(
    x_ptr,          # [seq_len, N]
    out_ptr,        # [seq_len, N]
    gamma_ptr,      # [N]
    N: tl.constexpr,
    stride: tl.constexpr,
    eps: tl.constexpr,
):
    """Grid: (seq_len,). One program per token row."""
    row = tl.program_id(0)
    offs = tl.arange(0, N)

    xv = tl.load(x_ptr + row * stride + offs).to(tl.float32)
    var = tl.sum(xv * xv, axis=0) / N
    rms = tl.sqrt(var + eps)
    g = tl.load(gamma_ptr + offs).to(tl.float32)
    out = xv / rms * g

    tl.store(out_ptr + row * stride + offs, out.to(tl.float16))


def invoke_rmsnorm(
    x: torch.Tensor,      # [seq_len, N]
    gamma: torch.Tensor,  # [N]
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm: single Triton kernel instead of pow+mean+rsqrt+mul."""
    seq_len, N = x.shape
    assert x.is_contiguous()
    out = torch.empty(seq_len, N, dtype=x.dtype, device=x.device)

    grid = (seq_len,)
    _rmsnorm_kernel[grid](
        x, out, gamma,
        N=N, stride=x.stride(0), eps=eps,
    )
    return out
