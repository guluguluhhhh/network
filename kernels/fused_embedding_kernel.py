"""
Fused embedding + LayerNorm triton kernel.

Combines token embedding lookup, position embedding lookup, element-wise
add, and LayerNorm into a single GPU kernel to eliminate CPU dispatch
overhead and kernel launch gaps.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_embedding_layernorm_kernel(
    # Output
    out_ptr,
    # Embedding weight tensors
    token_emb_ptr,       # [vocab_size, embed_dim]
    pos_emb_ptr,         # [max_seq_len, embed_dim]
    ln_weight_ptr,       # [embed_dim]
    ln_bias_ptr,         # [embed_dim] or dummy
    # Input
    input_ids_ptr,       # [seq_len] int64
    # Dimensions (constexpr for efficient indexing)
    embed_dim: tl.constexpr,
    # Strides
    stride_token_emb_0,
    stride_pos_emb_0,
    stride_out_0,
    # Options
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Each program processes one token: gather embeddings, add, then LayerNorm.

    Two-pass algorithm:
      1. Gather token_emb[input_ids[pid]] + pos_emb[pid], accumulate sum/sum_sq
      2. Normalize with precomputed mean/var, apply affine, write output
    """
    pid = tl.program_id(0)

    token_id = tl.load(input_ids_ptr + pid).to(tl.int64)

    # Precompute offset ranges used by both passes
    offs = tl.arange(0, BLOCK_SIZE)

    # ---- Pass 1: compute mean & variance in fp32 ----
    sum_val = 0.0
    sum_sq = 0.0

    for block_start in range(0, embed_dim, BLOCK_SIZE):
        offsets = block_start + offs
        mask = offsets < embed_dim

        t_val = tl.load(
            token_emb_ptr + token_id * stride_token_emb_0 + offsets,
            mask=mask, other=0.0,
        )
        p_val = tl.load(
            pos_emb_ptr + pid * stride_pos_emb_0 + offsets,
            mask=mask, other=0.0,
        )

        val = (t_val + p_val).to(tl.float32)
        sum_val += tl.sum(val, axis=0)
        sum_sq += tl.sum(val * val, axis=0)

    mean = sum_val / embed_dim
    var = sum_sq / embed_dim - mean * mean
    inv_std = tl.math.rsqrt(var + EPS)

    # ---- Pass 2: normalize & write ----
    for block_start in range(0, embed_dim, BLOCK_SIZE):
        offsets = block_start + offs
        mask = offsets < embed_dim

        t_val = tl.load(
            token_emb_ptr + token_id * stride_token_emb_0 + offsets,
            mask=mask, other=0.0,
        )
        p_val = tl.load(
            pos_emb_ptr + pid * stride_pos_emb_0 + offsets,
            mask=mask, other=0.0,
        )

        val = (t_val + p_val).to(tl.float32)
        normalized = (val - mean) * inv_std

        w = tl.load(ln_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(ln_bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            result = normalized * w + b
        else:
            result = normalized * w

        # Cast back to original dtype (fp16/bf16) before store
        result = result.to(t_val.dtype)

        tl.store(
            out_ptr + pid * stride_out_0 + offsets,
            result, mask=mask,
        )


def invoke_fused_embedding_layernorm(
    input_ids: torch.Tensor,        # [seq_len] int64
    token_emb_weight: torch.Tensor, # [vocab_size, embed_dim]
    pos_emb_weight: torch.Tensor,   # [max_seq_len, embed_dim]
    ln_weight: torch.Tensor,        # [embed_dim]
    ln_bias: torch.Tensor | None,   # [embed_dim] or None
    output: torch.Tensor,           # [seq_len, embed_dim] pre-allocated
    eps: float = 1e-5,
) -> None:
    """
    Launch the fused embedding + LayerNorm kernel.

    Args:
        input_ids: token IDs of shape [seq_len], dtype int64.
        token_emb_weight: embedding table [vocab_size, embed_dim].
        pos_emb_weight: position embedding table [max_seq_len, embed_dim].
        ln_weight: LayerNorm weight (gamma) [embed_dim].
        ln_bias: LayerNorm bias (beta) [embed_dim], or None.
        output: pre-allocated output tensor [seq_len, embed_dim].
        eps: epsilon for LayerNorm numerical stability.
    """
    seq_len = input_ids.shape[0]
    embed_dim = token_emb_weight.shape[1]
    has_bias = ln_bias is not None

    # Block size: balance between register pressure and loop iterations.
    # embed_dim=512 with BLOCK=128 → 4 iterations per pass, good balance.
    BLOCK_SIZE = 128

    grid = (seq_len,)

    fused_embedding_layernorm_kernel[grid](
        output,
        token_emb_weight,
        pos_emb_weight,
        ln_weight,
        ln_bias if has_bias else ln_weight,  # dummy pointer, never accessed
        input_ids,
        embed_dim=embed_dim,
        stride_token_emb_0=token_emb_weight.stride(0),
        stride_pos_emb_0=pos_emb_weight.stride(0),
        stride_out_0=output.stride(0),
        HAS_BIAS=has_bias,
        BLOCK_SIZE=BLOCK_SIZE,
        EPS=eps,
    )
