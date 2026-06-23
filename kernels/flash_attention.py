"""
FlashAttention kernels: Prefill (varlen) + Decode (FlashDecoding split-KV).

Prefill: Q has multiple tokens per seq, compute-bound. Uses varlen kernel with BLOCK_M=64.
Decode:  Q=1 per seq, memory-bound. Uses FlashDecoding: split KV into segments,
         compute partial attention in parallel, then reduce.
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
# Prefill Kernel (varlen, zero-copy from cache)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _flash_attn_prefill_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
    cu_seqlens_q_ptr, ctx_lens_ptr, tile_seq_ids_ptr,
    total_q_len,
    stride_q_seq, stride_q_head,
    stride_kc_seq, stride_kc_pos, stride_kc_head,
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
    """Varlen prefill: multiple Q tokens per seq. Grid: (num_q_tiles, Q_HEAD)."""
    pid_m = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // (Q_HEAD // KV_HEAD)

    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    qm_mask = offs_m < total_q_len

    seq_idx = tl.load(tile_seq_ids_ptr + pid_m).to(tl.int32)
    seq_q_start = tl.load(cu_seqlens_q_ptr + seq_idx).to(tl.int32)
    kv_seq_len = tl.load(ctx_lens_ptr + seq_idx).to(tl.int32)

    q_base = q_ptr + q_head_idx * stride_q_head
    q = tl.load(q_base + offs_m[:, None] * stride_q_seq + offs_d[None, :],
                mask=qm_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    k_base = k_cache_ptr + seq_idx * stride_kc_seq + kv_head_idx * stride_kc_head
    v_base = v_cache_ptr + seq_idx * stride_vc_seq + kv_head_idx * stride_vc_head

    n_blocks = tl.cdiv(kv_seq_len, BLOCK_N)
    for n_start in range(0, n_blocks * BLOCK_N, BLOCK_N):
        kn_mask = offs_n < (kv_seq_len - n_start)
        k = tl.load(k_base + (n_start + offs_n[:, None]) * stride_kc_pos + offs_d[None, :],
                    mask=kn_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * softmax_scale
        s = tl.where(kn_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + (n_start + offs_n[:, None]) * stride_vc_pos + offs_d[None, :],
                    mask=kn_mask[:, None], other=0.0)
        acc += tl.dot(p.to(COMPUTE_DTYPE), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_base = o_ptr + q_head_idx * stride_o_head
    tl.store(o_base + offs_m[:, None] * stride_o_seq + offs_d[None, :],
             acc.to(q.dtype), mask=qm_mask[:, None])


# ═══════════════════════════════════════════════════════════════════════════════
# FlashDecoding Decode Kernel (split-KV + reduce)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _flash_decode_split_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr,
    partial_o_ptr, partial_lse_ptr,
    ctx_lens_ptr,
    stride_q_batch, stride_q_head,
    stride_kc_seq, stride_kc_pos, stride_kc_head,
    stride_vc_seq, stride_vc_pos, stride_vc_head,
    stride_po_batch, stride_po_head, stride_po_split,
    stride_plse_batch, stride_plse_head,
    softmax_scale: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    Q_HEAD: tl.constexpr,
    KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    """
    Split kernel: each program handles one (head, batch, split) triple.
    Grid: (Q_HEAD, batch_size, num_splits)
    """
    q_head_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    kv_head_idx = q_head_idx // (Q_HEAD // KV_HEAD)

    kv_seq_len = tl.load(ctx_lens_ptr + batch_idx).to(tl.int32)
    kv_start = split_idx * SPLIT_SIZE
    kv_end = tl.minimum(kv_start + SPLIT_SIZE, kv_seq_len)

    # Early exit if this split has no work
    if kv_start >= kv_seq_len:
        # Write zeros so reduce doesn't read garbage
        offs_d = tl.arange(0, HEAD_DIM)
        po_base = partial_o_ptr + batch_idx * stride_po_batch + q_head_idx * stride_po_head + split_idx * stride_po_split
        tl.store(po_base + offs_d, tl.zeros([HEAD_DIM], dtype=tl.float16))
        plse_base = partial_lse_ptr + batch_idx * stride_plse_batch + q_head_idx * stride_plse_head
        tl.store(plse_base + split_idx, float("-inf"))
        return

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q [HEAD_DIM]
    q = tl.load(q_ptr + batch_idx * stride_q_batch + q_head_idx * stride_q_head + offs_d)

    # Online softmax over [kv_start, kv_end)
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    k_base = k_cache_ptr + batch_idx * stride_kc_seq + kv_head_idx * stride_kc_head
    v_base = v_cache_ptr + batch_idx * stride_vc_seq + kv_head_idx * stride_vc_head

    local_len = kv_end - kv_start
    n_blocks = tl.cdiv(local_len, BLOCK_N)
    for block_start in range(0, n_blocks * BLOCK_N, BLOCK_N):
        abs_pos = kv_start + block_start
        kn_mask = offs_n < (local_len - block_start)

        k = tl.load(k_base + (abs_pos + offs_n[:, None]) * stride_kc_pos + offs_d[None, :],
                    mask=kn_mask[:, None], other=0.0)
        s = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        s = tl.where(kn_mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0, keep_dims=True))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - tl.broadcast_to(m_new, [BLOCK_N]))
        l_i = alpha * l_i + tl.sum(p, axis=0, keep_dims=True)
        acc = acc * tl.broadcast_to(alpha, [HEAD_DIM])

        v = tl.load(v_base + (abs_pos + offs_n[:, None]) * stride_vc_pos + offs_d[None, :],
                    mask=kn_mask[:, None], other=0.0)
        acc += tl.sum(p[:, None].to(COMPUTE_DTYPE) * v, axis=0)
        m_i = m_new

    # Normalize acc before storing (so reduce can use lse-weighted merge correctly)
    acc = acc / tl.broadcast_to(l_i, [HEAD_DIM])

    # Store partial results (normalized)
    po_base = partial_o_ptr + batch_idx * stride_po_batch + q_head_idx * stride_po_head + split_idx * stride_po_split
    tl.store(po_base + offs_d, acc.to(tl.float16))

    # Store log-sum-exp: lse = m + log(l)
    lse_val = tl.sum(m_i, axis=0) + tl.log(tl.sum(l_i, axis=0))
    plse_base = partial_lse_ptr + batch_idx * stride_plse_batch + q_head_idx * stride_plse_head
    tl.store(plse_base + split_idx, lse_val)


@triton.jit
def _flash_decode_reduce_kernel(
    partial_o_ptr, partial_lse_ptr, out_ptr,
    stride_po_batch, stride_po_head, stride_po_split,
    stride_plse_batch, stride_plse_head,
    stride_o_batch, stride_o_head,
    Q_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """
    Reduce kernel: merge partial results from all splits.
    Grid: (Q_HEAD, batch_size)
    """
    q_head_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)

    # Load all partial LSEs for this (batch, head)
    plse_base = partial_lse_ptr + batch_idx * stride_plse_batch + q_head_idx * stride_plse_head
    
    # Find global max across splits
    global_m = tl.full([1], float("-inf"), dtype=tl.float32)
    for si in range(NUM_SPLITS):
        lse_i = tl.load(plse_base + si)
        global_m = tl.maximum(global_m, lse_i)

    # Weighted sum of partial outputs
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    total_w = tl.zeros([1], dtype=tl.float32)

    po_base = partial_o_ptr + batch_idx * stride_po_batch + q_head_idx * stride_po_head
    for si in range(NUM_SPLITS):
        lse_i = tl.load(plse_base + si)
        w_i = tl.exp(lse_i - tl.broadcast_to(global_m, [1]))
        partial = tl.load(po_base + si * stride_po_split + offs_d).to(tl.float32)
        acc += tl.broadcast_to(w_i, [HEAD_DIM]) * partial
        total_w += w_i

    acc = acc / tl.broadcast_to(total_w, [HEAD_DIM])

    # Write final output
    o_base = out_ptr + batch_idx * stride_o_batch + q_head_idx * stride_o_head
    tl.store(o_base + offs_d, acc.to(COMPUTE_DTYPE))


# ═══════════════════════════════════════════════════════════════════════════════
# Python wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def _invoke_flash_prefill(q, k_cache, v_cache, cu_seqlens_q, ctx_lens, softmax_scale):
    """Prefill path: varlen kernel."""
    total_q_len, q_head, head_dim = q.shape
    kv_head = k_cache.shape[2]
    compute_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16

    o = torch.empty(total_q_len, q_head, head_dim, dtype=q.dtype, device=q.device)
    BLOCK_M, BLOCK_N = 128, 128
    num_tiles = triton.cdiv(total_q_len, BLOCK_M)

    tile_seq_ids = torch.searchsorted(
        cu_seqlens_q[1:], torch.arange(num_tiles, device=q.device) * BLOCK_M, right=True
    ).to(torch.int32)

    assert q.is_contiguous()
    _flash_attn_prefill_kernel[(num_tiles, q_head)](
        q, k_cache, v_cache, o,
        cu_seqlens_q, ctx_lens, tile_seq_ids,
        total_q_len,
        stride_q_seq=q.stride(0), stride_q_head=q.stride(1),
        stride_kc_seq=k_cache.stride(0), stride_kc_pos=k_cache.stride(1), stride_kc_head=k_cache.stride(2),
        stride_vc_seq=v_cache.stride(0), stride_vc_pos=v_cache.stride(1), stride_vc_head=v_cache.stride(2),
        stride_o_seq=o.stride(0), stride_o_head=o.stride(1),
        softmax_scale=softmax_scale,
        COMPUTE_DTYPE=compute_dtype,
        Q_HEAD=q_head, KV_HEAD=kv_head, HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return o.reshape(total_q_len, q_head * head_dim)


def _invoke_flash_decode(q, k_cache, v_cache, ctx_lens, softmax_scale):
    """Decode path: FlashDecoding with split-KV + reduce."""
    batch_size, q_head, head_dim = q.shape
    kv_head = k_cache.shape[2]
    compute_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16

    max_kv = ctx_lens.max().item()
    NUM_SPLITS = max(1, triton.cdiv(max_kv, 256))
    SPLIT_SIZE = triton.cdiv(max_kv, NUM_SPLITS)
    BLOCK_N = 128

    # Allocate partial buffers
    partial_o = torch.empty(batch_size, q_head, NUM_SPLITS, head_dim, dtype=q.dtype, device=q.device)
    partial_lse = torch.full((batch_size, q_head, NUM_SPLITS), float('-inf'), dtype=torch.float32, device=q.device)

    assert q.is_contiguous()

    # Phase 1: Split kernel
    _flash_decode_split_kernel[(q_head, batch_size, NUM_SPLITS)](
        q, k_cache, v_cache,
        partial_o, partial_lse,
        ctx_lens,
        stride_q_batch=q.stride(0), stride_q_head=q.stride(1),
        stride_kc_seq=k_cache.stride(0), stride_kc_pos=k_cache.stride(1), stride_kc_head=k_cache.stride(2),
        stride_vc_seq=v_cache.stride(0), stride_vc_pos=v_cache.stride(1), stride_vc_head=v_cache.stride(2),
        stride_po_batch=partial_o.stride(0), stride_po_head=partial_o.stride(1), stride_po_split=partial_o.stride(2),
        stride_plse_batch=partial_lse.stride(0), stride_plse_head=partial_lse.stride(1),
        softmax_scale=softmax_scale,
        COMPUTE_DTYPE=compute_dtype,
        Q_HEAD=q_head, KV_HEAD=kv_head, HEAD_DIM=head_dim,
        BLOCK_N=BLOCK_N, SPLIT_SIZE=SPLIT_SIZE,
    )

    # Phase 2: Reduce kernel
    o = torch.empty(batch_size, q_head, head_dim, dtype=q.dtype, device=q.device)
    _flash_decode_reduce_kernel[(q_head, batch_size)](
        partial_o, partial_lse, o,
        stride_po_batch=partial_o.stride(0), stride_po_head=partial_o.stride(1), stride_po_split=partial_o.stride(2),
        stride_plse_batch=partial_lse.stride(0), stride_plse_head=partial_lse.stride(1),
        stride_o_batch=o.stride(0), stride_o_head=o.stride(1),
        Q_HEAD=q_head, HEAD_DIM=head_dim, NUM_SPLITS=NUM_SPLITS,
        COMPUTE_DTYPE=compute_dtype,
    )
    return o.reshape(batch_size, q_head * head_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified entry point (routes internally)
# ═══════════════════════════════════════════════════════════════════════════════

def invoke_flash_attention(
    q: torch.Tensor,            # [total_q, q_head, dim]
    k_cache: torch.Tensor,      # [num_seqs, max_seq_len, kv_head, dim]
    v_cache: torch.Tensor,      # [num_seqs, max_seq_len, kv_head, dim]
    cu_seqlens_q: torch.Tensor, # [num_seqs+1] int32
    ctx_lens: torch.Tensor,     # [num_seqs] int32
    num_decode_seqs: int = 0,   # split index: seqs[:num_decode] are decode, seqs[num_decode:] are prefill
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """
    Unified attention with index-based routing.
    Batch layout: [decode_seqs | prefill_seqs]
    num_decode_seqs splits the two regions.
    Returns: [total_q, q_head * dim]
    """
    head_dim = q.shape[2]
    scale = softmax_scale if softmax_scale is not None else (head_dim ** -0.5)
    num_seqs = cu_seqlens_q.shape[0] - 1

    outputs = []

    # Decode part: seqs [0, num_decode_seqs)
    if num_decode_seqs > 0:
        q_end = cu_seqlens_q[num_decode_seqs].item()
        q_decode = q[:q_end]  # [num_decode_seqs, q_head, dim] since each has 1 token
        k_decode = k_cache[:num_decode_seqs]
        v_decode = v_cache[:num_decode_seqs]
        ctx_decode = ctx_lens[:num_decode_seqs]
        out_decode = _invoke_flash_decode(q_decode, k_decode, v_decode, ctx_decode, scale)
        outputs.append(out_decode)

    # Prefill part: seqs [num_decode_seqs, num_seqs)
    num_prefill_seqs = num_seqs - num_decode_seqs
    if num_prefill_seqs > 0:
        q_start = cu_seqlens_q[num_decode_seqs].item()
        q_prefill = q[q_start:]  # remaining Q tokens
        k_prefill = k_cache[num_decode_seqs:num_seqs]
        v_prefill = v_cache[num_decode_seqs:num_seqs]
        ctx_prefill = ctx_lens[num_decode_seqs:num_seqs]
        # Rebuild cu_seqlens_q for prefill subset (re-base to 0)
        cu_prefill = cu_seqlens_q[num_decode_seqs:] - cu_seqlens_q[num_decode_seqs]
        out_prefill = _invoke_flash_prefill(q_prefill, k_prefill, v_prefill, cu_prefill, ctx_prefill, scale)
        outputs.append(out_prefill)

    return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
