"""
FlashAttention kernels: Prefill (varlen) + Decode (FlashDecoding split-KV).
Paged KV cache with block_tables addressing (FlashInfer-compatible).

Prefill: Q has multiple tokens per seq, compute-bound. Uses varlen kernel with BLOCK_M=128.
Decode:  Q=1 per seq, memory-bound. Uses FlashDecoding: split KV into segments,
         compute partial attention in parallel, then reduce.

KV access: pool[block_tables[seq, page_i], pos_in_page, head, dim]
page_size = BLOCK_N = 128 (kernel tile aligns with physical page).
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
# Prefill Kernel (varlen, paged KV cache)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _flash_attn_prefill_kernel(
    q_ptr, k_pool_ptr, v_pool_ptr, o_ptr,
    block_tables_ptr,
    cu_seqlens_q_ptr, ctx_lens_ptr, tile_seq_ids_ptr,
    total_q_len,
    stride_q_seq, stride_q_head,
    page_stride, stride_kc_pos, stride_kc_head,
    stride_vc_pos, stride_vc_head,
    stride_o_seq, stride_o_head,
    max_blocks_per_seq,
    softmax_scale: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    Q_HEAD: tl.constexpr,
    KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Varlen prefill with paged KV. Grid: (num_q_tiles, Q_HEAD).
    Each BLOCK_N iteration reads exactly one physical page (page_size = BLOCK_N).
    """
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

    # Block table row for this seq
    bt_row = block_tables_ptr + seq_idx * max_blocks_per_seq

    n_pages = tl.cdiv(kv_seq_len, BLOCK_N)
    for page_i in range(0, n_pages):
        # Load physical page ID from block_tables
        phys_page = tl.load(bt_row + page_i).to(tl.int64)
        k_base = k_pool_ptr + phys_page * page_stride + kv_head_idx * stride_kc_head
        v_base = v_pool_ptr + phys_page * page_stride + kv_head_idx * stride_vc_head

        # Mask for last page (may be partially filled)
        page_start = page_i * BLOCK_N
        kn_mask = offs_n < (kv_seq_len - page_start)

        k = tl.load(k_base + offs_n[:, None] * stride_kc_pos + offs_d[None, :],
                    mask=kn_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * softmax_scale
        s = tl.where(kn_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + offs_n[:, None] * stride_vc_pos + offs_d[None, :],
                    mask=kn_mask[:, None], other=0.0)
        acc += tl.dot(p.to(COMPUTE_DTYPE), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_base = o_ptr + q_head_idx * stride_o_head
    tl.store(o_base + offs_m[:, None] * stride_o_seq + offs_d[None, :],
             acc.to(q.dtype), mask=qm_mask[:, None])


# ═══════════════════════════════════════════════════════════════════════════════
# FlashDecoding Decode Kernel (split-KV + reduce, paged KV cache)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _flash_decode_split_kernel(
    q_ptr, k_pool_ptr, v_pool_ptr,
    partial_o_ptr, partial_lse_ptr,
    block_tables_ptr,
    ctx_lens_ptr,
    stride_q_batch, stride_q_head,
    page_stride, stride_kc_pos, stride_kc_head,
    stride_vc_pos, stride_vc_head,
    stride_po_batch, stride_po_head, stride_po_split,
    stride_plse_batch, stride_plse_head,
    max_blocks_per_seq,
    softmax_scale: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    Q_HEAD: tl.constexpr,
    KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    """
    Split kernel with paged KV. Each program handles one (head, batch, split) triple.
    Grid: (Q_HEAD, batch_size, num_splits)
    SPLIT_SIZE must be a multiple of PAGE_SIZE for aligned page iteration.
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

    # Block table row for this seq
    bt_row = block_tables_ptr + batch_idx * max_blocks_per_seq

    # Iterate over pages within this split
    start_page = kv_start // PAGE_SIZE
    end_page = tl.cdiv(kv_end, PAGE_SIZE)

    for page_i in range(start_page, end_page):
        phys_page = tl.load(bt_row + page_i).to(tl.int64)
        k_base = k_pool_ptr + phys_page * page_stride + kv_head_idx * stride_kc_head
        v_base = v_pool_ptr + phys_page * page_stride + kv_head_idx * stride_vc_head

        # Determine valid range within this page
        page_abs_start = page_i * PAGE_SIZE
        local_start = tl.maximum(kv_start - page_abs_start, 0)
        local_end = tl.minimum(kv_end - page_abs_start, PAGE_SIZE)
        kn_mask = (offs_n >= local_start) & (offs_n < local_end)

        k = tl.load(k_base + offs_n[:, None] * stride_kc_pos + offs_d[None, :],
                    mask=kn_mask[:, None], other=0.0)
        s = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        s = tl.where(kn_mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0, keep_dims=True))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - tl.broadcast_to(m_new, [BLOCK_N]))
        l_i = alpha * l_i + tl.sum(p, axis=0, keep_dims=True)
        acc = acc * tl.broadcast_to(alpha, [HEAD_DIM])

        v = tl.load(v_base + offs_n[:, None] * stride_vc_pos + offs_d[None, :],
                    mask=kn_mask[:, None], other=0.0)
        acc += tl.sum(p[:, None].to(COMPUTE_DTYPE) * v, axis=0)
        m_i = m_new

    # Normalize acc before storing
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
    """Reduce kernel: merge partial results from all splits. Grid: (Q_HEAD, batch_size)"""
    q_head_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)

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

def _invoke_flash_prefill(q, k_pool, v_pool, block_tables, cu_seqlens_q, ctx_lens,
                          softmax_scale, tile_seq_ids, page_size=128):
    """Prefill path: varlen kernel with paged KV cache."""
    total_q_len, q_head, head_dim = q.shape
    kv_head = k_pool.shape[2]
    compute_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
    max_blocks_per_seq = block_tables.shape[1]

    o = torch.empty(total_q_len, q_head, head_dim, dtype=q.dtype, device=q.device)
    BLOCK_M, BLOCK_N = 128, page_size
    num_tiles = triton.cdiv(total_q_len, BLOCK_M)

    assert tile_seq_ids is not None and tile_seq_ids.shape[0] == num_tiles
    assert q.is_contiguous()

    _flash_attn_prefill_kernel[(num_tiles, q_head)](
        q, k_pool, v_pool, o,
        block_tables,
        cu_seqlens_q, ctx_lens, tile_seq_ids,
        total_q_len,
        stride_q_seq=q.stride(0), stride_q_head=q.stride(1),
        page_stride=k_pool.stride(0), stride_kc_pos=k_pool.stride(1), stride_kc_head=k_pool.stride(2),
        stride_vc_pos=v_pool.stride(1), stride_vc_head=v_pool.stride(2),
        stride_o_seq=o.stride(0), stride_o_head=o.stride(1),
        max_blocks_per_seq=max_blocks_per_seq,
        softmax_scale=softmax_scale,
        COMPUTE_DTYPE=compute_dtype,
        Q_HEAD=q_head, KV_HEAD=kv_head, HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return o.reshape(total_q_len, q_head * head_dim)


def _invoke_flash_decode(q, k_pool, v_pool, block_tables, ctx_lens,
                         softmax_scale, max_kv_len: int, page_size: int = 128):
    """Decode path: FlashDecoding with paged KV cache."""
    batch_size, q_head, head_dim = q.shape
    kv_head = k_pool.shape[2]
    compute_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
    max_blocks_per_seq = block_tables.shape[1]

    # SPLIT_SIZE must be multiple of page_size for aligned page iteration
    NUM_SPLITS = max(1, triton.cdiv(max_kv_len, page_size * 2))  # each split covers ~2 pages
    SPLIT_SIZE = triton.cdiv(max_kv_len, NUM_SPLITS)
    # Round SPLIT_SIZE up to multiple of page_size
    SPLIT_SIZE = ((SPLIT_SIZE + page_size - 1) // page_size) * page_size
    BLOCK_N = page_size

    # Allocate partial buffers
    partial_o = torch.empty(batch_size, q_head, NUM_SPLITS, head_dim, dtype=q.dtype, device=q.device)
    partial_lse = torch.full((batch_size, q_head, NUM_SPLITS), float('-inf'), dtype=torch.float32, device=q.device)

    assert q.is_contiguous()

    # Phase 1: Split kernel
    _flash_decode_split_kernel[(q_head, batch_size, NUM_SPLITS)](
        q, k_pool, v_pool,
        partial_o, partial_lse,
        block_tables,
        ctx_lens,
        stride_q_batch=q.stride(0), stride_q_head=q.stride(1),
        page_stride=k_pool.stride(0), stride_kc_pos=k_pool.stride(1), stride_kc_head=k_pool.stride(2),
        stride_vc_pos=v_pool.stride(1), stride_vc_head=v_pool.stride(2),
        stride_po_batch=partial_o.stride(0), stride_po_head=partial_o.stride(1), stride_po_split=partial_o.stride(2),
        stride_plse_batch=partial_lse.stride(0), stride_plse_head=partial_lse.stride(1),
        max_blocks_per_seq=max_blocks_per_seq,
        softmax_scale=softmax_scale,
        COMPUTE_DTYPE=compute_dtype,
        Q_HEAD=q_head, KV_HEAD=kv_head, HEAD_DIM=head_dim,
        BLOCK_N=BLOCK_N, SPLIT_SIZE=SPLIT_SIZE, PAGE_SIZE=page_size,
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
    k_pool: torch.Tensor,       # [num_pages, page_size, kv_head, dim]
    v_pool: torch.Tensor,       # same shape
    block_tables: torch.Tensor, # [num_seqs, max_blocks_per_seq] int32
    cu_seqlens_q: torch.Tensor, # [num_seqs+1] int32
    ctx_lens: torch.Tensor,     # [num_seqs] int32
    num_decode_seqs: int = 0,   # split index: seqs[:num_decode] are decode
    num_decode_tokens: int = 0, # total decode Q tokens
    max_kv_len: int = 0,        # max KV length across all seqs
    cu_seqlens_prefill: torch.Tensor | None = None,
    tile_seq_ids_prefill: torch.Tensor | None = None,
    page_size: int = 128,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """
    Unified attention with paged KV cache and index-based routing.
    Batch layout: [decode_seqs | prefill_seqs]
    Returns: [total_q, q_head * dim]
    """
    total_q = q.shape[0]
    head_dim = q.shape[2]
    q_head = q.shape[1]
    scale = softmax_scale if softmax_scale is not None else (head_dim ** -0.5)
    num_seqs = cu_seqlens_q.shape[0] - 1

    # Pre-allocate output
    out = torch.empty(total_q, q_head * head_dim, dtype=q.dtype, device=q.device)

    # Decode part: seqs [0, num_decode_seqs)
    if num_decode_seqs > 0:
        q_decode = q[:num_decode_tokens]
        bt_decode = block_tables[:num_decode_seqs]
        ctx_decode = ctx_lens[:num_decode_seqs]
        out[:num_decode_tokens] = _invoke_flash_decode(
            q_decode, k_pool, v_pool, bt_decode, ctx_decode, scale, max_kv_len, page_size
        )

    # Prefill part: seqs [num_decode_seqs, num_seqs)
    num_prefill_seqs = num_seqs - num_decode_seqs
    if num_prefill_seqs > 0:
        q_prefill = q[num_decode_tokens:]
        bt_prefill = block_tables[num_decode_seqs:]
        ctx_prefill = ctx_lens[num_decode_seqs:num_seqs]
        cu_prefill = cu_seqlens_prefill if cu_seqlens_prefill is not None else cu_seqlens_q
        out[num_decode_tokens:] = _invoke_flash_prefill(
            q_prefill, k_pool, v_pool, bt_prefill, cu_prefill, ctx_prefill, scale,
            tile_seq_ids_prefill, page_size,
        )

    return out
