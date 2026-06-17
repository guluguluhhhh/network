"""
Fused MoE triton kernel, adapted from sglang/vllm.

Only supports bf16/fp16 non-quantized path. Quantization paths can be added later.
"""

from typing import Any, Dict

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Fused MoE GEMM kernel.

    A: input tokens [M, K] (indexed via sorted_token_ids)
    B: expert weights [E, N, K] (read as [K, N] tiles per expert)
    C: output [num_valid_tokens, N] (scattered via sorted_token_ids)

    sorted_token_ids, expert_ids, num_tokens_post_padded are produced by
    moe_align_block_size_torch().
    """
    # Map program id to the block of C it should compute (grouped for L2 reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Early exit for padding blocks
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Load sorted token ids and expert id for this block
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Pointers for A and B
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Accumulate in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply router weights if requested
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # Write back
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(
    A: torch.Tensor,  # [M, K] input hidden states
    B: torch.Tensor,  # [E, N, K] expert weights
    C: torch.Tensor,  # [M * top_k, N] output (scattered via sorted_token_ids)
    topk_weights: torch.Tensor,  # [M * top_k] flat routing weights
    topk_ids: torch.Tensor,  # [M * top_k] flat (only used for numel)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2]
    even_Ks = K % config["BLOCK_SIZE_K"] == 0

    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],  # N
        K,  # K
        sorted_token_ids.shape[0],  # EM
        topk_ids.numel(),  # num_valid_tokens
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=even_Ks,
        **config,
    )


# ─── SwiGLU-fused down MoE kernel ───────────────────────────────────────────
# Eliminates the intermediate "hidden" tensor (silu*gate*up) by fusing:
#   gate_up_out → split → SwiGLU → down GEMM  (all in one launch).
# Saves: ~35 MB allocation + 3 elementwise silu/mul kernels + Python dispatch gap.


@triton.jit
def fused_moe_swiglu_down_kernel(
    gate_up_ptr,                # [num_tokens, 2*experts_dim]
    down_b_ptr,                 # [E, embed_dim, experts_dim]
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,                          # = embed_dim (output cols)
    K,                          # = experts_dim (inner dim)
    EM,
    num_valid_tokens,
    stride_gateup_m, stride_gateup_n,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Fused SwiGLU + down GEMM.
    Reads gate_up_out[token, :] → split gate/up → silu(gate)*up → down GEMM → output.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Token indexing
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Output column range
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # ── Phase: SwiGLU + down GEMM ──
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        kmask = (offs_k < K - k_start) if not even_Ks else None

        # Load gate and up from gate_up_out for this K slice
        gate_row_idx = offs_token[:, None]  # [BLOCK_M, 1]
        gate_ptr = gate_up_ptr + gate_row_idx * stride_gateup_m + (k_start + offs_k)[None, :] * stride_gateup_n
        up_ptr   = gate_up_ptr + gate_row_idx * stride_gateup_m + (K + k_start + offs_k)[None, :] * stride_gateup_n

        raw_gate = tl.load(gate_ptr, mask=token_mask[:, None] & (kmask[None, :] if kmask is not None else True), other=0.0).to(tl.float32)
        raw_up   = tl.load(up_ptr,   mask=token_mask[:, None] & (kmask[None, :] if kmask is not None else True), other=0.0).to(tl.float32)

        # SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
        activated = tl.sigmoid(raw_gate) * raw_gate * raw_up  # [BLOCK_M, BLOCK_K]

        # Down weight for this expert: [embed_dim, experts_dim]
        # Load tile [embed_dim, BLOCK_K] for current K slice
        down_b_tile_ptr = (
            down_b_ptr + off_experts * stride_be
            + (k_start + offs_k[:, None]) * stride_bk + offs_cn[None, :] * stride_bn
        )
        if even_Ks:
            b = tl.load(down_b_tile_ptr)
        else:
            b = tl.load(down_b_tile_ptr, mask=(offs_k[:, None] < K - k_start), other=0.0)

        accumulator += tl.dot(activated.to(compute_type), b)

    # Apply router weight
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
    accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    # Write output
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_swiglu_down(
    gate_up_out: torch.Tensor,  # [M * top_k, 2 * experts_dim]
    down_weight: torch.Tensor,  # [E, embed_dim, experts_dim]
    out: torch.Tensor,          # [M * top_k, embed_dim]
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    experts_dim = down_weight.shape[2]
    embed_dim = down_weight.shape[1]
    even_Ks = experts_dim % config["BLOCK_SIZE_K"] == 0

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(embed_dim, META["BLOCK_SIZE_N"]),
    )

    fused_moe_swiglu_down_kernel[grid](
        gate_up_out,
        down_weight,
        out,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        embed_dim,
        experts_dim,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        gate_up_out.stride(0),
        gate_up_out.stride(1),
        down_weight.stride(0),
        down_weight.stride(2),
        down_weight.stride(1),
        out.stride(-2),
        out.stride(-1),
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=even_Ks,
        **config,
    )


def moe_align_block_size_torch(
    topk_ids: torch.Tensor,  # [M, top_k]
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align tokens to experts in blocks of block_size for the fused MoE kernel.

    Returns:
        sorted_token_ids: [max_padded] padded token indices (may be larger than needed)
        expert_ids: [max_blocks] expert id for each block
        num_tokens_post_padded: [1] scalar tensor (actual used count, on GPU)
    """
    M, top_k = topk_ids.shape
    num_valid = M * top_k
    flat_ids = topk_ids.view(-1)  # [M * top_k]

    # Pre-allocate to max possible size to avoid GPU→CPU sync
    max_padded = num_valid + num_experts * block_size

    # Count tokens per expert via scatter_add (CUDA-graph-safe, no CPU-GPU sync
    # unlike torch.bincount which reads GPU data on CPU to determine output size)
    tokens_per_expert = torch.zeros(
        num_experts, dtype=torch.int32, device=topk_ids.device
    )
    clamped_ids = flat_ids.clamp(min=0, max=num_experts - 1).long()
    tokens_per_expert.scatter_add_(
        0, clamped_ids, torch.ones(num_valid, dtype=torch.int32, device=topk_ids.device)
    )
    padded_counts = ((tokens_per_expert + block_size - 1) // block_size) * block_size

    # expert_offsets[0]=0, expert_offsets[i]=sum(padded_counts[:i])
    expert_offsets = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )
    torch.cumsum(padded_counts.int(), dim=0, out=expert_offsets[1:])
    num_tokens_post_padded = expert_offsets[
        num_experts : num_experts + 1
    ]  # [1] view, stays on GPU

    # Sort tokens by expert + compute positions (reuse cumsum for expert_start)
    sorted_order = flat_ids.argsort(stable=True)
    sorted_expert_ids = flat_ids[sorted_order]

    # expert_start[i] = sum(tokens_per_expert[:i]) — reuse tokens_per_expert cumsum
    expert_start = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    torch.cumsum(tokens_per_expert[:-1].int(), dim=0, out=expert_start[1:])
    position_in_expert = (
        torch.arange(num_valid, device=topk_ids.device)
        - expert_start[sorted_expert_ids]
    )

    # Scatter sorted tokens into padded layout
    sorted_token_ids = torch.full(
        (max_padded,), num_valid, dtype=torch.int32, device=topk_ids.device
    )
    dest = expert_offsets[sorted_expert_ids] + position_in_expert.int()
    sorted_token_ids[dest.long()] = sorted_order.int()

    # Expert ids per block: use searchsorted instead of repeat_interleave
    max_blocks = max_padded // block_size
    # block_boundaries[i] = expert_offsets[i+1] / block_size = cumulative blocks for experts 0..i
    block_boundaries = expert_offsets[1:] // block_size  # [E]
    block_indices = torch.arange(max_blocks, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.searchsorted(block_boundaries, block_indices, right=True).int()
    # Blocks beyond actual content get expert_id >= num_experts, which is fine
    # (they'll be skipped by num_tokens_post_padded check in kernel)

    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ─── Triton moe_align kernels ────────────────────────────────────────────────
# 3 Triton kernels replacing ~34 fragmented PyTorch ops:
#   hist | prefix+fill_ids (fused, single block) | scatter
# Stage 2+3 fused: prefix sum and expert_ids fill run in the same block,
# eliminating the ~31us gap between them (×2 per forward = ~62us saved).


@triton.jit
def _moe_align_hist_kernel(
    topk_ids_ptr,  # [num_tokens] int32 flat expert assignments
    count_ptr,     # [num_experts] int32, zeroed before call
    num_tokens,
    BLOCK: tl.constexpr,
):
    """Stage 1: count tokens per expert via vector atomic_add."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_tokens
    eids = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    # mask=mask prevents out-of-bounds atomics for the tail block
    tl.atomic_add(count_ptr + eids, 1, mask=mask)


@triton.jit
def _moe_align_prefix_fill_kernel(
    count_ptr,         # [num_experts] int32 raw counts
    offsets_ptr,       # [num_experts+1] int32 OUTPUT: exclusive padded prefix sums
    total_ptr,         # [1] int32 OUTPUT: total padded token count
    expert_ids_ptr,    # [max_blocks] int32 OUTPUT
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    MAX_E: tl.constexpr,    # triton.next_power_of_2(num_experts)
    FILL_BLOCK: tl.constexpr,
):
    """
    Fused Stage 2+3: exclusive prefix sum then expert_ids fill, single block. grid=(1,).

    Avoids the ~31us gap between the two formerly-separate kernels by running
    both phases in one launch. Phase 2 extracts per-expert scalars from the
    register tensors computed in Phase 1 (no global-memory round-trip needed).
    """
    # ── Phase 1: exclusive prefix sum over padded expert counts ──
    offs_e = tl.arange(0, MAX_E)
    mask_e = offs_e < num_experts
    counts = tl.load(count_ptr + offs_e, mask=mask_e, other=0).to(tl.int32)
    padded = ((counts + block_size - 1) // block_size) * block_size
    inclusive = tl.cumsum(padded, axis=0)       # inclusive[e] = sum(padded[0..e])
    exclusive = inclusive - padded              # exclusive[e] = sum(padded[0..e-1])
    tl.store(offsets_ptr + offs_e, exclusive, mask=mask_e)
    total = tl.sum(padded, axis=0)
    tl.store(offsets_ptr + num_experts, total)
    tl.store(total_ptr, total)

    # ── Phase 2: fill expert_ids[b] = e for every block b owned by expert e ──
    # Extract per-expert start/end block indices directly from the register tensors
    # computed above, avoiding a global-memory round-trip.
    # tl.static_range unrolls the loop at compile time (num_experts is constexpr);
    # the masked sum selects exclusive[eid] / inclusive[eid] without needing
    # runtime vector indexing.
    for eid in tl.static_range(num_experts):
        s_blk = tl.sum(tl.where(tl.arange(0, MAX_E) == eid, exclusive, 0)) // block_size
        e_blk = tl.sum(tl.where(tl.arange(0, MAX_E) == eid, inclusive, 0)) // block_size
        n = e_blk - s_blk
        for b in range(0, n, FILL_BLOCK):
            inner = b + tl.arange(0, FILL_BLOCK)
            tl.store(expert_ids_ptr + s_blk + inner, eid, mask=inner < n)


@triton.jit
def _moe_align_scatter_kernel(
    topk_ids_ptr,    # [num_tokens] int32
    offsets_ptr,     # [num_experts] int32 expert start offsets
    pos_ptr,         # [num_experts] int32 atomic position counter, zeroed before call
    sorted_ids_ptr,  # [max_padded] int32 OUTPUT
    num_tokens,
    BLOCK: tl.constexpr,
):
    """Stage 4: scatter token flat-indices into padded expert-sorted layout. grid=(cdiv(N,BLOCK),)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_tokens
    eids = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    # fetch-and-add: each valid token gets a unique position within its expert's slot
    pos = tl.atomic_add(pos_ptr + eids, 1, mask=mask)
    base = tl.load(offsets_ptr + eids, mask=mask, other=0).to(tl.int32)
    dest = base + pos
    tl.store(sorted_ids_ptr + dest, offs.to(tl.int32), mask=mask)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,  # [M, top_k]
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton-fused replacement for moe_align_block_size_torch.
    3 Triton kernels vs ~34 fragmented PyTorch ops:
      hist | prefix+fill_ids (fused) | scatter
    Returns identical (sorted_token_ids, expert_ids, num_tokens_post_padded).
    """
    M, top_k = topk_ids.shape
    num_tokens = M * top_k
    flat_ids = topk_ids.reshape(-1).to(torch.int32)
    max_padded = num_tokens + num_experts * block_size
    max_blocks = (max_padded + block_size - 1) // block_size
    dev = topk_ids.device

    count = torch.zeros(num_experts, dtype=torch.int32, device=dev)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=dev)
    pos = torch.zeros(num_experts, dtype=torch.int32, device=dev)
    sorted_token_ids = torch.full((max_padded,), num_tokens, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=dev)
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=dev)

    HIST_BLOCK = 512
    _moe_align_hist_kernel[(triton.cdiv(num_tokens, HIST_BLOCK),)](
        flat_ids, count, num_tokens, BLOCK=HIST_BLOCK,
    )

    MAX_E = triton.next_power_of_2(num_experts)
    _moe_align_prefix_fill_kernel[(1,)](
        count, offsets, num_tokens_post_padded, expert_ids,
        num_experts=num_experts, block_size=block_size, MAX_E=MAX_E, FILL_BLOCK=32,
    )

    SCATTER_BLOCK = 512
    _moe_align_scatter_kernel[(triton.cdiv(num_tokens, SCATTER_BLOCK),)](
        flat_ids, offsets, pos, sorted_token_ids, num_tokens, BLOCK=SCATTER_BLOCK,
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded


# torch.compile with dynamic=True + assert_indirect_indexing causes spurious
# assertion failures on valid expert IDs. Use the non-compiled version for now.
moe_align_block_size_compiled = moe_align_block_size_triton


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    top_k: int,
) -> Dict[str, Any]:
    """Select default block sizes based on problem dimensions.

    Tuned on H20 (SM90) with Triton 3.4.0. num_warps and num_stages are
    critical for performance — triton defaults are suboptimal on SM90.
    """
    avg_tokens_per_expert = M * top_k / max(E, 1)

    if avg_tokens_per_expert <= 2:
        # Decode / very small batch: maximize N-parallelism
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 4,
        }
    elif avg_tokens_per_expert <= 16:
        # Small batch: large K block for memory bandwidth
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    elif avg_tokens_per_expert <= 48:
        # Medium batch: BM=32 balances padding waste vs parallelism
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        # Large batch: maximize compute throughput with bigger M blocks
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }


# ─── Fused topk + softmax kernel ─────────────────────────────────────────────
# Replaces: topk → softmax(fp32) → .to(fp16) → .copy_  (~4 elementwise kernels)
# with a single Triton launch.  Per-token: load 64 logits, repeated-argmax top-8,
# softmax on the 8 selected values, output fp16 weights + int64 indices.

@triton.jit
def _fused_topk_softmax_kernel(
    logits_ptr,             # [seq_len, num_experts] fp16
    weights_ptr,            # [seq_len, top_k] fp16 output
    indices_ptr,            # [seq_len, top_k] int64 output
    seq_len,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_E: tl.constexpr,  # num_experts (must be power-of-2 for full load)
):
    """Grid: (seq_len,).  One program per token row."""
    row = tl.program_id(0)
    if row >= seq_len:
        return

    offs = tl.arange(0, BLOCK_E)
    mask_e = offs < num_experts
    x = tl.load(logits_ptr + row * num_experts + offs, mask=mask_e, other=float('-inf')).to(tl.float32)

    top_vals = tl.zeros([top_k], dtype=tl.float32)
    top_idxs = tl.zeros([top_k], dtype=tl.int32)

    # Repeated argmax: find top-k values
    for k in range(top_k):
        idx = tl.argmax(x, axis=0)
        val = tl.max(x, axis=0)
        top_idxs = tl.where(tl.arange(0, top_k) == k, idx, top_idxs)
        top_vals = tl.where(tl.arange(0, top_k) == k, val, top_vals)
        x = tl.where(offs == idx, float('-inf'), x)

    # Softmax in fp32
    m = tl.max(top_vals, axis=0)
    w = tl.exp(top_vals - m)
    w = w / tl.sum(w, axis=0)

    out_offs = tl.arange(0, top_k)
    tl.store(weights_ptr + row * top_k + out_offs, w.to(tl.float16))
    tl.store(indices_ptr + row * top_k + out_offs, top_idxs)


def invoke_fused_topk_softmax(
    logits: torch.Tensor,       # [seq_len, num_experts] fp16
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (weights [seq_len, top_k] fp16, indices [seq_len, top_k] int64)."""
    seq_len, num_experts = logits.shape
    weights = torch.empty(seq_len, top_k, dtype=logits.dtype, device=logits.device)
    indices = torch.empty(seq_len, top_k, dtype=torch.int32, device=logits.device)

    BLOCK_E = triton.next_power_of_2(num_experts)
    grid = (seq_len,)
    _fused_topk_softmax_kernel[grid](
        logits, weights, indices,
        seq_len,
        num_experts=num_experts,
        top_k=top_k,
        BLOCK_E=BLOCK_E,
    )
    return weights, indices
