import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
import triton.language as tl
from kernels.fused_moe_kernel import (
    invoke_fused_moe_kernel,
    moe_align_block_size_triton,
    get_default_config,
    invoke_fused_moe_swiglu_down,
    invoke_fused_topk_softmax,
)
from kernels.fused_embedding_kernel import invoke_fused_embedding_layernorm
from kernels.attn_data_prep import invoke_attn_data_prep
from kernels.flash_attention import invoke_flash_attention
from kernels.skip_rmsnorm import invoke_skip_rmsnorm, invoke_rmsnorm
from components import KVCache


@dataclass
class AttentionMetadata:
    """Scheduler-precomputed attention metadata. Zero GPU compute at runtime."""
    positions: torch.Tensor       # [num_tokens] absolute position per token
    seq_ids: torch.Tensor         # [num_tokens] int64 - which seq each token belongs to
    cu_seqlens_q: torch.Tensor    # [num_seqs+1] int32 - cumulative Q token boundaries
    ctx_lens: torch.Tensor        # [num_seqs] int32 - total KV len per seq (context + new)
    context_lens: torch.Tensor    # [num_seqs] int32 - existing context lengths
    num_decode_seqs: int          # split index for decode/prefill routing
    num_decode_tokens: int        # total Q tokens for decode part (avoids .item() D2H)
    max_kv_len: int               # max KV length across all seqs (avoids .max().item() D2H)
    cu_seqlens_prefill: Optional[torch.Tensor] = None  # pre-rebased cu_seqlens for prefill (avoids subtraction)
    tile_seq_ids_prefill: Optional[torch.Tensor] = None  # [num_tiles] int32 - tile-to-seq mapping for prefill FA

class TokenEmbedding(nn.Module):
    """Token embedding lookup + learned position embedding + LayerNorm."""

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """input_ids: [num_tokens], positions: [num_tokens] -> [num_tokens, embed_dim]"""
        num_tokens = input_ids.size(0)
        embed_dim = self.token_emb.weight.shape[1]
        output = torch.empty(
            num_tokens, embed_dim,
            dtype=self.token_emb.weight.dtype,
            device=input_ids.device,
        )
        invoke_fused_embedding_layernorm(
            input_ids=input_ids,
            position_ids=positions,
            token_emb_weight=self.token_emb.weight,
            pos_emb_weight=self.pos_emb.weight,
            ln_weight=self.norm.weight,
            ln_bias=self.norm.bias,
            output=output,
        )
        return output


class Attention(nn.Module):
    """GQA attention with QK-Norm and partial RoPE. Supports KV cache for decode."""

    def __init__(self, head_dim: int, q_head: int, kv_head: int, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        self.q_head = q_head
        self.kv_head = kv_head
        self.embed_dim = head_dim * q_head
        self.rope_dim = 32  # partial RoPE on last 32 dims

        # Fused QKV projection: single GEMM instead of 3 separate
        self.qkv_dim = head_dim * (q_head + 2 * kv_head)
        self.qkv_proj = nn.Linear(self.embed_dim, self.qkv_dim, bias=False)
        self.o_proj = nn.Linear(head_dim * q_head, self.embed_dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

        # Precompute RoPE cos/sin buffers: duplicate freqs → [max_seq_len, rope_dim]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)           # [max_seq_len, rope_dim // 2]
        emb = torch.cat([freqs, freqs], dim=-1)     # [max_seq_len, rope_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

        # Split sizes for q, k, v
        self._q_size = head_dim * q_head
        self._kv_size = head_dim * kv_head

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int = 0,
        kv_cache: Optional[KVCache] = None,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """x: [num_tokens, embed_dim] → [num_tokens, embed_dim]

        All metadata (positions, seq_ids, cu_seqlens_q, ctx_lens) is
        pre-computed by the scheduler and passed via AttentionMetadata.
        Attention.forward has ZERO metadata computation.
        """
        num_seqs = metadata.cu_seqlens_q.size(0) - 1

        # Op 1: QKV GEMM
        qkv = self.qkv_proj(x)

        if kv_cache is not None:
            # Cache views (pure view, zero kernel launch)
            k_cache = kv_cache.cache[:num_seqs, layer_idx, 0]
            v_cache = kv_cache.cache[:num_seqs, layer_idx, 1]

            # Op 2: attn_data_prep — all metadata pre-computed by scheduler
            q = invoke_attn_data_prep(
                qkv, self.cos_cached, self.sin_cached,
                metadata.positions, metadata.seq_ids,
                self.q_norm.weight, self.k_norm.weight,
                self.q_head, self.kv_head, self.head_dim,
                k_cache, v_cache,
            )

            # Op 3: flash_attention — all metadata pre-computed by scheduler
            out = invoke_flash_attention(
                q, k_cache, v_cache,
                metadata.cu_seqlens_q, metadata.ctx_lens,
                num_decode_seqs=metadata.num_decode_seqs,
                num_decode_tokens=metadata.num_decode_tokens,
                max_kv_len=metadata.max_kv_len,
                cu_seqlens_prefill=metadata.cu_seqlens_prefill,
                tile_seq_ids_prefill=metadata.tile_seq_ids_prefill,
            )

        # Op 4: O projection GEMM
        return self.o_proj(out)


class FFN(nn.Module):
    """MoE FFN using fused_moe_kernel: gate_up fused GEMM + SwiGLU + down fused GEMM."""

    def __init__(self, embed_dim: int, num_experts: int, active_experts: int, experts_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.embed_dim = embed_dim
        self.experts_dim = experts_dim

        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        # Fused gate_up weight: [E, 2*experts_dim, embed_dim] (layout: [E, N, K])
        self.gate_up_weight = nn.Parameter(
            torch.empty(num_experts, 2 * experts_dim, embed_dim)
        )
        # Down weight: [E, embed_dim, experts_dim] (layout: [E, N, K])
        self.down_weight = nn.Parameter(
            torch.empty(num_experts, embed_dim, experts_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_up_weight, std=0.02)
        nn.init.normal_(self.down_weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, embed_dim] → [seq_len, embed_dim]"""
        seq_len = x.size(0)
        top_k = self.active_experts
        config_up = get_default_config(
            M=seq_len, E=self.num_experts, N=2 * self.experts_dim,
            K=self.embed_dim, top_k=top_k
        )

        # Routing: top-k + softmax fused in single Triton kernel
        router_logits = self.gate(x)
        topk_weights, topk_indices = invoke_fused_topk_softmax(router_logits, top_k)

        # Align tokens to block boundaries for fused kernel
        sorted_token_ids, expert_ids, num_tokens_post_padded = \
            moe_align_block_size_triton(topk_indices, config_up["BLOCK_SIZE_M"], self.num_experts)

        # Step 1: Fused gate_up GEMM — [M, K] x [E, 2N, K] → [M*top_k, 2N]
        gate_up_out = torch.empty(
            seq_len * top_k, 2 * self.experts_dim,
            dtype=x.dtype, device=x.device
        )
        invoke_fused_moe_kernel(
            A=x,
            B=self.gate_up_weight,
            C=gate_up_out,
            topk_weights=topk_weights.view(-1),
            topk_ids=topk_indices.view(-1),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=config_up,
            compute_type=tl.float16,
        )

        # Step 2+3: SwiGLU + Down GEMM fused — [M*top_k, 2N] → [M*top_k, embed_dim]
        config_down = get_default_config(
            M=seq_len, E=self.num_experts, N=self.embed_dim,
            K=self.experts_dim, top_k=top_k
        )
        down_out = torch.empty(
            seq_len * top_k, self.embed_dim,
            dtype=x.dtype, device=x.device
        )
        invoke_fused_moe_swiglu_down(
            gate_up_out=gate_up_out,
            down_weight=self.down_weight,
            out=down_out,
            topk_weights=topk_weights.view(-1),
            topk_ids=topk_indices.view(-1),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            top_k=top_k,
            config=config_down,
            compute_type=tl.float16,
        )

        # Step 4: Reduce across top_k experts per token
        output = down_out.view(seq_len, top_k, self.embed_dim).sum(dim=1)
        return output


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: Norm→Attn + Norm→FFN(MoE)."""

    def __init__(self, head_dim, q_head, kv_head, num_experts, active_experts, experts_dim, max_seq_len):
        super().__init__()
        embed_dim = head_dim * q_head
        self.attn_norm_weight = nn.Parameter(torch.ones(embed_dim))
        self.attn = Attention(head_dim, q_head, kv_head, max_seq_len)
        self.ffn_norm = nn.RMSNorm(embed_dim)
        self.ffn = FFN(embed_dim, num_experts, active_experts, experts_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int = 0,
        kv_cache: Optional[KVCache] = None,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        # attn_norm (fused Triton kernel) + attention
        normed = invoke_rmsnorm(x, self.attn_norm_weight)
        attn_out = self.attn(normed, layer_idx=layer_idx, kv_cache=kv_cache, metadata=metadata)

        # skip_rmsnorm: fused x+=attn_out + ffn_norm → 1 kernel (was 2)
        ffn_in = invoke_skip_rmsnorm(x, attn_out, self.ffn_norm.weight)

        # FFN (unchanged)
        ffn_out = self.ffn(ffn_in)

        # residual add
        x.add_(ffn_out)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        num_of_layer: int = 6,
        head_dim: int = 64,
        q_head: int = 8,
        kv_head: int = 2,
        num_of_experts: int = 64,
        active_experts: int = 8,
        experts_dim: int = 128,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.embed_dim = head_dim * q_head
        self.vocab_size = vocab_size
        self.num_of_layer = num_of_layer
        self.kv_head = kv_head
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(vocab_size, self.embed_dim, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(head_dim, q_head, kv_head, num_of_experts, active_experts, experts_dim, max_seq_len)
            for _ in range(num_of_layer)
        ])
        self.final_norm_weight = nn.Parameter(torch.ones(self.embed_dim))
        # LM Head: project hidden states to vocab logits
        self.lm_head = nn.Linear(self.embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Pure computation forward (vLLM V1 style).
        input_ids: [num_tokens], positions: [num_tokens],
        kv_cache: multi-seq cache, metadata: scheduler-built.
        Returns: [num_tokens, vocab_size] logits.

        No scheduling logic. Scheduler handles metadata build + cache advance.
        """
        h = self.token_embedding(input_ids, positions)
        for i, layer in enumerate(self.layers):
            h = layer(h, positions, layer_idx=i, kv_cache=kv_cache, metadata=metadata)
        h = invoke_rmsnorm(h, self.final_norm_weight)
        return self.lm_head(h)


if __name__ == "__main__":
    from components import Scheduler

    model = Transformer(num_of_layer=1).half().cuda()
    scheduler = Scheduler(model, max_batch_size=4)

    # Test 1: Generate (prefill + decode, fills to max_seq_len=128)
    prompt = torch.randint(0, 1000, [32]).cuda()
    output = scheduler.generate(prompt, temperature=0)
    print(f"Generate: prompt=32, output={output.shape[0]} (generated {output.shape[0]-32})")

    # Test 2: Batched decode (4 seqs)
    scheduler.kv_cache.reset()
    # Prefill each seq individually
    for i in range(4):
        plen = 16 + i * 8
        p = torch.randint(0, 1000, [plen]).cuda()
        # Only use slot i: temporarily set other context_lens, then restore
        scheduler.kv_cache.context_lens[i] = 0
        positions = torch.arange(plen).cuda()
        seq_lens = torch.tensor([plen], dtype=torch.int32).cuda()
        # Build metadata for slot i only (single-seq prefill)
        # Use a temporary single-slot view approach
        from network import AttentionMetadata
        import triton
        cu = torch.tensor([0, plen], dtype=torch.int32, device='cuda')
        ctx = torch.tensor([plen], dtype=torch.int32, device='cuda')
        num_tiles = triton.cdiv(plen, 128)
        tile_ids = torch.zeros(num_tiles, dtype=torch.int32, device='cuda')
        meta = AttentionMetadata(
            positions=positions,
            seq_ids=torch.zeros(plen, dtype=torch.int64, device='cuda'),
            cu_seqlens_q=cu,
            ctx_lens=ctx,
            context_lens=torch.zeros(1, dtype=torch.int32, device='cuda'),
            num_decode_seqs=0, num_decode_tokens=0, max_kv_len=plen,
            cu_seqlens_prefill=cu, tile_seq_ids_prefill=tile_ids,
        )
        # Use a single-seq cache view
        k_view = scheduler.kv_cache.cache[i:i+1]
        single_kv = KVCache.__new__(KVCache)
        single_kv.cache = k_view
        single_kv.context_lens = torch.zeros(1, dtype=torch.int32, device='cuda')
        model.forward(p, positions, single_kv, meta)
        scheduler.kv_cache.cache[i:i+1] = k_view
        scheduler.kv_cache.context_lens[i] = plen

    # Now batched decode step
    tokens = torch.randint(0, 1000, [4]).cuda()
    positions = scheduler.kv_cache.context_lens[:4].long()
    seq_lens = torch.ones(4, dtype=torch.int32).cuda()
    logits = scheduler.step(tokens, positions, seq_lens, num_decode_seqs=4)
    print(f"Batched decode: 4 seqs, logits={logits.shape}")