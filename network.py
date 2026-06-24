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
from kernels.skip_rmsnorm import invoke_skip_rmsnorm
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
        self.attn_norm = nn.RMSNorm(embed_dim)
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
        # attn_norm + attention
        attn_out = self.attn(self.attn_norm(x), layer_idx=layer_idx, kv_cache=kv_cache, metadata=metadata)

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
        self.final_norm = nn.RMSNorm(self.embed_dim)
        # LM Head: project hidden states to vocab logits
        self.lm_head = nn.Linear(self.embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        seq_lens: Optional[torch.Tensor] = None,
        num_decode_seqs: int = 0,
    ) -> torch.Tensor:
        """
        Token-flat forward (vLLM V1 style).
        input_ids: [num_tokens], positions: [num_tokens],
        kv_cache: multi-seq cache, seq_lens: [num_seqs].
        num_decode_seqs: split index (decode first, prefill after).
        Returns: [num_tokens, vocab_size] logits.
        """
        # ── Scheduler pre-computes all attention metadata (zero per-layer cost) ──
        metadata = self._build_metadata(positions, kv_cache, seq_lens, num_decode_seqs)

        h = self.token_embedding(input_ids, positions)
        for i, layer in enumerate(self.layers):
            h = layer(h, positions, layer_idx=i, kv_cache=kv_cache, metadata=metadata)
        h = self.final_norm(h)
        logits = self.lm_head(h)

        # Advance cache after all layers processed
        if kv_cache is not None and seq_lens is not None:
            kv_cache.advance(seq_lens)

        return logits

    def _build_metadata(
        self,
        positions: torch.Tensor,
        kv_cache: Optional[KVCache],
        seq_lens: Optional[torch.Tensor],
        num_decode_seqs: int,
    ) -> Optional[AttentionMetadata]:
        """Simulate scheduler: pre-compute ALL attention metadata on GPU once.

        In production, this would be computed on CPU by the scheduler and
        uploaded as part of the batch descriptor — zero GPU compute.
        """
        if kv_cache is None or seq_lens is None:
            return None

        device = positions.device
        num_seqs = seq_lens.size(0)
        num_tokens = positions.size(0)

        # seq_ids: which sequence each token belongs to
        seq_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=device, dtype=torch.int64), seq_lens.long()
        )

        # cu_seqlens_q: cumulative Q token boundaries
        cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        cu_seqlens_q[1:] = seq_lens.to(torch.int32).cumsum(0)

        # ctx_lens: total KV length per seq after this forward
        context_lens = kv_cache.context_lens[:num_seqs]
        ctx_lens = context_lens + seq_lens.to(torch.int32)

        # Scalars the FA wrapper needs (avoids D2H syncs at runtime)
        num_decode_tokens = int(seq_lens[:num_decode_seqs].sum().item()) if num_decode_seqs > 0 else 0
        max_kv_len = int(ctx_lens.max().item()) if num_seqs > 0 else 0

        # Pre-rebased cu_seqlens for prefill subset (avoids elementwise subtraction)
        num_prefill_seqs = num_seqs - num_decode_seqs
        cu_seqlens_prefill = None
        tile_seq_ids_prefill = None
        if num_prefill_seqs > 0:
            if num_decode_seqs > 0:
                cu_seqlens_prefill = cu_seqlens_q[num_decode_seqs:] - cu_seqlens_q[num_decode_seqs]
            else:
                cu_seqlens_prefill = cu_seqlens_q  # no rebase needed
            # Pre-compute tile_seq_ids for prefill FA (avoids arange+mul+searchsorted+cast at runtime)
            total_prefill_tokens = num_tokens - num_decode_tokens
            BLOCK_M = 128
            import triton
            num_tiles = triton.cdiv(total_prefill_tokens, BLOCK_M)
            tile_seq_ids_prefill = torch.searchsorted(
                cu_seqlens_prefill[1:],
                torch.arange(num_tiles, device=device) * BLOCK_M,
                right=True,
            ).to(torch.int32)

        return AttentionMetadata(
            positions=positions,
            seq_ids=seq_ids,
            cu_seqlens_q=cu_seqlens_q,
            ctx_lens=ctx_lens,
            context_lens=context_lens,
            num_decode_seqs=num_decode_seqs,
            num_decode_tokens=num_decode_tokens,
            max_kv_len=max_kv_len,
            cu_seqlens_prefill=cu_seqlens_prefill,
            tile_seq_ids_prefill=tile_seq_ids_prefill,
        )

    def create_kv_cache(self, num_seqs: int = 1, device: str = "cuda") -> KVCache:
        """Create a fresh KV cache."""
        return KVCache(
            num_seqs=num_seqs,
            num_layers=self.num_of_layer,
            max_seq_len=self.max_seq_len,
            kv_head=self.kv_head,
            head_dim=self.head_dim,
            dtype=next(self.parameters()).dtype,
            device=device,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 0,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV cache.

        Args:
            prompt_ids: [prompt_len] token IDs on device.
            max_new_tokens: maximum number of tokens to generate.
            temperature: sampling temperature (1.0 = no change, <1 sharper, >1 flatter).
            top_k: keep only top-k logits before sampling.
            top_p: nucleus sampling threshold.
            eos_token_id: stop generation when this token is produced.

        Returns:
            [prompt_len + generated_len] full sequence of token IDs.
        """
        device = prompt_ids.device
        kv_cache = self.create_kv_cache(num_seqs=1, device=device)

        # If max_new_tokens=0, fill to max_seq_len
        prompt_len = prompt_ids.size(0)
        if max_new_tokens == 0:
            max_new_tokens = self.max_seq_len - prompt_len
        positions = torch.arange(prompt_len, device=device)
        seq_lens = torch.tensor([prompt_len], device=device, dtype=torch.int32)
        logits = self.forward(prompt_ids, positions, kv_cache, seq_lens)
        next_token_logits = logits[-1]  # [vocab]

        generated_ids = [prompt_ids]

        # ── Phase 2: Decode loop ──
        for _ in range(max_new_tokens):
            # Sample next token
            next_token = self._sample(next_token_logits, temperature, top_k, top_p)

            generated_ids.append(next_token.unsqueeze(0))

            # Stop on EOS
            if next_token.item() == eos_token_id:
                break

            # Run single token through model with KV cache
            pos = kv_cache.context_lens[0:1].long()
            sl = torch.ones(1, device=device, dtype=torch.int32)
            logits = self.forward(next_token.unsqueeze(0), pos, kv_cache, sl, num_decode_seqs=1)
            next_token_logits = logits[0]  # [vocab]

        return torch.cat(generated_ids, dim=0)

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample a single token from logits with temperature, top-k, and top-p."""
        if temperature <= 0:
            # Greedy
            return logits.argmax(dim=-1)

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_val = logits.topk(top_k, dim=-1).values[..., -1]
            logits = logits.where(logits >= kth_val, torch.full_like(logits, float('-inf')))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative prob > top_p
            remove_mask = cumulative_probs > top_p
            # Shift mask right so first token above threshold is kept
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_logits[remove_mask] = float('-inf')
            # Scatter back
            logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


if __name__ == "__main__":
    model = Transformer(num_of_layer=1).half().cuda()

    # Test 1: Generate (prefill + decode, fills to max_seq_len=128)
    prompt = torch.randint(0, 1000, [32]).cuda()
    output = model.generate(prompt, temperature=0)
    print(f"Generate: prompt=32, output={output.shape[0]} (generated {output.shape[0]-32})")

    # Test 2: Batched decode (4 seqs)
    batch_kv = model.create_kv_cache(num_seqs=4)
    for i in range(4):
        plen = 16 + i * 8
        p = torch.randint(0, 1000, [plen]).cuda()
        single_kv = model.create_kv_cache(num_seqs=1)
        model.forward(p, torch.arange(plen).cuda(), single_kv, torch.tensor([plen]).cuda())
        batch_kv.cache[i, :, :, :plen] = single_kv.cache[0, :, :, :plen]
        batch_kv.context_lens[i] = plen
    tokens = torch.randint(0, 1000, [4]).cuda()
    positions = batch_kv.context_lens.long()
    seq_lens = torch.ones(4, dtype=torch.int32).cuda()
    logits = model.forward(tokens, positions, batch_kv, seq_lens, num_decode_seqs=4)
    print(f"Batched decode: 4 seqs, logits={logits.shape}")