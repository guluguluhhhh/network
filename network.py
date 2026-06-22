import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class RopeEmbedding(nn.Module):
    """Partial RoPE applied only to the last 32 dimensions of each head."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, rope_dim: int = 32):
        super().__init__()
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        assert rope_dim <= head_dim, "rope_dim cannot exceed head_dim"
        assert rope_dim % 2 == 0, "rope_dim must be even"

        # Precompute sin/cos for the rope dimensions
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, rope_dim // 2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, rope_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        x: [batch, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
        Applies RoPE only to the LAST rope_dim dimensions; leaves the rest unchanged.
        """
        cos = self.cos_cached[:seq_len].to(x.dtype)  # [seq_len, rope_dim]
        sin = self.sin_cached[:seq_len].to(x.dtype)

        # Reshape for broadcasting: [seq_len, 1, rope_dim]
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        # Split: non-rope prefix | rope suffix
        prefix = x[..., : self.head_dim - self.rope_dim]
        suffix = x[..., self.head_dim - self.rope_dim :]

        # Standard rotary embedding on suffix
        d = suffix.shape[-1]
        x1, x2 = suffix[..., : d // 2], suffix[..., d // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1) * sin + suffix * cos

        return torch.cat([prefix, rotated], dim=-1)


class TokenEmbedding(nn.Module):
    """Token embedding lookup + learned position embedding + LayerNorm."""

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """input_ids: [seq_len] → output: [seq_len, embed_dim]

        Args:
            position_offset: starting position for position embeddings.
                             0 for prefill, kv_cache.seq_len for decode.
        """
        seq_len = input_ids.size(0)
        embed_dim = self.token_emb.weight.shape[1]
        output = torch.empty(
            seq_len, embed_dim,
            dtype=self.token_emb.weight.dtype,
            device=input_ids.device,
        )
        # Slice pos_emb_weight so kernel's pid=0 maps to position_offset
        pos_emb_weight = self.pos_emb.weight[position_offset:]
        invoke_fused_embedding_layernorm(
            input_ids=input_ids,
            token_emb_weight=self.token_emb.weight,
            pos_emb_weight=pos_emb_weight,
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
    ) -> torch.Tensor:
        """x: [seq_len, embed_dim] → [seq_len, embed_dim]

        Args:
            layer_idx: index of this layer in the model (for kv_cache addressing).
            kv_cache: if provided, uses KV cache for incremental decoding.
                      - prefill: writes all K/V to cache, uses causal attn.
                      - decode: writes new K/V, attends to full cached history.
        """
        seq_len = x.size(0)

        # Position offset: where in the sequence are we?
        if kv_cache is not None:
            position_offset = kv_cache.seq_len
        else:
            position_offset = 0

        # Single fused GEMM for Q, K, V
        qkv = self.qkv_proj(x)

        # ── Kernel 1: Data preparation with position-aware RoPE ──
        cos = self.cos_cached[position_offset:position_offset + seq_len]
        sin = self.sin_cached[position_offset:position_offset + seq_len]
        q, k, v = invoke_attn_data_prep(
            qkv, cos, sin,
            self.q_norm.weight, self.k_norm.weight,
            self.q_head, self.kv_head, self.head_dim,
        )

        # ── KV Cache handling ──
        if kv_cache is not None:
            # Write new K/V to cache
            kv_cache.update(layer_idx, k, v)
            # Read full history (including newly written)
            # Note: seq_len hasn't advanced yet, so we read [0:seq_len+new]
            # We manually include new tokens by reading up to current write position
            total_len = position_offset + seq_len
            k_full = kv_cache.cache[layer_idx, 0, :total_len].contiguous()
            v_full = kv_cache.cache[layer_idx, 1, :total_len].contiguous()
        else:
            k_full = k
            v_full = v

        # ── Kernel 2: FlashAttention V2 with GQA ──
        # prefill with cache: causal mask needed
        # decode (seq_len=1): no causal needed (single Q attends to all history)
        # no cache: no causal (original benchmark behavior)
        is_causal = (kv_cache is not None) and (seq_len > 1)
        out = invoke_flash_attention(q, k_full, v_full, is_causal=is_causal)

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
        layer_idx: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # attn_norm + attention
        attn_out = self.attn(self.attn_norm(x), layer_idx=layer_idx, kv_cache=kv_cache)

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
        vocab_size: int = 220000,
        num_of_layer: int = 6,
        head_dim: int = 64,
        q_head: int = 8,
        kv_head: int = 2,
        num_of_experts: int = 64,
        active_experts: int = 8,
        experts_dim: int = 128,
        max_seq_len: int = 2048,
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
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        x: [seq_len] token IDs (single sequence, no batch dim).
        kv_cache: if provided, uses incremental KV cache.
        Returns: [seq_len, vocab_size] logits.
        """
        h = self.token_embedding(x, position_offset=kv_cache.seq_len if kv_cache is not None else 0)
        for i, layer in enumerate(self.layers):
            h = layer(h, layer_idx=i, kv_cache=kv_cache)
        h = self.final_norm(h)
        logits = self.lm_head(h)

        # Advance cache after all layers processed
        if kv_cache is not None:
            kv_cache.advance(x.size(0))

        return logits

    def create_kv_cache(self, device: str = "cuda") -> KVCache:
        """Create a fresh KV cache for this model."""
        return KVCache(
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
        max_new_tokens: int = 128,
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
        kv_cache = self.create_kv_cache(device=device)

        # ── Phase 1: Prefill ──
        # Run entire prompt through the model, filling the KV cache
        logits = self.forward(prompt_ids, kv_cache=kv_cache)  # [prompt_len, vocab]
        # Only the last token's logits are used for next-token prediction
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
            logits = self.forward(next_token.unsqueeze(0), kv_cache=kv_cache)  # [1, vocab]
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
    model = Transformer(num_of_layer=1, max_seq_len=8192).half().cuda()

    # ── Test 1: Original forward (no cache, benchmark mode) ──
    x = torch.randint(low=0, high=220000, size=[4455]).cuda()
    logits = model.forward(x)
    print(f"Forward (no cache): input={x.shape}, output={logits.shape}")

    # ── Test 2: Generate with KV cache ──
    prompt = torch.randint(low=0, high=220000, size=[64]).cuda()
    output = model.generate(prompt, max_new_tokens=32, temperature=0.8)
    print(f"Generate: prompt={prompt.shape[0]}, output={output.shape[0]} "
          f"(generated {output.shape[0] - prompt.shape[0]} new tokens)")