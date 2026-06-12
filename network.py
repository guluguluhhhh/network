import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton.language as tl
from kernels.fused_moe_kernel import (
    invoke_fused_moe_kernel,
    moe_align_block_size_torch,
    get_default_config,
)


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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [seq_len] → output: [seq_len, embed_dim]"""
        seq_len = input_ids.size(0)
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        return self.norm(x)


class Attention(nn.Module):
    """GQA attention with QK-Norm and partial RoPE. No KV cache (single-seq only)."""

    def __init__(self, head_dim: int, q_head: int, kv_head: int, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        self.q_head = q_head
        self.kv_head = kv_head
        self.embed_dim = head_dim * q_head

        self.q_proj = nn.Linear(self.embed_dim, head_dim * q_head, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, head_dim * kv_head, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, head_dim * kv_head, bias=False)
        self.o_proj = nn.Linear(head_dim * q_head, self.embed_dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.rope = RopeEmbedding(head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, embed_dim] → [seq_len, embed_dim]"""
        seq_len = x.size(0)

        q = self.q_proj(x).view(seq_len, self.q_head, self.head_dim)
        k = self.k_proj(x).view(seq_len, self.kv_head, self.head_dim)
        v = self.v_proj(x).view(seq_len, self.kv_head, self.head_dim)

        # QK-Norm before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply partial RoPE
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        # Reshape to [1, heads, seq_len, head_dim] for SDPA
        q = q.transpose(0, 1).unsqueeze(0)   # [1, q_head, seq_len, head_dim]
        k = k.transpose(0, 1).unsqueeze(0)   # [1, kv_head, seq_len, head_dim]
        v = v.transpose(0, 1).unsqueeze(0)   # [1, kv_head, seq_len, head_dim]

        # GQA: expand KV heads to match Q heads
        n_rep = self.q_head // self.kv_head
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.squeeze(0).transpose(0, 1).contiguous().view(seq_len, self.embed_dim)
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

        # Routing
        router_logits = self.gate(x)
        topk_weights, topk_indices = torch.topk(router_logits, top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # Align tokens to block boundaries for fused kernel
        config_up = get_default_config(
            M=seq_len, E=self.num_experts, N=2 * self.experts_dim,
            K=self.embed_dim, top_k=top_k
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = \
            moe_align_block_size_torch(topk_indices, config_up["BLOCK_SIZE_M"], self.num_experts)

        # Step 1: Fused gate_up GEMM — [M, K] x [E, 2N, K] → [M*top_k, 2N]
        gate_up_out = torch.zeros(
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

        # Step 2: SwiGLU activation
        gate_out = gate_up_out[:, :self.experts_dim]
        up_out = gate_up_out[:, self.experts_dim:]
        hidden = F.silu(gate_out) * up_out  # [M*top_k, experts_dim]

        # Step 3: Down GEMM with router weight — [M*top_k, experts_dim] x [E, embed_dim, experts_dim] → [M*top_k, embed_dim]
        config_down = get_default_config(
            M=seq_len, E=self.num_experts, N=self.embed_dim,
            K=self.experts_dim, top_k=top_k
        )
        # Re-align for down_proj (may use different BLOCK_SIZE_M)
        sorted_token_ids_d, expert_ids_d, num_tokens_post_padded_d = \
            moe_align_block_size_torch(topk_indices, config_down["BLOCK_SIZE_M"], self.num_experts)

        down_out = torch.zeros(
            seq_len * top_k, self.embed_dim,
            dtype=x.dtype, device=x.device
        )
        invoke_fused_moe_kernel(
            A=hidden,
            B=self.down_weight,
            C=down_out,
            topk_weights=topk_weights.view(-1),
            topk_ids=topk_indices.view(-1),
            sorted_token_ids=sorted_token_ids_d,
            expert_ids=expert_ids_d,
            num_tokens_post_padded=num_tokens_post_padded_d,
            mul_routed_weight=True,  # Fuse weight multiply into GEMM
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
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

        self.token_embedding = TokenEmbedding(vocab_size, self.embed_dim, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(head_dim, q_head, kv_head, num_of_experts, active_experts, experts_dim, max_seq_len)
            for _ in range(num_of_layer)
        ])
        self.final_norm = nn.RMSNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len] token IDs (single sequence, no batch dim, no KV cache).
        Returns: [seq_len, vocab_size] logits.
        """
        h = self.token_embedding(x)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits

if __name__ == "__main__":
    model = Transformer(num_of_layer=1, max_seq_len=8192).half().cuda()
    x = torch.randint(low=0, high=220000, size=[4455]).cuda()
    y = model.forward(x)
    print(y)