"""
Static Batching Scheduler.

Responsibilities:
- KV cache lifecycle management
- AttentionMetadata construction (zero runtime compute for model)
- Generate loop control
- Request state tracking

In production, this would be the Engine/Scheduler layer that
dynamically manages batches. Here we implement static batching
as the simplest starting point.
"""

import torch
import triton
from typing import Optional

from components import KVCache


# Import AttentionMetadata from network (avoid circular import by lazy import)
def _get_attention_metadata_class():
    from network import AttentionMetadata
    return AttentionMetadata


class Scheduler:
    """Static batching scheduler. Manages KV cache and metadata construction."""

    def __init__(self, model, max_batch_size: int = 4):
        self.model = model
        self.max_batch_size = max_batch_size
        self.device = next(model.parameters()).device

        # KV cache: pre-allocated for max_batch_size sequences
        self.kv_cache = KVCache(
            num_seqs=max_batch_size,
            num_layers=model.num_of_layer,
            max_seq_len=model.max_seq_len,
            kv_head=model.kv_head,
            head_dim=model.head_dim,
            dtype=next(model.parameters()).dtype,
            device=self.device,
        )

    def build_metadata(
        self,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decode_seqs: int = 0,
    ):
        """Build AttentionMetadata from scheduling decisions.

        Batch layout: seq i uses kv_cache slot i (1:1 mapping).
        All derived tensors and scalars are pre-computed here so that
        model.forward and kernel wrappers have ZERO metadata compute.
        """
        AttentionMetadata = _get_attention_metadata_class()

        device = positions.device
        num_seqs = seq_lens.size(0)
        num_tokens = positions.size(0)

        # seq_ids: maps each token to its seq index in the batch (0..num_seqs-1)
        seq_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=device, dtype=torch.int64), seq_lens.long()
        )

        # cu_seqlens_q: cumulative Q token boundaries
        cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        cu_seqlens_q[1:] = seq_lens.to(torch.int32).cumsum(0)

        # ctx_lens: total KV length per seq after this forward
        context_lens = self.kv_cache.context_lens[:num_seqs]
        ctx_lens = context_lens + seq_lens.to(torch.int32)

        # Scalars (avoids D2H syncs at runtime)
        num_decode_tokens = int(seq_lens[:num_decode_seqs].sum().item()) if num_decode_seqs > 0 else 0
        max_kv_len = int(ctx_lens.max().item()) if num_seqs > 0 else 0

        # Pre-rebased cu_seqlens + tile_seq_ids for prefill (avoids runtime ops)
        num_prefill_seqs = num_seqs - num_decode_seqs
        cu_seqlens_prefill = None
        tile_seq_ids_prefill = None
        if num_prefill_seqs > 0:
            if num_decode_seqs > 0:
                cu_seqlens_prefill = cu_seqlens_q[num_decode_seqs:] - cu_seqlens_q[num_decode_seqs]
            else:
                cu_seqlens_prefill = cu_seqlens_q
            # tile_seq_ids for prefill FA kernel
            total_prefill_tokens = num_tokens - num_decode_tokens
            BLOCK_M = 128
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

    def step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decode_seqs: int = 0,
    ) -> torch.Tensor:
        """Single forward step: build metadata -> model.forward -> advance cache.

        Seq i uses kv_cache slot i (1:1). Returns: [num_tokens, vocab_size] logits.
        """
        metadata = self.build_metadata(positions, seq_lens, num_decode_seqs)
        logits = self.model.forward(input_ids, positions, self.kv_cache, metadata)
        # Advance context_lens for active seqs
        num_seqs = seq_lens.size(0)
        self.kv_cache.context_lens[:num_seqs] += seq_lens.to(torch.int32)
        return logits

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Single-sequence generate (convenience wrapper over batch_generate)."""
        results = self.batch_generate([prompt_ids], **kwargs)
        return results[0]

    @torch.inference_mode()
    def batch_generate(
        self,
        prompts: list,
        max_new_tokens: int = 0,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ) -> list:
        """Batched generation with mixed prefill/decode scheduling.

        Supports up to max_batch_size concurrent sequences.
        Batch layout: [decode seqs | prefill seqs] per step.

        Returns: list of [prompt_len + generated_len] tensors.
        """
        device = self.device
        batch_size = min(len(prompts), self.max_batch_size)
        max_seq_len = self.model.max_seq_len

        # Reset slots
        self.kv_cache.context_lens[:batch_size] = 0

        # Request state
        generated = [[] for _ in range(batch_size)]  # generated token ids
        prompt_lens = [p.size(0) for p in prompts[:batch_size]]
        is_done = [False] * batch_size

        if max_new_tokens == 0:
            max_new_tokens = max_seq_len - max(prompt_lens)

        # ── Phase 1: Prefill all sequences (pure prefill batch) ──
        # Concat all prompts into token-flat layout
        all_tokens = torch.cat(prompts[:batch_size], dim=0)
        all_positions = torch.cat([torch.arange(pl, device=device) for pl in prompt_lens])
        seq_lens = torch.tensor(prompt_lens, dtype=torch.int32, device=device)

        logits = self.step(all_tokens, all_positions, seq_lens, num_decode_seqs=0)

        # Extract last-token logits for each sequence
        cu = seq_lens.cumsum(0)
        next_logits = logits[cu.long() - 1]  # [batch_size, vocab]

        # ── Phase 2: Decode loop (all seqs decode together) ──
        for _ in range(max_new_tokens):
            # Sample next tokens for all active seqs
            next_tokens = []
            for i in range(batch_size):
                if is_done[i]:
                    # Placeholder token for done seqs (won't affect others)
                    next_tokens.append(torch.zeros(1, dtype=torch.long, device=device))
                else:
                    t = self._sample(next_logits[i], temperature, top_k, top_p)
                    generated[i].append(t.unsqueeze(0))
                    if t.item() == eos_token_id or self.kv_cache.context_lens[i].item() >= max_seq_len:
                        is_done[i] = True
                    next_tokens.append(t.unsqueeze(0))

            if all(is_done):
                break

            # Build decode batch: all active seqs (decode in front, layout trivial since all are decode)
            active_ids = [i for i in range(batch_size) if not is_done[i]]
            num_active = len(active_ids)
            if num_active == 0:
                break

            tokens = torch.cat([next_tokens[i] for i in active_ids])
            positions = self.kv_cache.context_lens[active_ids].long()
            sl = torch.ones(num_active, dtype=torch.int32, device=device)

            # All are decode sequences
            logits = self.step(tokens, positions, sl, num_decode_seqs=num_active)

            # Scatter logits back to per-seq next_logits
            for j, i in enumerate(active_ids):
                next_logits[i] = logits[j]

        # Assemble outputs
        results = []
        for i in range(batch_size):
            gen_ids = torch.cat(generated[i]) if generated[i] else torch.tensor([], dtype=torch.long, device=device)
            results.append(torch.cat([prompts[i], gen_ids], dim=0))
        return results

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample a single token from logits with temperature, top-k, and top-p."""
        if temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_val = logits.topk(top_k, dim=-1).values[..., -1]
            logits = logits.where(logits >= kth_val, torch.full_like(logits, float('-inf')))

        if top_p < 1.0:
            sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_logits[remove_mask] = float('-inf')
            logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
