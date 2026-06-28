"""
PID Continuous Batching Scheduler.

Control theory driven admission control:
- PV: sum(context_lens) — total KV tokens currently in use
- SP: target_ratio * max_total_tokens — target KV utilization
- CV: admission_budget (tokens) — how many prompt tokens to admit per step

Every decode step:
1. All active sequences decode 1 token (mandatory)
2. EOS sequences evicted (slot freed)
3. PID measures PV, computes budget
4. Admits new requests from waiting queue within budget
5. Assembles mixed batch: [decode_seqs | new_prefill_seqs]
6. model.forward()
"""

import torch
import triton
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from components.kv_cache import KVCache
from components.pid_controller import PIDController


# Lazy import to avoid circular dependency
def _get_attention_metadata_class():
    from network import AttentionMetadata
    return AttentionMetadata


@dataclass
class Request:
    """A single inference request."""
    id: int
    prompt_ids: torch.Tensor       # [prompt_len] on device
    generated_ids: list = field(default_factory=list)  # list of scalar tensors
    is_prefilled: bool = False
    is_done: bool = False
    slot_id: int = -1              # assigned KV cache slot


class PIDScheduler:
    """Continuous batching scheduler with PID-controlled admission.

    Batch layout per step: [decode_seqs | prefill_seqs]
    Kernel routes via num_decode_seqs index split.
    """

    def __init__(
        self,
        model,
        max_batch_size: int = 32,
        target_ratio: float = 0.8,
        kp: float = 1.0,
        ki: float = 0.01,
        kd: float = 0.1,
        eos_token_id: int = 2,
        temperature: float = 0.0,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_batch_size * model.max_seq_len
        self.eos_token_id = eos_token_id
        self.temperature = temperature
        self.device = next(model.parameters()).device

        # KV Cache — pool-based, zero-copy
        self.kv_cache = KVCache(
            num_layers=model.num_of_layer,
            max_seq_len=model.max_seq_len,
            kv_head=model.kv_head,
            head_dim=model.head_dim,
            dtype=next(model.parameters()).dtype,
            device=self.device,
            max_slots=max_batch_size,
        )

        # PID Controller — controls active_seqs count
        # SP = target_active = target_ratio * max_slots (direct: 80% of pool capacity)
        target_active = target_ratio * max_batch_size
        self.pid = PIDController(
            kp=kp, ki=ki, kd=kd,
            setpoint=target_active,
            output_min=0.0,
            output_max=float(max_batch_size),
        )

        # Request management
        self.waiting_queue: deque = deque()
        self.active_pool: dict[int, Request] = {}  # slot_id -> Request
        self._next_id = 0

    # ═══════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════

    def add_request(self, prompt_ids: torch.Tensor) -> int:
        """Submit a new request. Returns request ID."""
        req = Request(id=self._next_id, prompt_ids=prompt_ids.to(self.device))
        self._next_id += 1
        self.waiting_queue.append(req)
        return req.id

    def has_work(self) -> bool:
        """True if there are active sequences or waiting requests."""
        return len(self.active_pool) > 0 or len(self.waiting_queue) > 0

    def step(self) -> list[Request]:
        """Execute one scheduling iteration.

        Returns: list of completed requests this step.
        """
        # Phase 1: PID admission — move requests from queue to active pool
        self._pid_admit()

        # If nothing active, nothing to do
        if not self.active_pool:
            return []

        # Phase 2: Separate decode vs prefill
        decode_slots = []   # slots with is_prefilled=True (need 1 decode token)
        prefill_slots = []  # slots with is_prefilled=False (need full prompt)
        for slot_id, req in sorted(self.active_pool.items()):
            if req.is_prefilled:
                decode_slots.append(slot_id)
            else:
                prefill_slots.append(slot_id)

        # Phase 3: Assemble token-flat batch [decode | prefill]
        # Batch layout: decode seqs first, prefill seqs after
        all_slots = decode_slots + prefill_slots  # physical ordering
        num_decode_seqs = len(decode_slots)

        tokens_list = []
        positions_list = []
        seq_lens_list = []

        for slot_id in decode_slots:
            req = self.active_pool[slot_id]
            # Latest token to decode
            last_token = req.generated_ids[-1] if req.generated_ids else req.prompt_ids[-1:]
            if isinstance(last_token, torch.Tensor) and last_token.dim() == 0:
                last_token = last_token.unsqueeze(0)
            tokens_list.append(last_token)
            pos = torch.tensor([self.kv_cache.used_slots[slot_id]], dtype=torch.long, device=self.device)
            positions_list.append(pos)
            seq_lens_list.append(1)

        for slot_id in prefill_slots:
            req = self.active_pool[slot_id]
            tokens_list.append(req.prompt_ids)
            plen = req.prompt_ids.size(0)
            positions_list.append(torch.arange(plen, device=self.device))
            seq_lens_list.append(plen)

        if not tokens_list:
            return []

        input_ids = torch.cat(tokens_list)
        positions = torch.cat(positions_list)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=self.device)

        # Phase 4: Build metadata + forward
        # prepare_batch builds contiguous cache tensor from active slots
        self.kv_cache.prepare_batch(all_slots)

        metadata = self._build_metadata(positions, seq_lens, num_decode_seqs, all_slots)
        logits = self.model.forward(input_ids, positions, self.kv_cache, metadata)

        # Writeback not needed — kernel writes directly to pool

        # Advance context_lens for active slots
        self.kv_cache.advance(seq_lens)

        # Phase 5: Sample + EOS check
        completed = []
        # Extract per-seq logits (last token of each seq)
        cu = seq_lens.cumsum(0).long()
        last_positions = cu - 1  # index of last token per seq

        for i, slot_id in enumerate(all_slots):
            req = self.active_pool[slot_id]
            token_logits = logits[last_positions[i]]

            # Sample
            next_token = self._sample(token_logits)
            req.generated_ids.append(next_token)
            req.is_prefilled = True

            # Check EOS or max length
            ctx_len = self.kv_cache.used_slots.get(slot_id, 0)
            if next_token.item() == self.eos_token_id or ctx_len >= self.model.max_seq_len:
                req.is_done = True
                completed.append(req)
                self._evict(slot_id)

        return completed

    @torch.inference_mode()
    def run_to_completion(self) -> list[Request]:
        """Run all requests until all complete. Returns all completed requests."""
        all_completed = []
        while self.has_work():
            completed = self.step()
            all_completed.extend(completed)
        return all_completed

    # ═══════════════════════════════════════════════════════════════════════
    # Internal
    # ═══════════════════════════════════════════════════════════════════════

    def _pid_admit(self):
        """PID controls active_seqs count (not KV tokens directly).

        PV = current active sequence count (instant feedback)
        SP = target_active = target_kv / (max_seq_len / 2)
        CV = num_to_admit this step

        This eliminates dead-time: admit +1 -> PV immediately +1.
        KV naturally follows: KV ≈ active × avg_context_len.
        """
        if not self.waiting_queue:
            return

        # PV = current active count (instant feedback, no delay)
        pv = float(len(self.active_pool))

        # PID computes how many requests to admit
        raw_output = self.pid.compute(pv)
        num_to_admit = max(0, int(raw_output))

        # Admit up to num_to_admit requests
        admitted = 0
        while admitted < num_to_admit and self.waiting_queue and self.kv_cache.num_free > 0:
            req = self.waiting_queue.popleft()
            slot = self.kv_cache.allocate()  # pool returns next free slot
            req.slot_id = slot
            self.active_pool[slot] = req
            admitted += 1

    def _evict(self, slot_id: int):
        """Remove completed sequence from active pool, free its GPU memory."""
        del self.active_pool[slot_id]
        self.kv_cache.release(slot_id)  # GPU memory freed immediately

    def _build_metadata(self, positions, seq_lens, num_decode_seqs, slot_ids):
        """Build AttentionMetadata for the current batch.

        NOTE: The model reads kv_cache.cache[:num_seqs, layer, ...].
        We must ensure batch seq i maps to kv_cache slot slot_ids[i].
        Since Attention uses metadata.seq_ids to index into cache views,
        and cache view is kv_cache.cache[:num_seqs], we need slot_ids
        to be exactly [0, 1, ..., num_seqs-1] for correctness.

        Current approach: slot_ids is sorted ascending, and we pass
        kv_cache directly. The kernel indexes cache[seq_id] which must
        match the physical slot. This works when active slots are
        the first N contiguous slots.
        """
        AttentionMetadata = _get_attention_metadata_class()

        device = positions.device
        num_seqs = seq_lens.size(0)
        num_tokens = positions.size(0)

        # seq_ids: maps each token to batch index (0..num_seqs-1)
        seq_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=device, dtype=torch.int64), seq_lens.long()
        )

        # cu_seqlens_q
        cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        cu_seqlens_q[1:] = seq_lens.to(torch.int32).cumsum(0)

        # ctx_lens: read from batch view (built by prepare_batch)
        context_lens = self.kv_cache.context_lens
        ctx_lens = context_lens + seq_lens.to(torch.int32)

        # Scalars
        num_decode_tokens = int(seq_lens[:num_decode_seqs].sum().item()) if num_decode_seqs > 0 else 0
        max_kv_len = int(ctx_lens.max().item()) if num_seqs > 0 else 0

        # Prefill metadata
        num_prefill_seqs = num_seqs - num_decode_seqs
        cu_seqlens_prefill = None
        tile_seq_ids_prefill = None
        if num_prefill_seqs > 0:
            if num_decode_seqs > 0:
                cu_seqlens_prefill = cu_seqlens_q[num_decode_seqs:] - cu_seqlens_q[num_decode_seqs]
            else:
                cu_seqlens_prefill = cu_seqlens_q
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

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy or temperature sampling."""
        if self.temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / self.temperature
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
