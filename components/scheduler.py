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
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from components.kv_cache import KVCache, PageOOMError
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
    max_gen_tokens: int = -1       # per-request generation cap (-1 = use max_seq_len)


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
        admit_mode: str = "pid",  # "pid" | "threshold" | "nopid"
        num_pages: int = 4096,
        stop_mu: float = 4.5,     # LogNormal mu for autonomous stop (IETF standard)
        stop_sigma: float = 1.2,  # LogNormal sigma for autonomous stop
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.eos_token_id = eos_token_id
        self.temperature = temperature
        self.device = next(model.parameters()).device

        # KV Cache — paged, FlashInfer-compatible
        from components.kv_cache import PAGE_SIZE
        self.kv_cache = KVCache(
            num_layers=model.num_of_layer,
            max_seq_len=model.max_seq_len,
            kv_head=model.kv_head,
            head_dim=model.head_dim,
            dtype=next(model.parameters()).dtype,
            device=self.device,
            num_pages=num_pages,
            page_size=PAGE_SIZE,
        )
        self.max_total_tokens = num_pages * PAGE_SIZE

        # PID Controller — controls page utilization towards target_ratio
        target_pages = target_ratio * num_pages
        self.pid = PIDController(
            kp=kp, ki=ki, kd=kd,
            setpoint=target_pages,
            output_min=0.0,
            output_max=float(max_batch_size),
        )

        # Admission mode
        self.admit_mode = admit_mode

        # Request management (keyed by seq_id = req.id)
        self.waiting_queue: deque = deque()
        self.active_pool: dict[int, Request] = {}  # req.id -> Request
        self._next_id = 0

        # Autonomous stop: LogNormal hazard rate for EOS injection
        self.stop_mu = stop_mu
        self.stop_sigma = stop_sigma
        self._rng = np.random.RandomState(0)

    # ═══════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════

    def add_request(self, prompt_ids: torch.Tensor, max_gen_tokens: int = -1) -> int:
        """Submit a new request. Returns request ID."""
        req = Request(id=self._next_id, prompt_ids=prompt_ids.to(self.device),
                      max_gen_tokens=max_gen_tokens)
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
        decode_ids = []   # seq_ids with is_prefilled=True
        prefill_ids = []  # seq_ids with is_prefilled=False
        for seq_id, req in sorted(self.active_pool.items()):
            if req.is_prefilled:
                decode_ids.append(seq_id)
            else:
                prefill_ids.append(seq_id)

        # Batch ordering: [decode_seqs | prefill_seqs]
        all_seq_ids = decode_ids + prefill_ids
        num_decode_seqs = len(decode_ids)

        # Phase 3: Assemble token-flat batch
        tokens_list = []
        positions_list = []
        seq_lens_list = []

        for seq_id in decode_ids:
            req = self.active_pool[seq_id]
            last_token = req.generated_ids[-1] if req.generated_ids else req.prompt_ids[-1:]
            if isinstance(last_token, torch.Tensor) and last_token.dim() == 0:
                last_token = last_token.unsqueeze(0)
            tokens_list.append(last_token)
            ctx_len = self.kv_cache.seq_lens[seq_id]
            pos = torch.tensor([ctx_len], dtype=torch.long, device=self.device)
            positions_list.append(pos)
            seq_lens_list.append(1)

        for seq_id in prefill_ids:
            req = self.active_pool[seq_id]
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
        self.kv_cache.prepare_batch(all_seq_ids)

        metadata = self._build_metadata(positions, seq_lens, num_decode_seqs, all_seq_ids)
        logits = self.model.forward(input_ids, positions, self.kv_cache, metadata)

        # Advance seq_lens in KV cache (allocates new pages if needed)
        advance_ids = all_seq_ids
        advance_counts = [int(seq_lens[i].item()) for i in range(len(all_seq_ids))]
        try:
            self.kv_cache.advance(advance_ids, advance_counts)
        except PageOOMError:
            # Preempt longest active sequences until advance succeeds
            self._preempt_until_advance_ok(advance_ids, advance_counts)

        # Phase 5: Sample + EOS check
        completed = []
        cu = seq_lens.cumsum(0).long()
        last_positions = cu - 1

        for i, seq_id in enumerate(all_seq_ids):
            if seq_id not in self.active_pool:
                continue  # preempted during OOM recovery
            req = self.active_pool[seq_id]
            token_logits = logits[last_positions[i]]

            next_token = self._sample(token_logits)
            req.generated_ids.append(next_token)
            req.is_prefilled = True

            # Check EOS or max length or per-request generation cap
            ctx_len = self.kv_cache.seq_lens[seq_id]
            gen_len = len(req.generated_ids)
            gen_limit = req.max_gen_tokens if req.max_gen_tokens > 0 else self.model.max_seq_len

            # Autonomous stop: inject EOS via LogNormal hazard rate
            # h(t) = f(t) / (1 - F(t)) where f,F are LogNormal PDF/CDF
            if gen_len >= 1 and req.max_gen_tokens <= 0:
                t = float(gen_len)
                # LogNormal CDF: F(t) = 0.5 * (1 + erf((ln(t) - mu) / (sigma * sqrt(2))))
                z = (math.log(t) - self.stop_mu) / self.stop_sigma
                cdf_t = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                # Hazard: h(t) = f(t) / (1 - F(t)), clamp survival > epsilon
                survival = max(1.0 - cdf_t, 1e-6)
                pdf_t = (1.0 / (t * self.stop_sigma * math.sqrt(2 * math.pi))) * \
                        math.exp(-0.5 * z * z)
                hazard = pdf_t / survival
                # Clamp hazard to [0, 1] and roll dice
                if self._rng.random() < min(hazard, 1.0):
                    next_token = torch.tensor(self.eos_token_id, device=next_token.device)

            if (next_token.item() == self.eos_token_id
                    or ctx_len >= self.model.max_seq_len
                    or gen_len >= gen_limit):
                req.is_done = True
                completed.append(req)
                self._evict(seq_id)

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
        """Admission control with three modes.

        Admission allocates initial pages for each new request's prompt.
        """
        if not self.waiting_queue:
            return

        import math
        page_size = self.kv_cache.page_size

        def _can_admit(req):
            """Check if pool has enough free pages for this request's prompt."""
            pages_needed = math.ceil(req.prompt_ids.size(0) / page_size)
            return self.kv_cache.num_free >= pages_needed

        def _do_admit(req):
            """Admit a request: allocate pages and add to active pool."""
            seq_id = req.id
            pages_needed = math.ceil(req.prompt_ids.size(0) / page_size)
            self.kv_cache.allocate_pages(seq_id, pages_needed)
            self.active_pool[seq_id] = req

        if self.admit_mode == "nopid":
            while self.waiting_queue and _can_admit(self.waiting_queue[0]):
                req = self.waiting_queue.popleft()
                _do_admit(req)
            return

        if self.admit_mode == "threshold":
            target = int(self.pid.setpoint)
            num_to_admit = max(0, target - len(self.active_pool))
            admitted = 0
            while admitted < num_to_admit and self.waiting_queue and _can_admit(self.waiting_queue[0]):
                req = self.waiting_queue.popleft()
                _do_admit(req)
                admitted += 1
            return

        # PID mode — PV is pages_used, SP is target_pages
        pv = float(self.kv_cache.num_pages - self.kv_cache.num_free)
        raw_output = self.pid.compute(pv)
        num_to_admit = max(0, int(raw_output))

        admitted = 0
        while admitted < num_to_admit and self.waiting_queue and _can_admit(self.waiting_queue[0]):
            req = self.waiting_queue.popleft()
            _do_admit(req)
            admitted += 1

    def _evict(self, seq_id: int):
        """Remove completed sequence from active pool, free its pages."""
        del self.active_pool[seq_id]
        self.kv_cache.release(seq_id)

    def _build_metadata(self, positions, seq_lens, num_decode_seqs, seq_ids):
        """Build AttentionMetadata for the current batch.

        Uses kv_cache.batch_seq_lens as context_lens (built by prepare_batch).
        block_tables is passed via kv_cache directly to Attention.forward.
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

        # ctx_lens: existing KV length + new tokens this step
        context_lens = self.kv_cache.batch_seq_lens
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

    def _preempt_until_advance_ok(self, advance_ids, advance_counts):
        """Evict longest active sequences until advance succeeds (OOM recovery)."""
        import math
        max_retries = 50
        for _ in range(max_retries):
            # Find longest active seq to preempt
            longest_id = max(
                self.active_pool.keys(),
                key=lambda sid: self.kv_cache.seq_lens.get(sid, 0)
            )
            # Evict: release pages, remove from active pool
            self.kv_cache.release(longest_id)
            evicted = self.active_pool.pop(longest_id)
            # Remove from advance list (it's already evicted)
            if longest_id in advance_ids:
                idx = advance_ids.index(longest_id)
                advance_ids.pop(idx)
                advance_counts.pop(idx)
            self._preempted_count = getattr(self, '_preempted_count', 0) + 1
            # Retry advance with remaining seqs
            try:
                self.kv_cache.advance(advance_ids, advance_counts)
                return
            except PageOOMError:
                continue
        # If still failing after retries, just skip advance
        pass
