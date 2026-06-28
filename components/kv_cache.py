"""
Pool-based KV Cache with slot_mapping indirect addressing.

Physical layout: one pre-allocated pool tensor [max_slots, layers, 2, max_seq_len, kv_head, head_dim]
Logical management: PID scheduler controls which slots are in use.
Kernel access: pool_ptr + slot_mapping[batch_idx] * slot_stride + pos * pos_stride + head * head_stride

Zero-copy per step: prepare_batch only constructs a small slot_mapping tensor (~N int32s).
No torch.stack, no writeback. Kernel reads/writes pool directly.
"""

import torch


class KVCache:
    """Pool-based KV Cache with slot_mapping for indirect addressing."""

    def __init__(self, num_layers: int, max_seq_len: int, kv_head: int, head_dim: int,
                 dtype: torch.dtype = torch.float16, device: str = "cuda",
                 max_slots: int = 16384):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.kv_head = kv_head
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.max_slots = max_slots

        # Pool: one-time allocation (physical memory limit)
        # Shape: [max_slots, num_layers, 2(K/V), max_seq_len, kv_head, head_dim]
        self.pool = torch.zeros(
            max_slots, num_layers, 2, max_seq_len, kv_head, head_dim,
            dtype=dtype, device=device,
        )

        # Slot management
        self.free_slots: set = set(range(max_slots))
        self.used_slots: dict[int, int] = {}  # slot_id -> context_len

        # Batch view (rebuilt each step by prepare_batch)
        self.slot_mapping: torch.Tensor | None = None  # [num_active] int32
        self.context_lens: torch.Tensor | None = None  # [num_active] int32
        self._batch_slot_ids: list[int] = []

    # ═══════════════════════════════════════════════════════════════════════
    # Slot lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    def allocate(self) -> int:
        """Allocate a slot from the pool. Called when PID admits a request."""
        slot = self.free_slots.pop()
        self.used_slots[slot] = 0
        return slot

    def release(self, slot_id: int):
        """Release a slot back to pool. Called on EOS."""
        if slot_id in self.used_slots:
            del self.used_slots[slot_id]
            self.free_slots.add(slot_id)

    @property
    def num_active(self) -> int:
        return len(self.used_slots)

    @property
    def num_free(self) -> int:
        return len(self.free_slots)

    @property
    def total_kv_tokens(self) -> int:
        """Total KV tokens in use. PID's process variable."""
        return sum(self.used_slots.values())

    # ═══════════════════════════════════════════════════════════════════════
    # Batch preparation (called once per step, zero-copy)
    # ═══════════════════════════════════════════════════════════════════════

    def prepare_batch(self, batch_slot_ids: list[int]):
        """Construct slot_mapping + context_lens tensors for kernel.

        This is the ONLY per-step operation. No data copy — just build
        two small index tensors. Kernel uses slot_mapping to index into pool.
        """
        self._batch_slot_ids = batch_slot_ids
        if not batch_slot_ids:
            self.slot_mapping = None
            self.context_lens = None
            return

        self.slot_mapping = torch.tensor(
            batch_slot_ids, dtype=torch.int32, device=self.device
        )
        self.context_lens = torch.tensor(
            [self.used_slots[s] for s in batch_slot_ids],
            dtype=torch.int32, device=self.device,
        )

    def advance(self, seq_lens: torch.Tensor):
        """Advance context_lens for current batch after forward."""
        for i, sid in enumerate(self._batch_slot_ids):
            self.used_slots[sid] += int(seq_lens[i].item())

    def reset(self):
        """Release all slots."""
        self.free_slots = set(range(self.max_slots))
        self.used_slots.clear()
        self.slot_mapping = None
        self.context_lens = None
        self._batch_slot_ids = []
