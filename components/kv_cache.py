"""
Paged KV Cache with block_tables indirect addressing (FlashInfer-compatible).

Physical layout: one pre-allocated pool tensor [num_pages, num_layers, 2, page_size, kv_head, head_dim]
Logical management: block_tables[seq_id, logical_page_idx] -> physical_page_id
Kernel access: pool[block_tables[seq, pos // page_size], layer, kv, pos % page_size, head, :]

FlashInfer compatibility:
  - Per-layer slice: pool[:, layer_idx] -> [num_pages, 2, page_size, kv_head, head_dim] (NHD layout)
  - block_tables: [batch_size, max_blocks_per_seq] int32
  - last_page_len: [batch_size] int32

Zero-copy per step: prepare_batch only constructs small index tensors.
"""

import math
import torch


PAGE_SIZE = 128  # = BLOCK_N, aligned with kernel tile size


class PageOOMError(RuntimeError):
    """Raised when the page pool is exhausted."""
    pass


class KVCache:
    """Paged KV Cache with block_tables for indirect addressing (FlashInfer-compatible)."""

    def __init__(self, num_layers: int, max_seq_len: int, kv_head: int, head_dim: int,
                 dtype: torch.dtype = torch.float16, device: str = "cuda",
                 num_pages: int = 4096, page_size: int = PAGE_SIZE):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.kv_head = kv_head
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.num_pages = num_pages
        self.page_size = page_size
        self.max_blocks_per_seq = math.ceil(max_seq_len / page_size)

        # Pool: one-time allocation (physical memory limit)
        # Shape: [num_pages, num_layers, 2(K/V), page_size, kv_head, head_dim]
        self.pool = torch.zeros(
            num_pages, num_layers, 2, page_size, kv_head, head_dim,
            dtype=dtype, device=device,
        )

        # Page management
        self.free_pages: set = set(range(num_pages))
        self.seq_page_table: dict[int, list[int]] = {}  # seq_id -> [phys_page_0, phys_page_1, ...]
        self.seq_lens: dict[int, int] = {}  # seq_id -> total KV length (context + new tokens)

        # Batch view (rebuilt each step by prepare_batch)
        self.block_tables: torch.Tensor | None = None   # [batch_size, max_blocks_per_seq] int32
        self.batch_seq_lens: torch.Tensor | None = None  # [batch_size] int32
        self.last_page_len: torch.Tensor | None = None   # [batch_size] int32
        self._batch_seq_ids: list[int] = []

    # ===================================================================
    # Page lifecycle
    # ===================================================================

    def allocate_pages(self, seq_id: int, num_new: int):
        """Allocate pages for a seq. Called on prefill or when last page is full."""
        if num_new > len(self.free_pages):
            raise PageOOMError(
                f"Need {num_new} pages but only {len(self.free_pages)} free")
        if seq_id not in self.seq_page_table:
            self.seq_page_table[seq_id] = []
            self.seq_lens[seq_id] = 0
        for _ in range(num_new):
            page = self.free_pages.pop()
            self.seq_page_table[seq_id].append(page)

    def release(self, seq_id: int):
        """Release all pages for a seq. Called on EOS."""
        if seq_id in self.seq_page_table:
            for page in self.seq_page_table[seq_id]:
                self.free_pages.add(page)
            del self.seq_page_table[seq_id]
            del self.seq_lens[seq_id]

    @property
    def num_free(self) -> int:
        return len(self.free_pages)

    @property
    def total_kv_tokens(self) -> int:
        """Total KV tokens in use across all seqs."""
        return sum(self.seq_lens.values())

    # ===================================================================
    # Batch preparation (called once per step, zero-copy)
    # ===================================================================

    def prepare_batch(self, batch_seq_ids: list[int]):
        """Construct block_tables + seq_lens + last_page_len tensors for kernel.

        Output tensors are FlashInfer-compatible:
          block_tables:  [batch_size, max_blocks_per_seq] int32
          batch_seq_lens: [batch_size] int32
          last_page_len: [batch_size] int32
        """
        self._batch_seq_ids = batch_seq_ids
        if not batch_seq_ids:
            self.block_tables = None
            self.batch_seq_lens = None
            self.last_page_len = None
            return

        batch_size = len(batch_seq_ids)
        # Build block_tables (pad with 0 for unused entries)
        bt = torch.zeros(batch_size, self.max_blocks_per_seq, dtype=torch.int32, device=self.device)
        sl = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        lpl = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

        for i, seq_id in enumerate(batch_seq_ids):
            pages = self.seq_page_table[seq_id]
            seq_len = self.seq_lens[seq_id]
            for j, page_id in enumerate(pages):
                bt[i, j] = page_id
            sl[i] = seq_len
            # last_page_len: how many tokens are valid in the last page
            if seq_len > 0:
                lpl[i] = ((seq_len - 1) % self.page_size) + 1
            else:
                lpl[i] = 0

        self.block_tables = bt
        self.batch_seq_lens = sl
        self.last_page_len = lpl

    def advance(self, seq_ids: list[int], token_counts: list[int]):
        """Advance seq_lens after forward. Allocate new page if needed."""
        for seq_id, count in zip(seq_ids, token_counts):
            new_len = self.seq_lens[seq_id] + count
            self.seq_lens[seq_id] = new_len
            # Allocate more pages only if current allocation is insufficient
            pages_needed = math.ceil(new_len / self.page_size) if new_len > 0 else 0
            pages_have = len(self.seq_page_table[seq_id])
            if pages_needed > pages_have:
                self.allocate_pages(seq_id, pages_needed - pages_have)

    def reset(self):
        """Release all pages."""
        self.free_pages = set(range(self.num_pages))
        self.seq_page_table.clear()
        self.seq_lens.clear()
        self.block_tables = None
        self.batch_seq_lens = None
        self.last_page_len = None
        self._batch_seq_ids = []
