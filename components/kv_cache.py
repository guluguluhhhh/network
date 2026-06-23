"""Multi-sequence Static KV Cache for token-flat inference."""

import torch


class KVCache:
    """Multi-sequence KV Cache. Each sequence has independent context length."""

    def __init__(self, num_seqs: int, num_layers: int, max_seq_len: int,
                 kv_head: int, head_dim: int,
                 dtype: torch.dtype = torch.float16, device: str = "cuda"):
        self.num_seqs = num_seqs
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.kv_head = kv_head
        self.head_dim = head_dim
        # [num_seqs, num_layers, 2(K/V), max_seq_len, kv_head, head_dim]
        self.cache = torch.zeros(
            num_seqs, num_layers, 2, max_seq_len, kv_head, head_dim,
            dtype=dtype, device=device)
        self.context_lens = torch.zeros(num_seqs, dtype=torch.int32, device=device)

    def update(self, layer_idx: int, seq_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """Write K/V for one sequence. new_k/v: [num_tokens, kv_head, head_dim]"""
        pos = self.context_lens[seq_idx].item()
        n = new_k.size(0)
        self.cache[seq_idx, layer_idx, 0, pos:pos+n] = new_k
        self.cache[seq_idx, layer_idx, 1, pos:pos+n] = new_v

    def get_kv(self, layer_idx: int, seq_idx: int):
        """Get K, V history for one sequence. Returns [ctx_len, kv_head, head_dim] each."""
        L = self.context_lens[seq_idx].item()
        return (self.cache[seq_idx, layer_idx, 0, :L].contiguous(),
                self.cache[seq_idx, layer_idx, 1, :L].contiguous())

    def advance(self, seq_lens: torch.Tensor):
        """Advance context_lens. seq_lens: [num_seqs]."""
        self.context_lens += seq_lens.to(self.context_lens.dtype)

    def reset(self):
        self.context_lens.zero_()
