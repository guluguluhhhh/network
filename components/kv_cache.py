"""
Static KV Cache for autoregressive decoding.

Pre-allocates contiguous buffers for all layers, supporting both
prefill (batch write) and decode (single-token append) modes.
"""

import torch


class KVCache:
    """Static KV Cache for autoregressive decoding.

    Pre-allocates contiguous buffers for all layers.
    Shape per layer: K=[max_seq_len, kv_head, head_dim], V=[max_seq_len, kv_head, head_dim]
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        kv_head: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.kv_head = kv_head
        self.head_dim = head_dim
        self.seq_len = 0  # current filled length

        # [num_layers, 2(K/V), max_seq_len, kv_head, head_dim]
        self.cache = torch.zeros(
            num_layers, 2, max_seq_len, kv_head, head_dim,
            dtype=dtype, device=device,
        )

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """Append new K/V tokens to the cache.

        Args:
            new_k: [new_tokens, kv_head, head_dim]
            new_v: [new_tokens, kv_head, head_dim]
        """
        new_len = new_k.size(0)
        assert self.seq_len + new_len <= self.max_seq_len, (
            f"KV cache overflow: {self.seq_len} + {new_len} > {self.max_seq_len}"
        )
        self.cache[layer_idx, 0, self.seq_len:self.seq_len + new_len] = new_k
        self.cache[layer_idx, 1, self.seq_len:self.seq_len + new_len] = new_v

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full K, V history for a layer. Returns contiguous views."""
        k = self.cache[layer_idx, 0, :self.seq_len].contiguous()
        v = self.cache[layer_idx, 1, :self.seq_len].contiguous()
        return k, v

    def advance(self, num_tokens: int):
        """Advance seq_len after all layers have been updated for this step."""
        self.seq_len += num_tokens

    def reset(self):
        """Reset cache for a new sequence."""
        self.seq_len = 0
