"""
KV-cache for efficient autoregressive transformer inference.

Stores computed K/V tensors to avoid recomputation during generation.
"""

import numpy as np
from typing import Optional, Tuple, List


class KVCache:
    """KV cache for a single attention layer. Shape: (B, h, T, d)."""

    def __init__(
        self,
        batch_size: int,
        n_heads: int,
        max_seq_len: int,
        d_head: int,
        dtype=np.float32,
    ) -> None:
        """Initialize empty cache with pre-allocated arrays."""
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.d_head = d_head
        self.dtype = dtype

        # Pre-allocate cache
        self.k_cache = np.zeros(
            (batch_size, n_heads, max_seq_len, d_head), dtype=dtype
        )
        self.v_cache = np.zeros(
            (batch_size, n_heads, max_seq_len, d_head), dtype=dtype
        )
        self.seq_len = 0

    def update(
        self, k_new: np.ndarray, v_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Append new K/V and return full cached tensors."""
        new_tokens = k_new.shape[2]
        new_seq_len = self.seq_len + new_tokens

        if new_seq_len > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: {new_seq_len} > max_seq_len {self.max_seq_len}"
            )

        # Store new K, V
        self.k_cache[:, :, self.seq_len : new_seq_len, :] = k_new
        self.v_cache[:, :, self.seq_len : new_seq_len, :] = v_new
        self.seq_len = new_seq_len

        # Return full cache up to current position
        return (
            self.k_cache[:, :, :self.seq_len, :],
            self.v_cache[:, :, :self.seq_len, :],
        )

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current cached K/V without updating."""
        return (
            self.k_cache[:, :, :self.seq_len, :],
            self.v_cache[:, :, :self.seq_len, :],
        )

    def reset(self) -> None:
        """Clear the cache."""
        self.seq_len = 0
        # Optionally zero out for cleaner debugging
        self.k_cache.fill(0.0)
        self.v_cache.fill(0.0)

    @property
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return self.seq_len == 0


class LayerKVCache:
    """KV cache manager for all transformer layers."""

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        max_seq_len: int,
        d_head: int,
        dtype=np.float32,
    ) -> None:
        """Initialize KV caches for all layers."""
        self.n_layers = n_layers
        self.caches: List[KVCache] = [
            KVCache(batch_size, n_heads, max_seq_len, d_head, dtype)
            for _ in range(n_layers)
        ]

    def __getitem__(self, layer_idx: int) -> KVCache:
        """Get cache for a specific layer."""
        return self.caches[layer_idx]

    def reset(self) -> None:
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()

    @property
    def seq_len(self) -> int:
        """Current sequence length (same for all layers)."""
        return self.caches[0].seq_len if self.caches else 0


def apply_kv_cache(
    k: np.ndarray,
    v: np.ndarray,
    cache: Optional[KVCache],
) -> Tuple[np.ndarray, np.ndarray]:
    """Update cache with new K/V and return full tensors, or pass through if no cache."""
    if cache is None:
        return k, v
    return cache.update(k, v)
