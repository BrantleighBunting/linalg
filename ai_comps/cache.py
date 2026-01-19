"""
Caching mechanisms for efficient transformer inference.

Currently implemented:
- KVCache: Key-Value cache for autoregressive generation

The KV cache stores previously computed key and value tensors, avoiding
redundant computation during autoregressive generation. This provides
O(1) computation per new token instead of O(n) where n is sequence length.
"""

import numpy as np
from typing import Optional, Tuple, List


class KVCache:
    """
    Key-Value cache for a single attention layer.

    During autoregressive generation, we only need to compute Q, K, V for the
    new token, then concatenate K, V with cached values from previous tokens.

    Attributes:
        k_cache: Cached keys, shape (B, n_heads, seq_len, d_head).
        v_cache: Cached values, shape (B, n_heads, seq_len, d_head).
        seq_len: Current sequence length in cache.
    """

    def __init__(
        self,
        batch_size: int,
        n_heads: int,
        max_seq_len: int,
        d_head: int,
        dtype=np.float32,
    ) -> None:
        """
        Initialize empty KV cache.

        Args:
            batch_size: Batch size B.
            n_heads: Number of attention heads.
            max_seq_len: Maximum sequence length to cache.
            d_head: Dimension per head.
            dtype: Data type for cache arrays.
        """
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
        """
        Append new K, V to cache and return full K, V for attention.

        Args:
            k_new: New keys (B, n_heads, new_tokens, d_head).
            v_new: New values (B, n_heads, new_tokens, d_head).

        Returns:
            k_full: All keys up to current position (B, n_heads, seq_len, d_head).
            v_full: All values up to current position (B, n_heads, seq_len, d_head).

        Raises:
            ValueError: If cache would exceed max_seq_len.
        """
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
        """
        Get current cached K, V without updating.

        Returns:
            k: Cached keys (B, n_heads, seq_len, d_head).
            v: Cached values (B, n_heads, seq_len, d_head).
        """
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
    """
    KV cache manager for all layers in a transformer.

    Manages a list of KVCache objects, one per layer.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        max_seq_len: int,
        d_head: int,
        dtype=np.float32,
    ) -> None:
        """
        Initialize KV caches for all layers.

        Args:
            n_layers: Number of transformer layers.
            batch_size: Batch size B.
            n_heads: Number of attention heads per layer.
            max_seq_len: Maximum sequence length.
            d_head: Dimension per head.
            dtype: Data type for cache arrays.
        """
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
    """
    Helper to apply KV cache if provided.

    Args:
        k: New keys (B, h, T_new, d).
        v: New values (B, h, T_new, d).
        cache: Optional KVCache to update and use.

    Returns:
        k_full: Keys for attention (B, h, T_full, d).
        v_full: Values for attention (B, h, T_full, d).
    """
    if cache is None:
        return k, v
    return cache.update(k, v)
