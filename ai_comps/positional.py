"""
Positional encoding schemes for transformers.

Currently implemented:
- Sinusoidal: Fixed positional encodings (Vaswani et al., 2017)
- Learned: Trainable positional embeddings

Planned for future implementation:
- RoPE: Rotary Position Embeddings (Su et al., 2021)
- ALiBi: Attention with Linear Biases
"""

import numpy as np
from typing import Optional, Tuple


def sinusoidal_encoding(max_len: int, d_model: int, dtype=np.float32) -> np.ndarray:
    """
    Fixed sinusoidal positional encodings (Vaswani et al., 2017).

    Formula:
        PE[pos, 2i]   = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

    Args:
        max_len: Maximum sequence length T.
        d_model: Model dimension D.
        dtype: Output dtype.

    Returns:
        PE: Positional encodings of shape (T, D).
    """
    pos = np.arange(max_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    PE = np.zeros((max_len, d_model), dtype=dtype)
    PE[:, 0::2] = np.sin(angle[:, 0::2])
    PE[:, 1::2] = np.cos(angle[:, 1::2])
    return PE


class LearnedPositionalEmbedding:
    """
    Learned positional embeddings (GPT-2 style).

    Simple lookup table for position embeddings, trained alongside the model.

    Attributes:
        W: Position embedding matrix (max_len, D).
        gradW: Accumulated gradients.
    """

    def __init__(self, max_len: int, d_model: int, seed: int = 0) -> None:
        """
        Args:
            max_len: Maximum sequence length.
            d_model: Embedding dimension.
            seed: RNG seed for reproducible init.
        """
        rng = np.random.default_rng(seed)
        self.max_len = max_len
        self.d_model = d_model
        self.W = rng.normal(0.0, 0.02, size=(max_len, d_model)).astype(np.float32)
        self.gradW = np.zeros_like(self.W)

    def forward(self, seq_len: int) -> np.ndarray:
        """
        Get position embeddings for a sequence.

        Args:
            seq_len: Length of sequence (must be <= max_len).

        Returns:
            PE: Position embeddings of shape (seq_len, D).
        """
        assert seq_len <= self.max_len, f"seq_len {seq_len} > max_len {self.max_len}"
        self._seq_len = seq_len
        return self.W[:seq_len]

    def backward(self, dPE: np.ndarray) -> None:
        """
        Backward pass - accumulate gradients.

        Args:
            dPE: Gradient wrt position embeddings (seq_len, D).
        """
        seq_len = self._seq_len
        self.gradW[:seq_len] += dPE.sum(axis=0) if dPE.ndim == 3 else dPE

    def step(self, lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        """SGD update for position embeddings."""
        if weight_decay != 0.0:
            self.gradW += weight_decay * self.W
        self.W -= lr * self.gradW
        self.gradW.fill(0.0)


class RotaryPositionalEmbedding:
    """
    Rotary Position Embeddings (RoPE) - Su et al., 2021.

    Instead of adding position information, RoPE rotates query and key vectors
    based on their position. This allows the model to learn relative positions
    naturally through the dot product.

    The rotation is applied to pairs of dimensions:
        [q_0, q_1] -> [q_0*cos(θ) - q_1*sin(θ), q_0*sin(θ) + q_1*cos(θ)]

    Benefits:
    - Encodes relative position in attention scores
    - Enables length extrapolation
    - No additional parameters
    """

    def __init__(self, d_head: int, max_len: int = 4096, base: float = 10000.0) -> None:
        """
        Args:
            d_head: Dimension per attention head (must be even).
            max_len: Maximum sequence length to precompute.
            base: Base for the frequency computation.
        """
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        self.d_head = d_head
        self.max_len = max_len
        self.base = base

        # Precompute frequency bands: theta_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head))
        self.inv_freq = inv_freq  # (d_head/2,)

        # Precompute sin/cos for all positions
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        """Precompute sin/cos tables for positions 0..seq_len-1."""
        pos = np.arange(seq_len, dtype=np.float32)[:, None]  # (T, 1)
        angles = pos * self.inv_freq[None, :]  # (T, d_head/2)
        self._cos_cache = np.cos(angles).astype(np.float32)  # (T, d_head/2)
        self._sin_cache = np.sin(angles).astype(np.float32)  # (T, d_head/2)

    def forward(
        self, q: np.ndarray, k: np.ndarray, offset: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (..., T, d_head).
            k: Key tensor of shape (..., T, d_head).
            offset: Position offset for KV-cache scenarios.

        Returns:
            q_rot: Rotated queries, same shape as q.
            k_rot: Rotated keys, same shape as k.
        """
        T = q.shape[-2]
        assert offset + T <= self.max_len, "Sequence too long for precomputed cache"

        cos = self._cos_cache[offset : offset + T]  # (T, d_head/2)
        sin = self._sin_cache[offset : offset + T]  # (T, d_head/2)

        # Split into pairs and rotate
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)

        return q_rot, k_rot

    def _apply_rotation(
        self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray
    ) -> np.ndarray:
        """
        Apply rotation to tensor x using precomputed cos/sin.

        Args:
            x: Input tensor (..., T, d_head).
            cos: Cosine values (T, d_head/2).
            sin: Sine values (T, d_head/2).

        Returns:
            Rotated tensor, same shape as x.
        """
        # Split x into even and odd indices
        x_even = x[..., 0::2]  # (..., T, d_head/2)
        x_odd = x[..., 1::2]  # (..., T, d_head/2)

        # Broadcast cos/sin to match x shape
        # cos, sin are (T, d_head/2), need to broadcast to (..., T, d_head/2)

        # Apply rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # Interleave back
        x_rot = np.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd

        return x_rot


# Factory function
def get_positional_encoding(
    name: str,
    max_len: int,
    d_model: int,
    **kwargs
):
    """
    Get positional encoding by name.

    Args:
        name: One of 'sinusoidal', 'learned', 'rope'.
        max_len: Maximum sequence length.
        d_model: Model/head dimension.
        **kwargs: Additional arguments.

    Returns:
        Positional encoding (array for sinusoidal, object for others).
    """
    if name == "sinusoidal":
        return sinusoidal_encoding(max_len, d_model, **kwargs)
    elif name == "learned":
        return LearnedPositionalEmbedding(max_len, d_model, **kwargs)
    elif name == "rope":
        return RotaryPositionalEmbedding(d_model, max_len, **kwargs)
    else:
        raise KeyError(f"Unknown positional encoding: {name}")
