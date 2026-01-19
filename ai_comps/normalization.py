"""
Normalization layers for neural networks.

Currently implemented:
- LayerNorm: Standard layer normalization with learnable scale/shift

Planned for future implementation:
- RMSNorm: Root Mean Square normalization (faster, no mean centering)
"""

import numpy as np
from typing import Dict, Tuple


class LayerNorm:
    """
    Layer Normalization over the last axis with learnable scale/shift.

    Normalizes per-token features to zero mean and unit variance:
        mu = mean(x, axis=-1, keepdims=True)
        sigma = sqrt(var(x) + eps)
        xhat = (x - mu) / sigma
        y = gamma * xhat + beta

    Attributes:
        gamma: Learnable scale, shape (D,).
        beta:  Learnable shift, shape (D,).
    """

    def __init__(self, d_model: int = 512) -> None:
        """
        Args:
            d_model: Feature dimension D (size of last axis to normalize).
        """
        self.d_model = d_model
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self._grads: Dict[str, np.ndarray] = {
            "gamma": np.zeros(d_model, dtype=np.float32),
            "beta": np.zeros(d_model, dtype=np.float32),
        }
        self._cache = None

    def forward(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (..., D).
            eps: Small constant for numerical stability.

        Returns:
            y: Output of same shape as x.
        """
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        sigma = np.sqrt(var + eps)
        xhat = (x - mu) / sigma
        y = xhat * self.gamma + self.beta
        self._cache = (xhat, sigma, self.gamma.copy())
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dy: Upstream gradient, same shape as y.

        Returns:
            dx: Gradient with respect to x, same shape as dy.
        """
        xhat, sigma, gamma = self._cache
        ghat = dy * gamma
        m1 = ghat.mean(axis=-1, keepdims=True)
        m2 = (ghat * xhat).mean(axis=-1, keepdims=True)
        dx = (ghat - m1 - xhat * m2) / sigma

        sum_axes = tuple(range(dy.ndim - 1))
        dgamma = (dy * xhat).sum(axis=sum_axes)
        dbeta = dy.sum(axis=sum_axes)
        self._grads["gamma"] = dgamma
        self._grads["beta"] = dbeta
        return dx

    def step(self, lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        """
        SGD parameter update for gamma and beta.

        Args:
            lr: Learning rate.
            weight_decay: L2 penalty coefficient (typically 0 for LN params).
        """
        if weight_decay != 0.0:
            self._grads["gamma"] += weight_decay * self.gamma
        self.gamma -= lr * self._grads["gamma"]
        self.beta -= lr * self._grads["beta"]
        self._grads["gamma"].fill(0.0)
        self._grads["beta"].fill(0.0)

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        """Access gradients for external optimizers."""
        return self._grads


class RMSNorm:
    """
    Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm - no mean centering:
        rms = sqrt(mean(x^2) + eps)
        y = (x / rms) * gamma

    Used in LLaMA, Gemma, and other modern architectures.

    Attributes:
        gamma: Learnable scale, shape (D,).
    """

    def __init__(self, d_model: int = 512, eps: float = 1e-6) -> None:
        """
        Args:
            d_model: Feature dimension D.
            eps: Small constant for numerical stability.
        """
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self._grads: Dict[str, np.ndarray] = {
            "gamma": np.zeros(d_model, dtype=np.float32),
        }
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (..., D).

        Returns:
            y: Normalized output of same shape as x.
        """
        rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + self.eps)
        xnorm = x / rms
        y = xnorm * self.gamma
        self._cache = (x, xnorm, rms)
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dy: Upstream gradient, same shape as y.

        Returns:
            dx: Gradient with respect to x, same shape as dy.
        """
        x, xnorm, rms = self._cache
        D = x.shape[-1]

        # dgamma
        sum_axes = tuple(range(dy.ndim - 1))
        dgamma = (dy * xnorm).sum(axis=sum_axes)
        self._grads["gamma"] = dgamma

        # dx: d/dx of (x / rms * gamma)
        # Let g = dy * gamma
        g = dy * self.gamma
        # dx = g/rms - x * mean(g * x / rms^3)
        dx = g / rms - xnorm * (g * xnorm).mean(axis=-1, keepdims=True)
        return dx

    def step(self, lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        """
        SGD parameter update.

        Args:
            lr: Learning rate.
            weight_decay: L2 penalty coefficient (typically 0 for norm params).
        """
        if weight_decay != 0.0:
            self._grads["gamma"] += weight_decay * self.gamma
        self.gamma -= lr * self._grads["gamma"]
        self._grads["gamma"].fill(0.0)

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        """Access gradients for external optimizers."""
        return self._grads


# Factory function
def get_norm(name: str, d_model: int, **kwargs):
    """
    Get normalization layer by name.

    Args:
        name: One of 'layernorm', 'rmsnorm'.
        d_model: Feature dimension.
        **kwargs: Additional arguments passed to the norm constructor.

    Returns:
        Normalization layer instance.
    """
    norms = {
        "layernorm": LayerNorm,
        "rmsnorm": RMSNorm,
    }
    if name not in norms:
        raise KeyError(f"Unknown norm: {name}. Available: {list(norms.keys())}")
    return norms[name](d_model, **kwargs)
