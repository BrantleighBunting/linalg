"""
Attention mechanisms for transformers.

Implements scaled dot-product attention and multi-head attention
with full forward/backward passes.
"""

import numpy as np
from typing import Optional, Tuple, Dict


def softmax_last(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Softmax along the last axis with numerical stabilization.

    Args:
        x: Input array (..., K).
        eps: Small constant to avoid division by zero.

    Returns:
        Softmax probabilities, same shape as x.
    """
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + eps)


def causal_mask(seq_len: int, fill: float = -1e9, dtype=np.float32) -> np.ndarray:
    """
    Build a causal (future-blocking) additive attention mask.

    Args:
        seq_len: Sequence length T.
        fill: Large negative value for blocked positions.
        dtype: Output dtype.

    Returns:
        Mask of shape (1, 1, T, T) where mask[..., i, j] = fill if j > i else 0.
    """
    i = np.arange(seq_len)
    m = (i[:, None] < i[None, :]).astype(dtype) * fill
    return m[None, None, :, :]


class ScaledDotProductAttention:
    """Scaled dot-product attention: O = softmax(QK^T / sqrt(d)) @ V."""

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Tuple]:
        """Forward pass. Returns (O, cache). Shape: (BH, T, d)."""
        BH, T, d = Q.shape
        scale = 1.0 / np.sqrt(d)

        S = scale * (Q @ K.transpose(0, 2, 1))  # (BH, T_q, T_kv)
        if mask is not None:
            S = S + mask
        P = softmax_last(S)  # (BH, T_q, T_kv)
        O = P @ V  # (BH, T_q, d)

        cache = (Q, K, V, P, d)
        return O, cache

    def backward(self, dO: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass. Returns (dQ, dK, dV)."""
        Q, K, V, P, d = cache
        scale = 1.0 / np.sqrt(d)

        # dV = P.T @ dO
        dV = P.transpose(0, 2, 1) @ dO

        # dP = dO @ V.T
        dP = dO @ V.transpose(0, 2, 1)

        # Softmax backward: dS = (dP - sum(dP*P)) * P
        rowdot = (dP * P).sum(axis=-1, keepdims=True)
        dS = (dP - rowdot) * P

        # dQ = dS @ K * scale
        dQ = (dS @ K) * scale

        # dK = dS.T @ Q * scale
        dK = (dS.transpose(0, 2, 1) @ Q) * scale

        return dQ, dK, dV


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Kaiming/He initialization for ReLU layers."""
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float32)


class MultiHeadAttention:
    """Multi-Head Attention. KV=None for self-attention, else cross-attention."""

    def __init__(self, d_model: int, n_heads: int, seed: int = 0) -> None:
        """
        Args:
            d_model: Model dimension D.
            n_heads: Number of attention heads h.
            seed: RNG seed for reproducible init.
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.D = d_model
        self.h = n_heads
        self.d = d_model // n_heads

        rng = np.random.default_rng(seed)
        hd = n_heads * self.d

        self.Wq = he_init(d_model, hd, rng)
        self.Wk = he_init(d_model, hd, rng)
        self.Wv = he_init(d_model, hd, rng)
        self.Wo = he_init(hd, d_model, rng)

        self.grads: Dict[str, np.ndarray] = {
            "Wq": np.zeros_like(self.Wq),
            "Wk": np.zeros_like(self.Wk),
            "Wv": np.zeros_like(self.Wv),
            "Wo": np.zeros_like(self.Wo),
        }
        self.attn = ScaledDotProductAttention()
        self._cache = None

    @staticmethod
    def split_heads(X: np.ndarray, h: int) -> np.ndarray:
        """Reshape (B, T, h*d) -> (B, h, T, d)."""
        B, T, HD = X.shape
        d = HD // h
        return X.reshape(B, T, h, d).transpose(0, 2, 1, 3)

    @staticmethod
    def combine_heads(H: np.ndarray) -> np.ndarray:
        """Reshape (B, h, T, d) -> (B, T, h*d)."""
        B, h, T, d = H.shape
        return H.transpose(0, 2, 1, 3).reshape(B, T, h * d)

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
        KV: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass. Shape: (B, T, D) -> (B, T, D)."""
        B, T, D = X.shape
        h, d = self.h, self.d
        X_kv = X if KV is None else KV
        T_kv = X_kv.shape[1]

        # Linear projections
        Q_lin = X @ self.Wq
        K_lin = X_kv @ self.Wk
        V_lin = X_kv @ self.Wv

        # Split heads: (B, T, h*d) -> (B, h, T, d) -> (B*h, T, d)
        Q = self.split_heads(Q_lin, h)
        K = self.split_heads(K_lin, h)
        V = self.split_heads(V_lin, h)

        BH = B * h
        Qr = Q.reshape(BH, T, d)
        Kr = K.reshape(BH, T_kv, d)
        Vr = V.reshape(BH, T_kv, d)

        # Prepare mask
        mask_r = None
        if mask is not None:
            mb = mask
            while mb.ndim < 4:
                mb = mb[None, ...]
            mb = np.broadcast_to(mb, (B, h, T, T_kv))
            mask_r = mb.reshape(BH, T, T_kv)

        # Attention
        O_r, attn_cache = self.attn.forward(Qr, Kr, Vr, mask=mask_r)

        # Combine heads
        O = O_r.reshape(B, h, T, d)
        H = self.combine_heads(O)

        # Output projection
        Y = H @ self.Wo

        self._cache = (X, X_kv, Q, K, V, H, attn_cache)
        return Y

    def backward(self, dY: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Backward pass. Returns (dX, dKV) where dKV is None for self-attention."""
        X, X_kv, Q, K, V, H, attn_cache = self._cache
        B, T, D = X.shape
        h, d = self.h, self.d
        BH = B * h
        T_kv = X_kv.shape[1]
        is_cross = X_kv is not X

        # Output projection backward
        Hf = H.reshape(B * T, h * d)
        dYf = dY.reshape(B * T, D)
        dWo = Hf.T @ dYf
        dHf = dYf @ self.Wo.T
        dH = dHf.reshape(B, T, h * d)

        # Split dH back to heads
        dO = dH.reshape(B, T, h, d).transpose(0, 2, 1, 3)
        dO_r = dO.reshape(BH, T, d)

        # Attention backward
        dQr, dKr, dVr = self.attn.backward(dO_r, attn_cache)

        # Reshape back
        dQ = dQr.reshape(B, h, T, d)
        dK = dKr.reshape(B, h, T_kv, d)
        dV = dVr.reshape(B, h, T_kv, d)

        # Combine heads for projection backward
        dQ_lin = self.combine_heads(dQ)
        dK_lin = self.combine_heads(dK)
        dV_lin = self.combine_heads(dV)

        # Projection gradients
        Xf = X.reshape(B * T, D)
        X_kvf = X_kv.reshape(B * T_kv, D)
        dQ_linf = dQ_lin.reshape(B * T, h * d)
        dK_linf = dK_lin.reshape(B * T_kv, h * d)
        dV_linf = dV_lin.reshape(B * T_kv, h * d)

        dWq = Xf.T @ dQ_linf
        dWk = X_kvf.T @ dK_linf
        dWv = X_kvf.T @ dV_linf

        # Input gradients
        dX_q = (dQ_linf @ self.Wq.T).reshape(B, T, D)
        dX_k = (dK_linf @ self.Wk.T).reshape(B, T_kv, D)
        dX_v = (dV_linf @ self.Wv.T).reshape(B, T_kv, D)

        # Store gradients
        self.grads["Wq"] = dWq
        self.grads["Wk"] = dWk
        self.grads["Wv"] = dWv
        self.grads["Wo"] = dWo

        if is_cross:
            dKV = dX_k + dX_v
            return dX_q, dKV
        else:
            dX = dX_q + dX_k + dX_v
            return dX, None

    def step(self, lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        """SGD update for attention parameters."""
        if weight_decay != 0.0:
            for name in ["Wq", "Wk", "Wv", "Wo"]:
                self.grads[name] += weight_decay * getattr(self, name)

        for name in ["Wq", "Wk", "Wv", "Wo"]:
            W = getattr(self, name)
            W -= lr * self.grads[name]
            setattr(self, name, W)
            self.grads[name].fill(0.0)


# Alias for backward compatibility
MHA = MultiHeadAttention
Attention = ScaledDotProductAttention
