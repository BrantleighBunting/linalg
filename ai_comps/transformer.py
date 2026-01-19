#!/usr/bin/python3
"""
Attention is all you need — NumPy, pre-LN, Encoder/Decoder/Transformer.

This file implements a minimal Transformer (encoder–decoder) in pure NumPy with:
- Pre-LayerNorm: LN → sublayer → residual
- Multi-Head Attention (self- and cross-attention)
- Position-wise Feed-Forward Networks
- Token embeddings + sinusoidal positional encodings
- Linear output head with softmax cross-entropy loss
- Full backward pass and simple SGD ".step()" updates

Key terms:
- Logits: raw, unnormalized scores (real numbers) before softmax.
- Broadcasting: NumPy's implicit expansion of size-1 axes to match shapes in elementwise ops.
- Shapes: (B, T, D) = batch, time (sequence length), model width.
          (B, h, T, d) = batch, heads, time, per-head width (D = h*d).
"""

import numpy as np


# -------------------------- utils --------------------------


def softmax_rows(Z: np.ndarray, eps=1e-12):
    """
    Row-wise softmax with numerical stabilization.

    Args:
        Z: Array of shape (..., K). The softmax is applied along the last axis.
        eps: Small constant to avoid division by zero.

    Returns:
        Array with the same shape as Z, where each row (last axis) sums to 1.

    Notes:
        For a row vector z, softmax(z)_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z)).
        Subtracting the max improves numerical stability.
    """
    Zs = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Zs)
    return E / (E.sum(axis=-1, keepdims=True) + eps)


def causal_mask(T, fill=-1e9, dtype=np.float32):
    """
    Build a causal (future-blocking) additive mask.

    Args:
        T: Sequence length (int).
        fill: Large negative number added to disallowed positions.
        dtype: Output dtype.

    Returns:
        Mask of shape (1, 1, T, T) where mask[..., r, c] = fill if c > r else 0.

    Usage:
        Add this mask to attention scores S before softmax: softmax(S + mask).
        Broadcasting then expands (1,1,T,T) to (B,h,T,T).
    """
    i = np.arange(T)
    m = (i[:, None] < i[None, :]).astype(dtype) * fill  # (T, T)
    return m[None, None, :, :]  # (1,1,T,T)


def sinusoidal_pos_encoding(T, D, dtype=np.float32):
    """
    Fixed sinusoidal positional encodings (Vaswani et al., 2017).

    Args:
        T: Sequence length.
        D: Model dimension (even or odd; sin/cos interleave).
        dtype: dtype of returned array.

    Returns:
        PE of shape (T, D) to be added to token embeddings.

    Formula:
        For position p and channel i:
            angle(p, i) = p / 10000^{2*floor(i/2) / D}
            PE[p, 2i]   = sin(angle(p, 2i))
            PE[p, 2i+1] = cos(angle(p, 2i+1))
    """
    pos = np.arange(T)[:, None]
    i = np.arange(D)[None, :]
    angle = pos / (10000 ** (2 * (i // 2) / D))
    PE = np.zeros((T, D), dtype=dtype)
    PE[:, 0::2] = np.sin(angle[:, 0::2])
    PE[:, 1::2] = np.cos(angle[:, 1::2])
    return PE  # (T, D)


def he_init(fan_in, fan_out, rng):
    """
    Kaiming/He initialization for ReLU layers: N(0, 2/fan_in).

    Args:
        fan_in: Number of inputs to the layer.
        fan_out: Number of outputs from the layer (not used in the variance).
        rng: np.random.Generator for reproducibility.

    Returns:
        Weight matrix of shape (fan_in, fan_out).

    Rationale:
        Keeps activation variance approximately constant across layers with ReLU.
    """
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))


# -------------------------- LayerNorm --------------------------


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
        _grads: Dict holding parameter gradients for gamma and beta.
    """

    def __init__(self, d_model=512) -> None:
        """
        Args:
            d_model: Feature dimension D (size of last axis to normalize).
        """
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self._grads = {
            "gamma": np.zeros(d_model, dtype=np.float32),
            "beta": np.zeros(d_model, dtype=np.float32),
        }

    def forward(self, x: np.ndarray, eps=1e-5):
        """
        Forward pass.

        Args:
            x: Input of shape (..., D).
            eps: Small constant for numerical stability.

        Returns:
            y: Output of same shape as x.

        Caches:
            (xhat, sigma, gamma) for backward.
        """
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        sigma = np.sqrt(var + eps)
        xhat = (x - mu) / sigma
        y = xhat * self.gamma + self.beta
        self._cache = (xhat, sigma, self.gamma)
        return y

    def backward(self, dy: np.ndarray):
        """
        Backward pass.

        Args:
            dy: Upstream gradient, same shape as y.

        Returns:
            dx: Gradient with respect to x, same shape as dy.

        Gradients:
            dgamma = sum(dy * xhat, over all but last axis)
            dbeta  = sum(dy,        over all but last axis)
            dx     = (d(y)/dx) @ dy using LN identities:
                     Let ghat = dy * gamma,
                         m1 = mean(ghat, axis=-1, keepdims=True),
                         m2 = mean(ghat * xhat, axis=-1, keepdims=True).
                     Then dx = (ghat - m1 - xhat * m2) / sigma
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

    def step(self, lr=1e-3, weight_decay=0.0):
        """
        SGD parameter update for gamma and beta.

        Args:
            lr: Learning rate.
            weight_decay: L2 penalty coefficient applied to gamma (commonly 0).

        Notes:
            Typical practice: no weight decay on LN parameters (AdamW-style).
        """
        if weight_decay != 0.0:
            self._grads["gamma"] += weight_decay * self.gamma
        self.gamma -= lr * self._grads["gamma"]
        self.beta -= lr * self._grads["beta"]
        self._grads["gamma"].fill(0.0)
        self._grads["beta"].fill(0.0)


# -------------------------- FFN (position-wise) --------------------------


class FFN:
    """
    Position-wise feed-forward network: ReLU(X W1 + b1) W2 + b2.

    Operates identically and independently at each time step (position).
    Uses He init suitable for ReLU activations.

    Attributes:
        W1, b1: First affine layer params (D -> Dff).
        W2, b2: Second affine layer params (Dff -> D).
        grads:  Dict caching gradients for step().
    """

    def __init__(self, d_model=512, d_ff=2048, activation="relu", seed=0) -> None:
        """
        Args:
            d_model: Model dimension D.
            d_ff:    Inner (expansion) dimension Dff.
            activation: Nonlinearity; 'relu' supported here.
            seed: RNG seed for reproducible init.
        """
        rng = np.random.default_rng(seed)
        self.W1 = he_init(d_model, d_ff, rng)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = he_init(d_ff, d_model, rng)
        self.b2 = np.zeros(d_model, dtype=np.float32)
        self.activation = activation
        self.grads = {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }

    @staticmethod
    def relu(x):
        """Elementwise ReLU: max(0, x)."""
        return np.maximum(0.0, x)

    @staticmethod
    def relu_prime(x):
        """Elementwise derivative of ReLU: 1 if x>0 else 0 (same shape as x)."""
        return (x > 0.0).astype(x.dtype)

    def _phi(self, U):
        """Apply the activation phi to pre-activations U."""
        if self.activation == "relu":
            return self.relu(U)
        raise NotImplementedError

    def _phi_prime(self, U):
        """Derivative of activation phi with respect to its input U."""
        if self.activation == "relu":
            return self.relu_prime(U)
        raise NotImplementedError

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: Input of shape (B, T, D).

        Returns:
            Y: Output of shape (B, T, D).

        Flow:
            U = X W1 + b1,  H = phi(U),  Y = H W2 + b2
        """
        U = X @ self.W1 + self.b1
        H = self._phi(U)
        Y = H @ self.W2 + self.b2
        self._cache = (X, U, H)
        return Y

    def backward(self, dY):
        """
        Backward pass.

        Args:
            dY: Upstream gradient of shape (B, T, D).

        Returns:
            dX: Gradient with respect to X, shape (B, T, D).

        Gradients:
            dW2 = H^T dY,   db2 = sum(dY)
            dH  = dY W2^T
            dU  = dH ⊙ phi'(U)
            dW1 = X^T dU,   db1 = sum(dU)
            dX  = dU W1^T
        """
        X, U, H = self._cache
        B, T, Dff = U.shape
        D = dY.shape[-1]

        # Flatten (BT,·) for proper matmuls
        Hf = H.reshape(B * T, Dff)
        dYf = dY.reshape(B * T, D)

        # last affine
        dW2 = Hf.T @ dYf  # (Dff, D)
        db2 = dYf.sum(axis=0)  # (D,)
        dHf = dYf @ self.W2.T  # (BT, Dff)
        dH = dHf.reshape(B, T, Dff)

        # nonlinearity
        dU = dH * self._phi_prime(U)  # (B, T, Dff)

        # first affine
        Xf = X.reshape(B * T, -1)  # (BT, D)
        dUf = dU.reshape(B * T, Dff)
        dW1 = Xf.T @ dUf  # (D, Dff)
        db1 = dUf.sum(axis=0)  # (Dff,)
        dXf = dUf @ self.W1.T  # (BT, D)
        dX = dXf.reshape(B, T, -1)

        # store grads
        self.grads["W1"], self.grads["b1"] = dW1, db1
        self.grads["W2"], self.grads["b2"] = dW2, db2
        return dX

    def step(self, lr=1e-3, weight_decay=0.0):
        """
        SGD parameter update.

        Args:
            lr: Learning rate.
            weight_decay: L2 penalty coefficient (applied to weights, not biases).
        """
        if weight_decay != 0.0:
            self.grads["W1"] += weight_decay * self.W1
            self.grads["W2"] += weight_decay * self.W2
        self.W1 -= lr * self.grads["W1"]
        self.grads["W1"].fill(0.0)
        self.b1 -= lr * self.grads["b1"]
        self.grads["b1"].fill(0.0)
        self.W2 -= lr * self.grads["W2"]
        self.grads["W2"].fill(0.0)
        self.b2 -= lr * self.grads["b2"]
        self.grads["b2"].fill(0.0)


# -------------------------- Scaled Dot-Product Attention --------------------------


class Attention:
    """
    Scaled dot-product attention on batched heads.

    For each row:
        S = (Q K^T) / sqrt(d)
        (optional) S += mask
        P = softmax(S)
        O = P V

    Shapes:
        Q, K, V: (BH, T, d) or (BH, T_q, d), (BH, T_kv, d)
        mask:    (BH, T_q, T_kv), additive (0 keep, -inf block)
        O:       (BH, T_q, d)
    """

    @staticmethod
    def softmax_last(x):
        """
        Softmax along the last axis with stabilization.

        Args:
            x: Array (..., K).

        Returns:
            Softmax(x) along last axis, same shape as x.
        """
        z = x - x.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of attention.

        Args:
            Q: Queries, shape (BH, T_q, d).
            K: Keys,    shape (BH, T_kv, d).
            V: Values,  shape (BH, T_kv, d).
            mask: Optional additive mask broadcastable to (BH, T_q, T_kv),
                  with 0 for keep and large negative for block.

        Returns:
            (O, cache): O shape (BH, T_q, d) and cache tuple for backward.
        """
        BH, T, d = Q.shape
        scale = 1.0 / np.sqrt(d)
        S = scale * (Q @ K.transpose(0, 2, 1))  # (BH, T_q, T_kv)
        if mask is not None:
            S = S + mask
        P = self.softmax_last(S)  # (BH, T_q, T_kv)
        O = P @ V  # (BH, T_q, d)
        cache = (Q, K, V, P, d)
        return O, cache

    def backward(self, dO, cache):
        """
        Backward pass of attention.

        Args:
            dO: Upstream gradient, shape (BH, T_q, d).
            cache: Tuple (Q, K, V, P, d) from forward.

        Returns:
            dQ, dK, dV: Gradients with same shapes as Q, K, V.

        Notes:
            Using softmax Jacobian-vector product per row:
                dS = (dP - sum(dP*P, axis=-1, keepdims=True)) * P
            Then:
                dQ = dS @ K / sqrt(d)
                dK = dS^T @ Q / sqrt(d)
                dV = P^T @ dO
        """
        Q, K, V, P, d = cache
        BH, T, _ = Q.shape
        scale = 1.0 / np.sqrt(d)

        dV = P.transpose(0, 2, 1) @ dO
        dP = dO @ V.transpose(0, 2, 1)
        rowdot = (dP * P).sum(axis=-1, keepdims=True)
        dS = (dP - rowdot) * P
        dQ = (dS @ K) * scale
        dK = (dS.transpose(0, 2, 1) @ Q) * scale
        return dQ, dK, dV


# -------------------------- MHA (supports self- and cross-attn) --------------------------


class MHA:
    """
    Multi-Head Attention (self- or cross-attention).

    If KV is None → self-attention, i.e., Q,K,V computed from X.
    If KV is not None → cross-attention, i.e., Q from X and K,V from KV.

    Projections:
        Q_lin = X    @ Wq   # (B,T,h*d)
        K_lin = X_kv @ Wk   # (B,T',h*d)
        V_lin = X_kv @ Wv   # (B,T',h*d)

    Head mechanics:
        Split heads: (B,T,h*d) → (B,h,T,d) → merge to (BH,T,d).
        Apply attention per head in batch, then combine heads back.

    Output:
        Y = concat(O_1..O_h) @ Wo  # (B,T,D)
    """

    def __init__(self, D, h, seed=0) -> None:
        """
        Args:
            D: Model dimension.
            h: Number of heads (D must be divisible by h).
            seed: RNG seed for reproducible init.
        """
        assert D % h == 0, "D must be divisible by number of heads"
        self.D = D
        self.h = h
        self.d = D // h
        rng = np.random.default_rng(seed)
        self.Wq = he_init(D, h * self.d, rng)
        self.Wk = he_init(D, h * self.d, rng)
        self.Wv = he_init(D, h * self.d, rng)
        self.Wo = he_init(h * self.d, D, rng)
        self.grads = {
            "Wq": np.zeros_like(self.Wq),
            "Wk": np.zeros_like(self.Wk),
            "Wv": np.zeros_like(self.Wv),
            "Wo": np.zeros_like(self.Wo),
        }
        self.attn = Attention()

    @staticmethod
    def split_heads(X, h):
        """
        Reshape (B,T,h*d) into (B,h,T,d) by splitting the last dim into heads.

        Args:
            X: Array of shape (B, T, h*d).
            h: Number of heads.

        Returns:
            Array of shape (B, h, T, d), where d = (h*d) / h.
        """
        B, T, HD = X.shape
        d = HD // h
        return X.reshape(B, T, h, d).transpose(0, 2, 1, 3)

    @staticmethod
    def combine_heads(H):
        """
        Inverse of split_heads: (B,h,T,d) → (B,T,h*d).

        Args:
            H: Array of shape (B, h, T, d).

        Returns:
            Array of shape (B, T, h*d).
        """
        B, h, T, d = H.shape
        return H.transpose(0, 2, 1, 3).reshape(B, T, h * d)

    def forward(self, X, mask=None, KV=None):
        """
        Forward pass for MHA.

        Args:
            X:  Queries source, shape (B, T, D).
            mask: Optional additive mask broadcastable to (B, 1, T, T').
                  0 for keep, large negative for block (causal or padding).
            KV: Optional keys/values source, shape (B, T', D).
                If None, self-attention (KV := X); else cross-attention.

        Returns:
            Y: Output of shape (B, T, D).

        Mask handling:
            Any of (T,T), (1,1,T,T), (B,1,T,T'), (B,h,T,T') are accepted by
            broadcasting to (B,h,T,T') and then reshaped to (BH,T,T').
        """
        B, T, D = X.shape
        h, d = self.h, self.d
        X_kv = X if KV is None else KV
        Tkv = X_kv.shape[1]

        # projections
        Q_lin = X @ self.Wq  # (B,T,h*d)
        K_lin = X_kv @ self.Wk  # (B,T',h*d)
        V_lin = X_kv @ self.Wv  # (B,T',h*d)

        # split heads
        Q = self.split_heads(Q_lin, h)  # (B,h,T,d)
        K = self.split_heads(K_lin, h)  # (B,h,T',d)
        V = self.split_heads(V_lin, h)  # (B,h,T',d)

        # reshape to (BH, T, d)
        BH = B * h
        Qr = Q.reshape(BH, T, d)
        Kr = K.reshape(BH, Tkv, d)
        Vr = V.reshape(BH, Tkv, d)

        # mask → (BH, T, Tkv)
        if mask is not None:
            mb = mask
            # ensure 4D (..., T, Tkv)
            while mb.ndim < 4:
                mb = mb[None, ...]                     # add leading dims
            # broadcast to (B, 1, T, Tkv)
            mb = np.broadcast_to(mb, (B, 1, T, Tkv))
            # broadcast to (B, h, T, Tkv)
            mb = np.broadcast_to(mb, (B, h, T, Tkv))
            # collapse heads into batch for attention kernel
            mask_r = mb.reshape(B * h, T, Tkv)
        else:
            mask_r = None

        # attention
        O_r, attn_cache = self.attn.forward(Qr, Kr, Vr, mask=mask_r)  # (BH,T,d)

        # combine heads
        O = O_r.reshape(B, h, T, d)
        H = self.combine_heads(O)  # (B,T,h*d)

        # output projection
        Y = H @ self.Wo  # (B,T,D)

        # cache everything (including X_kv for cross-attn)
        self._cache = (X, X_kv, Q, K, V, H, attn_cache)
        return Y

    def backward(self, dY):
        """
        Backward pass for MHA.

        Args:
            dY: Upstream gradient of shape (B, T, D).

        Returns:
            dX:   Gradient into the query source X, shape (B, T, D).
            dMem: Gradient into the KV source when cross-attending, shape (B, T', D),
                  or None when self-attending (then gradients are added into dX).

        Parameter grads:
            From Y = H Wo:
                dWo = H^T dY, dH = dY Wo^T.
            Split dH → per-head dO, then attention backward:
                dQ, dK, dV.
            Map back to linear projections:
                dWq = X^T dQ_lin, dWk = X_kv^T dK_lin, dWv = X_kv^T dV_lin.
            Input grads:
                dX_q = dQ_lin Wq^T, dX_k = dK_lin Wk^T, dX_v = dV_lin Wv^T.
                If self-attn: dX += (dX_k + dX_v).
        """
        X, X_kv, Q, K, V, H, attn_cache = self._cache
        B, T, D = X.shape
        h, d = self.h, self.d
        BH = B * h

        # ---- Y = H @ Wo ----
        Hf  = H.reshape(B*T, h*d)            # (B*T, h*d)  H is (B,T,h*d)
        dYf = dY.reshape(B*T, D)             # (B*T, D)
        dWo = Hf.T @ dYf                     # (h*d, D)
        dHf = dYf @ self.Wo.T                # (B*T, h*d)
        dH  = dHf.reshape(B, T, h*d)         # (B, T, h*d)

        # ---- split dH back into heads: (B,h,T,d) ----
        dO = dH.reshape(B, T, h, d).transpose(0, 2, 1, 3)  # (B,h,T,d)
        dO_r = dO.reshape(BH, T, d)                        # (BH,T,d)

        # ---- attention backward (per head, batched) ----
        Qr = Q.reshape(BH, T, d)                           # (BH,T,d)
        Kr = K.reshape(BH, K.shape[2], d)                  # (BH,T',d)
        Vr = V.reshape(BH, V.shape[2], d)                  # (BH,T',d)
        dQr, dKr, dVr = self.attn.backward(dO_r, attn_cache)  # each (BH,T(’),d)

        # ---- restore shapes ----
        dQ = dQr.reshape(B, h, T, d)
        dK = dKr.reshape(B, h, K.shape[2], d)
        dV = dVr.reshape(B, h, V.shape[2], d)

        # combine heads back to (B,T,h*d)
        def _combine(Hhh):  # (B,h,T,d) -> (B,T,h*d)
            return Hhh.transpose(0, 2, 1, 3).reshape(B, Hhh.shape[2], h*d)

        dQ_lin = _combine(dQ)                # (B,T ,h*d)
        dK_lin = _combine(dK)                # (B,T',h*d)
        dV_lin = _combine(dV)                # (B,T',h*d)

        # ---- parameter grads for input projections ----
        Xf      = X.reshape(B*T, D)
        dQf     = dQ_lin.reshape(B*T, h*d)
        dWq     = Xf.T @ dQf                 # (D, h*d)

        Tkv     = X_kv.shape[1]
        Xkv_f   = X_kv.reshape(B*Tkv, D)
        dKf     = dK_lin.reshape(B*Tkv, h*d)
        dVf     = dV_lin.reshape(B*Tkv, h*d)
        dWk     = Xkv_f.T @ dKf              # (D, h*d)
        dWv     = Xkv_f.T @ dVf              # (D, h*d)

        # ---- input grads to X and X_kv ----
        dX_q  = dQ_lin @ self.Wq.T           # (B,T ,D)
        dX_k  = dK_lin @ self.Wk.T           # (B,T',D)
        dX_v  = dV_lin @ self.Wv.T           # (B,T',D)
        dX    = dX_q
        dKV   = dX_k + dX_v                  # grads into encoder memory when cross-attn

        # self-attn: X_kv is X → add both branches back to X
        if X_kv is X:
            dX += dKV

        # save grads
        self.grads["Wq"] = dWq
        self.grads["Wk"] = dWk
        self.grads["Wv"] = dWv
        self.grads["Wo"] = dWo

        # return (dX, dMemory) for cross-attn; None for self-attn
        return dX, (None if X_kv is X else dKV)
    
    def step(self, lr=1e-3, weight_decay=0.0):
        """
        SGD parameter update for all projection matrices.

        Args:
            lr: Learning rate.
            weight_decay: L2 penalty coefficient applied to weights.
        """
        if weight_decay != 0.0:
            self.grads["Wq"] += weight_decay * self.Wq
            self.grads["Wk"] += weight_decay * self.Wk
            self.grads["Wv"] += weight_decay * self.Wv
            self.grads["Wo"] += weight_decay * self.Wo
        for name in ["Wq", "Wk", "Wv", "Wo"]:
            W = getattr(self, name)
            W -= lr * self.grads[name]
            setattr(self, name, W)
            self.grads[name].fill(0.0)


# -------------------------- Encoder/Decoder Layers --------------------------


class EncoderLayer:
    """
    One encoder block (pre-LN):
        Y = X + MHA(LN(X))
        Y = Y + FFN(LN(Y))
    """

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, seed=0):
        """
        Args:
            d_model: Model dimension D.
            n_heads: Number of attention heads h.
            d_ff: Inner FFN dimension Dff.
            seed: RNG base seed; submodules offset internally.
        """
        self.ln1 = LayerNorm(d_model)
        self.mha = MHA(d_model, n_heads, seed=seed)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, activation="relu", seed=seed + 1)
        self._cache = None

    def forward(self, X, src_mask=None):
        """
        Forward pass.

        Args:
            X: Input (B,T,D).
            src_mask: Optional encoder mask, broadcastable to (B,1,T,T).

        Returns:
            Y: Output (B,T,D).
        """
        # Pre-LN → MHA → Residual
        Xn = self.ln1.forward(X)
        A = self.mha.forward(Xn, mask=src_mask, KV=None)  # self-attn
        Y1 = X + A

        # Pre-LN → FFN → Residual
        Y1n = self.ln2.forward(Y1)
        F = self.ffn.forward(Y1n)
        Y = Y1 + F

        self._cache = (X, Y1, Xn, Y1n, src_mask)
        return Y

    def backward(self, dY):
        """
        Backward pass.

        Args:
            dY: Upstream gradient (B,T,D).

        Returns:
            dX: Gradient with respect to X (B,T,D).
        """
        X, Y1, Xn, Y1n, src_mask = self._cache

        # Residual 2
        dY1 = dY.copy()
        dF = dY

        # FFN
        dY1n = self.ffn.backward(dF)
        dY1_ln2 = self.ln2.backward(dY1n)
        dY1 += dY1_ln2

        # Residual 1
        dX = dY1.copy()
        dA = dY1

        # MHA
        dXn, _ = self.mha.backward(dA)
        dX_ln1 = self.ln1.backward(dXn)
        dX += dX_ln1

        return dX

    def step(self, lr=1e-3, weight_decay=0.0):
        """
        SGD updates for submodules. No weight decay for LayerNorm by default.
        """
        self.mha.step(lr, weight_decay)
        self.ffn.step(lr, weight_decay)
        self.ln1.step(lr, 0.0)
        self.ln2.step(lr, 0.0)


class DecoderLayer:
    """
    One decoder block (pre-LN):
        Y1 = X  + SelfAttn(LN(X))         # masked causal self-attn
        Y2 = Y1 + CrossAttn(LN(Y1), mem)  # queries=Y1, keys/values=memory
        Y  = Y2 + FFN(LN(Y2))
    """

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, seed=0):
        """
        Args:
            d_model: Model dimension D.
            n_heads: Number of attention heads h.
            d_ff: Inner FFN dimension Dff.
            seed: RNG base seed; submodules offset internally.
        """
        self.ln1 = LayerNorm(d_model)
        self.self_attn = MHA(d_model, n_heads, seed=seed)
        self.ln2 = LayerNorm(d_model)
        self.cross_attn = MHA(d_model, n_heads, seed=seed + 1)
        self.ln3 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, activation="relu", seed=seed + 2)
        self._cache = None

    def forward(self, X, memory, tgt_mask=None, mem_mask=None):
        """
        Forward pass.

        Args:
            X: Decoder input (B,T,D).
            memory: Encoder memory (B,T_src,D).
            tgt_mask: Causal mask for decoder self-attn, broadcastable to (B,1,T,T).
            mem_mask: Optional mask for cross-attn (B,1,T,T_src) style.

        Returns:
            Y: Output (B,T,D).
        """
        # Pre-LN → masked self-attn → residual
        Xn = self.ln1.forward(X)
        A = self.self_attn.forward(Xn, mask=tgt_mask, KV=None)
        Y1 = X + A

        # Pre-LN → cross-attn(Q=Y1n, K/V=memory) → residual
        Y1n = self.ln2.forward(Y1)
        C = self.cross_attn.forward(Y1n, mask=mem_mask, KV=memory)
        Y2 = Y1 + C

        # Pre-LN → FFN → residual
        Y2n = self.ln3.forward(Y2)
        F = self.ffn.forward(Y2n)
        Y = Y2 + F

        self._cache = (X, Y1, Y2, Xn, Y1n, Y2n, memory, tgt_mask, mem_mask)
        return Y

    def backward(self, dY):
        """
        Backward pass.

        Args:
            dY: Upstream gradient (B,T,D).

        Returns:
            dX:        Gradient wrt decoder input X (B,T,D).
            dMemory:   Gradient wrt encoder memory (B,T_src,D).
        """
        X, Y1, Y2, Xn, Y1n, Y2n, memory, tgt_mask, mem_mask = self._cache

        # Residual 3
        dY2 = dY.copy()
        dF = dY

        # FFN
        dY2n = self.ffn.backward(dF)
        dY2_ln3 = self.ln3.backward(dY2n)
        dY2 += dY2_ln3

        # Residual 2 (cross-attn)
        dY1 = dY2.copy()
        dC = dY2

        dY1n, dMem_part = self.cross_attn.backward(dC)  # returns (dQ, dKV)
        dY1_ln2 = self.ln2.backward(dY1n)
        dY1 += dY1_ln2
        dMemory = dMem_part if dMem_part is not None else np.zeros_like(memory)

        # Residual 1 (self-attn)
        dX = dY1.copy()
        dA = dY1

        dXn, _ = self.self_attn.backward(dA)  # self-attn → only dX
        dX_ln1 = self.ln1.backward(dXn)
        dX += dX_ln1

        return dX, dMemory

    def step(self, lr=1e-3, weight_decay=0.0):
        """
        SGD updates for submodules. No weight decay for LayerNorm by default.
        """
        self.self_attn.step(lr, weight_decay)
        self.cross_attn.step(lr, weight_decay)
        self.ffn.step(lr, weight_decay)
        self.ln1.step(lr, 0.0)
        self.ln2.step(lr, 0.0)
        self.ln3.step(lr, 0.0)


# -------------------------- Encoder / Decoder / Transformer --------------------------


class Encoder:
    """
    Stack of EncoderLayer blocks.

    Forward:
        H = X
        for each layer:
            H = layer.forward(H, src_mask)
        return H  # encoder "memory"
    """

    def __init__(self, num_layers=6, d_model=512, n_heads=8, d_ff=2048, seed=0):
        """
        Args:
            num_layers: Number of encoder layers.
            d_model: Model dimension D.
            n_heads: Number of attention heads h.
            d_ff: Inner FFN dimension Dff.
            seed: RNG base seed.
        """
        self.layers = [
            EncoderLayer(d_model, n_heads, d_ff, seed=seed + i * 3)
            for i in range(num_layers)
        ]

    def forward(self, X, src_mask=None):
        """
        Run the encoder stack.

        Args:
            X: Input (B,T,D).
            src_mask: Optional source mask broadcastable to (B,1,T,T).

        Returns:
            H: Encoder memory (B,T,D).
        """
        H = X
        for layer in self.layers:
            H = layer.forward(H, src_mask=src_mask)
        return H  # "memory"

    def backward(self, dH):
        """
        Backprop through the encoder stack.

        Args:
            dH: Gradient wrt encoder outputs (B,T,D).

        Returns:
            dX: Gradient wrt encoder inputs (B,T,D).
        """
        dX = dH
        for layer in reversed(self.layers):
            dX = layer.backward(dX)
        return dX

    def step(self, lr=1e-3, weight_decay=0.0):
        """
        SGD updates for all encoder layers.
        """
        for layer in self.layers:
            layer.step(lr, weight_decay)


class Decoder:
    """
    Stack of DecoderLayer blocks.

    Forward:
        H = X
        for each layer:
            H = layer.forward(H, memory, tgt_mask, mem_mask)
        return H
    """

    def __init__(self, num_layers=6, d_model=512, n_heads=8, d_ff=2048, seed=1000):
        """
        Args:
            num_layers: Number of decoder layers.
            d_model: Model dimension D.
            n_heads: Number of attention heads h.
            d_ff: Inner FFN dimension Dff.
            seed: RNG base seed.
        """
        self.layers = [
            DecoderLayer(d_model, n_heads, d_ff, seed=seed + i * 4)
            for i in range(num_layers)
        ]

    def forward(self, X, memory, tgt_mask=None, mem_mask=None):
        """
        Run the decoder stack.

        Args:
            X: Decoder input (B,T,D).
            memory: Encoder memory (B,T_src,D).
            tgt_mask: Causal/self-attn mask for decoder (B,1,T,T).
            mem_mask: Optional cross-attn mask (B,1,T,T_src).

        Returns:
            H: Decoder outputs (B,T,D).
        """
        H = X
        for layer in self.layers:
            H = layer.forward(H, memory, tgt_mask=tgt_mask, mem_mask=mem_mask)
        return H

    def backward(self, dH):
        """
        Backprop through decoder stack.

        Args:
            dH: Gradient wrt decoder outputs (B,T,D).

        Returns:
            dX:       Gradient wrt decoder inputs (B,T,D).
            dMemSum:  Sum of gradients wrt encoder memory across layers (B,T_src,D).
        """
        dX = dH
        dMem_total = 0
        for layer in reversed(self.layers):
            dX, dMem = layer.backward(dX)
            dMem_total = dMem_total + dMem
        return dX, dMem_total

    def step(self, lr=1e-3, weight_decay=0.0):
        """
        SGD updates for all decoder layers.
        """
        for layer in self.layers:
            layer.step(lr, weight_decay)


class Transformer:
    """
    Minimal encoder–decoder Transformer core with pre-LN blocks.

    Interfaces:
        forward(src, tgt, ...)
            -> (decoder_outputs, encoder_memory)
        backward(dout)
            -> (dsrc, dtgt)  # grads for src and tgt streams via cross-attn
        step(...)
            -> apply SGD to all submodules
    """

    def __init__(
        self,
        num_enc_layers=6,
        num_dec_layers=6,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        seed=0,
    ):
        """
        Args:
            num_enc_layers: Number of encoder layers.
            num_dec_layers: Number of decoder layers.
            d_model: Model dimension D.
            n_heads: Number of attention heads h.
            d_ff: Inner FFN dimension Dff.
            seed: RNG base seed.
        """
        self.encoder = Encoder(num_enc_layers, d_model, n_heads, d_ff, seed=seed)
        self.decoder = Decoder(num_dec_layers, d_model, n_heads, d_ff, seed=seed + 999)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, mem_mask=None):
        """
        Forward pass through encoder then decoder.

        Args:
            src: Source embeddings (B,T_src,D).
            tgt: Target embeddings (B,T_tgt,D).
            src_mask: Optional encoder mask (B,1,T_src,T_src).
            tgt_mask: Decoder causal mask (B,1,T_tgt,T_tgt).
            mem_mask: Optional cross-attn mask (B,1,T_tgt,T_src).

        Returns:
            out:    Decoder outputs (B,T_tgt,D).
            memory: Encoder memory (B,T_src,D).
        """
        memory = self.encoder.forward(src, src_mask=src_mask)
        out = self.decoder.forward(tgt, memory, tgt_mask=tgt_mask, mem_mask=mem_mask)
        return out, memory

    def backward(self, dout):
        """
        Backward pass through decoder then encoder.

        Args:
            dout: Gradient wrt decoder outputs (B,T_tgt,D).

        Returns:
            dsrc: Gradients wrt encoder inputs (B,T_src,D).
            ddec: Gradients wrt decoder inputs (B,T_tgt,D).
        """
        ddec, dmem = self.decoder.backward(dout)
        dsrc = self.encoder.backward(dmem)  # propagate cross-attn grads into encoder
        return dsrc, ddec

    def step(self, lr=1e-3, weight_decay=0.0):
        """
        SGD updates for the whole model (encoder + decoder).

        Args:
            lr: Learning rate.
            weight_decay: L2 penalty coefficient applied to weights.
        """
        self.encoder.step(lr, weight_decay)
        self.decoder.step(lr, weight_decay)


# ----------------- embeddings & head -----------------
class TokenEmbedding:
    """
    Simple token embedding layer: look up rows from a weight matrix.

    Forward:
        Y[b,t,:] = W[idx[b,t], :]
    Backward:
        Accumulate dY into gradW at the looked-up rows (supports repeats).

    Attributes:
        W:     Embedding matrix (V, D).
        gradW: Accumulated gradients (V, D).
        _idx:  Last used integer indices (B, T).
    """

    def __init__(self, vocab_size, d_model, seed=0):
        """
        Args:
            vocab_size: Vocabulary size V.
            d_model:    Embedding dimension D.
            seed: RNG seed for reproducible init.
        """
        rng = np.random.default_rng(seed)
        # small init is fine; can also use 1/sqrt(d)
        self.W = rng.normal(0.0, 0.02, size=(vocab_size, d_model)).astype(np.float32)
        self.gradW = np.zeros_like(self.W)
        self._idx = None

    def forward(self, idx):  # idx: (B,T) int
        """
        Embedding lookup.

        Args:
            idx: Integer ids of shape (B, T).

        Returns:
            Y: Embeddings of shape (B, T, D).
        """
        self._idx = idx
        return self.W[idx]  # (B,T,D)

    def backward(self, dX):  # dX: (B,T,D)
        """
        Backward pass.

        Args:
            dX: Gradient wrt embedding outputs (B,T,D).

        Side effects:
            Accumulates into gradW using np.add.at for repeated indices.
        """
        self.gradW.fill(0.0)
        B, T, D = dX.shape
        flat_idx = self._idx.reshape(-1)
        flat_grad = dX.reshape(B * T, D)
        # accumulate per token row
        np.add.at(self.gradW, flat_idx, flat_grad)

    def step(self, lr=1e-2, weight_decay=0.0):
        """
        SGD update for embeddings.

        Args:
            lr: Learning rate.
            weight_decay: L2 penalty coefficient applied to W.
        """
        if weight_decay != 0.0:
            self.gradW += weight_decay * self.W
        self.W -= lr * self.gradW
        self.gradW.fill(0.0)


class OutputHead:
    """
    Linear classifier head + softmax cross-entropy loss.

    Logits:
        Z = Y W + b     # (B,T,V)
    Probabilities:
        P = softmax_rows(Zf).reshape(B,T,V)
    Loss:
        CE = - mean(log P[b,t,y[b,t]]), averaged over B*T

    Backward wrt logits:
        dZ = (P - one_hot(y)) / (B*T)
    """

    def __init__(self, d_model, vocab_size, seed=1):
        """
        Args:
            d_model: Model dimension D (input width).
            vocab_size: Number of classes V.
            seed: RNG seed for reproducible init.
        """
        rng = np.random.default_rng(seed)
        # Glorot init (simple)
        std = np.sqrt(2.0/(d_model+vocab_size))
        self.W = rng.normal(0.0, std, size=(d_model, vocab_size)).astype(np.float32)
        self.b = np.zeros(vocab_size, dtype=np.float32)
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        self._Y = None  # cache inputs to the head

    def logits(self, Y):  # Y: (B,T,D)
        """
        Compute logits from hidden states.

        Args:
            Y: Hidden states (B,T,D).

        Returns:
            Z: Logits (B,T,V).
        """
        self._Y = Y
        return Y @ self.W + self.b  # (B,T,V)

    def loss_and_dlogits(self, Z, targets):
        """
        Cross-entropy loss and gradient wrt logits.

        Args:
            Z: Logits (B,T,V).
            targets: Integer class indices (B,T).

        Returns:
            loss: Scalar mean cross-entropy over B*T positions.
            dZ:   Gradient wrt logits, shape (B,T,V).

        Implementation detail:
            We reshape to (B*T,V) to apply softmax_rows, and compute:
                loss = -mean(log P[range(N), y])
                dZ   = (P - one_hot(y)) / N
        """
        B, T, V = Z.shape
        Zf = Z.reshape(B*T, V)
        y = targets.reshape(B*T)
        P = softmax_rows(Zf)                   # (B*T, V)
        # CE
        loss = -np.log(P[np.arange(B*T), y] + 1e-12).mean()
        # dZ = (P - one_hot)/N
        dZ = P
        dZ[np.arange(B*T), y] -= 1.0
        dZ /= (B*T)
        return loss, dZ.reshape(B, T, V)

    def backward(self, dZ):   # dZ: (B,T,V), returns dY
        """
        Backprop into Y and collect parameter grads.

        Args:
            dZ: Gradient wrt logits (B,T,V).

        Returns:
            dY: Gradient wrt hidden states Y (B,T,D).

        Parameter grads:
            dW = Y^T dZ, db = sum(dZ)
        """
        Y = self._Y
        B, T, V = dZ.shape
        D = Y.shape[-1]
        Yf = Y.reshape(B*T, D)
        dZf = dZ.reshape(B*T, V)
        self.gradW = Yf.T @ dZf                     # (D,V)
        self.gradb = dZf.sum(axis=0)                # (V,)
        dY = dZf @ self.W.T                         # (B*T, D)
        return dY.reshape(B, T, D)

    def step(self, lr=1e-2, weight_decay=0.0):
        """
        SGD update for the output head.

        Args:
            lr: Learning rate.
            weight_decay: L2 penalty coefficient applied to W.
        """
        if weight_decay != 0.0:
            self.gradW += weight_decay * self.W
        self.W -= lr * self.gradW
        self.b -= lr * self.gradb
        self.gradW.fill(0.0)
        self.gradb.fill(0.0)

# ----------------- data: reverse task -----------------
def make_batch(B, T, V, bos_id=0, rng=None):
    """
    Create a synthetic batch for the reversal task with teacher forcing.

    Args:
        B: Batch size.
        T: Sequence length.
        V: Vocabulary size (0..V-1), where 0 is reserved for BOS.
        bos_id: Integer used as BOS token (default 0).
        rng: np.random.Generator or None for a new generator.

    Returns:
        src:     (B,T) integers in [1..V-1] (source).
        tgt_in:  (B,T) decoder inputs = [BOS, reversed[:-1]].
        tgt_out: (B,T) decoder targets = reversed.

    Notes:
        The model learns to output the source sequence in reverse order.
        Teacher forcing uses tgt_in (shifted targets) as decoder inputs.
    """
    rng = np.random.default_rng() if rng is None else rng
    src = rng.integers(1, V, size=(B, T), dtype=np.int32)  # exclude 0 for BOS
    rev = np.flip(src, axis=1)                             # reversed sequence
    tgt_out = rev.copy()
    tgt_in = np.concatenate([np.full((B,1), bos_id, dtype=np.int32), rev[:, :-1]], axis=1)
    return src, tgt_in, tgt_out

# ----------------- training loop -----------------
def train_reverse_demo(TransformerClass):
    """
    Train the Transformer on a toy sequence-reversal task with teacher forcing.

    Pipeline:
        - Build batch (src, tgt_in, tgt_out)
        - Embed + add sinusoidal positions
        - Forward through Transformer (encoder→decoder)
        - Output head → logits → CE loss
        - Backprop: head → decoder → encoder → embeddings
        - Step: update all parameters with SGD

    Hyperparameters here are small so it converges quickly on CPU.
    """
    # hyperparams (small so it trains fast)
    B, Tsrc, Ttgt = 64, 8, 8
    V = 32           # vocab size (id 0 reserved for BOS)
    D = 64           # model dim
    H = 4            # heads
    Dff = 4*D
    L_enc = 2
    L_dec = 2
    epochs = 5000
    lr_model = 5e-3
    lr_embed = 5e-3
    lr_head = 5e-3
    wd = 0.0

    rng = np.random.default_rng(42)

    # modules
    tok_emb_src = TokenEmbedding(V, D, seed=1)
    tok_emb_tgt = TokenEmbedding(V, D, seed=2)  # separate src/tgt embeddings (simplest)
    head = OutputHead(D, V, seed=3)

    model = TransformerClass(num_enc_layers=L_enc, num_dec_layers=L_dec,
                             d_model=D, n_heads=H, d_ff=Dff, seed=123)

    # fixed sinusoidal positions
    PE_src = sinusoidal_pos_encoding(Tsrc, D)   # (Tsrc,D)
    PE_tgt = sinusoidal_pos_encoding(Ttgt, D)

    # masks
    tgt_mask = causal_mask(Ttgt)  # (1,1,Ttgt,Ttgt)

    for ep in range(1, epochs+1):
        # ----- batch -----
        src_idx, tgt_in_idx, tgt_out_idx = make_batch(B, Tsrc, V, bos_id=0, rng=rng)

        # ----- forward -----
        src_emb = tok_emb_src.forward(src_idx).astype(np.float32) + PE_src[None, :, :]
        tgt_emb = tok_emb_tgt.forward(tgt_in_idx).astype(np.float32) + PE_tgt[None, :, :]

        out, memory = model.forward(src_emb, tgt_emb, src_mask=None, tgt_mask=tgt_mask, mem_mask=None)  # (B,Ttgt,D)

        logits = head.logits(out)                   # (B,Ttgt,V)
        loss, dZ = head.loss_and_dlogits(logits, tgt_out_idx)

        # ----- backward -----
        dOut = head.backward(dZ)                    # (B,Ttgt,D)

        dsrc, dtgt = model.backward(dOut)           # grads into src & tgt streams

        # Backprop to embeddings (PE is sinusoidal → no params)
        tok_emb_tgt.backward(dtgt)                  # (B,Ttgt,D)
        tok_emb_src.backward(dsrc)                  # (B,Tsrc,D)

        # ----- step -----
        head.step(lr=lr_head, weight_decay=wd)
        tok_emb_src.step(lr=lr_embed, weight_decay=wd)
        tok_emb_tgt.step(lr=lr_embed, weight_decay=wd)
        model.step(lr=lr_model, weight_decay=wd)

        if ep % 20 == 0 or ep == 1:
            # quick accuracy on the current batch (teacher forcing)
            with np.errstate(over="ignore"):
                P = softmax_rows(logits.reshape(B * Ttgt, V)).reshape(B, Ttgt, V)
            pred = P.argmax(axis=-1)
            acc = (pred == tgt_out_idx).mean()
            print(f"epoch {ep:3d}  loss {loss:.4f}  token-acc {acc:.3f}")

    # ----- demo: greedy decode on a few examples -----
    def greedy_decode(src_idx_single):
        """
        Greedy autoregressive decode on a single source sequence.

        Args:
            src_idx_single: Array of shape (Tsrc,) with token ids.

        Returns:
            pred: Array of shape (Ttgt,) with predicted token ids.

        Notes:
            Uses the learned model, embeddings, positional encodings, and
            the causal mask. Decodes step-by-step feeding back the argmax.
        """
        # encode source
        src_emb = tok_emb_src.forward(src_idx_single[None, :]) + PE_src[None, :, :]

        # decoder inputs start with BOS=0, rest dummy (won't be read due to causal mask)
        y_in = np.full((1, Ttgt), 0, dtype=np.int32)

        pred = []
        for t in range(Ttgt):
            tgt_emb = tok_emb_tgt.forward(y_in) + PE_tgt[None, :, :]
            out, _ = model.forward(
                src_emb, tgt_emb, src_mask=None, tgt_mask=tgt_mask, mem_mask=None
            )
            z_t = head.logits(out)[0, t]  # (V,)
            z_t = z_t - z_t.max()  # stable
            p_t = np.exp(z_t)
            p_t /= p_t.sum() + 1e-12
            token = int(p_t.argmax())
            pred.append(token)
            if t + 1 < Ttgt:
                y_in[0, t + 1] = token  # feed next step
        return np.array(pred, dtype=np.int32)

    # show a couple of predictions
    for _ in range(10):
        s, _, t = make_batch(1, Tsrc, V, bos_id=0, rng=rng)
        pred = greedy_decode(s[0])
        print("src: ", s[0].tolist())
        print("tgt: ", t[0].tolist(), "(reversed)")
        print("pred:", pred.tolist())
        print("---")

# ---- run it ----
if __name__ == "__main__":
    # Smoke Test
    # B, Tsrc, Ttgt, D, h = 2, 5, 4, 32, 4
    # rng = np.random.default_rng(0)
    # src = rng.normal(size=(B, Tsrc, D)).astype(np.float32)
    # tgt = rng.normal(size=(B, Ttgt, D)).astype(np.float32)
    #
    # # causal mask for decoder self-attention:
    # # shape (B,1,Ttgt,Ttgt), 0 for keep, -1e9 for block
    # i = np.arange(Ttgt)
    # causal = (i[None, :] < i[:, None]).astype(np.float32) * -1e9
    # tgt_mask = causal[None, None, :, :]
    #
    # model = Transformer(
    #     num_enc_layers=2, num_dec_layers=2, d_model=D, n_heads=h, d_ff=4 * D, seed=42
    # )
    # out, mem = model.forward(src, tgt, tgt_mask=tgt_mask)
    #
    # # pretend loss = sum(out). Backprop all ones.
    # dout = np.ones_like(out, dtype=np.float32)
    # dsrc, dtgt = model.backward(dout)
    #
    # # one SGD step
    # model.step(lr=1e-3, weight_decay=0.0)

    train_reverse_demo(Transformer)
