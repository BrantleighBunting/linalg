import numpy as np
import operator
import functools
# -----------------------
# Helpers: one-hot, softmax, loss
# -----------------------
def one_hot(y, C):
    Y = np.zeros((y.size, C), dtype=float)
    Y[np.arange(y.size), y] = 1.0
    return Y

def stable_softmax(Z):
    Zs = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Zs)
    return E / (E.sum(axis=1, keepdims=True) + 1e-12)

def cross_entropy(P, Y):
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))

# -----------------------
# MLP: ReLU -> Softmax
# -----------------------
def he_init(fan_in, fan_out, rng):
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))

class XORMLP:
    """
    Two-layer classifier for 2-bit XOR:
      input:  (B, 2)
      hidden: ReLU of width H
      logits: (B, 2) -> softmax probabilities for classes {0,1}
    """
    def __init__(self, H=8, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = he_init(2, H, rng); self.b1 = np.zeros(H)
        self.W2 = he_init(H, 2, rng); self.b2 = np.zeros(2)

    def forward(self, X):
        self.X = X
        self.U = X @ self.W1 + self.b1          # (B,H)
        self.H = np.maximum(0.0, self.U)        # ReLU
        self.Z = self.H @ self.W2 + self.b2     # (B,2)
        self.P = stable_softmax(self.Z)         # (B,2)
        return self.P

    def backward(self, Y):
        B = Y.shape[0]
        dZ = (self.P - Y) / B                   # (B,2)

        # last affine
        self.dW2 = self.H.T @ dZ                # (H,2)
        self.db2 = dZ.sum(axis=0)               # (2,)
        dH = dZ @ self.W2.T                     # (B,H)

        # ReLU backprop
        dU = dH
        dU[self.U <= 0.0] = 0.0

        # first affine
        self.dW1 = self.X.T @ dU                # (2,H)
        self.db1 = dU.sum(axis=0)               # (H,)

    def step(self, lr=0.1, weight_decay=0.0):
        if weight_decay:
            self.dW1 += weight_decay * self.W1
            self.dW2 += weight_decay * self.W2
        self.W1 -= lr * self.dW1; self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2; self.b2 -= lr * self.db2

    def predict_proba(self, X):
        P = self.forward(X)
        return P

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

# -----------------------
# Dataset: XOR truth table replicated & shuffled
# -----------------------
def make_xor_dataset(repeats=200, seed=1):
    X_base = np.array([[0.,0.],
                       [0.,1.],
                       [1.,0.],
                       [1.,1.]], dtype=float)
    y_base = np.array([0, 1, 1, 0], dtype=int)  # XOR labels
    X = np.tile(X_base, (repeats, 1))
    y = np.tile(y_base, repeats)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

# -----------------------
# Mini-batch training loop
# -----------------------
def train_xor(model, epochs=800, lr=0.1, batch_size=16, weight_decay=0.0, seed=123):
    X, y = make_xor_dataset(repeats=250, seed=seed)
    Y = one_hot(y, C=2)

    n = len(X)
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        # shuffle each epoch
        idx = rng.permutation(n)
        X, Y, y = X[idx], Y[idx], y[idx]

        # mini-batches
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size]
            yb = Y[i:i+batch_size]

            P = model.forward(xb)
            model.backward(yb)
            model.step(lr=lr, weight_decay=weight_decay)

        # occasional logging on full truth table
        if ep % 100 == 0 or ep == epochs-1:
            X_tt = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
            y_tt = np.array([0,1,1,0])
            P_tt = model.predict_proba(X_tt)
            loss = cross_entropy(P_tt, one_hot(y_tt,2))
            acc = (P_tt.argmax(1) == y_tt).mean()
            print(f"epoch {ep:4d}  loss(tt) {loss:.6f}  acc(tt) {acc:.3f}  probs={np.round(P_tt,3)}")

    return model

# -----------------------
# Use the learned gate on sequences
# -----------------------
def xor_gate(model, a, b):
    """
    a, b are bits 0/1 (ints or floats 0./1.)
    Returns predicted bit (0 or 1) from the trained model.
    """
    x = np.array([[float(a), float(b)]])
    return int(model.predict(x)[0])

def xor_reduce(model, bits):
    """
    Reduce a sequence of bits using the learned XOR gate:
    (((b0 XOR b1) XOR b2) XOR ...).
    Also returns the per-step intermediate predictions.
    """
    bits = [int(b) for b in bits]
    acc = bits[0]
    intermediates = [acc]
    for nxt in bits[1:]:
        acc = xor_gate(model, acc, nxt)
        intermediates.append(acc)
    return acc, intermediates

# -----------------------
# Demo
# -----------------------
if __name__ == "__main__":
    model = XORMLP(H=8, seed=0)
    model = train_xor(model, epochs=800, lr=0.1, batch_size=16, weight_decay=1e-4, seed=42)

    # Verify on truth table
    X_tt = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y_pred = model.predict(X_tt)
    print("Truth table preds:", y_pred)  # expect [0,1,1,0]

    # Use on sequences
    seq = [1, 0, 1, 1, 0]
    final_bit, steps = xor_reduce(model, seq)
    print(f"Sequence {seq}  -> XOR result {final_bit},  steps={steps}")

    assert xor_gate(model, 1, 0) == 1
    assert xor_gate(model, 1, 1) == 0
    assert xor_gate(model, 0, 1) == 1
    assert xor_gate(model, 0, 0) == 0
    assert xor_reduce(model, seq)[0] == functools.reduce(operator.xor, seq)
