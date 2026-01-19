from typing import Tuple
import numpy as np
import operator
import functools


def one_hot(y: np.ndarray, C: int) -> np.ndarray:
    """Return one-hot rows for integer labels y in {0,...,C-1}."""
    Y = np.zeros((y.size, C), dtype=float)
    Y[np.arange(y.size), y] = 1.0
    return Y


def softmax(Z: np.ndarray, epsilon=1e-12) -> np.ndarray:
    # find a maximum along rows
    Zs = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Zs)
    Sk = E.sum(axis=1, keepdims=True)
    return E / (Sk + epsilon)


def cross_entropy(P, Y, epsilon=1e-12):
    """
    Measure of the difference in probability distributions
    """
    return -np.mean(np.sum(Y * np.log(P + epsilon), axis=1))


class MLP:
    """
    Two-layer classifier for 2-bit OR:
    input: (B, 2)
    hidden: ReLU of width H
    logits: (B, 2) -> softmax probabilities for classes {0, 1}
    """

    def he_init(self, fan_in, fan_out, rng):
        """
        Kaiming/He initialization for optimal weight variance
        Inputs:
        fan_in: Number of inputs feeding into one neuron of the layer
        fan_out: Number of output neurons (columns) of the layer
        """
        # compute standard deviation (fan-in mode)
        std = np.sqrt(2.0 / fan_in)
        # compute normal distribution sampling an independent
        # identically distributed random variable weight matrix
        return rng.normal(0.0, std, size=(fan_in, fan_out))

    def __init__(self, H=8, seed=0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = self.kai_init(2, H, rng)
        self.b1 = np.zeros(H)
        self.W2 = self.kai_init(H, 2, rng)
        self.b2 = np.zeros(2)

    def forward(self, X) -> np.ndarray:
        self.X = X
        self.U = X @ self.W1 + self.b1
        # ReLU
        self.H = np.maximum(0.0, self.U)
        self.Z = self.H @ self.W2 + self.b2
        self.P = softmax(self.Z)
        return self.P

    def backward(self, Y):
        B = Y.shape[0]
        grad_Z = (self.P - Y) / B

        self.grad_W2 = self.H.T @ grad_Z
        self.grad_b2 = grad_Z.sum(axis=0)
        grad_H = grad_Z @ self.W2.T

        # ReLU Backpropagation
        grad_U = grad_H
        grad_U[self.U <= 0.0] = 0.0

        self.grad_W1 = self.X.T @ grad_U
        self.grad_b1 = grad_U.sum(axis=0)

    def step(self, lr=0.1, weight_decay=0.0):
        if weight_decay:
            self.grad_W1 += weight_decay * self.W1
            self.grad_W2 += weight_decay * self.W2
        self.W1 -= lr * self.grad_W1
        self.b1 -= lr * self.grad_b1
        self.W2 -= lr * self.grad_W2
        self.b2 -= lr * self.grad_b2

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def make_or_dataset(repeats=500, seed=9000) -> Tuple[np.ndarray, np.ndarray]:
    X_base = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    # OR labels
    y_base = np.array([0, 1, 1, 1], dtype=int)

    X = np.tile(X_base, (repeats, 1))
    y = np.tile(y_base, repeats)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train_or(model, epochs=1000, lr=0.1, batch_size=16, weight_decay=0.0, seed=9000):
    X, y = make_or_dataset(repeats=250, seed=seed)
    Y = one_hot(y, C=2)
    n = len(X)
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        idx = rng.permutation(n)
        X, Y, y = X[idx], Y[idx], y[idx]

        # mini-batches
        for i in range(0, n, batch_size):
            xb = X[i : i + batch_size]
            yb = Y[i : i + batch_size]
            P = model.forward(xb)
            model.backward(yb)
            model.step(lr=lr, weight_decay=weight_decay)

        # log
        if ep % 100 == 0 or ep == epochs - 1:
            X_tt = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            y_tt = np.array([0, 1, 1, 1])

            P_tt = model.forward(X_tt)
            loss = cross_entropy(P_tt, one_hot(y_tt, 2))
            acc = (P_tt.argmax(1) == y_tt).mean()
            print(
                f"epoch: {ep:4d} loss(tt): {loss:.6f} acc(tt): {acc:.3f} probs={np.round(P_tt, 3)}"
            )
    return model


def or_gate(model, a, b):
    """
    a, b are bits 0/1
    Returns the predicted bit (0 or 1) from the trained model
    """
    x = np.array([[a, b]])
    return int(model.predict(x)[0])


def or_reduce(model, bits):
    """
    Reduce a sequence of bits using the learned OR gate
    (((b0 OR b1) OR b2) OR ...)
    Also returns the per-step intermediate predictions
    """
    bits = [int(b) for b in bits]
    acc = bits[0]
    intermediates = [acc]
    for nxt in bits[1:]:
        acc = or_gate(model, acc, nxt)
        intermediates.append(acc)
    return acc, intermediates


if __name__ == "__main__":
    model = train_or(
        MLP(H=8, seed=9000),
        epochs=800,
        lr=0.1,
        batch_size=16,
        weight_decay=1e-4,
        seed=9000,
    )

    # Verify on truth table
    X_tt = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_pred = model.predict(X_tt)
    print(f"Input truth table:\n{X_tt}\nModel Prediction:\n{y_pred}\n")

    seq = [1, 0, 1, 1, 0]
    final_bit, steps = or_reduce(model, seq)
    print(f"Sequence {seq}  -> OR result {final_bit},  steps={steps}")

    assert or_gate(model, 1, 0) == 1
    assert or_gate(model, 1, 1) == 1
    assert or_gate(model, 0, 1) == 1
    assert or_gate(model, 0, 0) == 0
    assert or_reduce(model, seq)[0] == functools.reduce(operator.or_, seq)
