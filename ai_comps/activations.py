"""
Activation functions for neural networks.

Currently implemented:
- ReLU: Standard rectified linear unit

Planned for future implementation:
- GELU: Gaussian Error Linear Unit (GPT-2/3 style)
- SwiGLU: Swish-Gated Linear Unit (LLaMA/PaLM style)
- GeGLU: GELU-Gated Linear Unit
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit: max(0, x).

    Args:
        x: Input array of any shape.

    Returns:
        Element-wise maximum of 0 and x.
    """
    return np.maximum(0.0, x)


def relu_backward(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU: 1 if x > 0 else 0.

    Args:
        x: Pre-activation values (same as forward input).

    Returns:
        Gradient mask, same shape as x.
    """
    return (x > 0.0).astype(x.dtype)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit (approximate).

    GELU(x) = x * Phi(x) where Phi is the CDF of standard normal.
    Using the tanh approximation from the original paper:
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input array of any shape.

    Returns:
        GELU activation applied element-wise.
    """
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def gelu_backward(x: np.ndarray) -> np.ndarray:
    """
    Derivative of GELU (approximate).

    Args:
        x: Pre-activation values.

    Returns:
        Gradient of GELU, same shape as x.
    """
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner**2
    inner_deriv = c * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * inner_deriv


# Registry for easy lookup by name
ACTIVATIONS = {
    "relu": (relu, relu_backward),
    "gelu": (gelu, gelu_backward),
}


def get_activation(name: str):
    """
    Get activation function and its derivative by name.

    Args:
        name: One of 'relu', 'gelu'.

    Returns:
        Tuple of (forward_fn, backward_fn).

    Raises:
        KeyError: If activation name is not recognized.
    """
    if name not in ACTIVATIONS:
        raise KeyError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]
