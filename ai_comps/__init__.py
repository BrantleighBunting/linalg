"""
ai_comps - Neural network components for transformer architectures.

This package provides modular, educational implementations of transformer
building blocks in pure NumPy with manual backpropagation.

Modules:
- activations: Activation functions (ReLU, GELU)
- normalization: Layer normalization variants (LayerNorm, RMSNorm)
- positional: Positional encoding schemes (sinusoidal, learned, RoPE)
- attention: Attention mechanisms (scaled dot-product, multi-head)
- tokenizers: Text tokenization (character-level, BPE placeholder)
- cache: KV caching for efficient inference
- transformer: Full transformer blocks and models
"""

# Activations
from .activations import (
    relu,
    relu_backward,
    gelu,
    gelu_backward,
    get_activation,
    ACTIVATIONS,
)

# Normalization
from .normalization import (
    LayerNorm,
    RMSNorm,
    get_norm,
)

# Positional encodings
from .positional import (
    sinusoidal_encoding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    get_positional_encoding,
)

# Attention
from .attention import (
    softmax_last,
    causal_mask,
    ScaledDotProductAttention,
    MultiHeadAttention,
    MHA,
    Attention,
)

# Tokenizers
from .tokenizers import (
    BaseTokenizer,
    CharTokenizer,
)

# Caching
from .cache import (
    KVCache,
    LayerKVCache,
    apply_kv_cache,
)

# Transformer components (from original transformer.py)
from .transformer import (
    softmax_rows,
    sinusoidal_pos_encoding,
    he_init,
    FFN,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
    Transformer,
    TokenEmbedding,
    OutputHead,
)

__all__ = [
    # Activations
    "relu",
    "relu_backward",
    "gelu",
    "gelu_backward",
    "get_activation",
    "ACTIVATIONS",
    # Normalization
    "LayerNorm",
    "RMSNorm",
    "get_norm",
    # Positional
    "sinusoidal_encoding",
    "sinusoidal_pos_encoding",  # alias from transformer.py
    "LearnedPositionalEmbedding",
    "RotaryPositionalEmbedding",
    "get_positional_encoding",
    # Attention
    "softmax_last",
    "softmax_rows",
    "causal_mask",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "MHA",
    "Attention",
    "he_init",
    # Tokenizers
    "BaseTokenizer",
    "CharTokenizer",
    # Cache
    "KVCache",
    "LayerKVCache",
    "apply_kv_cache",
    # Transformer
    "FFN",
    "EncoderLayer",
    "DecoderLayer",
    "Encoder",
    "Decoder",
    "Transformer",
    "TokenEmbedding",
    "OutputHead",
]
