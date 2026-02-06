"""
Tokenizers for text encoding/decoding.

Implements character-level tokenization with encode/decode methods.
"""

import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Convert text to token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: np.ndarray) -> str:
        """Convert token IDs back to text."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer with stoi/itos mappings."""

    def __init__(self, text: Optional[str] = None, vocab: Optional[List[str]] = None):
        """Build vocabulary from text corpus or explicit char list."""
        if vocab is not None:
            chars = vocab
        elif text is not None:
            chars = sorted(list(set(text)))
        else:
            raise ValueError("Must provide either text or vocab")

        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_pretrained(cls, stoi: Dict[str, int], itos: Dict[int, str]) -> "CharTokenizer":
        """Create tokenizer from existing stoi/itos mappings."""
        tokenizer = cls.__new__(cls)
        tokenizer.stoi = stoi
        tokenizer.itos = {int(k): v for k, v in itos.items()}  # ensure int keys
        return tokenizer

    def encode(self, text: str, drop_unknown: bool = True) -> np.ndarray:
        """Encode text to token IDs. Returns int32 array."""
        if drop_unknown:
            ids = [self.stoi[ch] for ch in text if ch in self.stoi]
        else:
            ids = [self.stoi[ch] for ch in text]
        return np.array(ids, dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        """Decode token IDs back to text."""
        return "".join(self.itos[int(i)] for i in ids)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.stoi)

    def __contains__(self, char: str) -> bool:
        """Check if character is in vocabulary."""
        return char in self.stoi

    def save(self) -> Dict:
        """Export stoi/itos mappings for serialization."""
        return {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},  # JSON needs str keys
        }

    @classmethod
    def load(cls, data: Dict) -> "CharTokenizer":
        """Load tokenizer from serialized state dict."""
        return cls.from_pretrained(data["stoi"], data["itos"])


# Placeholder for future BPE implementation
class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding tokenizer (not yet implemented)."""

    def __init__(self):
        raise NotImplementedError(
            "BPE tokenizer not yet implemented. "
            "Consider using tiktoken: pip install tiktoken"
        )

    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def decode(self, ids: np.ndarray) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
