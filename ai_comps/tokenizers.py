"""
Tokenizers for text encoding/decoding.

Currently implemented:
- CharTokenizer: Simple character-level tokenization

Planned for future implementation:
- BPETokenizer: Byte-Pair Encoding (GPT-2 style)
- SentencePieceTokenizer: Unigram/BPE via SentencePiece
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
    """
    Character-level tokenizer.

    Simple tokenization where each unique character is a token.
    Useful for educational purposes and small-scale experiments.

    Attributes:
        stoi: String-to-integer mapping (char -> id).
        itos: Integer-to-string mapping (id -> char).
    """

    def __init__(self, text: Optional[str] = None, vocab: Optional[List[str]] = None):
        """
        Initialize tokenizer from text corpus or explicit vocabulary.

        Args:
            text: Text corpus to build vocabulary from.
            vocab: Explicit list of characters (alternative to text).

        Raises:
            ValueError: If neither text nor vocab is provided.
        """
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
        """
        Create tokenizer from pre-existing mappings (e.g., loaded from checkpoint).

        Args:
            stoi: String-to-integer mapping.
            itos: Integer-to-string mapping.

        Returns:
            CharTokenizer instance.
        """
        tokenizer = cls.__new__(cls)
        tokenizer.stoi = stoi
        tokenizer.itos = {int(k): v for k, v in itos.items()}  # ensure int keys
        return tokenizer

    def encode(self, text: str, drop_unknown: bool = True) -> np.ndarray:
        """
        Encode text to token IDs.

        Args:
            text: Input string.
            drop_unknown: If True, skip characters not in vocabulary.
                         If False, raises KeyError on unknown chars.

        Returns:
            Array of token IDs (int32).
        """
        if drop_unknown:
            ids = [self.stoi[ch] for ch in text if ch in self.stoi]
        else:
            ids = [self.stoi[ch] for ch in text]
        return np.array(ids, dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: Array or list of token IDs.

        Returns:
            Decoded string.
        """
        return "".join(self.itos[int(i)] for i in ids)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.stoi)

    def __contains__(self, char: str) -> bool:
        """Check if character is in vocabulary."""
        return char in self.stoi

    def save(self) -> Dict:
        """
        Export tokenizer state for serialization.

        Returns:
            Dict with 'stoi' and 'itos' mappings.
        """
        return {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},  # JSON needs str keys
        }

    @classmethod
    def load(cls, data: Dict) -> "CharTokenizer":
        """
        Load tokenizer from serialized state.

        Args:
            data: Dict with 'stoi' and 'itos' keys.

        Returns:
            CharTokenizer instance.
        """
        return cls.from_pretrained(data["stoi"], data["itos"])


# Placeholder for future BPE implementation
class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding tokenizer.

    TODO: Implement BPE algorithm:
    1. Start with character vocabulary
    2. Iteratively merge most frequent pairs
    3. Build merge rules table
    4. Apply merges during encoding

    For now, consider using tiktoken or sentencepiece as external dependencies.
    """

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
