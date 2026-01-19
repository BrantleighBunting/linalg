#!/usr/bin/env python3
"""
Transformer (pre-LN, encoder–decoder) in **PyTorch**, drop-in replacement for your NumPy version.

Components
- Pre-LayerNorm blocks (LN → sublayer → residual)
- Multi-Head Attention (self & cross) via nn.MultiheadAttention(batch_first=True)
- Position-wise FFN (GeLU or ReLU)
- Sinusoidal positional encodings
- Token embeddings (separate src/tgt) + optional output head with **weight tying** to tgt embeddings
- Causal masking for decoder self-attention
- Toy reversal task training loop (teacher forcing) to sanity-check

This mirrors your shapes and flow: (B,T,D).
Tested with PyTorch ≥ 2.1. Works on CPU/GPU.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- Utils --------------------------

def sinusoidal_pos_encoding(T: int, D: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return (T, D) fixed sinusoidal encodings.
    PE[p, 2i]   = sin(p / 10000^{2i/D})
    PE[p, 2i+1] = cos(p / 10000^{2i/D})
    """
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T,1)
    i = torch.arange(D, device=device, dtype=dtype).unsqueeze(0)    # (1,D)
    angle = pos / (10000 ** (2 * torch.div(i, 2, rounding_mode='floor') / D))
    pe = torch.zeros((T, D), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


def causal_mask(T: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return an additive causal mask of shape (T, T) with -inf above diagonal.
    Suitable for nn.MultiheadAttention(attn_mask=...).
    """
    m = torch.full((T, T), float('-inf'), device=device, dtype=dtype)
    m = torch.triu(m, diagonal=1)  # upper triangle (strict)
    return m


# -------------------------- Blocks --------------------------

class PreLNEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, activation: str = 'relu'):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN → self-attn → residual
        x_norm = self.ln1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        # Pre-LN → FFN → residual
        y_norm = self.ln2(x)
        y = self.ffn(y_norm)
        x = x + self.dropout(y)
        return x


class PreLNDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, activation: str = 'relu'):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # masked self-attn (causal)
        x_norm = self.ln1(x)
        self_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)
        x = x + self.dropout(self_out)
        # cross-attn (Q from decoder stream, K/V from encoder memory)
        y_norm = self.ln2(x)
        cross_out, _ = self.cross_attn(y_norm, memory, memory, attn_mask=mem_mask, key_padding_mask=memory_key_padding_mask, need_weights=False)
        x = x + self.dropout(cross_out)
        # FFN
        z_norm = self.ln3(x)
        z = self.ffn(z_norm)
        x = x + self.dropout(z)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, activation: str = 'relu'):
        super().__init__()
        self.layers = nn.ModuleList([
            PreLNEncoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, activation: str = 'relu'):
        super().__init__()
        self.layers = nn.ModuleList([
            PreLNDecoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, mem_mask=mem_mask, memory_key_padding_mask=memory_key_padding_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_enc_layers: int,
        num_dec_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = 'relu',
    ):
        super().__init__()
        self.encoder = Encoder(num_enc_layers, d_model, n_heads, d_ff, dropout, activation)
        self.decoder = Decoder(num_dec_layers, d_model, n_heads, d_ff, dropout, activation)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            mem_mask=mem_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out, memory


# -------------------------- Embeddings & Head --------------------------

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


class OutputHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, tie_with: Optional[nn.Embedding] = None):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=True)
        if tie_with is not None:
            # Weight tying: proj.weight shares storage with embedding.weight
            self.proj.weight = tie_with.weight  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# -------------------------- Toy data (reversal) --------------------------

def make_batch(B: int, T: int, V: int, bos_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.randint(1, V, (B, T), device=device)  # [1..V-1]
    rev = torch.flip(src, dims=[1])
    tgt_out = rev.clone()
    bos = torch.full((B, 1), bos_id, device=device, dtype=torch.long)
    tgt_in = torch.cat([bos, rev[:, :-1]], dim=1)
    return src, tgt_in, tgt_out


# -------------------------- Training demo --------------------------

@dataclass
class TrainConfig:
    B: int = 64
    Tsrc: int = 8
    Ttgt: int = 8
    V: int = 32
    D: int = 64
    H: int = 4
    Dff: int = 256
    Lenc: int = 2
    Ldec: int = 2
    epochs: int = 2000
    dropout: float = 0.0
    activation: str = 'relu'  # 'relu' or 'gelu'
    lr: float = 5e-3
    wd: float = 0.0
    tie_weights: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_reverse_demo_torch(cfg: TrainConfig = TrainConfig()):
    device = torch.device(cfg.device)
    torch.manual_seed(42)

    # Modules
    tok_src = TokenEmbedding(cfg.V, cfg.D).to(device)
    tok_tgt = TokenEmbedding(cfg.V, cfg.D).to(device)
    model = Transformer(cfg.Lenc, cfg.Ldec, cfg.D, cfg.H, cfg.Dff, dropout=cfg.dropout, activation=cfg.activation).to(device)
    head = OutputHead(cfg.D, cfg.V, tie_with=(tok_tgt.emb if cfg.tie_weights else None)).to(device)

    # Positional encodings
    PE_src = sinusoidal_pos_encoding(cfg.Tsrc, cfg.D, device)
    PE_tgt = sinusoidal_pos_encoding(cfg.Ttgt, cfg.D, device)

    # Causal mask for decoder self-attn
    tgt_causal = causal_mask(cfg.Ttgt, device)

    params = list(model.parameters()) + list(tok_src.parameters()) + list(tok_tgt.parameters()) + list(head.parameters())
    # If tied, head.proj.weight shares storage with tok_tgt.emb.weight; optimizer handles correctly.
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.wd)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, cfg.epochs + 1):
        src_idx, tgt_in_idx, tgt_out_idx = make_batch(cfg.B, cfg.Tsrc, cfg.V, bos_id=0, device=device)

        src_emb = tok_src(src_idx) + PE_src.unsqueeze(0)  # (B,Tsrc,D)
        tgt_emb = tok_tgt(tgt_in_idx) + PE_tgt.unsqueeze(0)  # (B,Ttgt,D)

        out, memory = model(src_emb, tgt_emb, src_mask=None, tgt_mask=tgt_causal, mem_mask=None)
        logits = head(out)  # (B,Ttgt,V)

        # Cross-entropy over (B*T) positions
        loss = loss_fn(logits.reshape(-1, cfg.V), tgt_out_idx.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        if ep % 20 == 0 or ep == 1:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == tgt_out_idx).float().mean().item()
            print(f"epoch {ep:4d}  loss {loss.item():.4f}  token-acc {acc:.3f}")

    # Greedy decode demo
    def greedy_decode(src_idx_single: torch.Tensor) -> torch.Tensor:
        src_idx_single = src_idx_single.to(device)
        with torch.no_grad():
            src_emb = tok_src(src_idx_single.unsqueeze(0)) + PE_src.unsqueeze(0)
            y_in = torch.zeros((1, cfg.Ttgt), dtype=torch.long, device=device)  # BOS=0
            for t in range(cfg.Ttgt):
                tgt_emb = tok_tgt(y_in) + PE_tgt.unsqueeze(0)
                out, _ = model(src_emb, tgt_emb, tgt_mask=tgt_causal)
                z_t = head(out)[:, t, :]  # (1,V)
                y_in[0, t] = z_t.argmax(dim=-1)
        return y_in[0]

    print("\n--- Greedy decode samples ---")
    for _ in range(5):
        s, _, t = make_batch(1, cfg.Tsrc, cfg.V, bos_id=0, device=device)
        pred = greedy_decode(s[0])
        print("src:", s[0].tolist())
        print("tgt:", t[0].tolist(), "(reversed)")
        print("pred:", pred.tolist())
        print("---")


if __name__ == "__main__":
    train_reverse_demo_torch()

