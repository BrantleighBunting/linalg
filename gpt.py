#!/usr/bin/env python3

import argparse, json, time, pathlib, sys, numpy as np
from typing import Dict, List, Generator
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizer worker threads
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # (unrelated, but quiets hub)
from datasets import load_dataset

# ---- import ONLY what we need from your original file (leave it unchanged) ----
# Change 'transformer_numpy' to your filename/module.
from ai_comps.transformer import (
    DecoderLayer,
    TokenEmbedding,
    OutputHead,
    sinusoidal_pos_encoding,
    causal_mask,
    LayerNorm,
    MHA,
    FFN,
)


# AdamW optimizer (Loshchilov & Hutter, "Decoupled Weight Decay Regularization")
# https://arxiv.org/abs/1711.05101
#
# High-level goal:
#   Given parameters p and their gradients g from backprop,
#   compute an update that:
#     • takes larger steps in directions with consistent gradients
#     • takes smaller steps where gradients are noisy
#     • prevents weights from growing without bound (regularization)
#
# Why we need this:
#   Simple SGD does:
#       p -= lr * g
#   which works, but:
#       • oscillates when gradients vary in sign
#       • requires very small lr in deep networks
#       • easily gets stuck or diverges
#
# AdamW improves this using *per-parameter adaptive learning rates* and
# *decoupled weight decay*. Each training step per parameter does:
#
#   1) Track a running average of gradients ("momentum")
#   2) Track a running average of squared gradients ("variance")
#   3) Normalize the update using these statistics
#   4) Apply weight decay separately
#
# Algorithm per parameter p with gradient g:
#
#   m  ← β1 * m + (1-β1) * g
#       (m tracks the *direction* of recent gradients, smoothing noise)
#
#   v  ← β2 * v + (1-β2) * g²
#       (v tracks the *magnitude* of recent gradients; large v means noisy or steep)
#
#   m̂ ← m / (1 - β1ᵗ)
#   v̂ ← v / (1 - β2ᵗ)
#       (bias correction: early steps underestimate m and v)
#
#   update = m̂ / (sqrt(v̂) + eps)
#       (normalize by recent gradient magnitude, forming an adaptive step)
#
#   p ← p - lr * update
#       (Adam update: large consistent gradients → larger step,
#                    noisy gradients → smaller step)
#
#   p ← p - lr * wd * p
#       (decoupled weight decay: shrink weights directly)
#
# IMPORTANT DISTINCTION:
#   Classical "L2 regularization" adds λ‖p‖² to the loss, meaning λp gets added
#   into g before momentum/variance tracking. This means Adam's m and v would
#   contain regularization effects, changing how much the optimizer trusts the
#   direction/magnitude of gradients. This causes:
#       • poorly tuned regularization strength
#       • inconsistent behavior across learning rates
#       • worse generalization
#
#   AdamW *decouples* weight decay from the gradient statistics:
#       weight decay does NOT enter m or v
#
# Why this matters:
#   • momentum tracks ONLY learning signal, not regularization signal
#   • variance estimates remain meaningful
#   • weight decay acts like true SGD shrinkage
#   • tuning wd behaves consistently across model sizes and learning rates
#
# Implementation details here:
#   • state[id(p)] stores m and v for each parameter
#   • param_groups let us apply weight_decay selectively
#       (standard practice: no decay on embeddings or LayerNorm γ/β)
#   • order:
#         p -= lr * wd * p
#         p -= lr * (m̂ / sqrt(v̂) + eps)
#     matches the decoupled formulation from the paper
#
# Result:
#   AdamW typically converges faster and generalizes better than Adam,
#   especially in Transformer-style architectures.
class AdamW:
    def __init__(self, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.state = {}  # id(param) -> {"m":..., "v":...}

    def _get_state(self, p):
        pid = id(p)
        if pid not in self.state:
            self.state[pid] = {
                "m": np.zeros_like(p, dtype=np.float32),
                "v": np.zeros_like(p, dtype=np.float32),
            }
        return self.state[pid]

    def step(self, param_groups):
        """
        param_groups: iterable of dicts with:
          {"p": ndarray, "g": ndarray, "weight_decay": float}
        """
        self.t += 1
        b1t = self.b1
        b2t = self.b2
        for pg in param_groups:
            p, g = pg["p"], pg["g"]
            wd = pg.get("weight_decay", self.wd)

            st = self._get_state(p)
            m, v = st["m"], st["v"]

            # Adam moments
            m *= b1t
            m += (1.0 - b1t) * g
            v *= b2t
            v += (1.0 - b2t) * (g * g)

            # bias correction
            mhat = m / (1.0 - (b1t**self.t))
            vhat = v / (1.0 - (b2t**self.t))

            # decoupled weight decay (skip for LN/embeddings by setting wd=0 in their group)
            if wd != 0.0:
                p -= self.lr * wd * p

            # param update
            p -= self.lr * (mhat / (np.sqrt(vhat) + self.eps))


class DecoderOnlyLayer:
    def __init__(self, d_model, n_heads, d_ff, seed=0):
        self.ln1 = LayerNorm(d_model)
        self.sa = MHA(d_model, n_heads, seed=seed)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, activation="relu", seed=seed + 1)

    def forward(self, X, tgt_mask):
        # pre-LN + masked self-attn + residual
        Xn = self.ln1.forward(X)
        A = self.sa.forward(Xn, mask=tgt_mask, KV=None)
        Y1 = X + A
        # pre-LN + FFN + residual
        Y1n = self.ln2.forward(Y1)
        F = self.ffn.forward(Y1n)
        return Y1 + F

    def backward(self, dY):
        # FFN branch
        dY2 = dY.copy()
        dF = dY
        dY1n = self.ffn.backward(dF)
        dY2 += self.ln2.backward(dY1n)
        # Self-attn branch
        dY1 = dY2.copy()
        dA = dY2
        dXn, _ = self.sa.backward(dA)  # KV=None → returns only dX
        dX = dY1 + self.ln1.backward(dXn)
        return dX

    def step(self, lr=3e-3, weight_decay=0.0):
        self.sa.step(lr, weight_decay)
        self.ffn.step(lr, weight_decay)
        self.ln1.step(lr, 0.0)
        self.ln2.step(lr, 0.0)


# ---------- tiny GPT stack (decoder-only, no cross-attn) ----------
class GPT:
    def __init__(self, num_layers=4, d_model=256, n_heads=4, d_ff=None, seed=123):
        if d_ff is None:
            d_ff = 4 * d_model
        self.layers = [
            DecoderOnlyLayer(d_model, n_heads, d_ff, seed=seed + i * 7)
            for i in range(num_layers)
        ]

    def forward(self, X, tgt_mask=None):
        H = X
        for lyr in self.layers:
            H = lyr.forward(H, tgt_mask)
        return H

    def backward(self, dH):
        g = dH
        for lyr in reversed(self.layers):
            g = lyr.backward(g)
        return g

    def step(self, lr=3e-3, weight_decay=0.0001):
        for lyr in self.layers:
            lyr.step(lr, weight_decay)


# ---------- data: load with Hugging Face + build char vocab ----------
def load_text() -> str:
    RAW_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # One big text doc; we’ll split ourselves 90/5/5
    raw = load_dataset("text", data_files={"dat": RAW_URL})
    full_text = "\n".join(x["text"] for x in raw["dat"])
    # dataset has a single split with a single row containing the entire corpus in 'text'
    return full_text


def build_char_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.int32)


def decode(ids: List[int], itos: Dict[int, str]) -> str:
    return "".join(itos[int(i)] for i in ids)


# Random window batching for next-token LM
def batch_stream(data_ids: np.ndarray, B: int, T: int, rng: np.random.Generator):
    L = len(data_ids)
    while True:
        ix = rng.integers(0, L - T - 1, size=B)
        x = np.stack([data_ids[i : i + T] for i in ix], axis=0)
        y = np.stack([data_ids[i + 1 : i + T + 1] for i in ix], axis=0)
        yield x, y


# ---------- training ----------
def train(args):
    rng = np.random.default_rng(args.seed)
    text = load_text()

    # model & layers
    D = args.d_model
    H = args.heads
    L = args.layers

    try:
        gpt, tok, head, PE, stoi, itos = load_ckpt(args.ckpt_dir)
    except Exception:
        print("Error loading checkpoint, starting from scratch")
        stoi, itos = build_char_vocab(text)
        V = len(stoi)
        Dff = 4 * D
        tok = TokenEmbedding(V, D, seed=1)
        head = OutputHead(D, V, seed=2)
        gpt = GPT(num_layers=L, d_model=D, n_heads=H, d_ff=Dff, seed=123)
    PE = sinusoidal_pos_encoding(args.ctx_len, D)
    mask = causal_mask(args.ctx_len)
    ids = encode(text, stoi)

    # split train/val
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    train_iter = batch_stream(train_ids, args.batch_size, args.ctx_len, rng)
    val_iter = batch_stream(val_ids, args.batch_size, args.ctx_len, rng)

    ckpt_dir = pathlib.Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "chars_gpt_best.npz"
    meta_path = ckpt_dir / "chars_gpt_meta.json"

    def lr_schedule(step, warmup=200, max_steps=args.steps, base=3e-3, min_lr=3e-4):
        if step < warmup:
            return base * (step / max(1, warmup))
        # cosine decay to min_lr
        t = (step - warmup) / max(1, max_steps - warmup)
        return min_lr + 0.5 * (base - min_lr) * (1 + np.cos(np.pi * t))

    def collect_param_groups(gpt, tok, head, wd):
        groups = []

        # --- embeddings (no decay) ---
        groups.append({"p": tok.W, "g": tok.gradW, "weight_decay": 0.0})

        # --- head: we tie head.W = tok.W.T so DO NOT step head.W.
        # Only step head.b (bias) with no decay.
        groups.append({"p": head.b, "g": head.gradb, "weight_decay": 0.0})

        # --- transformer layers ---
        for lyr in gpt.layers:
            sa = lyr.sa
            ffn = lyr.ffn
            ln1 = lyr.ln1
            ln2 = lyr.ln2

            # MHA weights (decay)
            groups += [
                {"p": sa.Wq, "g": sa.grads["Wq"], "weight_decay": wd},
                {"p": sa.Wk, "g": sa.grads["Wk"], "weight_decay": wd},
                {"p": sa.Wv, "g": sa.grads["Wv"], "weight_decay": wd},
                {"p": sa.Wo, "g": sa.grads["Wo"], "weight_decay": wd},
            ]
            # FFN (decay on weights, none on biases)
            groups += [
                {"p": ffn.W1, "g": ffn.grads["W1"], "weight_decay": wd},
                {"p": ffn.b1, "g": ffn.grads["b1"], "weight_decay": 0.0},
                {"p": ffn.W2, "g": ffn.grads["W2"], "weight_decay": wd},
                {"p": ffn.b2, "g": ffn.grads["b2"], "weight_decay": 0.0},
            ]
            # LayerNorm (no decay)
            groups += [
                {"p": ln1.gamma, "g": ln1._grads["gamma"], "weight_decay": 0.0},
                {"p": ln1.beta, "g": ln1._grads["beta"], "weight_decay": 0.0},
                {"p": ln2.gamma, "g": ln2._grads["gamma"], "weight_decay": 0.0},
                {"p": ln2.beta, "g": ln2._grads["beta"], "weight_decay": 0.0},
            ]

        return groups

    def zero_all_grads():
        tok.gradW.fill(0.0)
        head.gradW.fill(0.0)
        head.gradb.fill(0.0)
        for lyr in gpt.layers:
            sa, ffn, ln1, ln2 = lyr.sa, lyr.ffn, lyr.ln1, lyr.ln2
            for k in ["Wq", "Wk", "Wv", "Wo"]:
                sa.grads[k].fill(0.0)
            for k in ["W1", "b1", "W2", "b2"]:
                ffn.grads[k].fill(0.0)
            ln1._grads["gamma"].fill(0.0)
            ln1._grads["beta"].fill(0.0)
            ln2._grads["gamma"].fill(0.0)
            ln2._grads["beta"].fill(0.0)

    # single persistent AdamW optimizer
    optimizer = AdamW(
        lr=3e-4,  # good starting point for AdamW
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,  # decoupled; groups override when needed
    )

    def step_batch_adamw(step, x_ids, y_ids):
        # forward
        emb = tok.forward(x_ids) + PE[None, :, :]
        # weight tying (forward path)
        head.W = tok.W.T

        h = gpt.forward(emb, tgt_mask=mask)
        logits = head.logits(h)
        loss, dZ = head.loss_and_dlogits(logits, y_ids)

        # backward
        dH = head.backward(dZ)

        # tie grads: push head.W grad into tok.W, clear head.W grad
        tok.gradW += head.gradW.T
        head.gradW.fill(0.0)

        dX = gpt.backward(dH)
        tok.backward(dX)

        # LR schedule drives AdamW.lr
        optimizer.lr = lr_schedule(
            step, warmup=200, max_steps=args.steps, base=3e-4, min_lr=3e-5
        )

        # Build groups and take an AdamW step
        groups = collect_param_groups(gpt, tok, head, wd=args.weight_decay)
        optimizer.step(groups)

        # zero grads
        zero_all_grads()
        return loss

    def step_batch(x_ids, y_ids):
        # (kept around for comparison; not used right now)
        emb = tok.forward(x_ids) + PE[None, :, :]
        h = gpt.forward(emb, tgt_mask=mask)
        head.W = tok.W.T
        logits = head.logits(h)
        loss, dZ = head.loss_and_dlogits(logits, y_ids)
        # backward
        dH = head.backward(dZ)
        # tie grads: add dW.T into embedding grads and zero head’s dW so it won’t step
        tok.gradW += head.gradW.T
        head.gradW.fill(0.0)
        dX = gpt.backward(dH)
        tok.backward(dX)
        # SGD step
        lr = lr_schedule(step)
        head.step(lr=lr, weight_decay=args.weight_decay)
        tok.step(lr=lr, weight_decay=args.weight_decay)
        gpt.step(lr=lr, weight_decay=args.weight_decay)
        return loss

    def eval_avg(it, batches=10):
        losses = []
        for _ in range(batches):
            x_ids, y_ids = next(it)
            emb = tok.forward(x_ids) + PE[None, :, :]
            # weight tying
            head.W = tok.W.T
            h = gpt.forward(emb, tgt_mask=mask)
            logits = head.logits(h)
            loss, _ = head.loss_and_dlogits(logits, y_ids)
            losses.append(loss)
        return float(np.mean(losses))

    best = 1e9
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = next(train_iter)
        # loss = step_batch(x, y)
        loss = step_batch_adamw(step, x, y)
        if step % 20 == 0 or step == 1:
            print(f"step {step:6d}  loss {loss:.4f}")
        if step % args.eval_every == 0:
            val_loss = eval_avg(val_iter, batches=20)
            print(f"[eval] step {step:6d}  val_loss {val_loss:.4f}")
            if val_loss < best:
                best = val_loss
                np.savez_compressed(
                    ckpt_path,
                    tok_W=tok.W,
                    head_W=head.W,
                    head_b=head.b,
                    # MHA
                    **{f"l{i}_Wq": lyr.sa.Wq for i, lyr in enumerate(gpt.layers)},
                    **{f"l{i}_Wk": lyr.sa.Wk for i, lyr in enumerate(gpt.layers)},
                    **{f"l{i}_Wv": lyr.sa.Wv for i, lyr in enumerate(gpt.layers)},
                    **{f"l{i}_Wo": lyr.sa.Wo for i, lyr in enumerate(gpt.layers)},
                    # FFN
                    **{f"l{i}_W1": lyr.ffn.W1 for i, lyr in enumerate(gpt.layers)},
                    **{f"l{i}_b1": lyr.ffn.b1 for i, lyr in enumerate(gpt.layers)},
                    **{f"l{i}_W2": lyr.ffn.W2 for i, lyr in enumerate(gpt.layers)},
                    **{f"l{i}_b2": lyr.ffn.b2 for i, lyr in enumerate(gpt.layers)},
                    # LayerNorms
                    **{
                        f"l{i}_ln1_g": lyr.ln1.gamma for i, lyr in enumerate(gpt.layers)
                    },
                    **{f"l{i}_ln1_b": lyr.ln1.beta for i, lyr in enumerate(gpt.layers)},
                    **{
                        f"l{i}_ln2_g": lyr.ln2.gamma for i, lyr in enumerate(gpt.layers)
                    },
                    **{f"l{i}_ln2_b": lyr.ln2.beta for i, lyr in enumerate(gpt.layers)},
                )
                with open(meta_path, "w") as f:
                    json.dump(
                        {
                            "stoi": stoi,
                            "itos": itos,
                            "d_model": D,
                            "heads": H,
                            "layers": L,
                            "ctx_len": args.ctx_len,
                        },
                        f,
                    )
                print(f"  saved best → {ckpt_path}  (val {best:.4f})")
    print(f"done in {time.time() - t0:.1f}s")


def load_ckpt(ckpt_dir):
    ckpt_dir = pathlib.Path(ckpt_dir)
    z = np.load(ckpt_dir / "chars_gpt_best.npz")
    meta = json.load(open(ckpt_dir / "chars_gpt_meta.json"))
    D = meta["d_model"]
    H = meta["heads"]
    L = meta["layers"]
    T = meta["ctx_len"]
    stoi, itos = meta["stoi"], {int(k): v for k, v in meta["itos"].items()}
    V = len(stoi)
    # rebuild model
    tok = TokenEmbedding(V, D, seed=1)
    tok.W = z["tok_W"]
    head = OutputHead(D, V, seed=2)
    head.W = z["head_W"]
    head.b = z["head_b"]
    gpt = GPT(num_layers=L, d_model=D, n_heads=H, d_ff=4 * D, seed=123)
    for i, lyr in enumerate(gpt.layers):
        # MHA
        lyr.sa.Wq = z[f"l{i}_Wq"]
        lyr.sa.Wk = z[f"l{i}_Wk"]
        lyr.sa.Wv = z[f"l{i}_Wv"]
        lyr.sa.Wo = z[f"l{i}_Wo"]
        # FFN
        lyr.ffn.W1 = z[f"l{i}_W1"]
        lyr.ffn.b1 = z[f"l{i}_b1"]
        lyr.ffn.W2 = z[f"l{i}_W2"]
        lyr.ffn.b2 = z[f"l{i}_b2"]
        # LayerNorms
        lyr.ln1.gamma = z[f"l{i}_ln1_g"]
        lyr.ln1.beta = z[f"l{i}_ln1_b"]
        lyr.ln2.gamma = z[f"l{i}_ln2_g"]
        lyr.ln2.beta = z[f"l{i}_ln2_b"]
    PE = sinusoidal_pos_encoding(T, D)
    return gpt, tok, head, PE, stoi, itos


def sample(
    gpt, tok, head, PE, ctx_ids, itos, steps=200, temperature=1.0, top_k=0
) -> Generator[str, None, None]:
    """Generate tokens one at a time, yielding each character as it's produced."""
    ids = ctx_ids.copy()
    for _ in range(steps):
        Tctx = len(ids)
        if Tctx > PE.shape[0]:
            ids = ids[-PE.shape[0] :]
            Tctx = len(ids)
        x = ids[None, :]
        emb = tok.forward(x) + PE[None, :Tctx, :]
        mask = causal_mask(Tctx)
        h = gpt.forward(emb, tgt_mask=mask)
        head.W = tok.W.T  # keep tying on sampling too
        z = head.logits(h)[0, -1]  # (V,)
        z = z / max(1e-6, float(temperature))
        if top_k > 0:
            k = min(top_k, z.size)
            idx = np.argpartition(z, -k)[-k:]
            m = np.full_like(z, -1e9)
            m[idx] = 0.0
            z = z + m
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum() + 1e-12
        nxt = int(np.random.choice(z.size, p=p))
        ids = np.append(ids, nxt)
        yield itos[nxt]


def repl(args):
    gpt, tok, head, PE, stoi, itos = load_ckpt(args.ckpt_dir)
    print("\nREPL — type a prompt, Ctrl+C to exit.\n")
    while True:
        try:
            s = input("> ")
        except KeyboardInterrupt:
            print("\nbye")
            break
        if not s.strip():
            continue
        # unseen chars → drop or map; here we drop
        s = "".join(ch for ch in s if ch in stoi)
        ctx = np.array([stoi[ch] for ch in s], dtype=np.int32)
        for ch in sample(
            gpt,
            tok,
            head,
            PE,
            ctx,
            itos,
            steps=args.gen_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            print(ch, end="", flush=True)
        print()  # newline after generation completes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--ctx_len", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--lr_model", type=float, default=3e-3)
    ap.add_argument("--lr_embed", type=float, default=3e-3)
    ap.add_argument("--lr_head", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints_np")
    ap.add_argument("--gen_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.train:
        train(args)
    if args.repl:
        repl(args)
    if not args.train and not args.repl:
        print("Nothing to do. Pass --train and/or --repl.")


if __name__ == "__main__":
    main()
