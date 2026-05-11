#!/usr/bin/env python3
"""
prune_local_heal.py — Qwen3.5 hybrid-arch attention head pruning with
                      backprop-free local healing (W_O lstsq refit).

Targets the Qwen3.5 hybrid architecture which mixes:
  - full_attention layers (Qwen3_5Attention with gated query, GQA 16:4, head_dim=256)
  - linear_attention layers (Qwen3_5GatedDeltaNet, 32 v_heads / 16 k_heads, head_dim=128)

Two prune levers:
  --prune-full-frac    : fraction of Q heads to drop in full_attention layers
  --prune-linear-frac  : fraction of V heads to drop in linear_attention layers

Phases:
  0. Capture per-layer o_proj-input (full_attn) and out_proj-input (linear_attn)
     on a calibration set — these are the targets for healing.
  1. Score per-head importance via Michel-style gradient on per-head scale α=1.
  2. For each layer, in forward order, on the *current* (already-modified) model:
       a. Re-capture pre-proj input
       b. Zero rows for dropped heads in q/k/v / in_proj_qkv (and corresponding
          rows in z/b/a/conv1d/A_log/dt_bias for linear_attn)
       c. lstsq refit the kept columns of o_proj / out_proj so the post-proj
          output matches the original target

The model stays full-size on disk (mask-style); a separate physical-resize
step converts to a smaller model after eval validation. Mask-pruning is what
lstsq refit operates on, and it preserves output quality identically — only
inference compute differs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- calibration ----------

CALIB_TEXTS = [
    # Code-leaning (HumanEval/MBPP-style prompts) + a few prose to keep general-purpose signal
    "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if there are any two numbers in the list that are closer than threshold.\"\"\"\n    for i, a in enumerate(numbers):\n        for j, b in enumerate(numbers):\n            if i != j and abs(a - b) < threshold:\n                return True\n    return False\n",
    "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n",
    "import re\n\n\ndef is_palindrome(s: str) -> bool:\n    s = re.sub(r'[^A-Za-z0-9]', '', s).lower()\n    return s == s[::-1]\n",
    "from collections import Counter\n\n\ndef most_common_word(text: str) -> str:\n    words = text.lower().split()\n    return Counter(words).most_common(1)[0][0]\n",
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])\n",
    "def merge_sorted(a, b):\n    \"\"\"Merge two pre-sorted lists into a sorted list.\"\"\"\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: out.append(a[i]); i += 1\n        else: out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out\n",
    "import heapq\n\n\ndef k_largest(nums, k):\n    return heapq.nlargest(k, nums)\n",
    "class Stack:\n    def __init__(self): self._items = []\n    def push(self, x): self._items.append(x)\n    def pop(self): return self._items.pop() if self._items else None\n    def peek(self): return self._items[-1] if self._items else None\n    def __len__(self): return len(self._items)\n",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        if arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1\n",
    "import json, sys\n\n\ndef sum_field(path: str, field: str) -> float:\n    with open(path) as f:\n        rows = json.load(f)\n    return sum(r.get(field, 0) for r in rows)\n",
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
    "In linear algebra, the determinant is a scalar value that is a function of the entries of a square matrix. It allows characterizing some properties of the matrix and the linear map represented by it.",
]


def build_calib(tokenizer, total_tokens: int, chunk_tokens: int, device: str, calib_file: str | None = None) -> list[torch.Tensor]:
    """
    Read calibration text and split into chunks of `chunk_tokens` each, totaling
    at least `total_tokens`. Each chunk is sampled from a different offset in
    the corpus to maximize diversity. Returns a list of (1, chunk_tokens)
    tensors on device.
    """
    if calib_file:
        with open(calib_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        log(f"calib source: {calib_file} ({len(text)} chars)")
    else:
        text = "\n\n".join(CALIB_TEXTS)
        log("calib source: embedded CALIB_TEXTS (narrow — prefer --calib-file)")
    # Tokenize whole corpus once
    full_ids = tokenizer(text, return_tensors="pt").input_ids[0]  # (N,)
    if full_ids.shape[0] < chunk_tokens:
        # Repeat until at least one chunk worth
        reps = (chunk_tokens + full_ids.shape[0] - 1) // full_ids.shape[0]
        full_ids = full_ids.repeat(reps + 1)
    n_chunks = max(1, (total_tokens + chunk_tokens - 1) // chunk_tokens)
    # Stride across the corpus
    available = full_ids.shape[0] - chunk_tokens
    if available <= 0 or n_chunks == 1:
        offsets = [0]
        n_chunks = 1
    else:
        step = max(1, available // n_chunks)
        offsets = [i * step for i in range(n_chunks)]
    chunks = []
    for off in offsets:
        ids = full_ids[off : off + chunk_tokens].unsqueeze(0).to(device)
        chunks.append(ids)
    total = sum(c.shape[1] for c in chunks)
    log(f"calib: {n_chunks} chunks × {chunk_tokens} tokens = {total} total tokens")
    return chunks


# ---------- model layout helpers ----------

def get_layers(model):
    """Return decoder layers list regardless of nesting."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "language_model"):
        return model.language_model.layers
    raise RuntimeError("could not locate decoder layers")


def text_config(model):
    cfg = model.config
    if hasattr(cfg, "text_config"):
        return cfg.text_config
    return cfg


def layer_kind(model, idx: int) -> str:
    """'full_attention' or 'linear_attention'."""
    return text_config(model).layer_types[idx]


def is_full(layer) -> bool:
    return hasattr(layer, "self_attn")


def is_linear(layer) -> bool:
    return hasattr(layer, "linear_attn")


# ---------- phase 0: capture targets ----------

def phase0_capture(model, calib_chunks):
    """
    Forward the un-modified model on each calibration chunk, capturing per
    attention-bearing layer the input to o_proj / out_proj. Captures from
    different chunks are concatenated along the time dim.
    """
    layers = get_layers(model)
    chunk_targets: dict[int, list[torch.Tensor]] = {}
    handles = []
    for L, layer in enumerate(layers):
        if is_full(layer):
            proj = layer.self_attn.o_proj
        elif is_linear(layer):
            proj = layer.linear_attn.out_proj
        else:
            continue
        handles.append(proj.register_forward_pre_hook(
            lambda m, inp, idx=L: chunk_targets.setdefault(idx, []).append(inp[0].detach().to(torch.float32).cpu())
        ))

    log(f"phase0: registered {len(handles)} pre-hooks; running forward on {len(calib_chunks)} chunks...")
    with torch.no_grad():
        for i, chunk in enumerate(calib_chunks):
            model(chunk)
    for h in handles:
        h.remove()
    # Concatenate chunks per layer along the time axis
    targets = {L: torch.cat(parts, dim=1) for L, parts in chunk_targets.items()}
    log(f"phase0: captured {len(targets)} layer targets, total time={next(iter(targets.values())).shape[1]}")
    return targets


# ---------- phase 1: per-head importance via Michel-style gradient ----------

class FullAttnHeadGate(nn.Module):
    """
    Wraps Qwen3_5Attention to insert a per-Q-head scaling factor α.
    α multiplies the gated, concatenated head output before o_proj. After
    fwd+bwd, |grad α| at α=1 ranks Q-head importance.
    """
    def __init__(self, attn, num_q_heads: int, head_dim: int):
        super().__init__()
        self.attn = attn
        dev = attn.o_proj.weight.device
        self.alpha = nn.Parameter(torch.ones(num_q_heads, device=dev))
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self._orig_o_proj_forward = attn.o_proj.forward

        def patched_o_proj(x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, n_q*head_dim]. Reshape to [B, T, n_q, head_dim], scale, flatten.
            B, T, _ = x.shape
            x_h = x.view(B, T, self.num_q_heads, self.head_dim)
            x_h = x_h * self.alpha.view(1, 1, -1, 1).to(x_h.dtype)
            x = x_h.view(B, T, -1)
            return self._orig_o_proj_forward(x)

        attn.o_proj.forward = patched_o_proj

    def restore(self):
        self.attn.o_proj.forward = self._orig_o_proj_forward


class LinearAttnHeadGate(nn.Module):
    """
    Wraps Qwen3_5GatedDeltaNet to insert a per-V-head scaling factor α.
    α scales the value-dim signal feeding out_proj (post-norm, pre-projection).
    """
    def __init__(self, lin_attn, num_v_heads: int, head_v_dim: int):
        super().__init__()
        self.lin_attn = lin_attn
        dev = lin_attn.out_proj.weight.device
        self.alpha = nn.Parameter(torch.ones(num_v_heads, device=dev))
        self.num_v_heads = num_v_heads
        self.head_v_dim = head_v_dim
        self._orig_out_proj_forward = lin_attn.out_proj.forward

        def patched_out_proj(x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, value_dim] = [B, T, n_v*head_v_dim]
            B, T, _ = x.shape
            x_h = x.view(B, T, self.num_v_heads, self.head_v_dim)
            x_h = x_h * self.alpha.view(1, 1, -1, 1).to(x_h.dtype)
            x = x_h.view(B, T, -1)
            return self._orig_out_proj_forward(x)

        lin_attn.out_proj.forward = patched_out_proj

    def restore(self):
        self.lin_attn.out_proj.forward = self._orig_out_proj_forward


def phase1_importance(model, calib_chunks):
    """
    Insert α per attention-bearing layer, run fwd+bwd on each calibration chunk,
    accumulating |grad α| per head per layer.
    """
    cfg = text_config(model)
    layers = get_layers(model)
    n_q_full = cfg.num_attention_heads
    head_dim_full = cfg.head_dim
    n_v_lin = cfg.linear_num_value_heads
    head_v_dim_lin = cfg.linear_value_head_dim

    gates = {}
    for L, layer in enumerate(layers):
        if is_full(layer):
            g = FullAttnHeadGate(layer.self_attn, n_q_full, head_dim_full)
            g.alpha.requires_grad_(True)
            gates[L] = g
        elif is_linear(layer):
            g = LinearAttnHeadGate(layer.linear_attn, n_v_lin, head_v_dim_lin)
            g.alpha.requires_grad_(True)
            gates[L] = g

    for p in model.parameters():
        p.requires_grad_(False)
    for g in gates.values():
        g.alpha.requires_grad_(True)

    log(f"phase1: gates installed on {len(gates)} layers; running fwd+bwd on {len(calib_chunks)} chunks...")
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            log("phase1: gradient checkpointing enabled")
        except Exception as e:
            log(f"phase1: grad ckpt enable failed ({e}); continuing without")

    total_loss = 0.0
    for i, chunk in enumerate(calib_chunks):
        # Zero α grads before each chunk so we can accumulate |grad| per chunk
        for g in gates.values():
            if g.alpha.grad is not None:
                g.alpha.grad.zero_()
        out = model(chunk)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        total_loss += loss.item()
        # Accumulate |grad| into a buffer attached to the gate
        for g in gates.values():
            cur = g.alpha.grad.detach().abs().to(torch.float32)
            if not hasattr(g, "_imp_acc"):
                g._imp_acc = cur.cpu().clone()
            else:
                g._imp_acc += cur.cpu()
        log(f"phase1: chunk {i+1}/{len(calib_chunks)} CE = {loss.item():.4f}")
    log(f"phase1: avg CE = {total_loss / len(calib_chunks):.4f}")

    importance = {}
    for L, g in gates.items():
        importance[L] = g._imp_acc
        g.restore()
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    model.eval()
    log(f"phase1: importance computed for {len(importance)} layers")
    return importance


# ---------- phase 2: per-layer prune + lstsq refit ----------

@torch.no_grad()
def prune_full_attention_inplace(layer, keep_q_heads, target_olin_input, model, calib_ids, device):
    """
    Mask-prune full_attention layer:
      - Zero out q_proj rows for dropped Q heads (both q-half and gate-half).
      - Optionally drop matching KV heads if a whole GQA group is dropped.
      - lstsq-refit o_proj cols for kept Q heads on `target_olin_input`
        (the original o_proj input that we want to reproduce at this layer).

    target_olin_input: [B, T, n_q * head_dim] — tensor on cpu fp32.
    """
    cfg = text_config(model)
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    group_size = n_q // n_kv  # 16/4 = 4

    keep_q = sorted(int(i) for i in keep_q_heads)
    drop_q = [i for i in range(n_q) if i not in set(keep_q)]
    log(f"  full L?: keep_q={keep_q}, drop_q={drop_q}")

    sa = layer.self_attn

    # Zero q_proj rows for dropped heads. q_proj is shape (n_q*head_dim*2, hidden).
    # The 2× factor: each head gets [q_h | gate_h]. Need to zero both halves.
    # Layout per head h of size 2*head_dim: rows [h*2*head_dim : (h+1)*2*head_dim]
    qw = sa.q_proj.weight.data  # (n_q*head_dim*2, hidden)
    for h in drop_q:
        qw[h * 2 * head_dim : (h + 1) * 2 * head_dim, :] = 0
    # Optional KV pruning: drop whole groups
    keep_kv_groups = sorted({h // group_size for h in keep_q})
    drop_kv_groups = [g for g in range(n_kv) if g not in set(keep_kv_groups)]
    if drop_kv_groups:
        log(f"  full L?: drop KV groups {drop_kv_groups}")
        for g in drop_kv_groups:
            sa.k_proj.weight.data[g * head_dim : (g + 1) * head_dim, :] = 0
            sa.v_proj.weight.data[g * head_dim : (g + 1) * head_dim, :] = 0

    # lstsq refit on o_proj
    # o_proj: y = W_O @ x where x = [B, T, n_q * head_dim], y = [B, T, hidden].
    # We want to find W_O' such that W_O' @ x' ≈ y_target, where x' has dropped-head
    # cols zeroed (since we zeroed q_proj rows above, those head cols of x are now 0).
    # But target was captured pre-prune. The refit uses the ACTUAL post-prune x,
    # which we get by re-running the layer's forward up to o_proj.
    #
    # Approach: re-forward the model with the zeroed q_proj, hook o_proj's input
    # at this layer, then solve lstsq(x_actual, y_target) for new W_O.
    # We do this in the caller (one re-fwd per layer is wasteful; instead we
    # re-forward once at the start of phase 2 after each prune).
    # Here we just receive x_actual via the caller after re-fwd.
    return  # caller does the lstsq with re-captured x


@torch.no_grad()
def prune_linear_attention_inplace(layer, keep_v_heads, model):
    """
    Mask-prune linear_attention (gated DeltaNet):
      - Zero rows of in_proj_qkv corresponding to dropped V heads (value portion only).
      - Zero rows of in_proj_z, in_proj_b, in_proj_a for dropped V heads.
      - Zero conv1d depthwise channels for the value rows.
      - Zero entries of A_log, dt_bias for dropped V heads.
      - K-head dropping enabled when whole groups are pruned.
    The lstsq refit of out_proj is done by the caller after re-forward.
    """
    cfg = text_config(model)
    n_v = cfg.linear_num_value_heads
    n_k = cfg.linear_num_key_heads
    head_v = cfg.linear_value_head_dim
    head_k = cfg.linear_key_head_dim
    key_dim = n_k * head_k

    keep_v = sorted(int(i) for i in keep_v_heads)
    drop_v = [i for i in range(n_v) if i not in set(keep_v)]
    log(f"  lin L?: keep_v={keep_v[:8]}{'...' if len(keep_v)>8 else ''}, drop_v={drop_v}")

    la = layer.linear_attn

    # in_proj_qkv: (2*key_dim + value_dim, hidden). Layout: [Q | K | V] along output.
    # Value section starts at 2*key_dim and goes to 2*key_dim + value_dim.
    iqkv = la.in_proj_qkv.weight.data
    v_off = 2 * key_dim
    for h in drop_v:
        iqkv[v_off + h * head_v : v_off + (h + 1) * head_v, :] = 0
    # in_proj_z: (value_dim, hidden) — same layout as the V section
    iz = la.in_proj_z.weight.data
    for h in drop_v:
        iz[h * head_v : (h + 1) * head_v, :] = 0
    # in_proj_b, in_proj_a: (n_v, hidden)
    for h in drop_v:
        la.in_proj_b.weight.data[h, :] = 0
        la.in_proj_a.weight.data[h, :] = 0
    # conv1d: depthwise, (conv_dim, 1, kernel). conv_dim = 2*key_dim + value_dim
    cw = la.conv1d.weight.data  # (conv_dim, 1, kernel)
    for h in drop_v:
        cw[v_off + h * head_v : v_off + (h + 1) * head_v, :, :] = 0
    # A_log, dt_bias: (n_v,)
    for h in drop_v:
        la.A_log.data[h] = 0  # log(1) = 0 means decay rate 1 — but unused since v rows are zeroed
        la.dt_bias.data[h] = 0

    # K-head group pruning if whole group dropped
    v_per_k = n_v // n_k  # 32/16 = 2
    # Each k-head h_k corresponds to v-heads [h_k*v_per_k, (h_k+1)*v_per_k]
    drop_k_groups = []
    for h_k in range(n_k):
        v_in_group = list(range(h_k * v_per_k, (h_k + 1) * v_per_k))
        if all(v in drop_v for v in v_in_group):
            drop_k_groups.append(h_k)
    if drop_k_groups:
        log(f"  lin L?: drop K groups {drop_k_groups}")
        for h_k in drop_k_groups:
            # K section in iqkv: [key_dim : 2*key_dim], offset 0 is Q
            for kk in range(2):  # Q and K share dims layout [Q | K]
                base = kk * key_dim
                iqkv[base + h_k * head_k : base + (h_k + 1) * head_k, :] = 0
                cw[base + h_k * head_k : base + (h_k + 1) * head_k, :, :] = 0
    return


@torch.no_grad()
def lstsq_refit_o_proj(proj: nn.Linear, x_actual: torch.Tensor, y_target: torch.Tensor,
                       keep_cols: list[int] | None = None, ridge_rel: float = 1e-4):
    """
    Refit only the kept input columns of `proj.weight`. Dropped-head columns
    (passed via keep_cols complement) are zeroed in the new weight, so they
    contribute nothing to the projection.

    x_actual: [B, T, in_features]  (post-prune o_proj input — dropped-head cols are 0)
    y_target: [B, T, out_features] (pre-prune o_proj output target = W_orig @ x_orig)
    keep_cols: list of input-feature indices to fit. If None, fit the full input.
    ridge_rel: ridge as a fraction of mean diagonal of XtX (scale-invariant).

    Math: solve W_keep ∈ R^{out × |keep|} minimizing
        || X[:, keep] W_keep^T - Y ||^2 + λ ||W_keep||^2
    via (XtX_keep + λI) W_keep^T = XtY_keep.
    """
    assert x_actual.dim() == 3 and y_target.dim() == 3
    in_f = proj.in_features
    out_f = proj.out_features
    X = x_actual.reshape(-1, in_f).to(torch.float32)
    Y = y_target.reshape(-1, out_f).to(torch.float32)

    if keep_cols is None:
        keep_idx = torch.arange(in_f)
    else:
        keep_idx = torch.tensor(sorted(keep_cols), dtype=torch.long)

    Xk = X[:, keep_idx]                                # (T, k)
    XtX = Xk.t() @ Xk                                  # (k, k)
    XtY = Xk.t() @ Y                                   # (k, out)
    diag_mean = XtX.diagonal().mean().item()
    lam = max(diag_mean * ridge_rel, 1e-6)
    XtX += lam * torch.eye(XtX.shape[0], dtype=XtX.dtype, device=XtX.device)
    # cholesky for symmetric PD; falls back to solve if not PD
    try:
        L = torch.linalg.cholesky(XtX)
        Wk_t = torch.cholesky_solve(XtY, L)            # (k, out)
    except Exception:
        Wk_t = torch.linalg.solve(XtX, XtY)
    Wk = Wk_t.t().contiguous()                         # (out, k)

    # Build new weight: zeros, then scatter Wk into kept cols
    W_new = torch.zeros(out_f, in_f, dtype=torch.float32)
    W_new[:, keep_idx] = Wk

    # Residual
    Y_hat = Xk @ Wk_t
    resid = (Y_hat - Y).pow(2).mean().sqrt().item()
    target_norm = Y.pow(2).mean().sqrt().item()
    rel = resid / max(target_norm, 1e-9)
    log(f"  lstsq refit: kept={len(keep_idx)}/{in_f}, lam={lam:.3e}, rel_resid={rel:.4f}, target_rms={target_norm:.4f}")
    proj.weight.data.copy_(W_new.to(proj.weight.dtype).to(proj.weight.device))


# ---------- main ----------

@torch.no_grad()
def capture_proj_input_at_layer(model, calib_chunks, layer_idx: int) -> torch.Tensor:
    """
    Forward the (possibly modified) model on each chunk, capture and concatenate
    the o_proj/out_proj input at one specific layer.
    """
    layer = get_layers(model)[layer_idx]
    if is_full(layer):
        proj = layer.self_attn.o_proj
    elif is_linear(layer):
        proj = layer.linear_attn.out_proj
    else:
        raise RuntimeError("layer has no attention")
    parts = []
    def hook(m, inp):
        parts.append(inp[0].detach().to(torch.float32).cpu())
    h = proj.register_forward_pre_hook(hook)
    try:
        for chunk in calib_chunks:
            model(chunk)
    finally:
        h.remove()
    return torch.cat(parts, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prune-full-frac", type=float, default=0.0,
                    help="fraction of Q heads to drop in full_attention layers")
    ap.add_argument("--prune-linear-frac", type=float, default=0.0,
                    help="fraction of V heads to drop in linear_attention layers")
    ap.add_argument("--calib-tokens", type=int, default=8192,
                    help="Total calibration tokens across all chunks")
    ap.add_argument("--chunk-tokens", type=int, default=2048,
                    help="Per-chunk size to fit in GPU memory")
    ap.add_argument("--calib-file", default=None,
                    help="Path to a plain-text corpus for calibration (preferred over embedded CALIB_TEXTS)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true",
                    help="only prune layer 3 (full) and/or layer 0 (linear) — sanity check")
    args = ap.parse_args()

    log(f"loading {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model = model.to(args.device)

    # Disable use_cache: hooks easier and we never call generate during pruning
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.eval()
    layers = get_layers(model)
    log(f"model loaded: {len(layers)} layers; cfg full_heads={text_config(model).num_attention_heads} lin_v_heads={text_config(model).linear_num_value_heads}")

    calib_chunks = build_calib(tok, args.calib_tokens, args.chunk_tokens, args.device, args.calib_file)

    # Phase 0: capture original o_proj/out_proj INPUT per layer (cpu fp32)
    targets = phase0_capture(model, calib_chunks)
    # Convert each target to o_proj OUTPUT (the real target for healing)
    # by running it through the ORIGINAL projection weights, in fp32.
    log("phase0: computing pre-prune o_proj OUTPUTs as healing targets")
    target_outputs = {}
    for L, x in targets.items():
        layer = layers[L]
        proj = layer.self_attn.o_proj if is_full(layer) else layer.linear_attn.out_proj
        W = proj.weight.data.to(torch.float32).cpu()  # (out, in)
        # x: [B, T, in], y = x @ W^T: [B, T, out]
        y = x @ W.t()
        target_outputs[L] = y
    log(f"phase0: {len(target_outputs)} target outputs prepared")

    # Phase 1: importance
    importance = phase1_importance(model, calib_chunks)

    # Phase 2: per-layer prune + lstsq refit, in forward order
    cfg = text_config(model)
    n_q_full = cfg.num_attention_heads
    n_v_lin = cfg.linear_num_value_heads

    smoke_layers = set()
    if args.smoke:
        # Pick first full layer and first linear layer
        for L, layer in enumerate(layers):
            if is_full(layer) and not any(is_full(layers[Li]) for Li in smoke_layers):
                smoke_layers.add(L)
            if is_linear(layer) and not any(is_linear(layers[Li]) for Li in smoke_layers):
                smoke_layers.add(L)
            if len(smoke_layers) >= 2:
                break
        log(f"SMOKE mode: only pruning layers {sorted(smoke_layers)}")

    n_drop_full = int(round(n_q_full * args.prune_full_frac))
    n_drop_lin = int(round(n_v_lin * args.prune_linear_frac))
    log(f"phase2: drop {n_drop_full}/{n_q_full} full Q-heads per layer, {n_drop_lin}/{n_v_lin} linear V-heads per layer")

    pruned_layers = []
    for L, layer in enumerate(layers):
        if args.smoke and L not in smoke_layers:
            continue
        if is_full(layer) and n_drop_full > 0:
            imp = importance[L]
            keep_idx = imp.argsort(descending=True)[: n_q_full - n_drop_full].tolist()
            log(f"layer {L} [full]: importance min={imp.min():.3e} max={imp.max():.3e} mean={imp.mean():.3e}")
            prune_full_attention_inplace(layer, keep_idx, None, model, None, args.device)
            # o_proj input layout: [n_q × head_dim], head h occupies [h*head_dim : (h+1)*head_dim]
            head_dim_full = cfg.head_dim
            keep_cols = []
            for h in keep_idx:
                keep_cols.extend(range(h * head_dim_full, (h + 1) * head_dim_full))
            x_actual = capture_proj_input_at_layer(model, calib_chunks, L)
            y_target = target_outputs[L]
            lstsq_refit_o_proj(layer.self_attn.o_proj, x_actual, y_target,
                               keep_cols=keep_cols, ridge_rel=args.ridge)
            pruned_layers.append((L, "full"))
        elif is_linear(layer) and n_drop_lin > 0:
            imp = importance[L]
            keep_idx = imp.argsort(descending=True)[: n_v_lin - n_drop_lin].tolist()
            log(f"layer {L} [lin]:  importance min={imp.min():.3e} max={imp.max():.3e} mean={imp.mean():.3e}")
            prune_linear_attention_inplace(layer, keep_idx, model)
            # out_proj input layout: [n_v × head_v_dim], head h at [h*head_v : (h+1)*head_v]
            head_v_dim = cfg.linear_value_head_dim
            keep_cols = []
            for h in keep_idx:
                keep_cols.extend(range(h * head_v_dim, (h + 1) * head_v_dim))
            x_actual = capture_proj_input_at_layer(model, calib_chunks, L)
            y_target = target_outputs[L]
            lstsq_refit_o_proj(layer.linear_attn.out_proj, x_actual, y_target,
                               keep_cols=keep_cols, ridge_rel=args.ridge)
            pruned_layers.append((L, "lin"))

    # Final CE check (averaged over chunks)
    ce_total = 0.0
    with torch.no_grad():
        for chunk in calib_chunks:
            out = model(chunk)
            logits = out.logits
            shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
            shift_labels = chunk[:, 1:].contiguous()
            ce_total += F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
    ce_post = ce_total / len(calib_chunks)
    log(f"final CE on calibration: {ce_post:.4f}")

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"saving to {out_dir}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # Drop a manifest of what we did
    manifest = {
        "source": str(args.model_path),
        "prune_full_frac": args.prune_full_frac,
        "prune_linear_frac": args.prune_linear_frac,
        "n_drop_full": n_drop_full,
        "n_drop_linear": n_drop_lin,
        "calib_tokens": args.calib_tokens,
        "ridge": args.ridge,
        "smoke": args.smoke,
        "pruned_layers": pruned_layers,
        "final_ce": ce_post,
    }
    (out_dir / "prune_manifest.json").write_text(json.dumps(manifest, indent=2))
    log("done")


if __name__ == "__main__":
    main()
