#!/usr/bin/env python3
"""
prune_local_heal.py — Gemma 4 31B-it attention head pruning with
                      backprop-free local healing (W_O lstsq refit).

Targets the dense Gemma 4 31B-it text architecture:
  - 60 layers: 50 sliding_attention + 10 full_attention (every 6th, last incl)
  - 32 query heads / 16 KV heads (sliding, head_dim=256, GQA 2:1)
  - 32 query heads /  4 KV heads (full,   head_dim=512=global_head_dim, k=v shared)
  - No linear/recurrent attention anywhere (all standard softmax)

Pruning lever:
  --prune-frac : fraction of Q heads to drop, applied uniformly to every layer
                 (12.5% = 4/32 heads/layer; 25% = 8/32 heads/layer)

At 12.5% we keep ALL KV heads to minimize secondary perturbation. KV-group
dropping is gated behind --drop-kv-groups (off by default).

Phases:
  0. Capture per-layer o_proj-input on a calibration set; compute pre-prune
     o_proj output as the healing target via the original W_O (fp32).
  1. Score per-Q-head importance via Michel-style gradient on per-head α=1.
  2. For each layer in forward order, on the *current* (already-modified)
     model:
       a. Zero q_proj rows for dropped Q heads (per-layer head_dim)
       b. Re-capture pre-o_proj input (post-prune)
       c. lstsq-refit kept-Q-head columns of o_proj so post-proj output
          matches the pre-prune target.

The model stays full-size on disk (mask-style); inference compute is reduced
when a downstream tool physically resizes the matrices. Mask pruning is what
lstsq refits, and it preserves output quality identically to a physical
resize at inference time.

Memory plan for solidPC RTX 3090 24GB:
  device_map="auto", max_memory={"0":"22GiB","cpu":"200GiB"}
  gradient_checkpointing for phase 1 backward pass
  Per-chunk size ~1024 tokens, 8 chunks total
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from accelerate.utils import align_module_device
    _HAS_ALIGN = True
except ImportError:
    _HAS_ALIGN = False


def _maybe_align(module):
    """
    Context manager that materializes a module's weights to a real device
    when accelerate has them offloaded to meta. No-op when align is unavailable
    or the module already has real tensors.
    """
    if _HAS_ALIGN:
        try:
            return align_module_device(module)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- calibration ----------

def build_calib(tokenizer, total_tokens: int, chunk_tokens: int, calib_file: str | None = None) -> list[torch.Tensor]:
    """
    Read a plain-text corpus and split into chunks of `chunk_tokens` each,
    striding across the corpus to maximize diversity. Returns a list of
    (1, chunk_tokens) CPU long tensors. Caller moves to device per chunk.
    """
    if not calib_file:
        raise ValueError("--calib-file is mandatory for 31B (corpus too large to embed)")
    with open(calib_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    log(f"calib source: {calib_file} ({len(text)} chars)")
    full_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if full_ids.shape[0] < chunk_tokens:
        reps = (chunk_tokens + full_ids.shape[0] - 1) // full_ids.shape[0]
        full_ids = full_ids.repeat(reps + 1)
    n_chunks = max(1, (total_tokens + chunk_tokens - 1) // chunk_tokens)
    available = full_ids.shape[0] - chunk_tokens
    if available <= 0 or n_chunks == 1:
        offsets = [0]
        n_chunks = 1
    else:
        step = max(1, available // n_chunks)
        offsets = [i * step for i in range(n_chunks)]
    chunks = [full_ids[off : off + chunk_tokens].unsqueeze(0) for off in offsets]
    total = sum(c.shape[1] for c in chunks)
    log(f"calib: {n_chunks} chunks × {chunk_tokens} tokens = {total} total tokens")
    return chunks


# ---------- model layout helpers ----------

def get_layers(model):
    """Return decoder layers list regardless of nesting (Gemma4: model.language_model.layers)."""
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("could not locate decoder layers")


def text_config(model):
    cfg = model.config
    if hasattr(cfg, "text_config"):
        return cfg.text_config
    return cfg


def has_self_attn(layer) -> bool:
    return hasattr(layer, "self_attn") and layer.self_attn is not None


def layer_attn_geom(model, idx: int) -> tuple[int, int, int, str]:
    """
    Return (head_dim, num_q_heads, num_kv_heads, layer_type) for the given
    decoder layer, accounting for Gemma4's per-layer geometry:
      - sliding_attention: head_dim=cfg.head_dim, kv=cfg.num_key_value_heads
      - full_attention   : head_dim=cfg.global_head_dim, kv=cfg.num_global_key_value_heads (k=v if attention_k_eq_v)
    """
    cfg = text_config(model)
    lt = cfg.layer_types[idx]
    n_q = cfg.num_attention_heads
    if lt == "full_attention":
        head_dim = cfg.global_head_dim if cfg.global_head_dim else cfg.head_dim
        n_kv = cfg.num_global_key_value_heads if cfg.attention_k_eq_v else cfg.num_key_value_heads
    else:  # sliding_attention
        head_dim = cfg.head_dim
        n_kv = cfg.num_key_value_heads
    return head_dim, n_q, n_kv, lt


# ---------- checkpoint helpers (resumable Phase 0' / Phase 2 / pre-canary) ----------
#
# Three save points, all opt-in via --checkpoint-dir:
#   (A) phase0/  — per-layer o_proj output + decoder residual + (opt) mlp targets
#                  captured during phase0_capture(). Disk: ~340 MB/layer × 60 ≈ 20 GB.
#                  Enabled only with --checkpoint-phase0 (large disk).
#   (B) phase2/  — per-layer self_attn state_dict + heal stats + keep indices
#                  after each layer's prune_q + heal completes. Disk: ~300 MB/layer
#                  bf16 × 60 ≈ 18 GB. Enabled by default when --checkpoint-dir is set.
#   (C) staged/  — full BF16 model.save_pretrained() AFTER phase3b reshape but BEFORE
#                  the AR canary. Disk: ~62 GB. Enabled with --save-before-canary.
#                  Resume detects this and skips straight to phase 2.5; saves the
#                  ~1-2 h of phase 0+2 work when the canary itself stalls/dies.
#
# Manifest (manifest.json) carries a fingerprint hashed from the run's load-bearing
# args + calib-file contents + drop selection. Mismatch ⇒ refuse resume (loud fail,
# never silent corruption). Atomic writes: tmp file → os.replace.

_CHECKPOINT_VERSION = "1.0"


def _ckpt_fingerprint(args) -> str:
    """SHA-256 hash of the load-bearing run config — used to reject stale resume dirs.

    Inputs hashed (any change invalidates the checkpoint):
      - model path string + the load-bearing CLI args below
      - SHA-256 of the calib file contents (--calib-file)
      - SHA-256 of the importance cache contents (--imp-cache) when present —
        this makes the head selection deterministic, so phase-2 checkpoints
        are reusable across replays of the SAME published variant.

    NOT hashed:
      - paths to --output / --checkpoint-dir / --canary-baseline-cache
      - GPU mem budgets (placement-only knobs that don't change weights)
      - canary thresholds (gate config; doesn't affect saved weights)
    """
    h = hashlib.sha256()
    h.update(f"v={_CHECKPOINT_VERSION}\n".encode())
    for k in ("model_path", "prune_frac", "ridge", "chunk_tokens", "calib_tokens",
              "heal", "prune_mode", "phase1_mode", "phase1_nf4_chunk_tokens",
              "ffn_prune_frac", "ffn_block_size", "ffn_heal"):
        v = getattr(args, k, None)
        h.update(f"{k}={v}\n".encode())
    for name, path_attr in (("calib", "calib_file"), ("imp", "imp_cache")):
        p_str = getattr(args, path_attr, None)
        if p_str and Path(p_str).exists():
            with open(p_str, "rb") as f:
                h.update(f"{name}:".encode())
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        else:
            h.update(f"{name}:none\n".encode())
    return h.hexdigest()[:16]


def _read_manifest(ckpt_dir: Path) -> dict | None:
    mf = ckpt_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text())
    except Exception:
        return None


def _write_manifest(ckpt_dir: Path, manifest: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(ckpt_dir / "manifest.json")


def _atomic_torch_save(obj, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(dst)


def _phase0_layer_path(ckpt_dir: Path, L: int) -> Path:
    return ckpt_dir / "phase0" / f"layer_{L:03d}.pt"


def _phase2_layer_path(ckpt_dir: Path, L: int) -> Path:
    return ckpt_dir / "phase2" / f"layer_{L:03d}.pt"


def _staged_dir(ckpt_dir: Path) -> Path:
    return ckpt_dir / "staged"


def _staged_complete(ckpt_dir: Path) -> bool:
    """True iff staged/ has a complete safetensors model (config + ≥1 shard)."""
    s = _staged_dir(ckpt_dir)
    if not (s / "config.json").exists():
        return False
    if not any(s.glob("*.safetensors")):
        return False
    # save_pretrained writes a sentinel index for sharded models; for single-shard
    # models, the presence of model.safetensors + config.json is enough.
    return True


# ---------- phase 0: capture targets ----------

def phase0_capture(model, calib_chunks, capture_mlp: bool = False,
                    ckpt_dir: Path | None = None, ckpt_phase0: bool = False,
                    resume_layers: set[int] | None = None):
    """
    Forward the un-modified model, capturing per-layer artifacts:
      - o_proj OUTPUT (decoder layer's attention contribution)   → phase 2 attn healing target
      - decoder layer OUTPUT (residual-stream after the layer)   → phase 1a L2 target
      - down_proj OUTPUT (MLP block contribution) — only when capture_mlp=True;
        used as phase 2 FFN healing target (T7.6).

    Outputs are immediately cast to fp32 and moved to CPU.
    """
    # Layer-import lazy because has_mlp lives lower in the file.
    layers = get_layers(model)
    chunk_o_outputs: dict[int, list[torch.Tensor]] = {}
    chunk_layer_outputs: dict[int, list[torch.Tensor]] = {}
    chunk_mlp_outputs: dict[int, list[torch.Tensor]] = {}
    handles = []
    for L, layer in enumerate(layers):
        if has_self_attn(layer):
            proj = layer.self_attn.o_proj
            handles.append(proj.register_forward_hook(
                lambda m, inp, out, idx=L: chunk_o_outputs.setdefault(idx, []).append(
                    out.detach().to(torch.float32).cpu()
                )
            ))
        if capture_mlp and has_mlp(layer):
            dp = layer.mlp.down_proj
            handles.append(dp.register_forward_hook(
                lambda m, inp, out, idx=L: chunk_mlp_outputs.setdefault(idx, []).append(
                    out.detach().to(torch.float32).cpu()
                )
            ))
        # Decoder layer hook — capture residual-stream output
        def _layer_hook(m, inp, out, idx=L):
            h = out[0] if isinstance(out, tuple) else out
            chunk_layer_outputs.setdefault(idx, []).append(
                h.detach().to(torch.float32).cpu()
            )
        handles.append(layer.register_forward_hook(_layer_hook))

    log(f"phase0: registered {len(handles)} forward hooks; running fwd on {len(calib_chunks)} chunks...")
    with torch.no_grad():
        for i, chunk in enumerate(calib_chunks):
            _ = model(chunk.to(next(model.parameters()).device))
            log(f"phase0: chunk {i+1}/{len(calib_chunks)} captured")
    for h in handles:
        h.remove()
    o_targets = {L: torch.cat(parts, dim=1) for L, parts in chunk_o_outputs.items()}
    h_targets = {L: torch.cat(parts, dim=1) for L, parts in chunk_layer_outputs.items()}
    mlp_targets = {L: torch.cat(parts, dim=1) for L, parts in chunk_mlp_outputs.items()} if capture_mlp else {}
    extra = f" + {len(mlp_targets)} down_proj targets" if capture_mlp else ""
    log(f"phase0: captured {len(o_targets)} o_proj targets and {len(h_targets)} residual-stream targets{extra} (CPU fp32)")

    # Merge in any resumed-from-disk layers — overwrite the fresh capture with
    # the saved values so we use one authoritative copy.
    if resume_layers and ckpt_dir is not None:
        loaded = 0
        for L in resume_layers:
            p = _phase0_layer_path(ckpt_dir, L)
            if not p.exists():
                continue
            d = torch.load(p, map_location="cpu", weights_only=True)
            if d.get("o") is not None:
                o_targets[L] = d["o"]
            if d.get("h") is not None:
                h_targets[L] = d["h"]
            if capture_mlp and d.get("mlp") is not None:
                mlp_targets[L] = d["mlp"]
            loaded += 1
        if loaded:
            log(f"phase0: resumed {loaded} layer(s) from {ckpt_dir / 'phase0'}")

    # Save per-layer captures (opt-in; ~340 MB/layer × 60 ≈ 20 GB on 31B).
    if ckpt_phase0 and ckpt_dir is not None:
        for L in o_targets:
            obj = {"o": o_targets[L], "h": h_targets.get(L)}
            if capture_mlp:
                obj["mlp"] = mlp_targets.get(L)
            _atomic_torch_save(obj, _phase0_layer_path(ckpt_dir, L))
        log(f"phase0: wrote {len(o_targets)} per-layer checkpoints to {ckpt_dir / 'phase0'}")

    return o_targets, h_targets, mlp_targets


# ---------- phase 1: per-head importance via Michel-style gradient ----------

class HeadGate(nn.Module):
    """
    Wraps a Gemma4TextAttention to insert a per-Q-head scaling factor α.
    α multiplies the attention output (head dim) before o_proj. After
    fwd+bwd, |grad α| at α=1 ranks Q-head importance.
    """
    def __init__(self, attn, num_q_heads: int, head_dim: int):
        super().__init__()
        self.attn = attn
        # α MUST be a real tensor on a real device — never `meta`.
        # accelerate offload may put o_proj.weight on `meta` so we can't
        # derive α's device from that. CUDA if available (forward shuttles
        # offloaded layers there); else CPU.
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = nn.Parameter(torch.ones(num_q_heads, device=dev, dtype=torch.float32))
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self._orig_o_proj_forward = attn.o_proj.forward

        def patched_o_proj(x: torch.Tensor) -> torch.Tensor:
            # x: [..., num_q_heads * head_dim] → reshape, scale, flatten
            *prefix, last = x.shape
            assert last == self.num_q_heads * self.head_dim, (
                f"unexpected o_proj input shape {x.shape}, expected last dim "
                f"{self.num_q_heads * self.head_dim}"
            )
            x_h = x.view(*prefix, self.num_q_heads, self.head_dim)
            alpha = self.alpha.view(*([1] * len(prefix)), -1, 1).to(
                device=x_h.device, dtype=x_h.dtype
            )
            x_h = x_h * alpha
            x = x_h.view(*prefix, last)
            return self._orig_o_proj_forward(x)

        attn.o_proj.forward = patched_o_proj

    def restore(self):
        self.attn.o_proj.forward = self._orig_o_proj_forward


def phase1_importance_windowed(model, calib_chunks, window_K: int, h_targets: dict):
    """
    Windowed L2-reconstruction Michel-style α-grad importance scoring.

    For each attention-bearing layer L:
      1. Install α gate at L (entry point for grad)
      2. Install capture-and-detach hook at L_end = min(L+K, N-1):
           - Captures h(α) at the window end WITH grad attached
           - Returns detached output to the next layer so layers L_end+1..N
             see no-grad input (no activation storage cost)
      3. Run forward(model), then compute
           loss = mean((h_cached[L_end] - h(α)[L_end])²)
         and backward — gradient flows through layers L..L_end to α[L].
      4. Capture |grad α[L]| per head, accumulate over chunks
      5. Remove gate and hook, move to L+1

    The forward still runs end-to-end (the tail L_end+1..N does pointless
    no-grad compute), but autograd memory is bounded:
      - Layers 0..L-1: no grad in path  (no activation storage)
      - Layers L..L_end: grad enabled  (~K × ~50 MB per 1K tokens)
      - Layers L_end+1..N: detached    (no activation storage)

    Trade-off vs end-to-end CE: L2 reconstruction over the residual stream
    is 'over-specified' — it asks ALL hidden_size dimensions to match,
    including ones that don't matter for next-token prediction. The two-tier
    pairing with nf4 global α-grad recovers CE-relevance via the second
    signal.

    Returns: dict {layer_idx: tensor of |grad α| per head, accumulated over
    all calib chunks, on CPU fp32}.
    """
    layers = get_layers(model)
    n_layers = len(layers)
    log(f"phase1 (windowed L2 K={window_K}): scoring {n_layers} layers, "
        f"backward through min(K, N-L) layers per scoring step")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    importance = {}
    first_dev = next(model.parameters()).device
    total_chunks = len(calib_chunks)
    chunk_devices = [c.to(first_dev) for c in calib_chunks]

    for L in range(n_layers):
        layer = layers[L]
        if not has_self_attn(layer):
            continue
        head_dim, n_q, _, layer_kind = layer_attn_geom(model, L)
        L_end = min(L + window_K, n_layers - 1)

        gate = HeadGate(layer.self_attn, n_q, head_dim)
        gate.alpha.requires_grad_(True)

        # Capture-and-detach hook at L_end
        captured: dict = {}

        def capture_and_detach_hook(module, inputs, output, _captured=captured):
            if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
                _captured["h"] = output[0]
                return (output[0].detach(),) + output[1:]
            if torch.is_tensor(output):
                _captured["h"] = output
                return output.detach()
            return output

        end_handle = layers[L_end].register_forward_hook(capture_and_detach_hook)

        # Pre-fetch the cached target for this L_end (one tensor on CPU fp32);
        # we'll move it to the device of h_alpha each chunk.
        h_cached_full = h_targets[L_end]  # [B, T_total, hidden]

        loss_total = 0.0
        chunk_offset = 0
        for ci, ids in enumerate(chunk_devices):
            if gate.alpha.grad is not None:
                gate.alpha.grad.zero_()
            captured.clear()
            T_ci = ids.shape[1]
            _ = model(ids)  # full forward; we only consume the captured h(α)
            h_alpha = captured["h"]  # [B, T_ci, hidden] with grad
            # Slice the cached target to this chunk's time window
            h_cached_ci = h_cached_full[:, chunk_offset : chunk_offset + T_ci, :].to(
                device=h_alpha.device, dtype=h_alpha.dtype
            )
            chunk_offset += T_ci
            loss = (h_alpha - h_cached_ci).pow(2).mean()
            loss.backward()
            loss_total += loss.item()
            cur = gate.alpha.grad.detach().abs().to(torch.float32).cpu()
            if not hasattr(gate, "_imp_acc"):
                gate._imp_acc = cur.clone()
            else:
                gate._imp_acc += cur

        importance[L] = gate._imp_acc
        gate.restore()
        end_handle.remove()
        log(f"phase1: L={L:02d} [{layer_kind}] window=[{L},{L_end}] "
            f"avg_L2={loss_total/total_chunks:.4e} "
            f"imp range=[{importance[L].min():.3e}, {importance[L].max():.3e}]")

    log(f"phase1: importance computed for {len(importance)} layers")
    return importance


# (Back-compat alias removed; main() calls phase1_importance_windowed directly with h_targets.)


def _rechunk_calib(calib_chunks, target_tokens: int):
    """Concatenate all calib chunks and re-split into chunks of `target_tokens`.
    Used to bound activation memory for the nf4 forward+backward pass which
    OOMs at >224 tokens on a 24GB GPU."""
    if target_tokens <= 0:
        return calib_chunks
    if not calib_chunks:
        return []
    flat = torch.cat([c.view(1, -1) for c in calib_chunks], dim=1)  # (1, T_total)
    T = flat.shape[1]
    rechunked = []
    for off in range(0, T - target_tokens + 1, target_tokens):
        rechunked.append(flat[:, off : off + target_tokens])
    if not rechunked:
        rechunked = [flat[:, :target_tokens]]
    return rechunked


def phase1_importance_nf4_global(model_path: str, calib_chunks, chunk_tokens: int = 192):
    """
    Global (full-stack, no horizon clip) Michel-style α-grad importance scoring
    using a fresh nf4-quantized model load.

    Memory footprint: 31B nf4 ≈ 16 GB weights + ~3-5 GB activations w/ grad
    checkpointing + transient ≈ 22 GB total. Fits a single 24 GB GPU without
    accelerate CPU offload.

    Quality vs bf16 global: nf4 introduces dequantization noise into the
    forward chain, which propagates into |grad α|. Empirically (QLoRA paper +
    downstream work) the per-layer head ranking is stable to within ~5%
    relative agreement with bf16 — sufficient for coarse selection at
    moderate prune ratios. Captures the FULL long-tail cross-layer signal
    that the windowed K=10 path clips off.

    Returns: dict {layer_idx: tensor of |grad α| per head, accumulated over
    all calib chunks, on CPU fp32}.

    Loads + unloads its own model so the caller can keep their bf16-with-
    offload model intact during phase 0/1a/2.
    """
    from transformers import BitsAndBytesConfig

    log(f"phase1_nf4: loading model at nf4 (compute dtype bf16, chunk_tokens={chunk_tokens})")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # Full GPU placement (no CPU offload). Empirically the nf4-quantized 31B
    # plus the small vision/embed pieces totals only ~17 GiB on GPU, leaving
    # ~7 GiB headroom for forward+backward. CPU-offloading via accelerate
    # would saturate GPU during pull-back of dequant buffers and OOM.
    # bnb 4-bit + accelerate CPU offload also has a known compatibility bug
    # in current versions (Params4bit __new__ unexpected kwarg `_is_hf_initialized`).
    model_nf4 = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
    )
    log(f"phase1_nf4: load complete, GPU={torch.cuda.memory_allocated()/(1024**3):.2f} GiB")
    # Re-chunk calibration to fit forward+backward memory budget.
    calib_chunks = _rechunk_calib(calib_chunks, target_tokens=chunk_tokens)
    log(f"phase1_nf4: re-chunked calib to {len(calib_chunks)} × {chunk_tokens}-token chunks "
        f"(total {sum(c.shape[1] for c in calib_chunks)} tokens)")
    model_nf4.config.use_cache = False
    if hasattr(model_nf4.config, "text_config"):
        model_nf4.config.text_config.use_cache = False
    model_nf4.eval()

    # Freeze all model params; only α leaves get grad
    for p in model_nf4.parameters():
        p.requires_grad_(False)

    # Per-layer scoring: ONE α gate at a time. Mathematically equivalent to
    # all-gates-simultaneously because at α=1 the gates are no-ops, so
    # ∂CE/∂α[L,h] is identical regardless of whether other layers' gates
    # exist. Trades 60× more forward passes for bounded memory: with all 60
    # gates installed, autograd kept 60 simultaneous bnb dequant buffers
    # cached → OOM. With 1 gate active, only that layer's o_proj path is
    # in the autograd graph, dequant cache stays small.
    layers_nf4 = get_layers(model_nf4)
    n_layers_nf4 = len(layers_nf4)
    log(f"phase1_nf4: per-layer scoring across {n_layers_nf4} layers, "
        f"{len(calib_chunks)} chunks each (no grad-ckpt — model fits forward+bwd at chunk_tokens={chunk_tokens})")

    importance_full: dict[int, torch.Tensor] = {}
    import time as _time
    t_start = _time.time()
    cumulative_ops = 0
    cum_loss = 0.0
    for L in range(n_layers_nf4):
        layer = layers_nf4[L]
        if not has_self_attn(layer):
            continue
        head_dim, n_q, _, layer_kind = layer_attn_geom(model_nf4, L)
        gate = HeadGate(layer.self_attn, n_q, head_dim)
        gate.alpha.requires_grad_(True)

        l_loss = 0.0
        for ci, chunk in enumerate(calib_chunks):
            if gate.alpha.grad is not None:
                gate.alpha.grad.zero_()
            ids = chunk.to("cuda:0")
            out = model_nf4(ids)
            logits = out.logits
            shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
            shift_labels = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss.backward()
            l_loss += loss.item()
            cur = gate.alpha.grad.detach().abs().to(torch.float32).cpu()
            if L in importance_full:
                importance_full[L] += cur
            else:
                importance_full[L] = cur.clone()
            del out, logits, loss
            torch.cuda.empty_cache()
            cumulative_ops += 1
        gate.restore()
        avg_l_loss = l_loss / max(1, len(calib_chunks))
        cum_loss += avg_l_loss
        if L % 5 == 0 or L == n_layers_nf4 - 1:
            elapsed = _time.time() - t_start
            log(f"phase1_nf4: L={L:02d} [{layer_kind}] avg_CE={avg_l_loss:.4f} "
                f"imp range=[{importance_full[L].min():.3e}, {importance_full[L].max():.3e}] "
                f"({cumulative_ops} ops, {elapsed:.1f}s)")

    log(f"phase1_nf4: avg CE across all layers = {cum_loss / len(importance_full):.4f}")
    log(f"phase1_nf4: importance computed for {len(importance_full)} layers")

    del model_nf4
    torch.cuda.empty_cache()
    return importance_full


def combine_importance(imp_local: dict, imp_full: dict, lam: float = 1.0) -> dict:
    """
    Combine windowed-K bf16 importance (clean local signal) with nf4 global
    importance (long-tail with quantization noise).

    Per-layer normalize each to [0, 1] (divide by max), then weighted sum.
    Normalization prevents the two signals' raw scales from dominating
    each other across layers.

    imp_combined[L, h] = imp_local_normalized[L, h] + lam * imp_full_normalized[L, h]

    Default lam=1.0 (equal weighting). Higher lam emphasizes the long-tail
    correction; lower lam keeps the local signal dominant.
    """
    combined = {}
    for L in imp_local:
        a = imp_local[L]
        b = imp_full.get(L)
        a_norm = a / (a.max() + 1e-12)
        if b is None:
            combined[L] = a_norm
            continue
        b_norm = b / (b.max() + 1e-12)
        combined[L] = a_norm + lam * b_norm
    return combined


# ---------- phase 2: per-layer prune + lstsq refit ----------

@torch.no_grad()
def prune_q_heads_inplace(model, layer_idx: int, keep_q_heads, drop_kv_groups: bool = False,
                          keep_kv_heads=None):
    """
    Mask-prune Q heads in a Gemma4 attention layer:
      - Zero q_proj rows for dropped Q heads (per-layer head_dim).
      - Optionally drop KV heads if a whole GQA group is dropped (off by default).
      - Or, if keep_kv_heads is provided (group-aware mode), zero KV rows whose
        index is not in keep_kv_heads.

    No norm changes needed — q_norm/k_norm/v_norm operate on head_dim, shared across heads.

    Uses accelerate.align_module_device so writes work whether the layer is on
    real GPU/CPU tensors or accelerate-offloaded meta tensors.
    """
    layer = get_layers(model)[layer_idx]
    sa = layer.self_attn
    head_dim, n_q, n_kv, lt = layer_attn_geom(model, layer_idx)
    keep_q = sorted(int(i) for i in keep_q_heads)
    keep_set = set(keep_q)
    drop_q = [i for i in range(n_q) if i not in keep_set]

    with _maybe_align(sa):
        qw = sa.q_proj.weight.data  # (n_q * head_dim, hidden)
        for h in drop_q:
            qw[h * head_dim : (h + 1) * head_dim, :] = 0
        if sa.q_proj.bias is not None:
            for h in drop_q:
                sa.q_proj.bias.data[h * head_dim : (h + 1) * head_dim] = 0

        # Group-aware path: explicit kept-KV indices passed in.
        if keep_kv_heads is not None and n_kv > 0:
            keep_kv_set = set(int(i) for i in keep_kv_heads)
            drop_kv = [g for g in range(n_kv) if g not in keep_kv_set]
            if drop_kv:
                log(f"  L{layer_idx} [{lt}]: group-aware drop KV {drop_kv}")
                for g in drop_kv:
                    sa.k_proj.weight.data[g * head_dim : (g + 1) * head_dim, :] = 0
                    if sa.v_proj is not None:
                        sa.v_proj.weight.data[g * head_dim : (g + 1) * head_dim, :] = 0
                    if sa.k_proj.bias is not None:
                        sa.k_proj.bias.data[g * head_dim : (g + 1) * head_dim] = 0
                    if sa.v_proj is not None and sa.v_proj.bias is not None:
                        sa.v_proj.bias.data[g * head_dim : (g + 1) * head_dim] = 0
        elif drop_kv_groups and n_kv > 0:
            group_size = n_q // n_kv  # sliding 2, full 8
            keep_groups = sorted({h // group_size for h in keep_q})
            drop_groups = [g for g in range(n_kv) if g not in set(keep_groups)]
            if drop_groups:
                log(f"  L{layer_idx} [{lt}]: drop KV groups {drop_groups}")
                for g in drop_groups:
                    sa.k_proj.weight.data[g * head_dim : (g + 1) * head_dim, :] = 0
                    if sa.v_proj is not None:
                        sa.v_proj.weight.data[g * head_dim : (g + 1) * head_dim, :] = 0


# ---------- group-aware head selection (T11) ----------


def select_keep_q_kv_group_aware(
    imp: torch.Tensor,
    n_q: int,
    n_kv: int,
    layer_type: str,
    target_n_q_keep: int,
    target_n_kv_keep_sliding: int,
) -> tuple[list[int], list[int]]:
    """
    Group-aware head selection for structural pruning. Guarantees that the
    resulting (kept_q, kept_kv) preserves an integer GQA ratio so the layer
    can be physically resized and loaded by transformers / llama.cpp.

    Three regimes, picked from (n_q, n_kv) of THIS layer:

    (1) n_kv == 1 (E2B sliding+full, any model with a single shared KV head):
        Free Q-drop by importance, KV unchanged. (n_q_new % 1 == 0) is trivial,
        so any target_n_q_keep is valid. target_n_kv_keep_sliding is ignored
        for these layers.

    (2) group_size == n_q // n_kv == 1 (no GQA, 1:1 Q:KV): drop matching
        Q+KV in pairs ranked by Q-head importance. Each Q-drop forces its
        paired KV-drop. target_n_q_keep == target_n_kv_keep (both == target).

    (3) group_size > 1 (31B-style: 32 Q / 16 KV → group_size=2): rank KV-groups
        by sum of member Q importance, drop lowest-summed groups whole. Result
        kept_q = group_size × kept_kv. target_n_q_keep MUST equal
        group_size × target_n_kv_keep_sliding.

    Full-attention layers in regime (3) keep all KV (so target_n_q_keep just
    needs n_kv divisibility) — this matches Gemma 4 31B's full layers
    (n_kv_full=4, never reduced).

    Returns (keep_q sorted ascending, keep_kv sorted ascending).
    """
    imp_list = imp.tolist()

    # Regime 1: single shared KV head — free Q drop, KV intact.
    if n_kv == 1:
        keep_kv = [0]
        ranked = sorted(range(n_q), key=lambda h: -imp_list[h])
        keep_q = sorted(ranked[:target_n_q_keep])
        return keep_q, keep_kv

    # Full-attention with multi-KV (e.g. 31B full layers, n_kv_full=4):
    # Free Q drop, KV unchanged. Constraint: target % n_kv == 0.
    if layer_type == "full_attention":
        keep_kv = list(range(n_kv))
        ranked = sorted(range(n_q), key=lambda h: -imp_list[h])
        keep_q = sorted(ranked[:target_n_q_keep])
        if target_n_q_keep % n_kv != 0:
            raise ValueError(
                f"full layer keeps {target_n_q_keep} Q heads not divisible by {n_kv} KV"
            )
        return keep_q, keep_kv

    # Sliding-attention with multi-KV.
    group_size = n_q // n_kv
    if group_size * n_kv != n_q:
        raise ValueError(f"sliding layer non-integer GQA: n_q={n_q} n_kv={n_kv}")

    # Regime 2: group_size==1 (1:1 Q:KV). Q-drop forces paired KV-drop.
    if group_size == 1:
        if target_n_q_keep != target_n_kv_keep_sliding:
            raise ValueError(
                f"sliding 1:1 Q:KV: n_q_keep={target_n_q_keep} must == "
                f"n_kv_keep={target_n_kv_keep_sliding}"
            )
        ranked = sorted(range(n_q), key=lambda h: -imp_list[h])
        keep_q = sorted(ranked[:target_n_q_keep])
        keep_kv = list(keep_q)  # 1:1 mapping
        return keep_q, keep_kv

    # Regime 3: group_size > 1, paired KV-aligned drops.
    if target_n_q_keep != group_size * target_n_kv_keep_sliding:
        raise ValueError(
            f"sliding target mismatch: n_q_keep={target_n_q_keep} != "
            f"{group_size}*n_kv_keep={group_size * target_n_kv_keep_sliding}"
        )
    group_score = []
    for g in range(n_kv):
        members = list(range(g * group_size, (g + 1) * group_size))
        group_score.append((g, sum(imp_list[m] for m in members), members))
    group_score.sort(key=lambda t: -t[1])
    kept_groups = group_score[:target_n_kv_keep_sliding]
    keep_kv = sorted(g for g, _, _ in kept_groups)
    keep_q = sorted(m for _, _, members in kept_groups for m in members)
    return keep_q, keep_kv


def auto_resolve_prune_mode(
    cfg, target_n_q_keep: int, allow_low_kv_structural: bool = False
) -> tuple[str, int, int, int]:
    """
    Inspect a Gemma 4 text_config + the requested keep target, decide whether
    a structural prune (group-aware) is feasible, and return:
        (resolved_mode, group_size_sliding, target_n_kv_keep_sliding,
         first_kv_shared_layer_idx)

    resolved_mode:
      - "group-aware": pipeline can produce a valid integer-ratio reshape
        with the requested keep target. Includes group_size=1 / n_kv>1
        special case (which select_keep_q_kv_group_aware handles).
      - "mask": no integer-ratio configuration possible at this keep target,
        OR Fix-B refusal (n_kv=1 architectures collapse autoregressively
        after structural reshape + lstsq heal — see T7.2 E2B catastrophic).
        Caller can override the n_kv=1 refusal with allow_low_kv_structural=True.

    first_kv_shared_layer_idx: layers with idx >= this value have shared KV
    (k_proj/v_proj weights are dead at inference). Producer should skip K/V
    modifications + slicing on those layers. None when the model has no
    KV sharing.

    Fix-B (architecture-aware gate, 2026-05-09):
      Empirically (E2B, T7.2 sweep), structural reshape on n_kv_s == 1
      models destroys generation: HE pass@1 = 0.000 (vs base 0.7378), all
      MBPP gens degenerate — confirmed via HF gen on saved BF16, not a GGUF
      artifact. The lstsq o_proj heal cannot recover the missing diversity
      when all surviving Q heads share a single KV pair. Default behavior
      is now to refuse structural reshape on n_kv_s == 1 architectures and
      fall back to mask mode. Override with allow_low_kv_structural=True
      AT YOUR OWN RISK (the 2.5 canary should still abort the save).
    """
    n_q = cfg.num_attention_heads
    n_kv_s = cfg.num_key_value_heads
    n_global_kv = cfg.num_global_key_value_heads if cfg.attention_k_eq_v else cfg.num_key_value_heads
    n_layers = cfg.num_hidden_layers
    n_kv_shared = getattr(cfg, "num_kv_shared_layers", 0) or 0
    first_kv_shared_idx = (n_layers - n_kv_shared) if n_kv_shared > 0 else None

    group_size = n_q // n_kv_s if n_kv_s else 1

    # Sliding feasibility: any of three regimes works.
    #   (1) n_kv_s == 1: free Q-drop, KV intact (n_q_new % 1 == 0 trivially).
    #   (2) group_size == 1: 1:1 Q:KV paired drop (n_q_new == n_kv_new always).
    #   (3) group_size > 1: paired drop, target_n_q_keep % group_size == 0.
    if n_kv_s == 1:
        sliding_ok, target_n_kv_keep_sliding = True, 1
    elif group_size == 1:
        sliding_ok, target_n_kv_keep_sliding = True, target_n_q_keep
    else:
        ok = (target_n_q_keep % group_size == 0) and (target_n_q_keep // group_size <= n_kv_s)
        sliding_ok = ok
        target_n_kv_keep_sliding = target_n_q_keep // group_size if ok else n_kv_s

    # Full feasibility: target_n_q_keep % n_global_kv == 0 (free Q-drop, KV intact).
    # n_global_kv == 1 makes this trivial.
    full_ok = (n_global_kv == 1) or (target_n_q_keep % n_global_kv == 0)

    if sliding_ok and full_ok:
        # Fix-B refusal: reshape catastrophically broken on n_kv_s == 1
        # (T7.2 E2B). Prefer mask unless caller explicitly opts in.
        if n_kv_s == 1 and not allow_low_kv_structural:
            return "mask", group_size, n_kv_s, first_kv_shared_idx
        return "group-aware", group_size, target_n_kv_keep_sliding, first_kv_shared_idx

    # Cannot satisfy structural reshape constraints — fall back to mask.
    return "mask", group_size, n_kv_s, first_kv_shared_idx


@torch.no_grad()
def physical_reshape_attn_layer(
    layer,
    keep_q: list[int],
    keep_kv: list[int],
    head_dim: int,
    hidden: int,
) -> dict:
    """
    Replace q_proj / k_proj / v_proj / o_proj on an attention sublayer with new
    Linear modules of the sliced dimensions, preserving values for kept heads.

    Caller must ensure keep_q/keep_kv are sorted ascending. Returns shape info
    for verification logging.
    """
    sa = layer.self_attn
    keep_q = sorted(int(h) for h in keep_q)
    keep_kv = sorted(int(g) for g in keep_kv)
    n_q_new = len(keep_q)
    n_kv_new = len(keep_kv)

    # --- Q projection (rows) ---
    old_q = sa.q_proj
    rows = []
    for h in keep_q:
        rows.append(old_q.weight.data[h * head_dim : (h + 1) * head_dim, :])
    new_qw = torch.cat(rows, dim=0).contiguous().clone()
    new_q = nn.Linear(hidden, n_q_new * head_dim, bias=old_q.bias is not None,
                      dtype=new_qw.dtype, device=new_qw.device)
    new_q.weight.data = new_qw
    if old_q.bias is not None:
        new_qb = torch.cat(
            [old_q.bias.data[h * head_dim : (h + 1) * head_dim] for h in keep_q]
        ).contiguous().clone()
        new_q.bias.data = new_qb
    sa.q_proj = new_q

    # --- K and V projections (rows) ---
    for projname in ("k_proj", "v_proj"):
        proj = getattr(sa, projname, None)
        if proj is None:
            continue
        rows = [proj.weight.data[g * head_dim : (g + 1) * head_dim, :] for g in keep_kv]
        new_w = torch.cat(rows, dim=0).contiguous().clone()
        new_p = nn.Linear(hidden, n_kv_new * head_dim, bias=proj.bias is not None,
                          dtype=new_w.dtype, device=new_w.device)
        new_p.weight.data = new_w
        if proj.bias is not None:
            new_b = torch.cat(
                [proj.bias.data[g * head_dim : (g + 1) * head_dim] for g in keep_kv]
            ).contiguous().clone()
            new_p.bias.data = new_b
        setattr(sa, projname, new_p)

    # --- O projection (cols, indexed by Q-heads) ---
    old_o = sa.o_proj
    cols = []
    for h in keep_q:
        cols.append(old_o.weight.data[:, h * head_dim : (h + 1) * head_dim])
    new_ow = torch.cat(cols, dim=1).contiguous().clone()
    new_o = nn.Linear(n_q_new * head_dim, hidden, bias=old_o.bias is not None,
                      dtype=new_ow.dtype, device=new_ow.device)
    new_o.weight.data = new_ow
    if old_o.bias is not None:
        new_o.bias.data = old_o.bias.data.clone()
    sa.o_proj = new_o

    return {
        "n_q_new": n_q_new,
        "n_kv_new": n_kv_new,
        "head_dim": head_dim,
        "q_shape": tuple(new_q.weight.shape),
        "k_shape": tuple(sa.k_proj.weight.shape) if sa.k_proj is not None else None,
        "o_shape": tuple(new_o.weight.shape),
    }


@torch.no_grad()
def lstsq_refit_o_proj(proj: nn.Linear, x_actual: torch.Tensor, y_target: torch.Tensor,
                       keep_cols: list[int], ridge_rel: float = 1e-3) -> dict:
    """
    Refit only the kept input columns of `proj.weight`. Dropped-head columns
    are zeroed in the new weight, contributing nothing to the projection.

    x_actual: [B, T, in_features] post-prune (dropped-head cols == 0)
    y_target: [B, T, out_features] pre-prune (W_orig @ x_orig)
    keep_cols: input-feature indices to fit
    ridge_rel: ridge as a fraction of mean diagonal of XtX (scale-invariant)
    """
    in_f = proj.in_features
    out_f = proj.out_features
    X = x_actual.reshape(-1, in_f).to(torch.float32)
    Y = y_target.reshape(-1, out_f).to(torch.float32)
    keep_idx = torch.tensor(sorted(keep_cols), dtype=torch.long)

    Xk = X[:, keep_idx]
    XtX = Xk.t() @ Xk
    XtY = Xk.t() @ Y
    diag_mean = XtX.diagonal().mean().item()
    lam = max(diag_mean * ridge_rel, 1e-6)
    XtX += lam * torch.eye(XtX.shape[0], dtype=XtX.dtype, device=XtX.device)
    try:
        L = torch.linalg.cholesky(XtX)
        Wk_t = torch.cholesky_solve(XtY, L)
    except Exception:
        Wk_t = torch.linalg.solve(XtX, XtY)
    Wk = Wk_t.t().contiguous()

    W_new = torch.zeros(out_f, in_f, dtype=torch.float32)
    W_new[:, keep_idx] = Wk

    Y_hat = Xk @ Wk_t
    resid_rms = (Y_hat - Y).pow(2).mean().sqrt().item()
    target_rms = Y.pow(2).mean().sqrt().item()
    rel = resid_rms / max(target_rms, 1e-9)
    with _maybe_align(proj):
        proj.weight.data.copy_(W_new.to(proj.weight.dtype).to(proj.weight.device))
    return {"kept": int(len(keep_idx)), "in": int(in_f), "lam": lam,
            "rel_resid": rel, "target_rms": target_rms}


# ---------- T13: LoRA-as-correction heal ----------
#
# T7.5 (L4-L7 head prune) showed lstsq local heal recovers AR coherence on 1
# of 3 canary prompts but not the 2 that demand specific token-level coherence
# (code closing braces, short factual recall). The lstsq solution is a single
# linear projection from kept-Q activations to o_proj output; it minimizes
# TF-MSE but doesn't capture the AR-relevant variance directions on prompts
# where the trajectory leaves the calib distribution.
#
# Replace lstsq with a rank-r LoRA correction trained via Adam against the
# same Phase-2 reconstruction target. Strictly more expressive (linear is the
# r=∞ degenerate case); init B=0 so step 0 ≡ no-correction; weight_decay acts
# as ridge stabilizer. Fold the correction into the kept cols of proj.weight
# before save — same on-disk shape as lstsq, no llama.cpp surgery needed.

def lora_heal_o_proj(
    proj: nn.Linear,
    x_actual: torch.Tensor,
    y_target: torch.Tensor,
    keep_cols: list[int],
    rank: int = 8,
    n_steps: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    """LoRA-as-correction heal: keep original W_kept, add trained δ=B·A.

    Init A ~ N(0, 1/√n_kept), B = 0 → step 0 reproduces base behavior. Train
    via Adam to minimize ‖(W_kept + B·A) Xk − Y‖². Fold (W_kept + B·A) into
    proj.weight[:, keep_cols], zero the dropped cols, copy back.

    Args mirror lstsq_refit_o_proj where overlapping. `weight_decay` acts as
    ridge: `1e-4` works well; tighten to `1e-3` if overfitting on small calib.
    """
    in_f = proj.in_features
    out_f = proj.out_features
    keep_idx = torch.tensor(sorted(keep_cols), dtype=torch.long)
    n_kept = len(keep_idx)

    X = x_actual.reshape(-1, in_f).to(torch.float32)
    Y = y_target.reshape(-1, out_f).to(torch.float32)
    Xk = X[:, keep_idx]                 # [N, n_kept]
    target_rms = Y.pow(2).mean().sqrt().item()

    # Snapshot the original kept-cols weights (frozen reference). Match X's
    # device — x_actual was captured on CPU by capture_proj_input_at_layer.
    with _maybe_align(proj):
        W_orig = proj.weight.data.detach().to(torch.float32).cpu().clone()
    Wk_orig = W_orig[:, keep_idx]       # [out_f, n_kept] on CPU

    # LoRA params. PEFT-standard init: A small random, B zero → δ=0 at step 0.
    device = X.device
    A = torch.randn(rank, n_kept, device=device, dtype=torch.float32) * (1.0 / (n_kept ** 0.5))
    B = torch.zeros(out_f, rank, device=device, dtype=torch.float32)
    A.requires_grad_(True)
    B.requires_grad_(True)
    opt = torch.optim.Adam([A, B], lr=lr, weight_decay=weight_decay)

    # Frozen weight in column-major (Xk @ Wk_orig.T form). Use full-precision
    # for the inner product; this is a single hidden×hidden_kept tensor, fine.
    Wk_orig_T = Wk_orig.t().contiguous()  # [n_kept, out_f]

    losses: list[float] = []
    for step in range(n_steps):
        opt.zero_grad()
        # Y_pred = Xk @ (W_kept + B·A).T = Xk @ W_kept.T + (Xk @ A.T) @ B.T
        # Use the factored form so we never materialize hidden×hidden_kept.
        XA = Xk @ A.t()                    # [N, r]
        Y_pred = Xk @ Wk_orig_T + XA @ B.t()  # [N, out_f]
        loss = (Y_pred - Y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # Fold into proj.weight: dropped cols zeroed, kept cols become W_kept + B·A.
    with torch.no_grad():
        delta = (B @ A).detach()             # [out_f, n_kept]
        Wk_new = Wk_orig + delta
        W_new = torch.zeros(out_f, in_f, dtype=torch.float32)
        W_new[:, keep_idx] = Wk_new
        # Final residual on the same target — reported in same units as lstsq.
        Y_hat = Xk @ Wk_new.t()
        resid_rms = (Y_hat - Y).pow(2).mean().sqrt().item()
        rel = resid_rms / max(target_rms, 1e-9)

    with _maybe_align(proj):
        proj.weight.data.copy_(W_new.to(proj.weight.dtype).to(proj.weight.device))

    return {
        "kept": int(n_kept),
        "in": int(in_f),
        "rank": int(rank),
        "n_steps": int(n_steps),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "rel_resid": rel,
        "target_rms": target_rms,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        # Compatibility keys with lstsq stats so downstream prints don't crash.
        "lam": float(weight_decay),
    }


@torch.no_grad()
def capture_proj_input_at_layer(model, calib_chunks, layer_idx: int) -> torch.Tensor:
    """Forward the (modified) model, capture+concatenate o_proj input at one layer."""
    return _capture_proj_input(model, calib_chunks, layer_idx, target="o_proj")


def capture_proj_input_at_layer_mlp(model, calib_chunks, layer_idx: int) -> torch.Tensor:
    """Forward the (modified) model, capture+concatenate down_proj input at one layer.

    T7.6: parallel of capture_proj_input_at_layer for FFN. Captures the SwiGLU
    output (gate(x) * up(x)) post-mask, which is exactly down_proj's input.
    """
    return _capture_proj_input(model, calib_chunks, layer_idx, target="down_proj")


@torch.no_grad()
def _capture_proj_input(model, calib_chunks, layer_idx: int, target: str) -> torch.Tensor:
    layer = get_layers(model)[layer_idx]
    if target == "o_proj":
        proj = layer.self_attn.o_proj
    elif target == "down_proj":
        proj = layer.mlp.down_proj
    else:
        raise ValueError(f"unknown target {target!r}")
    parts = []

    def hook(m, inp):
        parts.append(inp[0].detach().to(torch.float32).cpu())

    h = proj.register_forward_pre_hook(hook)
    first_dev = next(model.parameters()).device
    try:
        for chunk in calib_chunks:
            _ = model(chunk.to(first_dev))
    finally:
        h.remove()
    return torch.cat(parts, dim=1)


# ---------- T7.7: AR-distributed lstsq helpers ----------

@torch.no_grad()
def capture_proj_output_at_layer(model, seqs, layer_idx: int, target: str = "o_proj") -> torch.Tensor:
    """Forward `model` on each seq; capture+concat the OUTPUT of o_proj/down_proj at one layer.

    Mirrors `_capture_proj_input` but uses register_forward_hook on the output side.
    Used by T7.7 ar-lstsq to capture teacher targets after restoring W_orig.
    """
    layer = get_layers(model)[layer_idx]
    if target == "o_proj":
        proj = layer.self_attn.o_proj
    elif target == "down_proj":
        proj = layer.mlp.down_proj
    else:
        raise ValueError(f"unknown target {target!r}")
    parts = []

    def hook(m, inp, out):
        parts.append(out.detach().to(torch.float32).cpu())

    h = proj.register_forward_hook(hook)
    first_dev = next(model.parameters()).device
    try:
        for seq in seqs:
            _ = model(seq.to(first_dev))
    finally:
        h.remove()
    return torch.cat(parts, dim=1)


@torch.no_grad()
def ar_rollout_seqs(model, tokenizer, calib_chunks, n_seed: int = 8,
                    n_gen: int = 128, n_prefix: int = 64) -> list[torch.Tensor]:
    """Slice n_prefix tokens from each of the first n_seed calib chunks; greedy-generate n_gen tokens.

    Returns a list of [1, n_prefix + n_gen_actual] CPU tensors. Generation is greedy
    (do_sample=False) so this captures the masked student's deterministic AR trajectory.
    """
    seqs = []
    first_dev = next(model.parameters()).device
    pad_id = getattr(tokenizer, "eos_token_id", None) or 0
    for chunk in calib_chunks[:n_seed]:
        prefix = chunk[:, :n_prefix].to(first_dev)
        full = model.generate(
            prefix,
            max_new_tokens=n_gen,
            do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=pad_id,
            use_cache=True,
        )
        seqs.append(full.detach().cpu())
    return seqs


def _snapshot_layer_attn(layer) -> dict:
    """Snapshot q/k/v_proj weights (and biases) for a self-attn layer to CPU fp32.

    Used by T7.7 ar-lstsq before mask-prune so we can restore teacher behavior
    later for the y_target_ar capture.
    """
    sa = layer.self_attn
    snap: dict = {}
    with _maybe_align(sa.q_proj):
        snap["q_proj_w"] = sa.q_proj.weight.data.detach().clone().cpu()
        if sa.q_proj.bias is not None:
            snap["q_proj_b"] = sa.q_proj.bias.data.detach().clone().cpu()
    if getattr(sa, "k_proj", None) is not None:
        with _maybe_align(sa.k_proj):
            snap["k_proj_w"] = sa.k_proj.weight.data.detach().clone().cpu()
            if sa.k_proj.bias is not None:
                snap["k_proj_b"] = sa.k_proj.bias.data.detach().clone().cpu()
    if getattr(sa, "v_proj", None) is not None:
        with _maybe_align(sa.v_proj):
            snap["v_proj_w"] = sa.v_proj.weight.data.detach().clone().cpu()
            if sa.v_proj.bias is not None:
                snap["v_proj_b"] = sa.v_proj.bias.data.detach().clone().cpu()
    return snap


def _restore_layer_attn(layer, snap: dict) -> None:
    """Restore q/k/v_proj weights+biases from a snapshot dict produced by _snapshot_layer_attn."""
    sa = layer.self_attn
    with _maybe_align(sa.q_proj):
        sa.q_proj.weight.data.copy_(
            snap["q_proj_w"].to(sa.q_proj.weight.device, sa.q_proj.weight.dtype)
        )
        if sa.q_proj.bias is not None and "q_proj_b" in snap:
            sa.q_proj.bias.data.copy_(
                snap["q_proj_b"].to(sa.q_proj.bias.device, sa.q_proj.bias.dtype)
            )
    if "k_proj_w" in snap and getattr(sa, "k_proj", None) is not None:
        with _maybe_align(sa.k_proj):
            sa.k_proj.weight.data.copy_(
                snap["k_proj_w"].to(sa.k_proj.weight.device, sa.k_proj.weight.dtype)
            )
            if sa.k_proj.bias is not None and "k_proj_b" in snap:
                sa.k_proj.bias.data.copy_(
                    snap["k_proj_b"].to(sa.k_proj.bias.device, sa.k_proj.bias.dtype)
                )
    if "v_proj_w" in snap and getattr(sa, "v_proj", None) is not None:
        with _maybe_align(sa.v_proj):
            sa.v_proj.weight.data.copy_(
                snap["v_proj_w"].to(sa.v_proj.weight.device, sa.v_proj.weight.dtype)
            )
            if sa.v_proj.bias is not None and "v_proj_b" in snap:
                sa.v_proj.bias.data.copy_(
                    snap["v_proj_b"].to(sa.v_proj.bias.device, sa.v_proj.bias.dtype)
                )


# ---------- Fix C2: AR generation canary ----------
#
# T7.2 (2026-05-09) showed that final_ce on calibration tokens (TF) and
# per-layer rel_resid both stayed near base while generation collapsed
# entirely on E2B he125-E (LAB:LAB:LAB: token loops, HE pass@1 = 0.000).
# The producer needs an autoregressive coherence gate that sees actual
# decode-mode behavior. Same lesson as v6I (memory): "loss-on-training ≠
# generation quality at T>1."
#
# C2 protocol:
#   Phase 0 add-on:  greedy-decode N short canary prompts on the *unpruned*
#                    model, record (input_ids, gen_ids, base_NLL_under_self).
#   Phase 2.5 gate:  TF-forward `prompt + base_gen` through the PRUNED model,
#                    compute NLL of the gen tokens. If
#                    pruned_NLL / base_NLL > threshold for any prompt, the
#                    pruned model has drifted off the base manifold (it
#                    finds clean text implausible). Refuse to save.
#
# Canary prompts span one short Python (catches code-gen collapse), one
# factual English (catches general LM drift), one narrative (catches token
# repetition / EOS cliffs). Cheap: 3 prompts × ~50 tokens, single TF pass.

CANARY_PROMPTS = (
    "def add_two_numbers(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return ",
    "The capital of France is",
    "Once upon a time, in a small village, there lived",
)


@torch.no_grad()
def capture_canary_baseline(
    model, tokenizer, prompts: tuple[str, ...] = CANARY_PROMPTS, n_gen: int = 50
) -> list[dict]:
    """Phase-0 add-on. Greedy-decode each prompt with the (still-unpruned)
    model and record (input_ids, gen_ids, NLL_per_token under itself). The
    pruned model later TF-evaluates the same `input + gen` and we compare
    NLLs — large ratio = autoregressive distribution drift.
    """
    device = next(model.parameters()).device
    out: list[dict] = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        # Greedy decode. Set sampling-mode args to None to silence HF warnings
        # since temperature=top_p=top_k are not used at do_sample=False.
        try:
            gen_ids = model.generate(
                ids, max_new_tokens=n_gen, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=getattr(tokenizer, "eos_token_id", None) or 0,
            )
        except Exception as e:
            log(f"canary: gen failed on prompt={p[:40]!r}: {e!r} — skipping")
            continue
        gen_only = gen_ids[:, ids.shape[1]:]
        if gen_only.shape[1] == 0:
            log(f"canary: empty gen on prompt={p[:40]!r} — skipping")
            continue
        full = torch.cat([ids, gen_only], dim=1)
        logits = model(full).logits[:, ids.shape[1] - 1:-1, :].float()
        labels = gen_only.squeeze(0)
        ll = F.log_softmax(logits.squeeze(0), dim=-1)
        nll = -ll[range(labels.size(0)), labels]
        out.append({
            "prompt": p,
            "input_ids": ids.cpu(),
            "gen_ids": gen_only.cpu(),
            "base_nll_mean": float(nll.mean().item()),
            "base_n_gen": int(gen_only.shape[1]),
        })
        log(f"canary: prompt={p[:40]!r} base_NLL_mean={out[-1]['base_nll_mean']:.4f}"
            f" gen_len={gen_only.shape[1]}")
    return out


def _gen_shape_diagnostic(
    text: str, token_ids: list[int], pad_token_id: int | None
) -> dict:
    """Compute generation-shape metrics that catch AR-only failures TF-NLL
    misses entirely. Returns: length_ok / repetition_ratio / nonprintable_ratio.
    """
    # Length: count non-pad tokens.
    if pad_token_id is not None:
        n_real = sum(1 for t in token_ids if t != pad_token_id)
    else:
        n_real = len(token_ids)
    # Bigram-repetition: fraction of consecutive (t_i, t_{i+1}) pairs whose
    # most-frequent bigram dominates. Token-loop collapse ('LAB:LAB:' or
    # '---\n---\n') puts the same bigram everywhere.
    rep_ratio = 0.0
    if len(token_ids) >= 4:
        bigrams = [(token_ids[i], token_ids[i + 1]) for i in range(len(token_ids) - 1)]
        from collections import Counter
        ctr = Counter(bigrams)
        rep_ratio = ctr.most_common(1)[0][1] / len(bigrams)
    # Non-printable / off-language fraction on the decoded string.
    if text:
        nonp = sum(1 for ch in text if ord(ch) > 127 or (ord(ch) < 32 and ch not in "\n\t\r"))
        nonp_ratio = nonp / len(text)
    else:
        nonp_ratio = 0.0
    return {
        "n_tokens": n_real,
        "rep_ratio": rep_ratio,
        "nonp_ratio": nonp_ratio,
    }


def _resolve_canary_runtime(args, model, log_fn):
    """Decide (canary_device, canary_dtype) from CLI knobs + system state.

    `canary_device`: 'gpu' | 'cpu'.  'gpu' implies accelerate offload via
       dispatch_model with a max_memory map honoring `--canary-gpu-mem-frac`
       (or the explicit `--canary-gpu-mem-gib`).
    `canary_dtype`: torch.bfloat16 | torch.float16 | torch.float32.
       The model is currently in its native save dtype (BF16 for Gemma 4);
       this function may pick a wider dtype for CPU forwards because BF16
       CPU matmul in torch<2.6 is single-thread, while FP32 is multi-thread.
    """
    import psutil  # stdlib-adjacent; in base pytorch image
    cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0

    # device
    if args.canary_device == "auto":
        canary_dev = "gpu" if cuda_ok else "cpu"
    elif args.canary_device == "gpu":
        if not cuda_ok:
            raise RuntimeError("--canary-device gpu requested but no CUDA device available")
        canary_dev = "gpu"
    else:
        canary_dev = "cpu"

    # dtype
    DT_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    native_dtype = next(model.parameters()).dtype
    if args.canary_dtype != "auto":
        canary_dtype = DT_MAP[args.canary_dtype]
    else:
        if canary_dev == "gpu":
            canary_dtype = native_dtype  # no upcast cost on GPU, native bf16 is fast
        else:
            # CPU: prefer FP32 if RAM allows. n_params × 4B for FP32 + headroom.
            n_params = sum(p.numel() for p in model.parameters())
            fp32_need = n_params * 4
            # Need both the current copy (bf16, 2 bytes) AND the new fp32 view
            # alive at the same time during in-place .to(dtype). Budget for both.
            ram_need = fp32_need + n_params * 2
            ram_have = psutil.virtual_memory().available
            if ram_have >= int(ram_need * 1.1):
                canary_dtype = torch.float32
                log_fn(f"canary dtype=auto → fp32 on CPU "
                       f"(avail={ram_have/1e9:.1f} GB ≥ need≈{ram_need/1e9:.1f} GB)")
            else:
                canary_dtype = native_dtype
                log_fn(f"canary dtype=auto → bf16 on CPU "
                       f"(avail={ram_have/1e9:.1f} GB < need≈{ram_need/1e9:.1f} GB for fp32 cast; "
                       f"single-thread bf16 CPU matmul — consider more RAM or --canary-device gpu)")
    return canary_dev, canary_dtype


def _enter_canary_runtime(args, model, canary_dev, canary_dtype, log_fn):
    """Move model to (canary_dev, canary_dtype). Returns a `restore()` callable.

    Pairs with _resolve_canary_runtime(). MUST always be followed by restore()
    in a finally block — saved-weights dtype/placement must match what the
    safetensors writer expects (bf16/CPU), not whatever the canary used.
    """
    orig_dtype = next(model.parameters()).dtype
    if canary_dtype is not None and canary_dtype != orig_dtype:
        log_fn(f"canary: casting model {orig_dtype} → {canary_dtype} (in place)")
        model.to(dtype=canary_dtype)

    if canary_dev == "gpu":
        from accelerate import dispatch_model, infer_auto_device_map
        total_vram = torch.cuda.get_device_properties(0).total_memory  # bytes
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        # Canary forward uses BOTH a teacher-forced pass over input+gen AND a
        # generate() call. Peak per-device budget must hold:
        #   - resident layer weights (driven by max_memory)
        #   - full-prompt KV cache for the longest single canary prompt
        #   - one layer's worth of activations (attention scores + residual)
        # plus CUDA context overhead. We compute the non-weight floor from
        # config + canary seq length and subtract it from the frac-budget.
        # If the user passed --canary-gpu-mem-gib explicitly, that overrides
        # everything below.
        def _attr(name, default):
            for c in (text_cfg, cfg):
                if hasattr(c, name):
                    return getattr(c, name)
            return default
        num_layers = _attr("num_hidden_layers", 60)
        num_kv = _attr("num_key_value_heads", _attr("num_attention_heads", 32))
        num_q  = _attr("num_attention_heads", 32)
        head_dim = _attr("head_dim", _attr("hidden_size", 5376) // max(num_q, 1))
        hidden   = _attr("hidden_size", 5376)
        # Canary sequence length: longest_prompt + n_gen + headroom.
        # The longest canary prompt is short (~80 tokens, hardcoded in
        # capture_canary_baseline) but be permissive in case the prompt set
        # grows. Use a 256-token floor on prompt length.
        canary_seq_max = max(256, args.canary_n_gen + 256)
        dtype_bytes = (2 if canary_dtype in (torch.bfloat16, torch.float16) else 4)
        # KV cache size, bytes (both K and V):
        kv_bytes = 2 * num_layers * num_kv * head_dim * canary_seq_max * dtype_bytes
        # Activation peak: attention scores (heads × seq²) for one layer
        # + residual + intermediate. Hold-out one layer's worth, doubled.
        act_bytes = 2 * (num_q * (canary_seq_max ** 2) * dtype_bytes
                         + canary_seq_max * hidden * dtype_bytes * 4)
        # CUDA context + cuBLAS workspace + accelerate overhead floor.
        cuda_overhead = 512 * (1 << 20)  # 512 MiB

        if args.canary_gpu_mem_gib is not None:
            gpu_budget_bytes = int(args.canary_gpu_mem_gib * (1 << 30))
            calc_note = f"explicit --canary-gpu-mem-gib={args.canary_gpu_mem_gib}"
        else:
            frac_cap   = int(total_vram * float(args.canary_gpu_mem_frac))
            reserve    = kv_bytes + act_bytes + cuda_overhead
            smart_cap  = max(int(0.50 * total_vram), total_vram - reserve)
            gpu_budget_bytes = min(frac_cap, smart_cap)
            calc_note = (f"frac={args.canary_gpu_mem_frac:.2f} → frac_cap={frac_cap/(1<<30):.2f}GiB; "
                         f"kv={kv_bytes/(1<<20):.0f}MiB act={act_bytes/(1<<20):.0f}MiB "
                         f"ctx={cuda_overhead/(1<<20):.0f}MiB → smart_cap={smart_cap/(1<<30):.2f}GiB; "
                         f"chose min")
        gpu_gib = gpu_budget_bytes / (1 << 30)
        cpu_mem = args.canary_cpu_mem or args.cpu_mem
        max_memory = {0: f"{gpu_gib:.2f}GiB", "cpu": cpu_mem}
        log_fn(f"canary: budgeting GPU={gpu_gib:.2f}/{total_vram/(1<<30):.2f}GiB ({calc_note})")
        log_fn(f"canary: dispatching to GPU offload "
               f"(max_memory={max_memory}, dtype={canary_dtype})")
        device_map = infer_auto_device_map(model, max_memory=max_memory, dtype=canary_dtype)
        # accelerate's infer_auto_device_map only walks the modules registered
        # as no_split / standard transformer blocks; on Gemma 4 it misses small
        # custom params/buffers like `layer_scalar`. Place any unmapped item
        # on CPU so dispatch_model accepts the map.
        def _all_param_buf_names(m, prefix=""):
            for n, _ in m.named_parameters(recurse=False):
                yield f"{prefix}{n}"
            for n, _ in m.named_buffers(recurse=False):
                yield f"{prefix}{n}"
            for n, sub in m.named_children():
                yield from _all_param_buf_names(sub, f"{prefix}{n}.")
        mapped_prefixes = tuple(
            (k + "." if k else "") for k in device_map
        )
        missing = [
            n for n in _all_param_buf_names(model)
            if not (n in device_map or n.startswith(mapped_prefixes))
        ]
        if missing:
            log_fn(f"canary: device_map missing {len(missing)} item(s); "
                   f"placing on cpu (e.g. {missing[:3]})")
            for n in missing:
                device_map[n] = "cpu"
        dispatch_model(model, device_map=device_map)
    else:
        log_fn(f"canary: running on CPU (dtype={canary_dtype})")
        model.to("cpu")
        # Recover CPU multi-thread for forwards even though MKL BF16 won't use
        # it — FP32 paths will, and other paths (RoPE, RMSNorm, softmax) do.
        try:
            n = os.cpu_count() or 1
            torch.set_num_threads(min(n, 64))
        except Exception:
            pass

    def _restore():
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except Exception:
            pass
        if canary_dtype is not None and canary_dtype != orig_dtype:
            log_fn(f"canary: casting model back to {orig_dtype} for save")
            model.to(dtype=orig_dtype)
        model.to("cpu")
        torch.cuda.empty_cache()
    return _restore


@torch.no_grad()
def gen_canary_check(
    model,
    canary: list[dict],
    tokenizer=None,
    n_gen: int = 50,
    ratio_threshold: float = 3.0,
    min_n_tokens: int = 5,
    max_rep_ratio: float = 0.40,
    max_nonp_ratio: float = 0.30,
) -> dict:
    """Phase-2.5 gate. Three checks per canary prompt; FAIL if ANY trips:

    (1) TF-NLL ratio: pruned_NLL_of_base_gen / base_NLL_of_base_gen.
        Catches global distribution drift. Threshold default 3.0×.
        NOTE: under teacher forcing the model never enters its own attractor,
        so this misses AR-only failures (token loops, early-EOS). Hence (2)+(3).

    (2) Pruned greedy-decode SHAPE check. Runs greedy decode under the PRUNED
        model on each canary prompt and inspects the output:
          - n_tokens < min_n_tokens → early-EOS (R1 mode)
          - rep_ratio > max_rep_ratio → token-loop collapse (he125-E mode)
          - nonp_ratio > max_nonp_ratio → off-language drift
        Catches the AR failures TF cannot see.

    (3) Self-vs-base NLL ratio under pruned: pruned_NLL_of_base_gen /
        pruned_NLL_of_pruned_own_gen. If the pruned model is in a token-loop
        attractor it assigns very low NLL to its own (degenerate) gen and
        high NLL to base's clean gen → ratio explodes. Threshold same as (1).

    All checks must pass.
    """
    device = next(model.parameters()).device
    pad_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    per_prompt: list[dict] = []
    for c in canary:
        ids = c["input_ids"].to(device)
        gen_base = c["gen_ids"].to(device)

        # (1) TF-NLL of base's gen under pruned.
        full_base = torch.cat([ids, gen_base], dim=1)
        logits = model(full_base).logits[:, ids.shape[1] - 1:-1, :].float()
        labels = gen_base.squeeze(0)
        ll = F.log_softmax(logits.squeeze(0), dim=-1)
        nll_base_under_pruned = float((-ll[range(labels.size(0)), labels]).mean().item())
        ratio_drift = nll_base_under_pruned / max(c["base_nll_mean"], 1e-9)

        # (2) Greedy decode under pruned, shape diagnostic.
        try:
            gen_pruned_full = model.generate(
                ids, max_new_tokens=n_gen, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=pad_id or 0,
            )
            gen_pruned = gen_pruned_full[:, ids.shape[1]:]
            text_pruned = tokenizer.decode(gen_pruned[0], skip_special_tokens=False) if tokenizer else ""
            shape = _gen_shape_diagnostic(text_pruned, gen_pruned[0].tolist(), pad_id)
        except Exception as e:
            shape = {"n_tokens": -1, "rep_ratio": -1.0, "nonp_ratio": -1.0,
                     "error": repr(e)[:120]}
            gen_pruned = None
            text_pruned = ""

        # (3) Self-NLL ratio under pruned: base-gen vs pruned-own-gen.
        # If pruned-own is a token loop it has very low NLL under pruned;
        # base-gen has high NLL → ratio explodes.
        ratio_self = float("nan")
        nll_pruned_own = float("nan")
        if gen_pruned is not None and gen_pruned.shape[1] > 0:
            full_pruned = torch.cat([ids, gen_pruned], dim=1)
            lp = model(full_pruned).logits[:, ids.shape[1] - 1:-1, :].float()
            la = gen_pruned.squeeze(0)
            llp = F.log_softmax(lp.squeeze(0), dim=-1)
            nll_pruned_own = float((-llp[range(la.size(0)), la]).mean().item())
            if nll_pruned_own > 1e-6:
                ratio_self = nll_base_under_pruned / nll_pruned_own

        # Per-prompt fail flags.
        fail_drift = ratio_drift > ratio_threshold
        fail_short = shape["n_tokens"] >= 0 and shape["n_tokens"] < min_n_tokens
        fail_loop = shape["rep_ratio"] > max_rep_ratio
        fail_nonp = shape["nonp_ratio"] > max_nonp_ratio
        fail_self = (not (ratio_self != ratio_self)) and ratio_self > ratio_threshold  # NaN-safe
        prompt_passed = not (fail_drift or fail_short or fail_loop or fail_nonp or fail_self)

        per_prompt.append({
            "prompt": c["prompt"][:60],
            "base_nll": c["base_nll_mean"],
            "pruned_nll": nll_base_under_pruned,
            "ratio_drift": ratio_drift,
            "ratio_self": ratio_self,
            "shape": shape,
            "gen_text_head": (text_pruned or "")[:80],
            "passed": prompt_passed,
            "fails": {
                "drift": fail_drift, "short": fail_short, "loop": fail_loop,
                "nonp": fail_nonp, "self": fail_self,
            },
        })

    overall_drift = max((p["ratio_drift"] for p in per_prompt), default=float("inf"))
    return {
        "passed": all(p["passed"] for p in per_prompt),
        "overall_ratio_drift": overall_drift,
        "threshold": ratio_threshold,
        "min_n_tokens": min_n_tokens,
        "max_rep_ratio": max_rep_ratio,
        "max_nonp_ratio": max_nonp_ratio,
        "per_prompt": per_prompt,
    }


# ---------- Fix A: leave-one-out importance ----------
#
# Single-pass gradient × activation importance (existing phase1_importance_*)
# is a marginal sensitivity signal that ranks heads under the assumption all
# others stay active. With low-n_kv architectures (E2B n_kv=1), all Q heads
# share the same KV memory and their gradient magnitudes are tightly
# correlated → ranking is near-degenerate. Leave-one-out actually drops the
# head and measures the CE delta on calibration. More expensive (n_q full
# forwards per layer) but architecture-agnostic and directly measures what
# we care about for k=1 drops. For k>1 drops, LOO is still a better marginal
# than gradient-based. Combine with the canary (Fix C2) and arch gate (Fix B)
# for full protection.

@torch.no_grad()
def phase1_importance_loo(
    model, calib_chunks: list[torch.Tensor], ce_baseline: float
) -> dict:
    """For each (layer L, head h): zero head h's q_proj rows, forward full
    model on calibration, measure CE; restore. Importance[L][h] = ΔCE.
    Higher ΔCE = more important head (its removal hurts CE more).

    Cost: n_layers × n_q full forwards on calib_chunks. For E2B 35×8 and
    1024 calib tokens → ~5 min on RTX 3090. For 31B 60×32 and 8192 tokens
    → ~30-60 min on A100.
    """
    layers = get_layers(model)
    cfg = text_config(model)
    n_q = cfg.num_attention_heads
    importance: dict[int, torch.Tensor] = {}
    device = next(model.parameters()).device

    for L, layer in enumerate(layers):
        if not has_self_attn(layer):
            continue
        head_dim, _, _, lt = layer_attn_geom(model, L)
        q_proj = layer.self_attn.q_proj
        original_w = q_proj.weight.data.clone()
        per_head_dce = torch.zeros(n_q)

        for h in range(n_q):
            q_proj.weight.data.copy_(original_w)
            q_proj.weight.data[h * head_dim:(h + 1) * head_dim].zero_()
            ce_total = 0.0
            for chunk in calib_chunks:
                ids = chunk.to(device)
                logits = model(ids).logits
                shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
                shift_labels = ids[:, 1:].contiguous()
                ce_total += F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).item()
            per_head_dce[h] = (ce_total / len(calib_chunks)) - ce_baseline

        q_proj.weight.data.copy_(original_w)
        importance[L] = per_head_dce
        log(f"phase1_loo: L{L:02d} [{lt}] ΔCE per-head: "
            f"min={per_head_dce.min():.4f} max={per_head_dce.max():.4f} "
            f"mean={per_head_dce.mean():.4f}")
    log(f"phase1_loo: importance computed for {len(importance)} layers")
    return importance


@torch.no_grad()
def measure_calib_ce(model, calib_chunks: list[torch.Tensor]) -> float:
    """Single-pass CE on calibration; used as the LOO baseline."""
    device = next(model.parameters()).device
    ce_total = 0.0
    for chunk in calib_chunks:
        ids = chunk.to(device)
        logits = model(ids).logits
        shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
        shift_labels = ids[:, 1:].contiguous()
        ce_total += F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).item()
    return ce_total / len(calib_chunks)


# ---------- T7.4: multi-class CD-map head importance ----------
#
# Single-axis LOO (Fix A above) gives one scalar per head: average ΔCE on a
# mixed calibration corpus. T7.2.1 showed even LOO-selected heads collapse the
# model on R1 scope — the scalar averages over class-specialists and noise.
# A head that is noise on prose but critical for code is ranked LOW under
# averaging, gets dropped, and code generation breaks (R1 early-EOS) or other
# downstream classes break too.
#
# Multi-class LOO computes ΔCE per (layer, head, class) by running LOO once
# per shard. The selection signal becomes "max-across-classes ΔCE" — a head's
# drop cost is its WORST-CASE class impact, not its average. This is an
# anti-specialist filter: heads that matter for ANY class are protected.
#
# Reusable: same per-(layer, head, class) importance tensor can drive head
# pruning, FFN-neuron pruning (when extended to neurons), expert dropping
# (T9 link to gemma4 98e CD-map), and CD differential quantization (Q5_K
# for class-specialists, Q3_K for cross-class generalists).

@torch.no_grad()
def phase1_importance_loo_multi(
    model, shards: dict[str, list[torch.Tensor]], ce_baselines: dict[str, float]
) -> dict[int, dict]:
    """Per-shard LOO ΔCE.

    For each (layer L, head h, shard s): zero head h's q_proj rows, forward
    the shard's calibration chunks, compute mean CE; restore. Per-shard
    importance is `ce_after - ce_baselines[s]` (positive = head matters for
    that shard's distribution).

    Returns: {L: {"per_class": Tensor[n_q, n_shards], "shard_names": [str, ...]}}
    """
    layers = get_layers(model)
    cfg = text_config(model)
    n_q = cfg.num_attention_heads
    shard_names = list(shards.keys())
    n_shards = len(shard_names)
    importance: dict[int, dict] = {}
    device = next(model.parameters()).device

    for L, layer in enumerate(layers):
        if not has_self_attn(layer):
            continue
        head_dim, _, _, lt = layer_attn_geom(model, L)
        q_proj = layer.self_attn.q_proj
        original_w = q_proj.weight.data.clone()
        per_class = torch.zeros(n_q, n_shards)

        for h in range(n_q):
            q_proj.weight.data.copy_(original_w)
            q_proj.weight.data[h * head_dim:(h + 1) * head_dim].zero_()
            for si, sname in enumerate(shard_names):
                ce_total = 0.0
                chunks = shards[sname]
                for chunk in chunks:
                    ids = chunk.to(device)
                    logits = model(ids).logits
                    shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
                    shift_labels = ids[:, 1:].contiguous()
                    ce_total += F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    ).item()
                per_class[h, si] = (ce_total / len(chunks)) - ce_baselines[sname]

        q_proj.weight.data.copy_(original_w)
        importance[L] = {"per_class": per_class, "shard_names": shard_names}
        # Compact log: per-shard ΔCE range across heads + worst-case (max-across-classes) ΔCE.
        max_per_head = per_class.max(dim=1).values
        log(f"phase1_loo_multi: L{L:02d} [{lt}] per-shard ΔCE: " + ", ".join(
            f"{sname}=[{per_class[:,si].min():.3f},{per_class[:,si].max():.3f}]"
            for si, sname in enumerate(shard_names)
        ) + f" | max-across-class min={max_per_head.min():.3f} max={max_per_head.max():.3f}")
    log(f"phase1_loo_multi: importance computed for {len(importance)} layers × "
        f"{n_shards} shards = {len(importance) * n_q * n_shards} cells")
    return importance


def aggregate_multi_class_importance(
    imp_multi: dict[int, dict], strategy: str = "max_class"
) -> dict[int, torch.Tensor]:
    """Convert per-(layer, head, class) tensor to per-(layer, head) scalar
    suitable for the existing selection paths (group-aware, mask).

    strategy="max_class": drop_cost[h] = max over classes of ΔCE[h, c].
        Anti-specialist — protects heads that matter for ANY class. Default.
    strategy="mean_class": drop_cost[h] = mean over classes (= single-axis LOO).
    strategy="positive_max": same as max_class but clipped to ≥ 0 (a head
        whose removal IMPROVES some class is treated as 0-importance for
        that class rather than negative-importance, which can distort max).
    """
    out: dict[int, torch.Tensor] = {}
    for L, blob in imp_multi.items():
        per_class = blob["per_class"]
        if strategy == "mean_class":
            out[L] = per_class.mean(dim=1)
        elif strategy == "positive_max":
            out[L] = per_class.clamp(min=0.0).max(dim=1).values
        else:  # max_class
            out[L] = per_class.max(dim=1).values
    return out


# ---------- T7.6: FFN-neuron LOO via T7.4 backbone (skeleton) ----------
#
# T7.4 validated the multi-class signal on Q-heads but importance choice was NOT
# the head-prune bottleneck — the lstsq local heal capacity was. FFN pruning is
# where multi-class signal SHOULD pay off:
#
#   - FFN intermediate channels specialize HARD (Wanda/REAP literature + the
#     v4 ContribDynamic data on Gemma 4 26B-A4B both show this empirically).
#   - The dynamic range across {code, prose, math} per channel is 5-10× wider
#     than for attention heads (most heads serve all classes; many neurons are
#     class-specific).
#   - The lstsq heal target is well-defined — same problem shape as o_proj
#     refit, just operating on down_proj rows when intermediate channels are
#     dropped.
#
# Cost discipline: full LOO over intermediate=10240 × 42 layers × 3 shards is
# ~1.3M forwards (untenable). Block-LOO over K=64 contiguous channels gets us
# to ~20k forwards (tractable). Refine the worst block at channel granularity
# only if we want fine-grained selection within a block.
#
# IMPLEMENTATION STATUS: skeleton + design notes only. Do NOT enable via CLI
# until the body is filled in and validated. Pivot decision is gated on T7.5
# (L4-L7 head prune): if head pruning passes canary at deeper layers, the
# capacity bottleneck is layer-depth-specific and FFN will likely behave the
# same (i.e. pruning shallow FFN bands also collapses, but deeper-FFN works).
# If head pruning still fails at L4-L7, the lstsq heal itself is the limiter
# and T13 LoRA-as-correction precedes FFN-LOO.

def has_mlp(layer) -> bool:
    """Layer has a feed-forward block we can prune intermediate channels of."""
    if not hasattr(layer, "mlp"):
        return False
    mlp = layer.mlp
    # Gemma 4 dense uses gate_proj + up_proj + down_proj (SwiGLU). MoE layers
    # have a different structure (experts list) and need a separate path —
    # skip them here; T9 handles the MoE expert dimension.
    return all(hasattr(mlp, k) for k in ("gate_proj", "up_proj", "down_proj"))


@torch.no_grad()
def phase1_importance_loo_ffn(
    model, calib_chunks: list[torch.Tensor], ce_baseline: float,
    block_size: int = 128,
    layer_filter: set[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Single-class block-LOO ΔCE for FFN intermediate channels.

    For each layer with an MLP, partition `intermediate_size` into contiguous
    blocks of `block_size` channels. For each block: zero the matching ROWS
    of gate_proj AND up_proj (both produce SwiGLU-paired outputs at the same
    channel index), forward all calib_chunks, compute mean CE, restore.

    importance[L][b] = mean_ce_after - ce_baseline. Higher = more important.

    Cost: n_layers × n_blocks full forwards. E4B with intermediate=8192 and
    block_size=128 gives 64 blocks × 42 layers = 2688 forwards. With
    8192-token calib at 0.5 s/forward → ~22 min on RTX 3090.
    """
    layers = get_layers(model)
    cfg = text_config(model)
    n_int = cfg.intermediate_size
    if n_int % block_size != 0:
        raise ValueError(
            f"intermediate_size={n_int} not divisible by block_size={block_size}"
        )
    n_blocks = n_int // block_size
    importance: dict[int, torch.Tensor] = {}
    device = next(model.parameters()).device

    for L, layer in enumerate(layers):
        if not has_mlp(layer):
            continue
        if layer_filter is not None and L not in layer_filter:
            continue
        gp = layer.mlp.gate_proj
        up = layer.mlp.up_proj
        gp_orig = gp.weight.data.clone()
        up_orig = up.weight.data.clone()
        per_block = torch.zeros(n_blocks)

        for b in range(n_blocks):
            gp.weight.data.copy_(gp_orig)
            up.weight.data.copy_(up_orig)
            lo, hi = b * block_size, (b + 1) * block_size
            gp.weight.data[lo:hi].zero_()
            up.weight.data[lo:hi].zero_()
            ce_total = 0.0
            for chunk in calib_chunks:
                ids = chunk.to(device)
                logits = model(ids).logits
                shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
                shift_labels = ids[:, 1:].contiguous()
                ce_total += F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).item()
            per_block[b] = (ce_total / len(calib_chunks)) - ce_baseline

        gp.weight.data.copy_(gp_orig)
        up.weight.data.copy_(up_orig)
        importance[L] = per_block
        log(f"phase1_loo_ffn: L{L:02d} ΔCE per-block (K={block_size}): "
            f"min={per_block.min():.4f} max={per_block.max():.4f} "
            f"mean={per_block.mean():.4f}")
    log(f"phase1_loo_ffn: importance computed for {len(importance)} layers × {n_blocks} blocks")
    return importance


@torch.no_grad()
def phase1_importance_loo_multi_ffn(
    model,
    shards: dict[str, list[torch.Tensor]],
    ce_baselines: dict[str, float],
    block_size: int = 128,
    layer_filter: set[int] | None = None,
) -> dict[int, dict]:
    """Multi-class block-LOO ΔCE per (layer, block, shard) for FFN intermediate.

    Mirrors phase1_importance_loo_multi but operating on FFN channel blocks
    instead of attention heads. Output shape is compatible with
    aggregate_multi_class_importance: each layer's "per_class" tensor has shape
    [n_blocks, n_shards].
    """
    layers = get_layers(model)
    cfg = text_config(model)
    n_int = cfg.intermediate_size
    if n_int % block_size != 0:
        raise ValueError(
            f"intermediate_size={n_int} not divisible by block_size={block_size}"
        )
    n_blocks = n_int // block_size
    shard_names = list(shards.keys())
    n_shards = len(shard_names)
    importance: dict[int, dict] = {}
    device = next(model.parameters()).device

    for L, layer in enumerate(layers):
        if not has_mlp(layer):
            continue
        if layer_filter is not None and L not in layer_filter:
            continue
        gp = layer.mlp.gate_proj
        up = layer.mlp.up_proj
        gp_orig = gp.weight.data.clone()
        up_orig = up.weight.data.clone()
        per_class = torch.zeros(n_blocks, n_shards)

        for b in range(n_blocks):
            gp.weight.data.copy_(gp_orig)
            up.weight.data.copy_(up_orig)
            lo, hi = b * block_size, (b + 1) * block_size
            gp.weight.data[lo:hi].zero_()
            up.weight.data[lo:hi].zero_()
            for si, sname in enumerate(shard_names):
                ce_total = 0.0
                chunks = shards[sname]
                for chunk in chunks:
                    ids = chunk.to(device)
                    logits = model(ids).logits
                    shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
                    shift_labels = ids[:, 1:].contiguous()
                    ce_total += F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    ).item()
                per_class[b, si] = (ce_total / len(chunks)) - ce_baselines[sname]

        gp.weight.data.copy_(gp_orig)
        up.weight.data.copy_(up_orig)
        importance[L] = {"per_class": per_class, "shard_names": shard_names}
        max_per_block = per_class.max(dim=1).values
        log(f"phase1_loo_multi_ffn: L{L:02d} per-shard ΔCE: " + ", ".join(
            f"{sname}=[{per_class[:,si].min():.3f},{per_class[:,si].max():.3f}]"
            for si, sname in enumerate(shard_names)
        ) + f" | max-across-class min={max_per_block.min():.3f} max={max_per_block.max():.3f}")
    log(f"phase1_loo_multi_ffn: importance computed for {len(importance)} layers × "
        f"{n_blocks} blocks × {n_shards} shards = "
        f"{len(importance) * n_blocks * n_shards} cells")
    return importance


def lstsq_refit_down_proj(
    down_proj,  # nn.Linear: in_features=intermediate_size, out_features=hidden_size
    x_actual: torch.Tensor,   # [B, T, intermediate], with dropped channels == 0
    y_target: torch.Tensor,   # [B, T, hidden] pre-prune down_proj output
    keep_cols: list[int],
    ridge_rel: float = 1e-2,
) -> dict:
    """lstsq heal for FFN. Same math as lstsq_refit_o_proj, applied to
    down_proj's input columns (kept channels) and output rows (hidden).
    """
    return lstsq_refit_o_proj(down_proj, x_actual, y_target,
                              keep_cols=keep_cols, ridge_rel=ridge_rel)


# ---------- W&B logging (optional) ----------
#
# Backward-compatible: if --wandb-project is unset OR WANDB_API_KEY is empty
# OR the wandb package is missing, _init_wandb returns a stub object whose
# .log()/.summary calls are silent no-ops. Same pattern as the CTD trainer.

class _WandbStub:
    """No-op shim: every attribute access is a callable that does nothing."""
    summary: dict = {}

    def log(self, *a, **kw):  # pragma: no cover
        pass

    def finish(self, *a, **kw):  # pragma: no cover
        pass

    def __bool__(self):  # for `if wb:` truthiness checks
        return False


def _init_wandb(args):
    """Return a wandb run object or a no-op stub.

    Disabled when --wandb-project is None, when WANDB_API_KEY is unset/empty,
    or when import fails. Logs a one-line status either way so the user
    knows whether metrics are flowing.
    """
    if not getattr(args, "wandb_project", None):
        log("wandb: disabled (no --wandb-project)")
        return _WandbStub()
    # Auth: env var OR ~/.netrc entry for api.wandb.ai. If neither is present,
    # wandb.init will prompt interactively — which would hang our headless run,
    # so we refuse to init and warn.
    if not (os.environ.get("WANDB_API_KEY")
            or (Path.home() / ".netrc").exists()):
        log("wandb: disabled (no WANDB_API_KEY and no ~/.netrc)")
        return _WandbStub()
    try:
        import wandb  # type: ignore
    except Exception as e:  # pragma: no cover
        log(f"wandb: import failed ({e}); disabled")
        return _WandbStub()
    name = args.wandb_name or Path(args.output).name
    tags = (
        [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        if args.wandb_tags else None
    )
    cfg = {
        "model_path": args.model_path,
        "output": args.output,
        "prune_frac": args.prune_frac,
        "prune_mode": args.prune_mode,
        "prune_layers": args.prune_layers,
        "phase1_mode": args.phase1_mode,
        "phase1_multi_strategy": args.phase1_multi_strategy,
        "phase1_shard_corpora": args.phase1_shard_corpora,
        "ridge": args.ridge,
        "heal": args.heal,
        "lora_steps": args.lora_steps,
        "lora_lr": args.lora_lr,
        "lora_weight_decay": args.lora_weight_decay,
        "calib_tokens": args.calib_tokens,
        "chunk_tokens": args.chunk_tokens,
        "canary_threshold": args.canary_ratio_threshold,
        "canary_n_gen": args.canary_n_gen,
        "no_canary": args.no_canary,
        "allow_low_kv_structural": args.allow_low_kv_structural,
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=name,
        tags=tags,
        config=cfg,
        reinit=True,
    )
    log(f"wandb: enabled — project={args.wandb_project} run={name} url={run.url}")
    return run


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prune-frac", type=float, default=0.125,
                    help="fraction of Q heads to drop per layer (uniform)")
    ap.add_argument("--calib-file", required=True,
                    help="path to a plain-text calibration corpus")
    ap.add_argument("--calib-tokens", type=int, default=8192)
    ap.add_argument("--chunk-tokens", type=int, default=1024)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--phase1-window-K", type=int, default=10,
                    help="Backward window size for phase 1 windowed α-grad. "
                         "Per-layer L scoring backward only flows through "
                         "layers L..L+K (forward goes all the way to compute CE). "
                         "K=10 covers ~80-95%% of cross-layer signal in a "
                         "60-layer dense transformer; K=N would be the global "
                         "α-grad (infeasible on 24GB for 31B at bf16).")
    ap.add_argument("--phase1-nf4-chunk-tokens", type=int, default=192,
                    help="Per-chunk size for nf4 phase 1b forward+backward. "
                         "Memory scales linearly: 192 tokens ≈ 21.7 GiB peak "
                         "on RTX 3090 (1 GiB headroom). 256+ OOMs.")
    ap.add_argument("--phase1-mode", choices=["windowed", "nf4_global", "two_tier", "loo", "loo_multi"],
                    default="nf4_global",
                    help="Importance scoring strategy. "
                         "windowed = K-windowed bf16 only (no long-tail). "
                         "nf4_global = single nf4-quantized full backward (long-tail with quant noise). "
                         "two_tier = both, normalized + weighted sum (defensive, slower). "
                         "loo = leave-one-out (Fix A): for each (layer, head) zero the "
                         "head and measure ΔCE on calibration. Architecture-agnostic, "
                         "directly measures what matters; more expensive (~n_q × full "
                         "forwards per layer) but cheap on small models. "
                         "loo_multi = T7.4 multi-class LOO: per-shard ΔCE, with selection "
                         "rule = max-across-shards (anti-specialist filter). Requires "
                         "--phase1-shard-corpora to define the shards. Reusable signal: "
                         "the per-(layer, head, class) tensor is also useful for FFN-prune, "
                         "expert-drop, and CD differential quantization.")
    ap.add_argument("--phase1-shard-corpora", default=None,
                    help="(loo_multi only) comma-separated 'name:path' pairs of shard "
                         "corpora. Each path is a plain-text file; per-shard tokenization "
                         "uses --calib-tokens / --chunk-tokens divided by n_shards. "
                         "Example: 'code:/path/code.txt,prose:/path/prose.txt,math:/path/math.txt'. "
                         "If omitted, falls back to single-axis LOO with the main --calib-file.")
    ap.add_argument("--phase1-multi-strategy",
                    choices=["max_class", "positive_max", "mean_class"],
                    default="max_class",
                    help="(loo_multi only) aggregation rule from per-class ΔCE to "
                         "per-head scalar. max_class (default) = anti-specialist filter, "
                         "drops heads whose worst-case class impact is smallest. "
                         "positive_max = same but clamps negative ΔCE to 0 first (treats "
                         "removal-helps as zero-importance, not negative). "
                         "mean_class = degenerate to single-axis LOO (avg across shards).")
    ap.add_argument("--phase1-combine-lambda", type=float, default=1.0,
                    help="Weight on the nf4-global term in two_tier combine. "
                         "imp = imp_K_local_norm + lambda * imp_full_nf4_norm")
    ap.add_argument("--imp-cache", default=None,
                    help="Optional path to cache imp_full_nf4 (so we can re-run "
                         "phase 1a + phase 2 without re-running nf4 phase 1b)")
    ap.add_argument("--phase2-max-chunks", type=int, default=1,
                    help="Use only the first N calib chunks for the per-layer "
                         "x_actual capture in phase 2 (cost grows linearly with this). "
                         "Phase-0 targets are sliced to match. 1 chunk × 1024 tokens "
                         "is plenty for a stable lstsq fit with ridge regularization.")
    ap.add_argument("--placement", choices=["cpu", "gpu", "auto"], default="cpu",
                    help="cpu = full model on CPU RAM (slow but safe, no meta tensors); "
                         "gpu = full model on a single GPU (device_map={'':'cuda:0'}); "
                         "fastest and bug-127-safe when VRAM fits the whole BF16 model "
                         "— required for any in-place weight modification flow (mask "
                         "prune + lstsq heal). Use for small dense models locally or "
                         "cloud pods (A100 80GB / H100 80GB) where 31B BF16 fits. "
                         "auto = device_map=auto with GPU+CPU offload (fast forward but "
                         "BUG-127: accelerate offload silently discards in-place writes "
                         "to CPU-offloaded layers at save_pretrained time. Only use for "
                         "read-only flows like phase-0 target capture.)")
    ap.add_argument("--gpu-mem", default="22GiB",
                    help="(auto-placement only) GPU memory budget for accelerate device_map")
    ap.add_argument("--cpu-mem", default="200GiB")
    ap.add_argument("--drop-kv-groups", action="store_true",
                    help="also drop KV heads when whole GQA group is pruned (off by default)")
    ap.add_argument("--prune-mode", choices=["auto", "mask", "group-aware"], default="auto",
                    help="auto = inspect model config and pick the best strategy: use "
                         "group-aware (with physical reshape) when the requested keep "
                         "target satisfies n_q_new %% n_kv == 0 for both sliding+full "
                         "layer types; otherwise fall back to mask. Handles E2B (n_kv=1, "
                         "free Q-drop with reshape), 31B (n_kv=16, paired KV-aligned "
                         "drop with reshape), and unusual fractions where neither fits "
                         "(legacy zero-rows mask, no reshape). "
                         "mask = force legacy zero-pad pruning (model stays full size on "
                         "disk, useful for accuracy-only ablations). "
                         "group-aware = force structural prune: select Q-heads in KV-aligned "
                         "groups for sliding layers, free Q drop on full layers (KV kept). "
                         "Tensors are physically resized AFTER lstsq heal and config is "
                         "rewritten. Errors out if constraints aren't satisfiable.")
    ap.add_argument("--target-n-q-keep", type=int, default=None,
                    help="(group-aware) target number of Q heads to keep per layer (uniform). "
                         "If omitted, derived from --prune-frac. Must satisfy "
                         "n%%2==0 (sliding alignment) and n%%num_global_kv==0 (full alignment).")
    ap.add_argument("--smoke", action="store_true",
                    help="only prune layer 0 (sliding) and layer 5 (full) — sanity check")
    ap.add_argument("--smoke-tokens", type=int, default=256)
    ap.add_argument("--prune-layers", default="all",
                    help="'all' (default) prunes every decoder layer with self-attn. "
                         "Otherwise comma-separated layer indices to restrict pruning to "
                         "(e.g. '0,1,2,3' for the R1 shallow-only-tuning-pass hypothesis). "
                         "Phase-0 target capture still runs across all layers (cheap); "
                         "only phase-2 prune+heal is restricted. Mutually informative with "
                         "--smoke (which is a fixed 2-layer subset).")
    # Fix B (2026-05-09): refuse structural reshape on n_kv=1 architectures
    # by default. T7.2 E2B catastrophic showed it cannot be healed.
    ap.add_argument("--allow-low-kv-structural", action="store_true",
                    help="Override Fix-B refusal: permit group-aware (structural reshape) "
                         "even on n_kv_s == 1 architectures (E2B-style). Catastrophically "
                         "broke E2B he125-E in T7.2; only enable if you have evidence the "
                         "specific architecture survives. Fix-C2 canary still gates the save.")
    # Fix C2 (2026-05-09): autoregressive coherence gate before save.
    ap.add_argument("--no-canary", action="store_true",
                    help="Disable Fix-C2 AR-NLL coherence gate (NOT recommended). The gate "
                         "captures greedy gens with the unpruned model at phase 0, then "
                         "after lstsq heal compares pruned-NLL vs base-NLL of the same "
                         "tokens. Ratio > --canary-ratio-threshold → refuse save.")
    ap.add_argument("--canary-ratio-threshold", type=float, default=3.0,
                    help="Max allowed worst-prompt ratio of pruned_NLL/base_NLL on canary "
                         "gens. 3.0 = base text is at most ~e^3 ≈ 20× less likely under "
                         "the pruned model. Tighten to 2.0 for safety, loosen to 4-5 to "
                         "tolerate larger fracs at cost of false negatives.")
    ap.add_argument("--canary-n-gen", type=int, default=50,
                    help="Greedy decode length for each canary prompt. Longer = more "
                         "sensitive to drift but slower. 50 catches token-loop collapse.")
    ap.add_argument("--canary-baseline-cache", default=None,
                    help="Path to cache the unpruned-model AR canary baseline "
                         "(prompts + n_gen + per-prompt {input_ids, gen_ids, base_nll_mean}). "
                         "First run captures+writes; subsequent runs with same prompts/n_gen "
                         "load and skip the ~17min capture under accelerate offload on 31B. "
                         "Cache is auto-invalidated if prompts or n_gen differ.")
    # Phase 2.5 runtime: where & in what dtype the AR canary runs. The recipe
    # materializes the full BF16 model on CPU before save (phase3a), and prior
    # to this flag the canary inherited that placement — meaning a 31B model
    # doing greedy decode strictly single-thread on CPU due to torch<2.6 BF16
    # CPU matmul falling back to a single MKL thread. On ssh3.vast.ai:10024
    # 2026-05-11 this turned a 30-90s GPU canary into ~25 minutes of wallclock
    # at 100% on one core out of 128. These knobs separate canary placement
    # from save placement.
    ap.add_argument("--canary-device", choices=["auto", "gpu", "cpu"], default="auto",
                    help="Where to run the Phase-2.5 AR canary forward + greedy decode. "
                         "'auto' = GPU with accelerate offload if CUDA is available, else CPU. "
                         "'gpu' = force GPU offload (errors if unavailable). "
                         "'cpu' = force CPU (always works; slow on BF16 with torch<2.6 — "
                         "consider --canary-dtype fp32 to recover multi-thread CPU matmul).")
    ap.add_argument("--canary-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="Dtype the canary forward uses. 'auto' picks BF16 on GPU (native) and "
                         "FP32 on CPU if free RAM ≥ 2.2× model size (FP32 CPU matmul is fully "
                         "multi-threaded in torch, BF16 is not until 2.6), falling back to BF16 "
                         "when CPU RAM is tight. Manual override for big-model-tight-RAM cases. "
                         "Cast is reverted to the saved model's native dtype before save.")
    ap.add_argument("--canary-gpu-mem-frac", type=float, default=0.85,
                    help="When --canary-device=gpu (or auto-picks gpu), fraction of total VRAM "
                         "to budget for the canary offload device_map. Defaults to 0.85 "
                         "(conservative; leaves room for activations + KV cache).")
    ap.add_argument("--canary-gpu-mem-gib", type=float, default=None,
                    help="Explicit GPU budget (GiB) for canary offload — overrides "
                         "--canary-gpu-mem-frac. Use when 0.85×VRAM autocalc is wrong for "
                         "your model (e.g. 31B on 24 GiB needs explicit 20 GiB to leave KV room).")
    ap.add_argument("--canary-cpu-mem", default=None,
                    help="CPU memory budget for canary offload (e.g. '200GiB'). "
                         "Defaults to --cpu-mem.")
    # ---- Resumable checkpoints (opt-in; covers phase0 + phase2 + pre-canary) ----
    # Burned a couple of hours on 2026-05-11 redoing 30+ min of phase 0' capture
    # and 30+ min of phase 2 lstsq heal after the AR canary stalled on a
    # single-threaded BF16 CPU forward. With --checkpoint-dir + --save-before-canary
    # the same restart becomes ~1 min (load staged BF16, re-run canary on GPU).
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Directory for resumable checkpoints. Disabled when unset. "
                         "Contains manifest.json (run fingerprint), phase2/layer_NN.pt "
                         "(default ON), optional phase0/layer_NN.pt (--checkpoint-phase0), "
                         "and optional staged/ (--save-before-canary).")
    ap.add_argument("--checkpoint-phase0", action="store_true",
                    help="Also checkpoint per-layer Phase 0 captures (o_proj outputs + "
                         "residual stream). ~20 GB disk for 31B but lets the whole calib "
                         "forward be skipped on restart. Off by default — usually the "
                         "staged/ + phase2/ checkpoints are enough.")
    ap.add_argument("--save-before-canary", action="store_true",
                    help="Save the full BF16 model under <checkpoint-dir>/staged/ AFTER "
                         "phase3b reshape but BEFORE the AR canary check. On a fresh "
                         "restart with the same checkpoint-dir, load() jumps directly to "
                         "phase 2.5 — bypassing phase 0/2 entirely. Disk: ~62 GB for 31B.")
    ap.add_argument("--resume", choices=["auto", "fresh", "force"], default="auto",
                    help="auto: reuse checkpoint dir if fingerprint matches the current "
                         "run config (default). fresh: ignore any existing checkpoint dir "
                         "(forces recompute; old checkpoint is left in place). force: "
                         "reuse checkpoint dir without fingerprint check (debug only — "
                         "you certify the saved data matches your current config).")
    # T13: LoRA-as-correction heal (alternative to lstsq).
    # T7.7: ar-lstsq — same lstsq math but with AR-distributed (X, Y) target pair.
    # T7.8: noheal — pure mask, no o_proj refit (informed by T13b/T7.6/T7.7 finding
    #       that every TF/AR/LoRA target damages shallow attention).
    ap.add_argument("--heal",
                    choices=["lstsq", "ar-lstsq", "noheal",
                             "lora-r1", "lora-r2", "lora-r4",
                             "lora-r8", "lora-r16", "lora-r32"],
                    default="lstsq",
                    help="Phase 2 heal kind. 'lstsq' (default) is the linear projection refit "
                         "used in T8/T11/T7.x. 'ar-lstsq' (T7.7) uses the same math but builds "
                         "(X, Y) on the masked student's AR rollouts vs teacher activations on "
                         "the same prefix+gen sequences — fixes the TF-distribution mismatch that "
                         "broke T13b lstsq on shallow attention. 'noheal' (T7.8) skips the heal "
                         "entirely — pure mask + saved kept-Q-cols of W_orig; informed by T13b/"
                         "T7.7 finding that every local correction damages E4B shallow attn. "
                         "'lora-rN' adds a trained rank-N correction to the original kept-cols "
                         "weights via Adam; folds into proj.weight at save time so on-disk "
                         "shape is identical to lstsq (llama.cpp-clean).")
    ap.add_argument("--ar-rollout-seeds", type=int, default=8,
                    help="(T7.7) Number of calib chunks to use as rollout prefixes for ar-lstsq.")
    ap.add_argument("--ar-rollout-gen", type=int, default=128,
                    help="(T7.7) max_new_tokens for each ar-lstsq rollout. Cap at 32-64 if the "
                         "student loops fast; 128 works at L4-L7 12.5%%.")
    ap.add_argument("--ar-rollout-prefix", type=int, default=64,
                    help="(T7.7) Prefix length sliced from each calib chunk before generation. "
                         "Total seq length per sample = prefix + ar-rollout-gen.")
    ap.add_argument("--lora-steps", type=int, default=200,
                    help="Adam steps for LoRA heal. 200 is a reasonable default for E4B; "
                         "raise to 400-500 for 31B if rel_resid plateaus high.")
    ap.add_argument("--lora-lr", type=float, default=1e-3,
                    help="Adam learning rate for LoRA heal. 1e-3 with weight_decay=1e-4 "
                         "matches PEFT defaults. Drop to 5e-4 if loss oscillates.")
    ap.add_argument("--lora-weight-decay", type=float, default=1e-4,
                    help="Adam weight decay for LoRA heal — acts as ridge stabilizer.")
    # T7.6 FFN-prune flags — disabled by default (frac=0.0). When set, runs in
    # parallel to attention prune (independent layer sets, independent heal).
    ap.add_argument("--ffn-prune-frac", type=float, default=0.0,
                    help="Fraction of FFN intermediate channels to prune per layer "
                         "(0 = disabled, T7.6 OFF). Selection is block-wise at "
                         "--ffn-block-size granularity.")
    ap.add_argument("--ffn-block-size", type=int, default=128,
                    help="Block size for FFN block-LOO importance and pruning. "
                         "intermediate_size must be divisible by this.")
    ap.add_argument("--ffn-prune-layers", default="all",
                    help="Same syntax as --prune-layers but for the FFN pass.")
    ap.add_argument("--ffn-heal",
                    choices=["noheal", "lstsq", "lora-r1", "lora-r2", "lora-r4",
                             "lora-r8", "lora-r16", "lora-r32"],
                    default="noheal",
                    help="FFN heal strategy. Default 'noheal' — T13c proved TF-fit "
                         "heal is harmful on shallow attn; FFN starts from the same "
                         "prior. Use lstsq/lora-* for explicit ablations.")
    ap.add_argument("--ffn-phase1-mode", choices=["loo", "loo_multi"],
                    default="loo",
                    help="FFN importance scoring. loo_multi uses --phase1-shard-corpora.")
    # Wandb logging — optional, no-op if --wandb-project unset or WANDB_API_KEY empty.
    ap.add_argument("--wandb-project", default=None,
                    help="W&B project name. Omit (or unset WANDB_API_KEY) to disable.")
    ap.add_argument("--wandb-entity", default=None,
                    help="W&B entity (user/team). Optional.")
    ap.add_argument("--wandb-name", default=None,
                    help="W&B run name. Defaults to basename of --output.")
    ap.add_argument("--wandb-tags", default=None,
                    help="Comma-separated W&B tags, e.g. 'gemma4,head-prune,T7.4'.")
    args = ap.parse_args()

    # ---- Resume state (populated below if --checkpoint-dir set) ----
    # Attached to `args` so downstream code can introspect without threading
    # a new positional parameter through every callee.
    args._ckpt_dir = None        # Path | None
    args._ckpt_fp = None         # current-run fingerprint
    args._ckpt_phase0_layers = set()  # which phase0 layers are already on disk
    args._ckpt_phase2_layers = set()  # which phase2 layers are already on disk
    args._ckpt_staged_ready = False   # whether staged/ has a complete BF16 model

    if args.checkpoint_dir:
        cdir = Path(args.checkpoint_dir)
        fp_now = _ckpt_fingerprint(args)
        manifest = _read_manifest(cdir) if args.resume != "fresh" else None
        if args.resume == "fresh" and cdir.exists():
            log(f"checkpoint: --resume fresh — ignoring existing dir {cdir} "
                "(left in place; rm manually if you want disk back)")
        if manifest is not None:
            saved_fp = manifest.get("fingerprint", "")
            if args.resume == "force":
                log(f"checkpoint: --resume force — reusing {cdir} without "
                    f"fingerprint check (saved={saved_fp} current={fp_now}). "
                    "Caller certifies the saved data matches the current config.")
            elif saved_fp != fp_now:
                log(f"checkpoint: fingerprint mismatch (saved={saved_fp} "
                    f"current={fp_now}) — REFUSING TO RESUME. Either pass "
                    "--resume fresh to start over, or pass --resume force if "
                    "you know the saved data is still valid for this run.")
                sys.exit(2)
            else:
                log(f"checkpoint: resuming from {cdir} (fp={fp_now})")
            # Inventory what's already on disk
            p0_dir = cdir / "phase0"
            if p0_dir.exists():
                args._ckpt_phase0_layers = {
                    int(p.stem.split("_")[1]) for p in p0_dir.glob("layer_*.pt")
                }
            p2_dir = cdir / "phase2"
            if p2_dir.exists():
                args._ckpt_phase2_layers = {
                    int(p.stem.split("_")[1]) for p in p2_dir.glob("layer_*.pt")
                }
            args._ckpt_staged_ready = _staged_complete(cdir)
            log(f"checkpoint: phase0_layers={len(args._ckpt_phase0_layers)} "
                f"phase2_layers={len(args._ckpt_phase2_layers)} "
                f"staged_ready={args._ckpt_staged_ready}")
        else:
            log(f"checkpoint: starting fresh in {cdir} (fp={fp_now})")
        args._ckpt_dir = cdir
        args._ckpt_fp = fp_now
        # Write/refresh the manifest so subsequent saves can update it.
        _write_manifest(cdir, {
            "version": _CHECKPOINT_VERSION,
            "fingerprint": fp_now,
            "model_path": str(args.model_path),
            "prune_frac": args.prune_frac,
            "heal": args.heal,
        })

    # Parse --prune-layers into a set or None for 'all'.
    if args.prune_layers == "all":
        args._prune_layer_set = None
    else:
        try:
            args._prune_layer_set = {int(x) for x in args.prune_layers.split(",") if x.strip()}
        except ValueError as e:
            raise ValueError(f"--prune-layers must be 'all' or comma-separated ints, got {args.prune_layers!r}") from e
        if not args._prune_layer_set:
            raise ValueError("--prune-layers parsed to empty set; pass 'all' or non-empty list")
    # T7.6: parse --ffn-prune-layers similarly.
    if args.ffn_prune_layers == "all":
        args._ffn_prune_layer_set = None
    else:
        try:
            args._ffn_prune_layer_set = {int(x) for x in args.ffn_prune_layers.split(",") if x.strip()}
        except ValueError as e:
            raise ValueError(f"--ffn-prune-layers must be 'all' or comma-separated ints, got {args.ffn_prune_layers!r}") from e
        if not args._ffn_prune_layer_set:
            raise ValueError("--ffn-prune-layers parsed to empty set; pass 'all' or non-empty list")

    # Wandb init — opt-in, silent no-op if disabled. Falls back to a stub object
    # so the rest of main() can call wb.log(...) unconditionally.
    wb = _init_wandb(args)

    import gc

    def _load_bf16_model():
        log(f"loading {args.model_path} (placement={args.placement})")
        if args.placement == "cpu":
            m = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
            )
        elif args.placement == "gpu":
            # Full single-GPU placement — no offload, no hooks. In-place writes
            # to module.weight.data persist trivially through save_pretrained.
            # User is responsible for VRAM budget: caller must ensure the BF16
            # model fits (Gemma 4 31B ≈ 58 GB BF16 → needs A100 80GB / H100 80GB
            # for 31B; 4B/8B dense fits 24GB cards; smaller models fit anything).
            if not torch.cuda.is_available():
                raise RuntimeError("--placement gpu requested but CUDA not available")
            m = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
            )
            log(f"  GPU={torch.cuda.memory_allocated()/(1024**3):.2f} GiB / "
                f"{torch.cuda.get_device_properties(0).total_memory/(1024**3):.0f} GiB")
        else:  # auto
            m = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: args.gpu_mem, "cpu": args.cpu_mem},
                low_cpu_mem_usage=True,
            )
        m.config.use_cache = False
        if hasattr(m.config, "text_config"):
            m.config.text_config.use_cache = False
        m.eval()
        return m

    tok = AutoTokenizer.from_pretrained(args.model_path)

    # NOTE on staged/ usage (v1 scope):
    #   --save-before-canary writes <ckpt>/staged/ with the full post-heal BF16
    #   model so a canary stall/crash never costs the phase 0/2 work. Auto-resume
    #   FROM staged (skipping phase 0/1/2/3 wholesale) is v2 work — for now, on
    #   a crash after phase 3, manually copy staged/ to the final output dir or
    #   point a follow-up tool at it. The per-layer phase0 + phase2 checkpoints
    #   handle the more common mid-phase crashes.
    args._skip_to_canary = False
    if args._ckpt_dir is not None and args._ckpt_staged_ready:
        log(f"checkpoint: staged/ is on disk at {_staged_dir(args._ckpt_dir)}. "
            "Auto-resume from staged is not wired in v1 — proceeding with full "
            "phase 0/1/2/3 (per-layer checkpoints will accelerate where they exist).")

    # Build calibration BEFORE loading any model — the calibration is just
    # tokenized text, doesn't need the model.
    calib_total = args.smoke_tokens if args.smoke else args.calib_tokens
    calib_chunk = min(args.chunk_tokens, args.smoke_tokens) if args.smoke else args.chunk_tokens
    calib_chunks = build_calib(tok, calib_total, calib_chunk, args.calib_file)

    # ---- Phase 1b: nf4 global α-grad importance ----
    # When in nf4_global / two_tier modes, we run this FIRST so the GPU is
    # clean (no leftover allocator residue from a prior bf16 load). After
    # nf4 phase 1b finishes and unloads, GPU is freed and we then load bf16
    # for phase 0/1a/2.
    imp_full = None
    if args.phase1_mode in ("nf4_global", "two_tier"):
        if args.imp_cache and Path(args.imp_cache).exists():
            log(f"phase1_nf4: loading cached importance from {args.imp_cache}")
            imp_full = {int(k): v for k, v in torch.load(args.imp_cache).items()}
        else:
            imp_full = phase1_importance_nf4_global(
                args.model_path, calib_chunks,
                chunk_tokens=args.phase1_nf4_chunk_tokens,
            )
            if args.imp_cache:
                torch.save({str(k): v for k, v in imp_full.items()}, args.imp_cache)
                log(f"phase1_nf4: cached imp_full to {args.imp_cache}")
        # Aggressive cleanup before bf16 load
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Load bf16 model for phase 0 (target capture), phase 1a (windowed
    # importance, optional), and phase 2 (prune + lstsq heal)
    model = _load_bf16_model()
    layers = get_layers(model)
    cfg = text_config(model)
    log(f"model loaded: {len(layers)} layers, {cfg.num_attention_heads} Q heads")
    log(f"layer_types: full={sum(1 for x in cfg.layer_types if x=='full_attention')} "
        f"sliding={sum(1 for x in cfg.layer_types if x=='sliding_attention')}")

    # Phase 0 — captures o_proj outputs (phase 2 attn healing targets), decoder
    # residual-stream outputs (phase 1a L2 target), and (when FFN prune is on)
    # down_proj outputs (phase 2 FFN healing targets).
    _capture_mlp = args.ffn_prune_frac > 0 and args.ffn_heal != "noheal"
    o_targets, h_targets, mlp_targets = phase0_capture(
        model, calib_chunks, capture_mlp=_capture_mlp,
        ckpt_dir=args._ckpt_dir,
        ckpt_phase0=args.checkpoint_phase0,
        resume_layers=args._ckpt_phase0_layers,
    )
    p2_chunks = calib_chunks[: max(1, args.phase2_max_chunks)]
    p2_tokens = sum(c.shape[1] for c in p2_chunks)
    log(f"phase2 will use {len(p2_chunks)} chunk(s) = {p2_tokens} tokens for x_actual capture + lstsq")
    targets = {L: y[:, :p2_tokens, :] for L, y in o_targets.items()}
    ffn_targets = {L: y[:, :p2_tokens, :] for L, y in mlp_targets.items()}

    # Fix C2: capture greedy canary baseline with the unpruned model. This is
    # the reference distribution against which the post-heal model will be
    # judged for autoregressive coherence. Skip on --smoke or --no-canary.
    #
    # T8noheal-scope: under accelerate offload on a 31B model, the canary
    # baseline AR generation costs ~6 min/prompt × 3 prompts = ~17 min per
    # run. Across a multi-run sweep on the same base model with the same
    # CANARY_PROMPTS + canary_n_gen, this is identical work — cache it.
    canary_baseline = None
    if not args.smoke and not args.no_canary:
        cache_path = args.canary_baseline_cache or None
        loaded_from_cache = False
        if cache_path:
            try:
                from pathlib import Path as _P
                if _P(cache_path).is_file():
                    cb = torch.load(cache_path, map_location="cpu", weights_only=False)
                    # Validate: prompts + n_gen must match what we're about to use.
                    if (cb.get("prompts") == list(CANARY_PROMPTS)
                            and cb.get("n_gen") == args.canary_n_gen
                            and isinstance(cb.get("baseline"), list)
                            and len(cb["baseline"]) == len(CANARY_PROMPTS)):
                        canary_baseline = cb["baseline"]
                        loaded_from_cache = True
                        log(f"phase0+: loaded canary baseline from cache {cache_path} "
                            f"({len(canary_baseline)} prompts) — skipping AR capture")
                    else:
                        log(f"phase0+: cache {cache_path} mismatch (prompts/n_gen "
                            f"differ) — recomputing")
            except Exception as e:
                log(f"phase0+: failed to load canary cache {cache_path!r}: {e!r} — "
                    f"recomputing")
        if canary_baseline is None:
            try:
                log(f"phase0+: capturing AR canary baseline ({len(CANARY_PROMPTS)} prompts × "
                    f"{args.canary_n_gen} gen tokens) on unpruned model")
                canary_baseline = capture_canary_baseline(
                    model, tok, CANARY_PROMPTS, n_gen=args.canary_n_gen
                )
            except Exception as e:
                log(f"warn: canary baseline capture failed: {e!r} — gate will be skipped")
                canary_baseline = None
            # Persist on first successful capture.
            if cache_path and canary_baseline is not None and not loaded_from_cache:
                try:
                    from pathlib import Path as _P
                    _P(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "prompts": list(CANARY_PROMPTS),
                        "n_gen": args.canary_n_gen,
                        "baseline": canary_baseline,
                    }, cache_path)
                    log(f"phase0+: wrote canary baseline cache → {cache_path}")
                except Exception as e:
                    log(f"phase0+: failed to write canary cache {cache_path!r}: {e!r}")

    # Phase 1a (windowed L2) — runs unless mode is nf4_global / loo only
    if args.phase1_mode in ("windowed", "two_tier"):
        imp_local = phase1_importance_windowed(
            model, calib_chunks, window_K=args.phase1_window_K, h_targets=h_targets
        )
    else:
        imp_local = None

    # Fix A: LOO mode runs after Phase 0 capture; baseline CE is measured
    # once with the unpruned model, then per-head zero+forward gives ΔCE.
    if args.phase1_mode == "loo" and args.prune_frac > 0:
        ce_baseline = measure_calib_ce(model, calib_chunks)
        log(f"phase1_loo: baseline CE on calibration = {ce_baseline:.4f}")
        imp_loo = phase1_importance_loo(model, calib_chunks, ce_baseline=ce_baseline)
    else:
        imp_loo = None
        if args.phase1_mode == "loo" and args.prune_frac == 0:
            log("phase1_loo: skipped (--prune-frac 0, FFN-only run)")

    # T7.4: multi-class LOO (loo_multi). Per-shard ΔCE → max-across-shards
    # selection. Requires --phase1-shard-corpora; falls back to single-axis
    # LOO with a warning if missing.
    imp_loo_multi_raw = None  # raw per-class tensor, kept for manifest
    imp_loo_multi = None      # aggregated per-head scalar via strategy
    if args.phase1_mode == "loo_multi":
        if not args.phase1_shard_corpora:
            log("phase1_mode=loo_multi but --phase1-shard-corpora missing — "
                "falling back to single-axis LOO on main calib")
            ce_baseline = measure_calib_ce(model, calib_chunks)
            log(f"phase1_loo (fallback): baseline CE = {ce_baseline:.4f}")
            imp_loo = phase1_importance_loo(model, calib_chunks, ce_baseline=ce_baseline)
            args.phase1_mode = "loo"  # reflect actual mode in manifest
        else:
            shard_specs = []
            for spec in args.phase1_shard_corpora.split(","):
                spec = spec.strip()
                if not spec:
                    continue
                if ":" not in spec:
                    raise ValueError(f"--phase1-shard-corpora entry {spec!r} missing ':'")
                name, path = spec.split(":", 1)
                shard_specs.append((name.strip(), path.strip()))
            log(f"phase1_loo_multi: {len(shard_specs)} shard(s): "
                + ", ".join(n for n, _ in shard_specs))
            # Per-shard token budget = total // n_shards (so total compute budget
            # matches single-axis LOO at first order; per-shard signal is noisier
            # but max-across-shards is what we need).
            per_shard_total = max(args.chunk_tokens, calib_total // max(1, len(shard_specs)))
            shards: dict[str, list[torch.Tensor]] = {}
            ce_baselines: dict[str, float] = {}
            for name, path in shard_specs:
                schunks = build_calib(tok, per_shard_total, args.chunk_tokens, path)
                shards[name] = schunks
                ce_baselines[name] = measure_calib_ce(model, schunks)
                log(f"  shard {name!r} ({path}): {len(schunks)} chunks × "
                    f"{args.chunk_tokens} tok = {sum(c.shape[1] for c in schunks)} tok, "
                    f"baseline CE={ce_baselines[name]:.4f}")
            imp_loo_multi_raw = phase1_importance_loo_multi(
                model, shards, ce_baselines
            )
            imp_loo_multi = aggregate_multi_class_importance(
                imp_loo_multi_raw, strategy=args.phase1_multi_strategy
            )
            log(f"phase1_loo_multi: aggregated per-head importance via "
                f"strategy={args.phase1_multi_strategy}")

    # Combine
    if args.phase1_mode == "two_tier":
        log(f"combining: imp = imp_K{args.phase1_window_K}_bf16_norm + "
            f"{args.phase1_combine_lambda} * imp_full_nf4_norm")
        importance = combine_importance(imp_local, imp_full, lam=args.phase1_combine_lambda)
    elif args.phase1_mode == "nf4_global":
        importance = imp_full
    elif args.phase1_mode == "loo":
        importance = imp_loo
    elif args.phase1_mode == "loo_multi":
        importance = imp_loo_multi  # aggregated per-head scalar via strategy
    else:  # windowed
        importance = imp_local

    # Phase 2
    n_q = cfg.num_attention_heads
    n_drop = int(round(n_q * args.prune_frac))
    log(f"phase2: dropping {n_drop}/{n_q} Q heads per layer ({args.prune_frac*100:.1f}%)")

    # Resolve prune strategy. `auto` inspects cfg + target keep count and picks
    # group-aware (with physical reshape) when feasible, else mask. Explicit
    # mask/group-aware bypass auto-resolution but still need targets computed.
    target_n_q_keep = None
    target_n_kv_keep_sliding = None
    first_kv_shared_idx = None  # set when model has KV-cache sharing (E2B-style)
    resolved_mode = args.prune_mode  # for manifest; tracks auto-resolution outcome
    if args.prune_mode in ("auto", "group-aware"):
        candidate_target = args.target_n_q_keep if args.target_n_q_keep is not None else (n_q - n_drop)
        resolved, group_size, target_n_kv_keep_sliding, first_kv_shared_idx = \
            auto_resolve_prune_mode(
                cfg, candidate_target,
                allow_low_kv_structural=args.allow_low_kv_structural,
            )
        resolved_mode = resolved
        if args.prune_mode == "auto":
            args.prune_mode = resolved
            log(f"prune_mode=auto resolved to '{resolved}' "
                f"(target_n_q_keep={candidate_target}, group_size_sliding={group_size}, "
                f"n_kv_sliding={cfg.num_key_value_heads}, "
                f"n_kv_full={cfg.num_global_key_value_heads if cfg.attention_k_eq_v else cfg.num_key_value_heads}, "
                f"kv_shared_from_layer={first_kv_shared_idx})")
            if resolved == "mask":
                log("  → constraints unsatisfiable for structural reshape; "
                    "falling back to mask (zero-rows, no reshape, no size win)")
        elif resolved == "mask" and args.prune_mode == "group-aware":
            raise ValueError(
                f"--prune-mode group-aware explicitly requested but constraints not satisfiable "
                f"for target_n_q_keep={candidate_target} given n_kv_sliding="
                f"{cfg.num_key_value_heads}, n_kv_full="
                f"{cfg.num_global_key_value_heads if cfg.attention_k_eq_v else cfg.num_key_value_heads}. "
                f"Adjust --prune-frac or --target-n-q-keep, or use --prune-mode auto."
            )

        if args.prune_mode == "group-aware":
            target_n_q_keep = candidate_target
            n_global_kv = cfg.num_global_key_value_heads if cfg.attention_k_eq_v else cfg.num_key_value_heads
            log(f"group-aware targets: n_q_keep={target_n_q_keep} "
                f"sliding_n_kv_keep={target_n_kv_keep_sliding} "
                f"full_n_kv_keep={n_global_kv} (unchanged) "
                f"group_size={group_size}")
            if first_kv_shared_idx is not None:
                log(f"  KV-shared layers: idx >= {first_kv_shared_idx} have dead k_proj/v_proj, "
                    f"will be skipped during K/V modification + reshape")

    smoke_layers = set()
    if args.smoke:
        smoke_layers = {0, 5}  # one sliding, one full
        log(f"SMOKE mode: pruning only layers {sorted(smoke_layers)}")

    pruned_layers = []
    refit_stats = {}
    keep_q_per_layer: dict[int, list[int]] = {}
    keep_kv_per_layer: dict[int, list[int]] = {}
    # T7.7: ar-lstsq defers per-layer heal until after the whole mask pass so the
    # AR rollout sees the fully-pruned student. Per-layer state we'll need later.
    ar_deferred: dict[int, dict] = {}
    for L, layer in enumerate(layers):
        if not has_self_attn(layer):
            continue
        if args.smoke and L not in smoke_layers:
            continue
        if args._prune_layer_set is not None and L not in args._prune_layer_set:
            continue
        if n_drop <= 0:
            break

        # Phase 2 resume: if a per-layer checkpoint exists, load its self_attn
        # state (which encodes the in-place prune + heal) and skip recompute.
        # Bypasses prune_q_heads_inplace + capture_proj_input_at_layer + lstsq.
        if (args._ckpt_dir is not None
                and L in args._ckpt_phase2_layers
                and args.heal != "ar-lstsq"):
            try:
                d = torch.load(_phase2_layer_path(args._ckpt_dir, L),
                               map_location="cpu", weights_only=True)
                layer.self_attn.load_state_dict(d["self_attn"], strict=True)
                refit_stats[L] = d["stats"]
                keep_q_per_layer[L] = d["keep_q"]
                keep_kv_per_layer[L] = d["keep_kv"]
                pruned_layers.append((L, d["lt"]))
                targets.pop(L, None)
                log(f"  L{L}: phase2 RESUMED from checkpoint "
                    f"(kept={d['stats'].get('kept')}/{d['stats'].get('in')} "
                    f"rel_resid={d['stats'].get('rel_resid')})")
                continue
            except Exception as _e:
                log(f"  L{L}: phase2 checkpoint load failed: {_e!r} — recomputing")

        head_dim, _, n_kv_layer, lt = layer_attn_geom(model, L)
        imp = importance[L]

        if args.prune_mode == "group-aware":
            keep_idx, keep_kv = select_keep_q_kv_group_aware(
                imp, n_q, n_kv_layer, lt,
                target_n_q_keep=target_n_q_keep,
                target_n_kv_keep_sliding=target_n_kv_keep_sliding,
            )
        else:
            keep_idx = imp.argsort(descending=True)[: n_q - n_drop].tolist()
            keep_kv = list(range(n_kv_layer))  # mask mode: don't touch KV (legacy)

        drop_idx = sorted(set(range(n_q)) - set(keep_idx))
        drop_kv_idx = sorted(set(range(n_kv_layer)) - set(keep_kv))
        log(f"layer {L:02d} [{lt}]: imp min={imp.min():.3e} max={imp.max():.3e} "
            f"drop_q={drop_idx} drop_kv={drop_kv_idx}")

        # Skip K/V modifications on KV-shared layers (E2B 15-34): k_proj/v_proj
        # exist but are dead at inference (KV is fetched from past_key_values
        # cache of an earlier non-shared layer). Q is still used → still pruned.
        is_kv_shared = (first_kv_shared_idx is not None and L >= first_kv_shared_idx)
        kv_keep_for_call = None if is_kv_shared else keep_kv
        # T7.7: snapshot W_orig BEFORE prune_q_heads_inplace mutates it.
        if args.heal == "ar-lstsq":
            ar_snap = _snapshot_layer_attn(layer)
        if args.prune_mode == "group-aware":
            prune_q_heads_inplace(model, L, keep_idx, keep_kv_heads=kv_keep_for_call)
        else:
            # Mask mode: still allow drop_kv_groups on non-shared layers.
            prune_q_heads_inplace(
                model, L, keep_idx,
                drop_kv_groups=(args.drop_kv_groups and not is_kv_shared),
            )

        keep_cols = []
        for h in keep_idx:
            keep_cols.extend(range(h * head_dim, (h + 1) * head_dim))

        if args.heal == "ar-lstsq":
            # T7.7: defer the heal until all layers are masked, then sample AR
            # rollouts and rebuild (X, Y) from the student's actual trajectory.
            ar_deferred[L] = {
                "snap": ar_snap,
                "keep_cols": keep_cols,
                "keep_idx": keep_idx,
                "kv_keep_for_call": kv_keep_for_call,
                "is_kv_shared": is_kv_shared,
                "drop_idx": drop_idx,
                "drop_kv_idx": drop_kv_idx,
                "head_dim": head_dim,
                "lt": lt,
            }
            # Bookkeeping that doesn't require heal stats yet
            pruned_layers.append((L, lt))
            keep_q_per_layer[L] = sorted(int(i) for i in keep_idx)
            keep_kv_per_layer[L] = sorted(int(i) for i in keep_kv)
            continue

        if args.heal == "noheal":
            # T7.8: pure mask, no o_proj refit. Skip x_actual capture entirely
            # — saves ~5 sec/layer on 60-layer 31B and leaves W_orig kept-cols
            # untouched (the prior T13b's rank-0 anchor proved superior).
            stats = {
                "kept": int(len(keep_cols)),
                "in": int(layer.self_attn.o_proj.in_features),
                "lam": 0.0,
                "rel_resid": None,
                "target_rms": None,
            }
            log(f"  L{L}: noheal kept={stats['kept']}/{stats['in']} (no refit)")
            wb.log({
                "phase2/layer": L,
                "phase2/kept": stats["kept"],
                "phase2/in": stats["in"],
                "phase2/n_drop_q": len(drop_idx),
                "phase2/n_drop_kv": len(drop_kv_idx),
            })
            targets.pop(L, None)
            pruned_layers.append((L, lt))
            refit_stats[L] = stats
            keep_q_per_layer[L] = sorted(int(i) for i in keep_idx)
            keep_kv_per_layer[L] = sorted(int(i) for i in keep_kv)
            continue

        x_actual = capture_proj_input_at_layer(model, p2_chunks, L)
        if args.heal == "lstsq":
            stats = lstsq_refit_o_proj(
                layer.self_attn.o_proj, x_actual, targets[L],
                keep_cols=keep_cols, ridge_rel=args.ridge,
            )
            log(f"  L{L}: lstsq kept={stats['kept']}/{stats['in']} "
                f"lam={stats['lam']:.3e} rel_resid={stats['rel_resid']:.4f} "
                f"target_rms={stats['target_rms']:.4f}")
        else:
            # T13: lora-rN — train rank-N correction via Adam, fold into o_proj.
            rank = int(args.heal.split("-r")[1])
            stats = lora_heal_o_proj(
                layer.self_attn.o_proj, x_actual, targets[L],
                keep_cols=keep_cols,
                rank=rank,
                n_steps=args.lora_steps,
                lr=args.lora_lr,
                weight_decay=args.lora_weight_decay,
            )
            _lf = stats.get('loss_first')
            _ll = stats.get('loss_last')
            _lf_s = f"{_lf:.4e}" if _lf is not None else "n/a"
            _ll_s = f"{_ll:.4e}" if _ll is not None else "n/a"
            log(f"  L{L}: lora-r{rank} kept={stats['kept']}/{stats['in']} "
                f"loss {_lf_s} → {_ll_s} "
                f"rel_resid={stats['rel_resid']:.4f} "
                f"target_rms={stats['target_rms']:.4f}")
        wb_log = {
            "phase2/layer": L,
            "phase2/rel_resid": stats["rel_resid"],
            "phase2/lam": stats["lam"],
            "phase2/kept": stats["kept"],
            "phase2/in": stats["in"],
            "phase2/target_rms": stats["target_rms"],
            "phase2/n_drop_q": len(drop_idx),
            "phase2/n_drop_kv": len(drop_kv_idx),
        }
        if "loss_first" in stats and stats["loss_first"] is not None:
            wb_log["phase2/lora_loss_first"] = stats["loss_first"]
            wb_log["phase2/lora_loss_last"] = stats["loss_last"]
            wb_log["phase2/lora_rank"] = stats.get("rank")
        wb.log(wb_log)
        # Free the per-layer target tensor — biggest CPU mem freed eagerly
        targets.pop(L, None)
        pruned_layers.append((L, lt))
        refit_stats[L] = stats
        keep_q_per_layer[L] = sorted(int(i) for i in keep_idx)
        keep_kv_per_layer[L] = sorted(int(i) for i in keep_kv)
        # Per-layer phase 2 checkpoint (default on whenever --checkpoint-dir is set).
        # Snapshot self_attn state_dict() — captures BOTH the in-place q/k/v_proj
        # prune from prune_q_heads_inplace AND the o_proj kept-cols overwrite from
        # the heal. On resume we load_state_dict() to reconstruct the layer.
        # Use _maybe_align to materialize accelerate-offloaded tensors before
        # .cpu() — state_dict() on an offloaded module returns meta tensors.
        if args._ckpt_dir is not None:
            with _maybe_align(layer.self_attn):
                sa_state = {
                    k: v.detach().to(dtype=v.dtype).cpu().clone()
                    for k, v in layer.self_attn.state_dict().items()
                }
            _atomic_torch_save({
                "self_attn": sa_state,
                "stats": stats,
                "keep_q": keep_q_per_layer[L],
                "keep_kv": keep_kv_per_layer[L],
                "drop_idx": drop_idx,
                "drop_kv_idx": drop_kv_idx,
                "lt": lt,
                "L": L,
            }, _phase2_layer_path(args._ckpt_dir, L))

    # ---- T7.7: AR-distributed lstsq deferred heal ----
    # All layers in `ar_deferred` are currently masked. Sample greedy rollouts
    # from this fully-masked student, then per-layer: restore W_orig → capture
    # teacher o_proj outputs (y_target_ar) → re-mask → capture o_proj inputs
    # (x_actual_ar) → solve lstsq.
    if args.heal == "ar-lstsq" and ar_deferred:
        log(f"phase2_ar: ar-lstsq deferred heal for {len(ar_deferred)} layer(s)")
        log(f"phase2_ar: sampling rollouts ({args.ar_rollout_seeds} seeds × "
            f"{args.ar_rollout_gen} gen tokens, prefix={args.ar_rollout_prefix}) "
            f"from fully-masked student")
        ar_seqs = ar_rollout_seqs(
            model, tok, p2_chunks,
            n_seed=args.ar_rollout_seeds,
            n_gen=args.ar_rollout_gen,
            n_prefix=args.ar_rollout_prefix,
        )
        ar_total_tokens = sum(int(s.shape[1]) for s in ar_seqs)
        log(f"phase2_ar: collected {len(ar_seqs)} rollout sequence(s), "
            f"{ar_total_tokens} total tokens")
        if tok is not None and len(ar_seqs) > 0:
            preview = tok.decode(ar_seqs[0][0].tolist(), skip_special_tokens=False)
            log(f"phase2_ar: rollout[0] head: {preview[:160]!r}")

        for L, info in ar_deferred.items():
            layer = layers[L]
            snap = info["snap"]
            keep_cols = info["keep_cols"]
            keep_idx = info["keep_idx"]

            # 1. Restore W_orig → teacher behavior on this layer.
            _restore_layer_attn(layer, snap)
            # 2. Capture teacher o_proj outputs on the AR rollouts.
            y_target_ar = capture_proj_output_at_layer(model, ar_seqs, L, target="o_proj")
            # 3. Re-mask the layer.
            if args.prune_mode == "group-aware":
                prune_q_heads_inplace(model, L, keep_idx,
                                      keep_kv_heads=info["kv_keep_for_call"])
            else:
                prune_q_heads_inplace(
                    model, L, keep_idx,
                    drop_kv_groups=(args.drop_kv_groups and not info["is_kv_shared"]),
                )
            # 4. Capture masked-student o_proj inputs on the SAME sequences.
            x_actual_ar = _capture_proj_input(model, ar_seqs, L, target="o_proj")
            # 5. Solve lstsq with the AR-distributed pair.
            stats = lstsq_refit_o_proj(
                layer.self_attn.o_proj, x_actual_ar, y_target_ar,
                keep_cols=keep_cols, ridge_rel=args.ridge,
            )
            # Log AR vs TF target divergence for diagnostic value.
            try:
                tf_y = targets.get(L)
                if tf_y is not None:
                    n_overlap = min(int(tf_y.shape[1]), int(y_target_ar.shape[1]))
                    if n_overlap > 0:
                        diff = (y_target_ar[:, :n_overlap, :] - tf_y[:, :n_overlap, :])
                        rel_div = float(diff.pow(2).mean().sqrt() /
                                        max(tf_y[:, :n_overlap, :].pow(2).mean().sqrt().item(), 1e-9))
                        stats["ar_vs_tf_rel"] = rel_div
                    else:
                        stats["ar_vs_tf_rel"] = None
            except Exception as _e:
                log(f"  L{L}: ar-vs-tf comparison skipped ({_e!r})")
                stats["ar_vs_tf_rel"] = None
            _div_s = (f"{stats['ar_vs_tf_rel']:.3f}"
                      if stats.get("ar_vs_tf_rel") is not None else "n/a")
            log(f"  L{L}: ar-lstsq kept={stats['kept']}/{stats['in']} "
                f"lam={stats['lam']:.3e} rel_resid={stats['rel_resid']:.4f} "
                f"target_rms={stats['target_rms']:.4f} ar_vs_tf={_div_s}")
            wb.log({
                "phase2/layer": L,
                "phase2/rel_resid": stats["rel_resid"],
                "phase2/lam": stats["lam"],
                "phase2/kept": stats["kept"],
                "phase2/in": stats["in"],
                "phase2/target_rms": stats["target_rms"],
                "phase2/n_drop_q": len(info["drop_idx"]),
                "phase2/n_drop_kv": len(info["drop_kv_idx"]),
                "phase2/ar_vs_tf_rel": stats.get("ar_vs_tf_rel"),
            })
            refit_stats[L] = stats
            targets.pop(L, None)
            # Free per-layer snapshot eagerly
            ar_deferred[L]["snap"] = None
            del y_target_ar, x_actual_ar
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del ar_seqs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- T7.6 Phase 1+2 FFN — runs only when --ffn-prune-frac > 0 ----
    # Structurally parallel to attention's phase 1 (LOO) + phase 2 (mask + heal)
    # but operates on FFN intermediate channel BLOCKS instead of Q heads.
    # Mask mode only in this revision; structural reshape deferred (T7.6+).
    ffn_imp_loo: dict[int, torch.Tensor] | None = None
    ffn_imp_loo_multi_raw: dict[int, dict] | None = None
    ffn_keep_per_layer: dict[int, list[int]] = {}
    ffn_drop_per_layer: dict[int, list[int]] = {}
    ffn_refit_stats: dict[int, dict] = {}
    if args.ffn_prune_frac > 0:
        # Free GPU + Python state from phase 0 / canary baseline / attn (if any)
        # before FFN phase 1 starts. FFN LOO does many forwards and is sensitive
        # to fragmentation. Without this we OOM at first forward on E4B (24 GB).
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log(f"phase1_ffn: pre-cleanup GPU mem = "
                f"{torch.cuda.memory_allocated()/(1024**3):.2f} GiB allocated, "
                f"{torch.cuda.memory_reserved()/(1024**3):.2f} GiB reserved")
        n_int = cfg.intermediate_size
        K = args.ffn_block_size
        if n_int % K != 0:
            raise ValueError(
                f"intermediate_size={n_int} not divisible by --ffn-block-size={K}"
            )
        n_blocks = n_int // K
        n_drop_blocks = int(round(n_blocks * args.ffn_prune_frac))
        log(f"phase1_ffn: {args.ffn_phase1_mode}, intermediate={n_int} K={K} "
            f"→ n_blocks={n_blocks}, will drop {n_drop_blocks}/{n_blocks} blocks "
            f"({args.ffn_prune_frac*100:.1f}%)")

        if args.ffn_phase1_mode == "loo":
            # Re-baseline CE here — model state already includes the attention
            # prune (if any), so the FFN ΔCE measurement is conditional on
            # the attention prune already in effect. This is the correct setup:
            # we want FFN block importance given the prune chain so far.
            ffn_baseline = measure_calib_ce(model, calib_chunks)
            log(f"phase1_loo_ffn: baseline CE (post-attn-prune) = {ffn_baseline:.4f}")
            ffn_imp_loo = phase1_importance_loo_ffn(
                model, calib_chunks, ce_baseline=ffn_baseline, block_size=K,
                layer_filter=args._ffn_prune_layer_set,
            )
            ffn_importance: dict[int, torch.Tensor] = ffn_imp_loo
        else:  # loo_multi
            if not args.phase1_shard_corpora:
                log("ffn_phase1_mode=loo_multi but --phase1-shard-corpora missing — "
                    "falling back to single-axis FFN LOO")
                ffn_baseline = measure_calib_ce(model, calib_chunks)
                ffn_imp_loo = phase1_importance_loo_ffn(
                    model, calib_chunks, ce_baseline=ffn_baseline, block_size=K,
                    layer_filter=args._ffn_prune_layer_set,
                )
                ffn_importance = ffn_imp_loo
                args.ffn_phase1_mode = "loo"  # reflect actual mode
            else:
                # Reuse shard parsing from attn loo_multi when available; else build.
                if 'shards' not in dir() or 'ce_baselines' not in dir():
                    shards = {}
                    ce_baselines = {}
                    for spec in args.phase1_shard_corpora.split(","):
                        spec = spec.strip()
                        if not spec or ":" not in spec:
                            continue
                        name, path = spec.split(":", 1)
                        per_shard_total = max(args.chunk_tokens, calib_total // 3)
                        schunks = build_calib(tok, per_shard_total, args.chunk_tokens, path.strip())
                        shards[name.strip()] = schunks
                        ce_baselines[name.strip()] = measure_calib_ce(model, schunks)
                ffn_imp_loo_multi_raw = phase1_importance_loo_multi_ffn(
                    model, shards, ce_baselines, block_size=K,
                    layer_filter=args._ffn_prune_layer_set,
                )
                ffn_importance = aggregate_multi_class_importance(
                    ffn_imp_loo_multi_raw, strategy=args.phase1_multi_strategy,
                )
                log(f"phase1_loo_multi_ffn: aggregated per-block importance via "
                    f"strategy={args.phase1_multi_strategy}")

        log(f"phase2_ffn: dropping {n_drop_blocks}/{n_blocks} blocks per layer")
        for L, layer in enumerate(layers):
            if not has_mlp(layer):
                continue
            if args._ffn_prune_layer_set is not None and L not in args._ffn_prune_layer_set:
                continue
            imp = ffn_importance[L]
            keep_blocks = imp.argsort(descending=True)[: n_blocks - n_drop_blocks].tolist()
            keep_blocks_sorted = sorted(int(b) for b in keep_blocks)
            keep_cols: list[int] = []
            for b in keep_blocks_sorted:
                keep_cols.extend(range(b * K, (b + 1) * K))
            drop_cols = sorted(set(range(n_int)) - set(keep_cols))
            log(f"phase2_ffn: L{L:02d} imp range=[{imp.min():.3e},{imp.max():.3e}] "
                f"drop_blocks={[int(b) for b in sorted(set(range(n_blocks)) - set(keep_blocks_sorted))]}")

            # Mask: zero rows of gate_proj + up_proj for dropped channels.
            with torch.no_grad():
                drop_idx_t = torch.tensor(drop_cols, dtype=torch.long,
                                          device=layer.mlp.gate_proj.weight.device)
                with _maybe_align(layer.mlp.gate_proj):
                    layer.mlp.gate_proj.weight.data.index_fill_(0, drop_idx_t, 0)
                with _maybe_align(layer.mlp.up_proj):
                    layer.mlp.up_proj.weight.data.index_fill_(0, drop_idx_t, 0)

            # Heal selection
            if args.ffn_heal == "noheal":
                stats = {"kept": len(keep_cols), "in": int(n_int), "lam": 0.0,
                         "rel_resid": None, "target_rms": None,
                         "loss_first": None, "loss_last": None}
            else:
                x_actual = capture_proj_input_at_layer_mlp(model, p2_chunks, L)
                if args.ffn_heal == "lstsq":
                    stats = lstsq_refit_down_proj(
                        layer.mlp.down_proj, x_actual, ffn_targets[L],
                        keep_cols=keep_cols, ridge_rel=args.ridge,
                    )
                    log(f"  L{L} ffn-lstsq kept={stats['kept']}/{stats['in']} "
                        f"lam={stats['lam']:.3e} rel_resid={stats['rel_resid']:.4f} "
                        f"target_rms={stats['target_rms']:.4f}")
                else:  # lora-rN
                    rank = int(args.ffn_heal.split("-r")[1])
                    stats = lora_heal_o_proj(
                        layer.mlp.down_proj, x_actual, ffn_targets[L],
                        keep_cols=keep_cols,
                        rank=rank,
                        n_steps=args.lora_steps,
                        lr=args.lora_lr,
                        weight_decay=args.lora_weight_decay,
                    )
                    _lf = stats.get('loss_first')
                    _ll = stats.get('loss_last')
                    _lf_s = f"{_lf:.4e}" if _lf is not None else "n/a"
                    _ll_s = f"{_ll:.4e}" if _ll is not None else "n/a"
                    log(f"  L{L} ffn-lora-r{rank} kept={stats['kept']}/{stats['in']} "
                        f"loss {_lf_s} → {_ll_s} rel_resid={stats['rel_resid']:.4f} "
                        f"target_rms={stats['target_rms']:.4f}")
                # Free the per-layer FFN target eagerly.
                ffn_targets.pop(L, None)
                del x_actual

            ffn_keep_per_layer[L] = keep_cols
            ffn_drop_per_layer[L] = drop_cols
            ffn_refit_stats[L] = stats
            wb_ffn = {
                "phase2_ffn/layer": L,
                "phase2_ffn/n_drop_blocks": len(drop_cols) // K,
                "phase2_ffn/n_keep_blocks": len(keep_cols) // K,
            }
            if stats.get("rel_resid") is not None:
                wb_ffn["phase2_ffn/rel_resid"] = stats["rel_resid"]
                wb_ffn["phase2_ffn/target_rms"] = stats["target_rms"]
            if "loss_first" in stats and stats["loss_first"] is not None:
                wb_ffn["phase2_ffn/lora_loss_first"] = stats["loss_first"]
                wb_ffn["phase2_ffn/lora_loss_last"] = stats["loss_last"]
                wb_ffn["phase2_ffn/lora_rank"] = stats.get("rank")
            wb.log(wb_ffn)
        # Free remaining FFN targets if any (no-op when empty / noheal).
        ffn_targets.clear()

    # Final CE on calibration
    ce_total = 0.0
    first_dev = next(model.parameters()).device
    with torch.no_grad():
        for chunk in calib_chunks:
            ids = chunk.to(first_dev)
            out = model(ids)
            logits = out.logits
            shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
            shift_labels = ids[:, 1:].contiguous()
            ce_total += F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).item()
    ce_post = ce_total / len(calib_chunks)
    log(f"final CE on calibration: {ce_post:.4f}")

    # ---- Phase 3a: detach accelerate hooks + materialize on CPU before save ----
    # Critical (bug-127, 2026-05-09): without this, accelerate's offload
    # weights_map silently restores the ORIGINAL pre-prune tensors at
    # save_pretrained time for any layer that was CPU-offloaded during the
    # prune+heal loop. Result: only GPU-resident layers' modifications stick.
    # Pre-2026-05-09 mask-mode runs (T8 he125/he25/he375) only persisted L0-L3.
    # This block must run for EVERY prune mode, not just group-aware.
    if not args.smoke:
        log("phase3a: detaching accelerate hooks + materializing model on CPU")
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except Exception as _e:
            log(f"warn: remove_hook_from_submodules failed: {_e!r}")
        # Force all weights onto CPU so no params remain on meta/offload.
        model.to("cpu")
        torch.cuda.empty_cache()

    # ---- Phase 3b (T11): physical reshape for group-aware mode ----
    # Skip when prune coverage is partial (--smoke / --prune-layers <subset>):
    # the global cfg.num_attention_heads update would conflict with un-touched
    # layers' nn.Linear shapes, producing an unloadable model. Partial prunes
    # are inherently mask-style (zero-rows on a subset, no global shape change).
    reshape_stats: dict[str, object] = {}
    n_attn_layers = sum(1 for layer in layers if has_self_attn(layer))
    full_coverage = (len(keep_q_per_layer) == n_attn_layers)
    if args.prune_mode == "group-aware" and not args.smoke and not full_coverage:
        log(f"phase3b: SKIPPED (partial prune coverage {len(keep_q_per_layer)}/{n_attn_layers} layers via "
            f"--prune-layers); saving as zero-rows mask, no global config rewrite, no physical reshape")
    if args.prune_mode == "group-aware" and not args.smoke and full_coverage:
        log(f"phase3b (group-aware): physical reshape on {len(keep_q_per_layer)} layers")
        hidden = cfg.hidden_size
        for L, layer in enumerate(layers):
            if not has_self_attn(layer):
                continue
            if L not in keep_q_per_layer:
                continue
            head_dim, _, n_kv_layer, lt = layer_attn_geom(model, L)
            # On KV-shared layers, k_proj/v_proj are dead — preserve their
            # full shape so loaded model still matches transformers' expected
            # arch when it tries to instantiate them. Pass full keep range so
            # physical_reshape produces an identity slice on K/V.
            is_kv_shared = (first_kv_shared_idx is not None and L >= first_kv_shared_idx)
            keep_kv_for_reshape = (
                list(range(n_kv_layer)) if is_kv_shared else keep_kv_per_layer[L]
            )
            stats = physical_reshape_attn_layer(
                layer,
                keep_q=keep_q_per_layer[L],
                keep_kv=keep_kv_for_reshape,
                head_dim=head_dim,
                hidden=hidden,
            )
            reshape_stats[str(L)] = stats
            if L < 3 or L == len(layers) - 1:
                log(f"  L{L} [{lt}] reshape: q={stats['q_shape']} k={stats['k_shape']} o={stats['o_shape']}")
        # Update config to reflect new uniform sizes.
        old_n_q = cfg.num_attention_heads
        old_n_kv = cfg.num_key_value_heads
        cfg.num_attention_heads = target_n_q_keep
        cfg.num_key_value_heads = target_n_kv_keep_sliding
        # full-layer KV count (num_global_key_value_heads) stays unchanged.
        # Sync top-level multimodal config too if present.
        if hasattr(model.config, "text_config") and model.config.text_config is not cfg:
            model.config.text_config.num_attention_heads = target_n_q_keep
            model.config.text_config.num_key_value_heads = target_n_kv_keep_sliding
        log(f"config updated: num_attention_heads {old_n_q} → {target_n_q_keep}, "
            f"num_key_value_heads {old_n_kv} → {target_n_kv_keep_sliding}, "
            f"num_global_key_value_heads {cfg.num_global_key_value_heads} (unchanged)")

    # ---- Pre-canary staged BF16 save (covers the canary-stall recovery case) ----
    # All the slow work (phase 0', phase 2 heal, phase 3 reshape) is now in `model`.
    # Persisting BF16/CPU here means a future restart can load from `staged/` and
    # jump straight to phase 2.5 — saving ~1-2 hours on 31B if the canary
    # itself dies/stalls (the 2026-05-11 CPU-canary-stall scenario).
    if args._ckpt_dir is not None and args.save_before_canary:
        staged = _staged_dir(args._ckpt_dir)
        log(f"phase3c (pre-canary): saving staged BF16 to {staged}")
        staged.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(staged, safe_serialization=True)
        tok.save_pretrained(staged)
        args._ckpt_staged_ready = True
        log(f"phase3c: staged save complete ({staged}).")

    # ---- Phase 2.5 (Fix C2): AR generation coherence gate ----
    # Compare pruned-NLL of base's canary gens against base's own NLL of the
    # same gens. If pruned distribution has drifted off the base manifold
    # (LAB:LAB: token loops on E2B he125-E type collapses), base's clean
    # output becomes very implausible → ratio explodes → refuse save.
    canary_result = None
    if canary_baseline:
        log(f"phase2.5: AR canary check (drift threshold={args.canary_ratio_threshold:.2f}× "
            f"+ shape: n_tokens≥5, rep<0.40, nonp<0.30)")
        # Resolve canary runtime (device + dtype) per CLI knobs. This is
        # decoupled from the save placement (BF16/CPU) — phase3a already did
        # the save-side move; canary may legitimately re-disperse to GPU offload
        # or upcast to FP32 to recover multi-thread CPU matmul. Always
        # reverted before save_pretrained() further below.
        _can_dev, _can_dtype = _resolve_canary_runtime(args, model, log)
        _restore_after_canary = _enter_canary_runtime(
            args, model, _can_dev, _can_dtype, log
        )
        try:
            canary_result = gen_canary_check(
                model, canary_baseline,
                tokenizer=tok,
                n_gen=args.canary_n_gen,
                ratio_threshold=args.canary_ratio_threshold,
            )
        finally:
            _restore_after_canary()
        for p in canary_result["per_prompt"]:
            sh = p["shape"]
            fails = ",".join(k for k, v in p["fails"].items() if v) or "OK"
            log(f"  canary {p['prompt']!r}: "
                f"drift={p['ratio_drift']:.2f}× self={p['ratio_self']:.2f}× "
                f"n={sh.get('n_tokens')} rep={sh.get('rep_ratio'):.2f} "
                f"nonp={sh.get('nonp_ratio'):.2f} → {fails}")
            log(f"    pruned-gen head: {p['gen_text_head']!r}")
        log(f"  → {'PASS' if canary_result['passed'] else 'FAIL'} "
            f"(worst drift={canary_result['overall_ratio_drift']:.3f}×)")

    # Pick the actual output dir based on the canary verdict — only create
    # the destination we will write to. This avoids leaving an empty stub
    # at args.output when we redirect to .broken/.
    out_dir = Path(args.output)
    canary_failed = canary_result is not None and not canary_result["passed"]
    if canary_failed:
        broken_dir = out_dir.with_name(out_dir.name + ".broken")
        log(f"CANARY FAILED — saving to {broken_dir} (not {out_dir}); will exit nonzero")
        out_dir = broken_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"saving to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    rel_resid_values = [
        v["rel_resid"] for v in refit_stats.values()
        if isinstance(v, dict) and isinstance(v.get("rel_resid"), (int, float))
    ]
    rel_resid_mean = sum(rel_resid_values) / len(rel_resid_values) if rel_resid_values else None
    manifest = {
        "source": str(args.model_path),
        "prune_frac": args.prune_frac,
        "n_drop_per_layer": n_drop,
        "prune_mode": args.prune_mode,
        "resolved_mode": resolved_mode,
        "full_coverage": full_coverage,
        "target_n_q_keep": target_n_q_keep,
        "target_n_kv_keep_sliding": target_n_kv_keep_sliding,
        "drop_kv_groups": args.drop_kv_groups,
        "phase1_mode": args.phase1_mode,
        "phase1_shard_corpora": args.phase1_shard_corpora,
        "phase1_multi_strategy": (
            args.phase1_multi_strategy if args.phase1_mode == "loo_multi" else None
        ),
        "allow_low_kv_structural": args.allow_low_kv_structural,
        # T7.4 per-class importance tensor — small (n_layers × n_q × n_shards floats)
        # so persist it for downstream reuse (FFN-prune, expert-drop, CD quant).
        "imp_loo_multi": (
            {
                str(L): {
                    "shard_names": blob["shard_names"],
                    "per_class": [
                        [round(float(x), 4) for x in row.tolist()]
                        for row in blob["per_class"]
                    ],
                }
                for L, blob in imp_loo_multi_raw.items()
            }
            if imp_loo_multi_raw is not None else None
        ),
        "calib_file": args.calib_file,
        "calib_tokens": calib_total,
        "chunk_tokens": calib_chunk,
        "ridge_rel": args.ridge,
        "heal": args.heal,
        "lora_steps": args.lora_steps if args.heal not in ("lstsq", "ar-lstsq") else None,
        "lora_lr": args.lora_lr if args.heal not in ("lstsq", "ar-lstsq") else None,
        "lora_weight_decay": args.lora_weight_decay if args.heal not in ("lstsq", "ar-lstsq") else None,
        # T7.7 ar-lstsq parameters (None for non-AR runs)
        "ar_rollout_seeds": args.ar_rollout_seeds if args.heal == "ar-lstsq" else None,
        "ar_rollout_gen": args.ar_rollout_gen if args.heal == "ar-lstsq" else None,
        "ar_rollout_prefix": args.ar_rollout_prefix if args.heal == "ar-lstsq" else None,
        "smoke": args.smoke,
        "prune_layers_arg": args.prune_layers,
        "first_kv_shared_idx": first_kv_shared_idx,
        "pruned_layers": pruned_layers,
        "refit_stats": {str(k): v for k, v in refit_stats.items()},
        "rel_resid_mean": rel_resid_mean,
        "canary_result": canary_result,
        "canary_failed": canary_failed,
        "keep_q_per_layer": {str(k): v for k, v in keep_q_per_layer.items()},
        "keep_kv_per_layer": {str(k): v for k, v in keep_kv_per_layer.items()},
        "reshape_stats": reshape_stats,
        "final_ce": ce_post,
        # T7.6 FFN-prune outputs (empty unless --ffn-prune-frac > 0).
        "ffn_prune_frac": args.ffn_prune_frac,
        "ffn_block_size": args.ffn_block_size,
        "ffn_heal": args.ffn_heal,
        "ffn_phase1_mode": args.ffn_phase1_mode,
        "ffn_prune_layers_arg": args.ffn_prune_layers,
        "ffn_keep_per_layer": {str(k): v for k, v in ffn_keep_per_layer.items()},
        "ffn_drop_per_layer": {str(k): v for k, v in ffn_drop_per_layer.items()},
        "ffn_refit_stats": {str(k): v for k, v in ffn_refit_stats.items()},
    }
    (out_dir / "prune_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Wandb summary — single record so the W&B sidebar shows the headline
    # numbers without scrolling through phase2 step plots.
    wb.summary["final_ce"] = ce_post
    wb.summary["rel_resid_mean"] = rel_resid_mean
    wb.summary["resolved_mode"] = resolved_mode
    wb.summary["full_coverage"] = full_coverage
    wb.summary["canary_failed"] = canary_failed
    if canary_result:
        wb.summary["canary_passed"] = canary_result["passed"]
        wb.summary["canary_overall_drift"] = canary_result["overall_ratio_drift"]
        wb.summary["canary_max_rep"] = canary_result.get("max_rep_ratio")
        wb.summary["canary_max_nonp"] = canary_result.get("max_nonp_ratio")
        wb.summary["canary_min_n_tokens"] = canary_result.get("min_n_tokens")
    wb.summary["out_dir"] = str(out_dir)
    wb.finish()

    log("done")
    # Fix C2: nonzero exit so wrappers don't silently quantize broken models.
    if canary_failed:
        log("EXIT 2: canary gate FAILED — see prune_manifest.json canary_result")
        sys.exit(2)


if __name__ == "__main__":
    main()
