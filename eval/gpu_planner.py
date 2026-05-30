"""Dynamic GPU selection + parallelism planning for omk_eval.

ONE canonical eval launcher must serve every host (single 24 GB 3090; the
2× 97 GB Blackwell box; cloud pods) without per-host edits or clones. This
module is the runtime brain: it probes the GPUs, picks the FREE one(s),
estimates how many request slots fit in their free VRAM, and computes the
context length — so templates can stay agnostic to parallelism and the runner
decides at launch.

Design rules:
  * stdlib-only (no torch / pynvml / aiohttp). It shells out to `nvidia-smi`.
    omk_eval is invoked with bare `python3` by the eval-suite shells and the
    pod bootstrap, so a third-party import here would break those callers.
  * SOFT-FAIL. Every probe/estimate degrades to a safe default. If
    `nvidia-smi` is missing, `probe_gpus()` returns [] and the caller falls
    back to exactly today's single-GPU / template-default behavior. The
    planner must never turn a working eval into a crash.
  * pure functions. No import of omk_eval (one-way dependency); callers pass
    their own `log` callable so messages land in the bench log.

Free-GPU policy is THRESHOLD (not zero-foreign-process): a GPU is usable when
it has enough free VRAM AND utilization is below a threshold. The operator
generally knows the box's state before launching, so a conservative
"refuse if any foreign process" policy would only make evals fail needlessly.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _noop(_msg: str) -> None:
    pass


# ── GPU discovery ──────────────────────────────────────────────────────────


@dataclass
class GpuInfo:
    index: int
    mem_total_mib: int
    mem_free_mib: int
    util_pct: int
    n_compute_procs: int  # foreign compute processes currently on this GPU


def probe_gpus(log=_noop) -> list[GpuInfo]:
    """Return per-GPU state via nvidia-smi, or [] if it can't be queried.

    [] is the regression anchor: the caller treats it as "no planner info —
    use the single-GPU / template-default path", i.e. today's behavior.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        if out.returncode != 0 or not out.stdout.strip():
            return []
    except Exception as e:  # FileNotFoundError, TimeoutExpired, …
        log(f"gpu_planner: nvidia-smi query failed ({e}); falling back")
        return []

    gpus: dict[int, GpuInfo] = {}
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx, tot, free, util = (int(parts[0]), int(parts[1]),
                                    int(parts[2]), int(float(parts[3])))
        except ValueError:
            continue
        gpus[idx] = GpuInfo(index=idx, mem_total_mib=tot, mem_free_mib=free,
                            util_pct=util, n_compute_procs=0)

    # Count foreign compute processes per GPU. compute-apps reports a GPU
    # *uuid*, not an index, so join through a second index→uuid query.
    try:
        uo = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        uuid2idx: dict[str, int] = {}
        for line in uo.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 2:
                uuid2idx[p[1]] = int(p[0])
        co = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        for line in co.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if not p or not p[0]:
                continue
            idx = uuid2idx.get(p[0])
            if idx is not None and idx in gpus:
                gpus[idx].n_compute_procs += 1
    except Exception:
        pass  # proc counts are advisory; threshold policy uses util+free

    return [gpus[i] for i in sorted(gpus)]


# ── Size estimation ────────────────────────────────────────────────────────

_MIB = 1024 * 1024
_CUDA_CTX_OVERHEAD_MIB = 700      # CUDA context + cublas workspaces per process
_VLLM_ACT_OVERHEAD_MIB = 4096     # vLLM activation + cudagraph capture headroom


def estimate_model_mib(model: str, backend: str, quant: str = "auto",
                       log=_noop) -> int:
    """Conservative estimate of weight VRAM for one full copy of the model.

    llama: GGUF on-disk size ≈ resident weights at -ngl 99.
    vLLM:  safetensors index total_size if present (exact), else
           params × dtype-bytes from config.json.
    Always rounds up and adds a fixed CUDA-context overhead. On any failure
    returns a large sentinel so the planner under-provisions parallelism
    rather than OOMing.
    """
    p = Path(model)
    try:
        if backend == "llama" or quant == "gguf" or p.suffix == ".gguf":
            gguf = p
            if p.is_dir():
                cands = sorted(p.glob("*.gguf"),
                               key=lambda f: f.stat().st_size, reverse=True)
                if not cands:
                    raise FileNotFoundError("no .gguf in model dir")
                gguf = cands[0]
            sz = gguf.stat().st_size
            return int(sz / _MIB) + _CUDA_CTX_OVERHEAD_MIB

        # vLLM / HF dir
        if p.is_dir():
            idx = p / "model.safetensors.index.json"
            if idx.is_file():
                meta = json.loads(idx.read_text()).get("metadata", {})
                tot = int(meta.get("total_size", 0))
                if tot > 0:
                    return (int(tot / _MIB) + _CUDA_CTX_OVERHEAD_MIB
                            + _VLLM_ACT_OVERHEAD_MIB)
            single = p / "model.safetensors"
            if single.is_file():
                return (int(single.stat().st_size / _MIB)
                        + _CUDA_CTX_OVERHEAD_MIB + _VLLM_ACT_OVERHEAD_MIB)
            cfg = p / "config.json"
            if cfg.is_file():
                d = json.loads(cfg.read_text())
                params = _approx_param_count(d)
                if params > 0:
                    dbytes = _dtype_bytes(quant, d)
                    return (int(params * dbytes / _MIB)
                            + _CUDA_CTX_OVERHEAD_MIB + _VLLM_ACT_OVERHEAD_MIB)
    except Exception as e:
        log(f"gpu_planner: model size estimate failed ({e}); using sentinel")
    # Unknown: assume it needs a big GPU (forces conservative parallel).
    return 48 * 1024


def _dtype_bytes(quant: str, cfg: dict) -> float:
    q = (quant or "auto").lower()
    if q in ("nvfp4a16", "fp4"):
        return 0.55
    if q in ("awq", "gptq", "int4", "q4"):
        return 0.55
    if q in ("int8", "q8"):
        return 1.05
    qc = (cfg.get("quantization_config") or {}).get("quant_method", "")
    if qc == "modelopt":
        return 0.55
    if qc in ("awq", "gptq", "gptqmodel"):
        return 0.55
    return 2.0  # bf16 / fp16


def _approx_param_count(cfg: dict) -> int:
    if cfg.get("num_parameters"):
        return int(cfg["num_parameters"])
    try:
        L = int(cfg.get("num_hidden_layers", 0))
        h = int(cfg.get("hidden_size", 0))
        inter = int(cfg.get("intermediate_size", h * 4))
        vocab = int(cfg.get("vocab_size", 0))
        if not (L and h):
            return 0
        # attn (~4 h^2) + mlp (~3 h*inter) per layer + embeddings (×2 tied-ish)
        per_layer = 4 * h * h + 3 * h * inter
        return L * per_layer + 2 * vocab * h
    except Exception:
        return 0


def estimate_kv_mib_per_slot(model_dir: str, ctx_tokens: int,
                             kv_dtype: str = "q8_0", log=_noop) -> int:
    """KV-cache MiB for ONE request slot at `ctx_tokens`.

    Reads config.json for layer/head geometry. Applies the Gemma-4
    sliding-window correction: sliding-attention layers cap their KV at the
    sliding window, only full-attention layers pay the whole ctx — without
    this a naive layers×ctx estimate over-counts Gemma 4 by ~5×. Falls back
    to a deliberately high constant when geometry is unknown (under-provision
    parallel rather than OOM).
    """
    bytes_per = 1.1 if kv_dtype.startswith("q8") else (
        0.6 if kv_dtype.startswith("q4") else 2.0)
    fallback = int(0.6 * (ctx_tokens / 1024.0)) + 1  # MiB; conservative-high
    try:
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.is_file():
            return fallback
        d = json.loads(cfg_path.read_text())
        L = int(d.get("num_hidden_layers", 0))
        hkv = int(d.get("num_key_value_heads",
                        d.get("num_attention_heads", 0)))
        head_dim = int(d.get("head_dim", 0)) or (
            int(d.get("hidden_size", 0)) //
            max(1, int(d.get("num_attention_heads", 1))))
        if not (L and hkv and head_dim):
            return fallback
        sw = int(d.get("sliding_window", 0) or 0)
        pattern = int(d.get("sliding_window_pattern", 0) or 0)
        if sw and pattern:
            n_full = max(1, L // pattern)
            n_slide = L - n_full
            eff_tokens = n_full * ctx_tokens + n_slide * min(ctx_tokens, sw)
        else:
            eff_tokens = L * ctx_tokens
        # K and V, both, per layer-token: hkv * head_dim elements each.
        bts = eff_tokens * hkv * head_dim * 2 * bytes_per
        return max(1, int(math.ceil(bts / _MIB)))
    except Exception as e:
        log(f"gpu_planner: kv estimate failed ({e}); using fallback")
        return fallback


# ── Selection + parallel/ctx planning ──────────────────────────────────────


def _parse_visible(env_val: str | None) -> set[int] | None:
    if not env_val:
        return None
    ids: set[int] = set()
    for tok in env_val.split(","):
        tok = tok.strip()
        if tok.isdigit():
            ids.add(int(tok))
    return ids or None


def select_gpus(gpus: list[GpuInfo], policy: str, need_mib: int,
                util_thresh: float = 0.15, log=_noop) -> list[int]:
    """THRESHOLD selection. A GPU is usable when free VRAM >= need_mib AND
    util < util_thresh*100. Honors an explicit id list as a filter and any
    preset CUDA_VISIBLE_DEVICES (never widens the caller's pin).
    """
    if not gpus:
        return []
    visible = _parse_visible(os.environ.get("CUDA_VISIBLE_DEVICES"))
    pol = (policy or "auto").strip().lower()
    explicit: set[int] | None = None
    if pol not in ("auto", "free"):
        explicit = {int(t) for t in pol.split(",") if t.strip().isdigit()}

    util_pct = util_thresh * 100.0
    usable: list[int] = []
    for g in gpus:
        if visible is not None and g.index not in visible:
            continue
        if explicit is not None and g.index not in explicit:
            continue
        if g.mem_free_mib < need_mib:
            log(f"gpu_planner: skip GPU{g.index} — free {g.mem_free_mib} MiB "
                f"< need {need_mib} MiB")
            continue
        if g.util_pct >= util_pct:
            log(f"gpu_planner: skip GPU{g.index} — util {g.util_pct}% "
                f">= thresh {util_pct:.0f}% (procs={g.n_compute_procs})")
            continue
        usable.append(g.index)

    if explicit is not None:
        dropped = explicit - set(usable)
        if dropped:
            log(f"gpu_planner: requested GPUs {sorted(dropped)} unusable "
                f"(busy/insufficient) — proceeding with {usable}")
    return usable


def plan_parallel(free_after_model_mib: int, kv_mib_per_slot: int,
                  requested: int | None, max_parallel: int = 8) -> int:
    """Slots that fit: clamp(requested or free//kv, 1, max_parallel)."""
    fits = max(1, free_after_model_mib // max(1, kv_mib_per_slot))
    want = requested if (requested and requested > 0) else fits
    return max(1, min(want, fits, max_parallel))


def plan_ctx(parallel: int, thinking_budget: int, content_headroom: int,
             ctx_max: int, requested_ctx: int, log=_noop) -> tuple[int, int]:
    """Per-slot ctx safety: guarantee (ctx // parallel) >= thinking_budget +
    content_headroom by bumping ctx UP. Only clamp parallel down as a last
    resort when even ctx_max can't fit the requested parallel.

    Extracted verbatim from the original omk_eval llama auto-bump so BOTH
    backends share identical math (the vLLM path had no auto-bump before).
    Returns (parallel, ctx); emits the same WARNING/AUTO-BUMP log lines.
    """
    ctx = requested_ctx
    if thinking_budget > 0:
        required_per_slot = thinking_budget + content_headroom
        required_ctx = parallel * required_per_slot
        if required_ctx > ctx_max:
            safe_parallel = max(1, ctx_max // required_per_slot)
            log(f"WARNING: llama_parallel={parallel} requires ctx "
                f"{required_ctx} > ctx_max {ctx_max}; clamping parallel "
                f"to {safe_parallel} (per-slot {ctx_max // safe_parallel} "
                f">= thinking_budget {thinking_budget} + headroom "
                f"{content_headroom})")
            parallel = safe_parallel
            required_ctx = parallel * required_per_slot
        if required_ctx > ctx:
            log(f"AUTO-BUMP llama_ctx: template={ctx} → required={required_ctx} "
                f"(parallel={parallel} × (thinking_budget={thinking_budget} "
                f"+ headroom={content_headroom}))")
            ctx = required_ctx
    return parallel, ctx
