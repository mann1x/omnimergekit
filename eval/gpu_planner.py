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
import time
from dataclasses import dataclass
from pathlib import Path


def _noop(_msg: str) -> None:
    pass


def _env_truthy(name: str) -> bool:
    """True when an env var is set to a truthy literal (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to `default` on unset/garbage."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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


# ── Plan assembly ──────────────────────────────────────────────────────────


@dataclass
class GpuPlan:
    """The launch decision produced once per eval and consumed by both backends.

    `source` is the regression tell: "fallback" means the planner had no usable
    GPU info (no nvidia-smi, or no GPU met the THRESHOLD) and the caller must
    reproduce today's behavior exactly — no pin, template-default parallel.
    """
    gpu_ids: list[int]       # chosen physical nvidia-smi indices ([] = fallback)
    replicas: int            # model copies to launch (fleet lands in P4; 1 here)
    parallel: int            # per-replica request slots (after VRAM clamp)
    ctx: int                 # context length after plan_ctx auto-bump
    tensor_parallel: int     # >1 only when one copy must span GPUs
    gpu_mem_util: float      # vLLM --gpu-memory-utilization (consumed in P3)
    need_mib: int            # estimated per-copy weight VRAM (logging)
    source: str              # "planner" | "fallback"

    @property
    def effective_concurrency(self) -> int:
        """Total in-flight request slots across the fleet (replicas × parallel).

        P5 feeds this to the lm-eval/dispatcher num_concurrent so every server
        slot is kept busy. In P2/P3 (single server) replicas is 1.
        """
        return max(1, self.replicas) * max(1, self.parallel)

    def cuda_visible(self) -> str | None:
        """CUDA_VISIBLE_DEVICES value for the whole fleet, or None in fallback."""
        return ",".join(str(i) for i in self.gpu_ids) if self.gpu_ids else None


def build_plan(*, model_dir: str, backend: str, quant: str = "auto",
               requested_gpus: str = "auto", requested_parallel: int | None = None,
               requested_replicas: str = "auto", util_thresh: float = 0.15,
               max_parallel: int = 8, thinking_budget: int = 0,
               max_gen_toks: int = 0, content_headroom: int = 4096,
               ctx_max: int = 262144, requested_ctx: int = 32768,
               default_parallel: int = 2, kv_dtype: str = "q8_0",
               default_gpu_mem_util: float = 0.92, log=_noop) -> GpuPlan:
    """Probe → select → size → plan parallel + ctx. Soft-fails to today's path.

    The single source of truth for GPU selection and parallelism. Both the
    llama and vLLM launch paths call this once and consume the returned plan, so
    the free-GPU policy and auto-bump are shared. The auto-bump is BACKEND-AWARE
    because the two engines treat context differently:

      * llama.cpp `-c` is ONE KV arena shared across `--parallel` slots, so
        per-slot ctx = ctx // parallel must hold the reasoning budget → ctx is
        bumped parallel-multiplied (plan_ctx). This is the SAT_COLLAPSE fix.
      * vLLM `--max-model-len` is PER-REQUEST (PagedAttention); concurrency
        comes from data-parallel replicas + --max-num-seqs, NOT from dividing
        max-model-len. So vLLM is bumped only to fit the largest SINGLE request
        (prompt + full generation incl. thinking) — never ×concurrency — which
        is what prevents vLLM's HTTP-400-on-long-reasoning failure.
    """
    def _par_ctx(parallel_req: int | None,
                 free_after: int | None = None,
                 kv: int | None = None) -> tuple[int, int]:
        if free_after is not None and kv is not None:
            par = plan_parallel(free_after, kv, parallel_req, max_parallel)
        else:
            par = parallel_req if parallel_req else default_parallel
        if backend == "vllm":
            base = max_gen_toks or thinking_budget
            ctx = requested_ctx
            if base > 0:
                ctx = min(max(requested_ctx, base + content_headroom), ctx_max)
            return par, ctx
        return plan_ctx(par, thinking_budget, content_headroom, ctx_max,
                        requested_ctx, log=log)

    def _terminal(need: int, source: str) -> GpuPlan:
        par, ctx = _par_ctx(requested_parallel)
        return GpuPlan(gpu_ids=[], replicas=1, parallel=par, ctx=ctx,
                       tensor_parallel=1, gpu_mem_util=default_gpu_mem_util,
                       need_mib=need, source=source)

    def _fallback(need: int) -> GpuPlan:
        return _terminal(need, "fallback")

    gpus = probe_gpus(log)
    if not gpus:
        # No nvidia-smi / unreadable → exactly today's single-GPU path. This is
        # a CAPABILITY gap (we can't see GPUs), not contention, so we keep the
        # legacy unpinned launch rather than refusing — refusing here would break
        # CPU-only / non-nvidia hosts that work fine today.
        return _fallback(0)

    need = estimate_model_mib(model_dir, backend, quant, log=log)
    chosen = select_gpus(gpus, requested_gpus, need, util_thresh, log=log)
    if not chosen:
        # GPUs exist but none meets the THRESHOLD policy (free VRAM + util) right
        # now — the box is under CONTENTION. Silently launching UNPINNED here
        # inherits whatever GPUs are visible and reliably OOMs/crashes the moment
        # a co-tenant (another eval, an ollama model, a quant sweep) is resident.
        # That is exactly the T87 mk1_256k crash (2026-06-08): a 14.5 GB ollama
        # model on GPU0 → unpinned TP=2 vLLM serve → exit=20. So: optionally WAIT
        # for a GPU to free, then REFUSE loudly (source="contended") unless the
        # operator explicitly opts into the legacy unpinned fallback.
        wait_s = _env_float("OMK_GPU_WAIT_S", 0.0)
        allow_unpinned = _env_truthy("OMK_ALLOW_UNPINNED")
        if wait_s > 0:
            poll = min(10.0, max(2.0, wait_s / 10.0))
            log(f"gpu_planner: no GPU meets THRESHOLD (need≈{need}MiB free, "
                f"util<{util_thresh}) — waiting up to {wait_s:.0f}s for one to "
                f"free (poll {poll:.0f}s; set OMK_GPU_WAIT_S to tune)")
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                time.sleep(poll)
                gpus = probe_gpus(log)
                if not gpus:
                    break
                chosen = select_gpus(gpus, requested_gpus, need, util_thresh, log=log)
                if chosen:
                    log(f"gpu_planner: GPU(s) {sorted(chosen)} freed after waiting "
                        "— proceeding with a pinned launch")
                    break
        if not chosen:
            if allow_unpinned:
                log("gpu_planner: no free GPU, but OMK_ALLOW_UNPINNED=1 — falling "
                    "back to UNPINNED launch (today's path; may OOM/crash under "
                    "contention)")
                return _fallback(need)
            log("gpu_planner: REFUSING to launch — no GPU meets THRESHOLD "
                f"(need≈{need}MiB free + util<{util_thresh}) after waiting "
                f"{wait_s:.0f}s. An unpinned launch under contention crashes "
                "(T87 mk1_256k, exit=20). Free a GPU, raise OMK_GPU_WAIT_S to "
                "wait longer, or set OMK_ALLOW_UNPINNED=1 to force the legacy "
                "unpinned fallback.")
            return _terminal(need, "contended")

    chosen_gpus = [g for g in gpus if g.index in chosen]
    n_gpu = len(chosen)
    min_free = min(g.mem_free_mib for g in chosen_gpus)
    tightest = min(chosen_gpus, key=lambda g: g.mem_free_mib)

    # gpu_mem_util from the tightest chosen GPU's free fraction (vLLM, P3).
    gpu_mem_util = round(
        min(0.95, max(0.30, tightest.mem_free_mib / max(1, tightest.mem_total_mib))), 2)

    # replicas: one model copy per usable GPU unless the operator caps it.
    rr = str(requested_replicas).strip().lower()
    if rr in ("auto", ""):
        replicas = n_gpu
    else:
        try:
            replicas = max(1, min(int(rr), n_gpu))
        except ValueError:
            replicas = n_gpu

    # KV for ONE request slot at the requested ctx, in the SERVING KV dtype (the
    # caller passes the right one per backend: bf16 for a vLLM bf16/nvfp4a16 serve,
    # q8_0/q4 for a llama.cpp quantized KV arena).
    kv = estimate_kv_mib_per_slot(model_dir, requested_ctx, kv_dtype, log=log)

    # Tensor-parallel split when ONE serving copy can't fit a single GPU. A copy
    # overflows one card two ways:
    #   (1) the WEIGHTS alone exceed free VRAM, OR
    #   (2) weights + the KV for a single request at the requested ctx exceed the
    #       gpu_mem_util budget vLLM is given — the long-context case the old
    #       weights-only trigger was BLIND to. A 26B (~52GB) model fits one 96GB
    #       GPU, but its 512k KV (~42GB bf16) does not: 52+42 > 0.92×96 → vLLM
    #       OOMs at KV-block alloc on a single card. Splitting across GPUs (TP)
    #       pools the VRAM so the one copy + KV fit. This is the 512k-serve fix.
    usable_one = int(tightest.mem_total_mib * gpu_mem_util)
    one_copy_mib = need + kv
    tensor_parallel = 1
    if n_gpu > 1 and (need > min_free or one_copy_mib > usable_one):
        tensor_parallel = n_gpu
        replicas = 1  # the whole fleet is one split copy
        log(f"gpu_planner: TP={n_gpu} — one copy {one_copy_mib}MiB (weights "
            f"{need} + kv@{requested_ctx}tok {kv}) exceeds one-GPU budget "
            f"{usable_one}MiB (util {gpu_mem_util}); splitting across {n_gpu} GPUs")

    # Per-replica parallel from the COMBINED VRAM after weights (TP pools VRAM
    # across the fleet; min_free × tensor_parallel is the shared pool).
    free_after = max(0, min_free * tensor_parallel - need)
    parallel, ctx = _par_ctx(requested_parallel, free_after, kv)

    return GpuPlan(gpu_ids=chosen, replicas=replicas, parallel=parallel, ctx=ctx,
                   tensor_parallel=tensor_parallel, gpu_mem_util=gpu_mem_util,
                   need_mib=need, source="planner")
