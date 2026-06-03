#!/usr/bin/env python3
"""REDIST — universal, driveable capability-redistribution for Gemma-4 MoE.

Redistribute the *function* of EXCLUDED experts INTO the surviving experts of a
pruned MoE (e.g. the 62e A2), driven by a target capability. The *same machine,
different driver corpus* yields a multilingual-/code-/science-/creative-recovered
62e. This is the T191 framework backbone; see
/root/.claude/plans/cozy-bubbling-kernighan.md.

Pipeline (5 stages, each a subcommand; `run` chains them):

  localize     which (layer,expert) cells carry the target capability, and is the
               capability LOCALIZED (a few experts -> closed-form fold works) or
               DIFFUSE (spread thin -> only trainable survivors can absorb it)?
               Two signal modes:
                 * loop-differential  (router_diff_bucket.py) for fail-by-LOOP caps
                   (multilingual, constrained-format)
                 * fluent divergence  (redist_localize_divergence.py) for fail-FLUENTLY
                   caps (code, science, math) where detect_loop finds nothing.
  capture      teacher-force the driver corpus through the 128e teacher and record
               the tensor substrate every method consumes (router softmax, per-expert
               SwiGLU gate/up/intermediate, MoE block output).
  redistribute the pluggable step: a RedistMethod consumes capture + keep-map and
               emits new survivor weights / router edits / an adapter delta.
  gate         AR-FIRST two-tier validation. Tier-1 = held-out block-output divergence
               + a small generation canary. NEVER reconstruction-MSE-first (off-manifold
               rule, feedback_calibration_signal_misleading).
  build        materialize the recovered 62e (expert_drop backend if the keep-set
               changed) + apply the method delta + full loop_screen.

Methods (registry; Phase-1 bodies):
  hcsmoe        output-space agglomerative cluster-merge (corrected DERN, arXiv:2410.08589)
  mergemoe      per-survivor functional least-squares T1=Q.P+  (arXiv:2510.14436)
  ream          router-weighted expert activation merging (arXiv:2604.04356, REAP successor)
  expert_kd     KD into TRAINABLE survivor experts (the only DIFFUSE-capable method)
  shared_mlp_kd KD into Gemma-4's always-on parallel dense mlp.* via LoRA (sidesteps routing)

Design notes:
  * One portable file. Sibling scripts (expert_drop / loop_screen / router_diff_bucket
    / redist_localize_divergence) are called as SUBPROCESSES via --scripts-dir
    (default = this file's dir), so host path differences are config, not code.
  * NEVER writes to /tmp. --workdir defaults to a persistent path.
  * Closed-form methods (trainable=False) run on one Blackwell; expert_kd is pod-scale.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# ─── env / paths ──────────────────────────────────────────────────────────────

SCRIPTS_DIR_DEFAULT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR_DEFAULT.parent

# Cross-directory dep scripts that do NOT live next to this file in the repo.
# Resolved host-independently by _resolve_dep() relative to REPO_ROOT; keep the
# basenames stable. Co-located deps (loop_screen / router_diff_bucket /
# redist_localize_divergence, all in scripts/) need no entry — probe step (2).
_DEP_MAP = {
    "expert_drop.py": "gemma4/expert_pruning/expert_drop.py",
    "generate_drop_map_v5.py": "gemma4/expert_pruning/generate_drop_map_v5.py",
    "router_shared_upweight.py": "recipes/gemma4/v5_moe_sweep/router_shared_upweight.py",
}


def _first_existing(*cands: str) -> str | None:
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def _resolve_dep(name: str, scripts_dir) -> Path:
    """Locate a dep/sibling script host-independently. Probe order:
      (1) operator override  <scripts_dir>/<name>
      (2) co-located         <this file's dir>/<name>
      (3) repo-relative      REPO_ROOT/_DEP_MAP[name]
      (4) PATH               shutil.which(name)
    Raise FileNotFoundError listing every probed path if none resolve."""
    probed: list[Path] = []
    for cand in (Path(scripts_dir) / name, SCRIPTS_DIR_DEFAULT / name):
        probed.append(cand)
        if cand.exists():
            return cand
    if name in _DEP_MAP:
        cand = REPO_ROOT / _DEP_MAP[name]
        probed.append(cand)
        if cand.exists():
            return cand
    which = shutil.which(name)
    if which:
        return Path(which)
    raise FileNotFoundError(
        "dep script not found: " + name + "\n  probed:\n    "
        + "\n    ".join(str(p) for p in probed)
        + "\n    PATH (shutil.which)\n  fix: pass --scripts-dir, or place it under "
        + "REPO_ROOT/" + _DEP_MAP.get(name, "<scripts>/" + name)
    )


# Persistent default workdir — NEVER /tmp (tmpfs). Env-var first, then known
# hosts (harmless off-host auto-detect -> None), then CWD-relative.
WORKDIR_DEFAULT = os.environ.get("REDIST_WORK") or _first_existing(
    "/srv/ml/redist_work",
    "/mnt/sdc/ml/redist_work",
) or str(Path.cwd() / "redist_work")

# 128e teacher + A2 student — env-var first, then per-host auto-detect (-> None
# on a foreign host; validated at use via _req(), never at import).
TEACHER_DEFAULT = os.environ.get("REDIST_TEACHER") or _first_existing(
    "/srv/ml/models/base/gemma-4-26B-A4B-it",
    "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/google/gemma-4-26B-A4B-it",
)
STUDENT_DEFAULT = os.environ.get("REDIST_STUDENT") or _first_existing(
    "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it",
)
KEEPMETA_DEFAULT = os.environ.get("REDIST_KEEP_META") or _first_existing(
    "/srv/ml/scripts/a2_keep_metadata.json",
)
SAMPLE_DEFAULT = os.environ.get("REDIST_SAMPLE") or _first_existing(
    "/mnt/sdc/ml/corpora/loop_screen_sample.jsonl",
)


def log(msg: str) -> None:
    print("[redist %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ─── drivers: capability spec + corpus + failure shape ────────────────────────


@dataclass(frozen=True)
class Driver:
    """A target capability to recover. The *driver* of the whole pipeline."""
    name: str
    # how the capability fails when the experts are dropped:
    #   "loop"   -> detect_loop fires; localize via loop-differential
    #   "fluent" -> wrong-but-fluent; localize via teacher-correct/student-failing divergence
    fail_mode: str
    # bucket key in the loop_screen sample for the loop-differential path
    loop_bucket: str | None = None
    # default corpus (jsonl of {prompt, bucket}); None => use the loop_screen sample
    corpus: str | None = None


DRIVERS: dict[str, Driver] = {
    "multilingual": Driver("multilingual", "loop", loop_bucket="multilingual"),
    "constrained": Driver("constrained", "loop", loop_bucket="constrained"),
    "code": Driver("code", "fluent"),
    "science": Driver("science", "fluent"),
    "math": Driver("math", "fluent"),
    "creative": Driver("creative", "fluent"),
}


# ─── keep-metadata utilities (absorbs the standalone regen helper) ────────────


def load_keep_meta(path: str | os.PathLike) -> dict:
    """Load an expert_drop_metadata.json / a2_keep_metadata.json sidecar.

    Returns the parsed dict with int-keyed per_layer_keep / per_layer_drop added
    as ``keep`` / ``drop`` for convenience.
    """
    d = json.load(open(path))
    if "per_layer_keep" not in d:
        raise SystemExit(f"FAIL: {path} has no per_layer_keep (not an expert_drop sidecar)")
    d["keep"] = {int(k): sorted(v) for k, v in d["per_layer_keep"].items()}
    if "per_layer_drop" in d:
        d["drop"] = {int(k): sorted(v) for k, v in d["per_layer_drop"].items()}
    return d


def keep_meta_from_drop_map(drop_map_path: str, num_experts: int = 128,
                            base_model: str = "google/gemma-4-26B-A4B-it") -> dict:
    """Build a keep-metadata dict from a bare drop map {layer:[dropped ids]}.

    Only needed when a variant has no expert_drop sidecar (expert_drop.py already
    emits one for every model it builds). Complement = survivors.
    """
    drop_map = {int(k): sorted(v) for k, v in json.load(open(drop_map_path)).items()}
    keep = {li: sorted(set(range(num_experts)) - set(ids)) for li, ids in drop_map.items()}
    target = num_experts - len(drop_map[next(iter(drop_map))])
    return {
        "base_model": base_model,
        "method": "keep_meta_from_drop_map",
        "drop_map_file": str(drop_map_path),
        "original_experts": num_experts,
        "target_experts": target,
        "per_layer_keep": {str(li): v for li, v in keep.items()},
        "per_layer_drop": {str(li): v for li, v in drop_map.items()},
        "keep": keep,
        "drop": drop_map,
    }


# ─── RedistMethod plugin contract ─────────────────────────────────────────────


@dataclass
class Artifacts:
    """What a RedistMethod produces in the redistribute stage."""
    # new stacked survivor expert weights per layer, ready to overwrite the built
    # model's experts.{gate_up_proj,down_proj}; absent => no weight edit.
    survivor_weights: dict | None = None
    # router edits (e.g. summed/centroid rows) keyed by tensor name; absent => keep.
    router_edits: dict | None = None
    # path to a saved LoRA / adapter delta (trainable methods); absent => none.
    adapter_path: str | None = None
    # the keep-set this artifact is built against (defaults to the student's A2 keep).
    keep_meta: dict | None = None
    meta: dict = field(default_factory=dict)


class RedistMethod(ABC):
    """Contract every redistribution technique implements.

    The framework calls plan() then fit() then emit(). Closed-form methods
    (trainable=False) implement fit() as a one-shot solve; gradient methods
    (trainable=True) run a KD loop in fit().
    """
    name: str = "base"
    # which captured tensors this method consumes:
    #   {"router", "swiglu_io", "block_out"}
    needs_tap: set[str] = set()
    # closed-form (False) vs gradient/KD (True). Trainable => pod-scale, AR-gate.
    trainable: bool = False
    # one-line human description.
    blurb: str = ""

    @abstractmethod
    def plan(self, localize: dict, keep_meta: dict) -> dict:
        """Decide assignment/clustering: which dropped experts fold into which
        survivor (per layer). Returns a JSON-able plan."""

    @abstractmethod
    def fit(self, capture: dict, plan: dict, keep_meta: dict, args: argparse.Namespace) -> Artifacts:
        """Consume captured tensors + plan; produce Artifacts."""

    def emit(self, art: Artifacts, model_dir: str) -> None:
        """Apply Artifacts into a built model dir (default: overwrite expert tensors
        / router rows / drop in an adapter). Overridable per method."""
        raise NotImplementedError(f"{self.name}.emit not implemented (Phase 1)")


class _Phase1Stub(RedistMethod):
    """Declares metadata (needs_tap/trainable) so the contract is testable now;
    bodies land in Phase 1 (plan file: Arm A / Arm B)."""

    def plan(self, localize, keep_meta):
        raise NotImplementedError(f"{self.name}.plan — Phase 1")

    def fit(self, capture, plan, keep_meta, args):
        raise NotImplementedError(f"{self.name}.fit — Phase 1")


# ─── closed-form merge infrastructure (HCSMoE / MergeMoE / REAM) ───────────────
#
# Topology (matches the user's "redistribute INTO the survivors" framing):
# survivors = A2's FIXED 62-expert keep set. Each of the 66 dropped experts is
# assigned to its nearest survivor (output-space cosine), and its function is
# folded into that survivor's weights. Keep set + router (incl. A2's PES) are
# unchanged, so the result is apples-to-apples with the A2 baseline — only the 62
# survivor experts' gate_up_proj/down_proj change. Closed-form methods are
# predicted to recover LOCALIZED capabilities (code/constrained) and to plateau
# on DIFFUSE multilingual — the AR gate measures which.

_EXP_PREFIX = "model.language_model.layers.%d.experts.%s"


def _st_index(model_dir: str) -> dict:
    import os
    p = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(p):
        return json.load(open(p))["weight_map"]
    # single-shard fallback
    import glob
    shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if len(shards) != 1:
        raise SystemExit(f"FAIL: no index.json and {len(shards)} shards in {model_dir}")
    from safetensors import safe_open
    with safe_open(shards[0], framework="pt") as f:
        return {k: os.path.basename(shards[0]) for k in f.keys()}


def _load_tensor(model_dir: str, name: str, index: dict, device="cpu"):
    import os
    from safetensors import safe_open
    shard = index[name]
    with safe_open(os.path.join(model_dir, shard), framework="pt", device=device) as f:
        return f.get_tensor(name)


def _act(x):
    import torch.nn.functional as F
    return F.gelu(x, approximate="tanh")   # gelu_pytorch_tanh


def _per_expert_mean_output(x, gate_up, down):
    """Mean SwiGLU output per expert over calib tokens x.
    x:[T,H] f32, gate_up:[E,2*M,H], down:[E,H,M]. Returns [E,H] f32 (CPU)."""
    import torch
    E = gate_up.shape[0]
    M = down.shape[-1]
    outs = torch.empty(E, x.shape[-1], dtype=torch.float32)
    for e in range(E):
        h = x @ gate_up[e].T.float()           # [T,2M]
        g, u = h[:, :M], h[:, M:]
        inter = _act(g) * u                    # [T,M]
        o = inter @ down[e].T.float()          # [T,H]
        outs[e] = o.mean(0).cpu()
    return outs


def _routing_freq(top_idx, num_experts):
    import torch
    return torch.bincount(top_idx.reshape(-1), minlength=num_experts).float()


def _assign_dropped(mean_out, keep_ids, drop_ids):
    """Assign each dropped expert -> nearest survivor by output cosine.
    Returns {survivor_id: [dropped ids]}."""
    import torch
    mo = torch.nn.functional.normalize(mean_out.float(), dim=-1)
    surv = torch.tensor(keep_ids)
    assign: dict = {i: [] for i in keep_ids}
    for j in drop_ids:
        sims = mo[surv] @ mo[j]
        i = keep_ids[int(sims.argmax())]
        assign[i].append(j)
    return assign


def _emit_merge(art: "Artifacts", src_dir: str, out_dir: str) -> None:
    """Write a recovered 62e: copy src (A2) aux files + non-expert tensors, overwrite
    the 62 survivor experts.{gate_up_proj,down_proj} with the merged weights."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    index = _st_index(src_dir)
    # aux files (configs/tokenizer/index)
    for fn in os.listdir(src_dir):
        if fn.endswith(".safetensors"):
            continue
        s = os.path.join(src_dir, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(out_dir, fn))
    over: dict = {}
    for li, w in art.survivor_weights.items():
        over[_EXP_PREFIX % (li, "gate_up_proj")] = w["gate_up_proj"]
        over[_EXP_PREFIX % (li, "down_proj")] = w["down_proj"]
    shards: dict = {}
    for name, shard in index.items():
        shards.setdefault(shard, []).append(name)
    for shard, names in shards.items():
        tensors = {}
        with safe_open(os.path.join(src_dir, shard), framework="pt") as f:
            for n in names:
                tensors[n] = over.get(n, f.get_tensor(n))
        save_file(tensors, os.path.join(out_dir, shard), metadata={"format": "pt"})
    print("[emit] wrote %d shards (%d expert tensors overwritten) -> %s"
          % (len(shards), len(over), out_dir))


class _MergeMethod(RedistMethod):
    """Shared plan/emit for closed-form merge-into-fixed-survivors methods."""
    trainable = False

    def plan(self, localize, keep_meta):
        # assignment is computed in fit (needs per-expert outputs from capture);
        # plan just records the topology + any localize-derived priorities.
        return {"topology": "merge_into_fixed_survivors",
                "n_layers": len(keep_meta["keep"]),
                "localize_shape": (localize or {}).get("shape")}

    def emit(self, art, model_dir, src_dir=None):
        _emit_merge(art, src_dir or art.meta.get("src_dir"), model_dir)


class HCSMoE(_MergeMethod):
    name = "hcsmoe"
    needs_tap = {"swiglu_io", "block_out"}
    blurb = "output-space agglomerative cluster-merge (corrected DERN, arXiv:2410.08589)"

    def _saliency(self, data, li, freq, mean_out, E):
        """Per-expert merge importance. HCSMoE = pure routing frequency."""
        return freq.float()

    def fit(self, capture, plan, keep_meta, args):
        import torch
        data = capture["data"]
        keep, drop = keep_meta["keep"], keep_meta["drop"]
        E = capture.get("num_experts", 128)
        index = _st_index(args.teacher)
        survivor_w = {}
        for li in sorted(keep):
            gu = _load_tensor(args.teacher, _EXP_PREFIX % (li, "gate_up_proj"), index, args.device)
            dn = _load_tensor(args.teacher, _EXP_PREFIX % (li, "down_proj"), index, args.device)
            x = data["swiglu_in"][li].to(args.device).float()
            mean_out = _per_expert_mean_output(x, gu, dn)
            freq = _routing_freq(data["top_idx"][li], E)
            sal = self._saliency(data, li, freq, mean_out, E)   # [E] merge importance
            assign = _assign_dropped(mean_out, keep[li], drop[li])
            new_gu, new_dn = [], []
            for i in keep[li]:                      # keep[li] already sorted -> A2 slot order
                members = [i] + assign[i]
                w = sal[members].clamp(min=1e-6)
                w = (w / w.sum()).to(args.device)
                mg = (w[:, None, None] * gu[members].float()).sum(0)
                md = (w[:, None, None] * dn[members].float()).sum(0)
                new_gu.append(mg.to(torch.bfloat16).cpu())
                new_dn.append(md.to(torch.bfloat16).cpu())
            survivor_w[li] = {"gate_up_proj": torch.stack(new_gu),
                              "down_proj": torch.stack(new_dn)}
            del gu, dn, x
            torch.cuda.empty_cache()
            if li % 6 == 0:
                log(f"  {self.name} fit L{li}: {len(keep[li])} survivors, "
                    f"{sum(len(v) for v in assign.values())} dropped folded")
        return Artifacts(survivor_weights=survivor_w, keep_meta=keep_meta,
                         meta={"method": self.name, "src_dir": args.student})


class MergeMoE(_MergeMethod):
    name = "mergemoe"
    needs_tap = {"swiglu_io"}
    blurb = "per-survivor functional least-squares T1=Q.P+ (arXiv:2510.14436)"

    def fit(self, capture, plan, keep_meta, args):
        import torch
        data = capture["data"]
        keep, drop = keep_meta["keep"], keep_meta["drop"]
        E = capture.get("num_experts", 128)
        ridge = float(getattr(args, "ridge", 1e-2))
        index = _st_index(args.teacher)
        survivor_w = {}
        for li in sorted(keep):
            gu = _load_tensor(args.teacher, _EXP_PREFIX % (li, "gate_up_proj"), index, args.device).float()
            dn = _load_tensor(args.teacher, _EXP_PREFIX % (li, "down_proj"), index, args.device).float()
            x = data["swiglu_in"][li].to(args.device).float()          # [T,H]
            M = dn.shape[-1]
            mean_out = _per_expert_mean_output(x, gu, dn)
            freq = _routing_freq(data["top_idx"][li], E).to(args.device)
            assign = _assign_dropped(mean_out, keep[li], drop[li])
            new_gu, new_dn = [], []
            for i in keep[li]:
                members = [i] + assign[i]
                fw = freq[members].clamp(min=1e-6)
                fw = fw / fw.sum()                     # [m] frequency weights
                # T2=T3 = freq-weighted gate/up block-average (fixed); survivor keeps its
                # gate_up. Target Q = sum_j fw_j * SwiGLU_j(x); solve down' = Q^+ matching.
                # We keep survivor i's gate_up (its routing identity) and solve the down
                # projection (T1 folded into down) so the merged output reproduces the
                # frequency-weighted cluster output on the calib activations.
                hi = x @ gu[i].T                       # [T,2M]
                gi, ui = hi[:, :M], hi[:, M:]
                Pi = _act(gi) * ui                     # [T,M]  survivor pre-activation
                # cluster target output Q = sum_j fw_j * down_j @ SwiGLU_j(x)   [T,H]
                Q = torch.zeros(x.shape[0], x.shape[-1], device=args.device)
                for k, j in enumerate(members):
                    hj = x @ gu[j].T
                    gj, uj = hj[:, :M], hj[:, M:]
                    Q = Q + fw[k] * ((_act(gj) * uj) @ dn[j].T)
                # solve down' : Pi @ down'^T ≈ Q  ->  down'^T = (Pi^T Pi + λI)^-1 Pi^T Q
                PtP = Pi.T @ Pi + ridge * torch.eye(M, device=args.device)
                downT = torch.linalg.solve(PtP, Pi.T @ Q)   # [M,H]
                new_gu.append(gu[i].to(torch.bfloat16).cpu())
                new_dn.append(downT.T.contiguous().to(torch.bfloat16).cpu())
            survivor_w[li] = {"gate_up_proj": torch.stack(new_gu),
                              "down_proj": torch.stack(new_dn)}
            del gu, dn, x
            torch.cuda.empty_cache()
            if li % 6 == 0:
                log(f"  mergemoe fit L{li}: T1 lstsq ridge={ridge}")
        return Artifacts(survivor_weights=survivor_w, keep_meta=keep_meta,
                         meta={"method": "mergemoe", "ridge": ridge, "src_dir": args.student})


class REAM(HCSMoE):
    name = "ream"
    needs_tap = {"swiglu_io", "block_out"}
    blurb = "router-weighted expert activation merging (arXiv:2604.04356, REAP successor)"

    def _saliency(self, data, li, freq, mean_out, E):
        # REAM saliency = applied router gate-mass x activation magnitude (REAP/REAM
        # importance). gate_mass_e = sum over calib tokens of the top-k weight actually
        # routed to e (not a raw count), scaled by e's mean SwiGLU output norm. The
        # survivor keeps its OWN router row (A2 unchanged) -> sidesteps REAP Theorem 1
        # (no un-gateable packed neurons). Frequency-weighting (HCSMoE) is the special
        # case gate_mass~count, output-norm~1.
        import torch
        ti = data["top_idx"][li].reshape(-1).long()
        tw = data["top_w"][li].reshape(-1).float()
        gate_mass = torch.zeros(E).scatter_add_(0, ti, tw)
        return gate_mass * mean_out.float().norm(dim=-1)


class ExpertKD(_Phase1Stub):
    name = "expert_kd"
    # offline rank-probe / KD consumes the frozen-router input (router_in), the expert
    # input (swiglu_in, always captured) and the teacher block target (block_out).
    needs_tap = {"swiglu_io", "block_out", "router_in"}
    trainable = True
    blurb = "KD into TRAINABLE survivor experts (the only DIFFUSE-capable method)"


class SharedMLPKD(_Phase1Stub):
    name = "shared_mlp_kd"
    # dense mlp.* is fed by the same pre-FFN residual as the router (router_in).
    needs_tap = {"block_out", "router_in"}
    trainable = True
    blurb = "KD into Gemma-4 always-on parallel dense mlp.* via LoRA (sidesteps routing)"


REGISTRY: dict[str, type[RedistMethod]] = {
    m.name: m for m in (HCSMoE, MergeMoE, REAM, ExpertKD, SharedMLPKD)
}


# ─── stage helpers ────────────────────────────────────────────────────────────


def _sibling(args, name: str) -> Path:
    """Resolve a dep script host-independently (see _resolve_dep)."""
    try:
        return _resolve_dep(name, args.scripts_dir)
    except FileNotFoundError as e:
        raise SystemExit(f"FAIL: {e}")


def _req(val, flag: str, env: str | None = None):
    """Validate-at-use: a stage needs `val`; if it resolved to None (no CLI arg,
    no env, no on-host auto-detect), fail loudly naming the flag + env override."""
    if not val:
        extra = f" (or set {env})" if env else ""
        raise SystemExit(f"FAIL: {flag} is required{extra}")
    return val


def _run(cmd: list[str], dry: bool) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    if dry:
        log("(dry-run: not executed)")
        return 0
    return subprocess.call([str(c) for c in cmd])


def _driver(args) -> Driver:
    if args.driver not in DRIVERS:
        raise SystemExit(f"FAIL: unknown driver '{args.driver}'. Known: {sorted(DRIVERS)}")
    return DRIVERS[args.driver]


def _workpath(args, *parts: str) -> Path:
    p = Path(args.workdir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ─── stage: localize ──────────────────────────────────────────────────────────


def do_localize(args) -> None:
    drv = _driver(args)
    _req(args.teacher, "--teacher", "REDIST_TEACHER")
    _req(args.student, "--student", "REDIST_STUDENT")
    out = _workpath(args, f"localize_{drv.name}.json")
    if drv.fail_mode == "loop":
        # loop-differential: reuse router_diff_bucket.py (teacher vs student dropped-mass).
        _req(args.keep_meta, "--keep-meta", "REDIST_KEEP_META")
        _req(args.sample, "--sample", "REDIST_SAMPLE")
        script = _sibling(args, "router_diff_bucket.py")
        cmd = [args.python, script,
               "--a2", args.student, "--base", args.teacher,
               "--keep-meta", args.keep_meta, "--sample", args.sample,
               "--loop-bucket", drv.loop_bucket,
               "--out", str(out)]
    elif drv.fail_mode == "fluent":
        # fluent divergence: teacher-correct vs student-failing block-output divergence.
        corpus = args.corpus or drv.corpus
        if not corpus:
            raise SystemExit(f"FAIL: driver '{drv.name}' (fluent) needs --corpus")
        script = _sibling(args, "redist_localize_divergence.py")
        cmd = [args.python, script,
               "--student", args.student, "--teacher", args.teacher,
               "--keep-meta", args.keep_meta, "--corpus", corpus,
               "--out", str(out)]
    else:
        raise SystemExit(f"FAIL: bad fail_mode {drv.fail_mode}")
    rc = _run(cmd, args.dry_run)
    if rc:
        raise SystemExit(f"FAIL: localize rc={rc}")
    log(f"localize -> {out}")


# ─── stage: capture ───────────────────────────────────────────────────────────


def do_capture(args) -> None:
    method = REGISTRY[args.method]()
    taps = method.needs_tap
    drv = _driver(args)
    corpus = args.corpus or drv.corpus or args.sample
    out = _workpath(args, f"capture_{drv.name}_{args.method}.pt")
    log(f"method={args.method} needs_tap={sorted(taps)} trainable={method.trainable}")
    log(f"capture from teacher={args.teacher} on corpus={corpus} -> {out}")
    if args.dry_run:
        log("capture (dry-run): would teacher-force the corpus through 128e, hooking "
            "each layer's .experts (input/routing/output) + .router (softmax-over-E)")
        return

    _req(args.teacher, "--teacher", "REDIST_TEACHER")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not corpus or not Path(corpus).exists():
        raise SystemExit(f"FAIL: capture needs a corpus (got {corpus})")
    rows = [json.loads(x) for x in open(corpus)]
    # loop drivers filter the corpus to their bucket if present
    if drv.fail_mode == "loop" and any("bucket" in r for r in rows[:5]):
        rows = [r for r in rows if r.get("bucket") == drv.loop_bucket] or rows
    prompts = [(r.get("prompt") or r.get("text") or "") for r in rows][: args.max_seqs]
    log(f"corpus rows -> {len(prompts)} prompts (max_seqs={args.max_seqs}, "
        f"max_tokens={args.max_tokens})")

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager",
        device_map={"": args.device}).eval()

    # discover MoE layers via the fused .experts / .router modules
    exp = sorted([(int(n.split("layers.")[1].split(".")[0]), m)
                  for n, m in model.named_modules() if n.endswith(".experts")],
                 key=lambda t: t[0])
    rtr = {int(n.split("layers.")[1].split(".")[0]): m
           for n, m in model.named_modules() if n.endswith(".router")}
    L = len(exp)
    log(f"MoE layers discovered: {L}")

    cur: dict = {}    # per-forward scratch, keyed by (li, what)

    def pre_hook(li):
        def h(_m, a):
            # a = (hidden_states, top_k_index, top_k_weights)
            cur[(li, "in")] = a[0].detach().to("cpu", torch.float16)
            cur[(li, "idx")] = a[1].detach().to("cpu")
            cur[(li, "w")] = a[2].detach().to("cpu", torch.float16)
        return h

    def post_hook(li):
        def h(_m, _a, out):
            cur[(li, "out")] = out.detach().to("cpu", torch.float16)
        return h

    def rtr_hook(li):
        def h(_m, _i, out):
            cur[(li, "prob")] = out[0].detach().to("cpu", torch.float16)
        return h

    def rtr_pre(li):
        def h(_m, a):
            # router input = raw residual (hidden_states_flat), [B*S, H]
            cur[(li, "rin")] = a[0].detach().to("cpu", torch.float16)
        return h

    handles = []
    for li, m in exp:
        handles.append(m.register_forward_pre_hook(pre_hook(li)))
        handles.append(m.register_forward_hook(post_hook(li)))
    for li, m in rtr.items():
        handles.append(m.register_forward_hook(rtr_hook(li)))
        if "router_in" in taps:
            handles.append(m.register_forward_pre_hook(rtr_pre(li)))

    store = {k: {li: [] for li in range(L)}
             for k in ("swiglu_in", "block_out", "top_idx", "top_w",
                       "router_prob", "router_in")}
    n_tok = 0
    for i, p in enumerate(prompts):
        chat = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        enc = tok(chat, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=args.max_tokens).to(args.device)
        cur.clear()
        with torch.no_grad():
            model(input_ids=enc["input_ids"],
                  attention_mask=torch.ones_like(enc["input_ids"]),
                  mm_token_type_ids=torch.zeros_like(enc["input_ids"]), use_cache=False)
        # experts I/O is token-major 2-D [B*S, *] (residual is reshape(-1,H)-flattened
        # before experts()), so store the full tensor — do NOT index [0] (that was the
        # first-token-only bug caught by the bs2 smoke).
        for li in range(L):
            store["swiglu_in"][li].append(cur[(li, "in")])        # [T,H]
            store["block_out"][li].append(cur[(li, "out")])       # [T,H]
            store["top_idx"][li].append(cur[(li, "idx")])         # [T,K]
            store["top_w"][li].append(cur[(li, "w")])             # [T,K]
            if "router" in taps:
                store["router_prob"][li].append(cur[(li, "prob")])  # [T,E]
            if "router_in" in taps:
                store["router_in"][li].append(cur[(li, "rin")])   # [T,H]
        n_tok += enc["input_ids"].shape[1]
        if (i + 1) % 16 == 0:
            log(f"  captured {i + 1}/{len(prompts)} seqs, {n_tok} tokens")
    for h in handles:
        h.remove()

    cat = {k: {li: __import__("torch").cat(v, 0) for li, v in d.items() if v}
           for k, d in store.items() if (k != "router_prob" or "router" in taps)}
    payload = {
        "taps": sorted(taps), "n_layers": L, "n_tokens": n_tok,
        "corpus": corpus, "teacher": args.teacher,
        "num_experts": int(model.config.text_config.num_experts),
        "hidden": cat["swiglu_in"][0].shape[-1],
        "data": cat,
    }
    import torch as _t
    _t.save(payload, out)
    log(f"capture done: {n_tok} tokens, {L} layers -> {out}")


# ─── stage: redistribute ──────────────────────────────────────────────────────


def do_redistribute(args) -> None:
    method = REGISTRY[args.method]()
    log(f"method={method.name}: {method.blurb}  (trainable={method.trainable})")
    if args.dry_run:
        log(f"would: load keep-meta+capture -> plan -> fit -> "
            f"{'emit recovered 62e to ' + args.emit if args.emit else 'save Artifacts .pt'}")
        return
    _req(args.keep_meta, "--keep-meta", "REDIST_KEEP_META")
    keep_meta = load_keep_meta(args.keep_meta)
    localize = json.load(open(args.localize)) if args.localize else {}
    plan = method.plan(localize, keep_meta)
    log("plan ready")
    capture = _load_capture(args.capture)
    art = method.fit(capture, plan, keep_meta, args)
    if args.emit:
        # closed-form merge artifacts are the full ~22 GB survivor weight set — emit
        # the recovered 62e directly instead of saving+reloading a giant .pt.
        method.emit(art, args.emit)
        log(f"emitted recovered 62e -> {args.emit}")
    else:
        out = _workpath(args, f"artifacts_{args.driver}_{args.method}.pt")
        _save_artifacts(art, out)
        log(f"artifacts -> {out}")


def _load_capture(path):
    import torch
    # our own capture/Artifacts files (dict-of-tensors / Artifacts dataclass)
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_artifacts(art: Artifacts, path):
    import torch
    torch.save(art, path)


# ─── stage: gate (AR-first, two-tier) ─────────────────────────────────────────


def do_gate(args) -> None:
    """Tier-1 cheap gate: a small generation canary via loop_screen.py on the
    target bucket. (Block-output divergence tier-1 is emitted by the method's fit
    and read here; Phase 1 wires the numeric comparison.) Tier-2 (full 200-prompt
    loop_screen + capability bench) is invoked with --tier 2."""
    drv = _driver(args)
    _req(args.model, "--model")
    _req(args.sample, "--sample", "REDIST_SAMPLE")
    script = _sibling(args, "loop_screen.py")
    res = _workpath(args, f"gate_{drv.name}_{args.method}_tier{args.tier}.json")
    max_new = 2048
    sample = args.sample
    cmd = [args.python, script, "--model", args.model, "--out", str(res),
           "--name", f"{args.driver}_{args.method}_t{args.tier}",
           "--sample", sample, "--bs", "16", "--max-new", str(max_new)]
    rc = _run(cmd, args.dry_run)
    if rc:
        raise SystemExit(f"FAIL: gate rc={rc}")
    log(f"gate tier-{args.tier} -> {res}  (AR-first: judge on loop% / divergence, "
        f"NEVER reconstruction MSE)")


# ─── stage: build ─────────────────────────────────────────────────────────────


def do_build(args) -> None:
    """Materialize a recovered 62e: if the keep-set changed (e.g. REAM centroids),
    run expert_drop.py to slice; then apply the method's Artifacts (emit); then
    screen. For methods that only edit survivor weights of the EXISTING A2 keep-set,
    the student dir is copied and overwritten in-place by emit."""
    method = REGISTRY[args.method]()
    _req(args.output_dir, "--output-dir")
    log(f"build: method={method.name} -> {args.output_dir}")
    if args.drop_map:
        _req(args.teacher, "--teacher", "REDIST_TEACHER")
        script = _sibling(args, "expert_drop.py")
        cmd = [args.python, script, "--source-dir", args.teacher,
               "--drop-map", args.drop_map, "--output-dir", args.output_dir]
        rc = _run(cmd, args.dry_run)
        if rc:
            raise SystemExit(f"FAIL: expert_drop rc={rc}")
    else:
        log("no --drop-map: redistribute-into-A2-survivors path "
            "(emit overwrites student expert tensors in a copy)")
    if args.artifacts and not args.dry_run:
        art = _load_capture(args.artifacts)
        method.emit(art, args.output_dir)
    log("build done (Phase 1 wires emit + post-build loop_screen)")


# ─── stage: run (chain) ───────────────────────────────────────────────────────


def do_run(args) -> None:
    log(f"=== REDIST run: driver={args.driver} method={args.method} (dry={args.dry_run}) ===")
    do_localize(args)
    do_capture(args)
    do_redistribute(args)
    log("run: localize+capture+redistribute staged; gate+build are explicit stages "
        "(need the built model dir). Phase 1 fills capture/fit/emit bodies.")


# ─── info ─────────────────────────────────────────────────────────────────────


def do_methods(args) -> None:
    print(f"{'method':<14} {'trainable':<10} {'needs_tap':<22} blurb")
    print("-" * 96)
    for name, cls in REGISTRY.items():
        m = cls()
        print(f"{name:<14} {str(m.trainable):<10} {','.join(sorted(m.needs_tap)):<22} {m.blurb}")
    print()
    print("drivers:")
    for d in DRIVERS.values():
        print(f"  {d.name:<14} fail_mode={d.fail_mode:<7} "
              f"loop_bucket={d.loop_bucket}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--driver", default="multilingual",
                   help=f"target capability {sorted(DRIVERS)}")
    p.add_argument("--method", default="hcsmoe", choices=sorted(REGISTRY))
    p.add_argument("--teacher", default=TEACHER_DEFAULT, help="128e teacher dir")
    p.add_argument("--student", default=STUDENT_DEFAULT, help="pruned 62e (A2) dir")
    p.add_argument("--keep-meta", default=KEEPMETA_DEFAULT,
                   help="expert_drop_metadata.json / a2_keep_metadata.json")
    p.add_argument("--sample", default=SAMPLE_DEFAULT, help="loop_screen sample jsonl")
    p.add_argument("--corpus", default=None, help="driver corpus jsonl (fluent caps)")
    p.add_argument("--workdir", default=WORKDIR_DEFAULT)
    p.add_argument("--scripts-dir", default=str(SCRIPTS_DIR_DEFAULT),
                   help="dir holding sibling scripts (default: this file's dir)")
    p.add_argument("--python", default=sys.executable, help="python for sibling subprocesses")
    p.add_argument("--device", default="cuda:0", help="device for capture/fit")
    p.add_argument("--max-seqs", type=int, default=128, help="capture: prompts to forward")
    p.add_argument("--max-tokens", type=int, default=512, help="capture: truncate each seq")
    p.add_argument("--dry-run", action="store_true",
                   help="print what each stage would do without GPU/subprocess")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="redist", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in [("localize", do_localize), ("capture", do_capture),
                     ("redistribute", do_redistribute), ("run", do_run)]:
        p = sub.add_parser(name)
        _add_common(p)
        if name == "redistribute":
            p.add_argument("--localize", default=None)
            p.add_argument("--capture", default=None)
            p.add_argument("--emit", default=None,
                           help="emit the recovered 62e directly to this dir "
                                "(merge methods; avoids a ~22 GB artifacts .pt)")
            p.add_argument("--ridge", type=float, default=1e-2,
                           help="mergemoe T1 lstsq ridge")
        p.set_defaults(func=fn)

    pg = sub.add_parser("gate")
    _add_common(pg)
    pg.add_argument("--model", required=True, help="built model dir to screen")
    pg.add_argument("--tier", type=int, default=1, choices=[1, 2])
    pg.set_defaults(func=do_gate)

    pb = sub.add_parser("build")
    _add_common(pb)
    pb.add_argument("--output-dir", required=True)
    pb.add_argument("--drop-map", default=None, help="if keep-set changed (e.g. REAM)")
    pb.add_argument("--artifacts", default=None, help="Artifacts .pt from redistribute")
    pb.set_defaults(func=do_build)

    pm = sub.add_parser("methods", help="list methods + drivers")
    pm.set_defaults(func=do_methods)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
