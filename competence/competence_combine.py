#!/usr/bin/env python3
"""Combine per-(source, task) competence maps into one per-source map.

Each per-source-per-task safetensors (from competence_extract.py) holds:
  <name>                  primary signal (Σ |W.grad|, default for Fisher consumer)
  <name>.grad_sq          Fisher diagonal
  <name>.weight_taylor    TaylorFO at weight level
  <name>.act_taylor       TaylorFO at input-channel level (1D, optional)

Combine recipe:
  1. Pick a `--signal` (default: weight_taylor — best balance per Han et al.).
  2. Per (source, task), normalize the signal by its own mean → unit-mean tensors.
     Without this, larger absolute grads on one task dominate.
  3. Per source, combine across tasks weighted by (success_rate − base_rate)+
     so a task contributes only the source's *above-floor* competence.
     If --raw-rate is set, weight by raw success rate instead.
  4. Output one safetensors per source, primary key = name. Drop into
     omnimergekit's --fisher consumer directly.

Usage:
  python competence_combine.py \
      --map jackrong-v2:he:eval_results/.../results.json:competence/jackrong-v2__he.safetensors \
      --map jackrong-v2:mbpp:...:competence/jackrong-v2__mbpp.safetensors \
      --map jackrong-v2:lcb:...:competence/jackrong-v2__lcb.safetensors \
      --map continuum-forged:he:...:competence/continuum-forged__he.safetensors \
      ... \
      --base-rates he=0.6037,mbpp=0.4600,lcb=0.1000 \
      --signal weight_taylor \
      --output-dir competence/combined/
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file, save_file


SIGNAL_SUFFIXES = {
    "grad_l1": "",  # primary key (no suffix)
    "grad_sq": ".grad_sq",
    "weight_taylor": ".weight_taylor",
    "act_taylor": ".act_taylor",
}


def parse_map_arg(s: str) -> Tuple[str, str, str, Path]:
    """Parse 'source:task:rate_or_results.json:safetensors_path'."""
    parts = s.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--map expected 4 colon-separated parts, got {len(parts)}: {s}")
    return parts[0], parts[1], parts[2], Path(parts[3])


def parse_base_rates(s: str) -> Dict[str, float]:
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def resolve_rate(rate_str: str, task: str) -> float:
    """Either a float, or a path to lm_eval results.json (extract pass@1 for the task)."""
    try:
        return float(rate_str)
    except ValueError:
        pass
    p = Path(rate_str)
    if not p.exists():
        raise FileNotFoundError(f"results file not found: {rate_str}")
    obj = json.loads(p.read_text())
    res = obj.get("results", {})
    # try common shapes
    if task in res:
        r = res[task]
        for k in ("pass@1,create_test", "pass_at_1,none", "pass@1", "pass_at_1", "exact_match,none"):
            if k in r:
                return float(r[k])
    raise ValueError(f"could not extract rate for task={task} from {p}")


def extract_signal(tensors: Dict[str, torch.Tensor], signal: str) -> Dict[str, torch.Tensor]:
    suffix = SIGNAL_SUFFIXES[signal]
    out = {}
    for k, v in tensors.items():
        if suffix == "":
            # primary key: name itself, exclude any name with a known suffix
            if any(k.endswith(sfx) for sfx in SIGNAL_SUFFIXES.values() if sfx):
                continue
            out[k] = v
        else:
            if k.endswith(suffix):
                out[k[: -len(suffix)]] = v
    return out


def normalize_unit_mean(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = t.float().mean()
    if m < eps:
        return t.float()
    return t.float() / m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", action="append", required=True, type=parse_map_arg,
                    help="One per (source, task): 'source:task:rate_or_results.json:safetensors_path'. "
                         "Pass repeatedly.")
    ap.add_argument("--base-rates", type=parse_base_rates, default=None,
                    help="Per-task base-model success rate to subtract: 'he=0.6037,mbpp=0.46,lcb=0.10'. "
                         "When set, source weight per task = max(source_rate − base_rate, 0). "
                         "Without this, --raw-rate must be set.")
    ap.add_argument("--raw-rate", action="store_true",
                    help="Weight by raw success rate (not above-base). Set this if you don't have base rates "
                         "or want pure competence rather than incremental.")
    ap.add_argument("--signal", choices=list(SIGNAL_SUFFIXES), default="weight_taylor",
                    help="Which signal to combine. Default: weight_taylor (TaylorFO at weight level).")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Per-source combined safetensors written here as <source>.safetensors.")
    ap.add_argument("--task-equal-floor", type=float, default=0.0,
                    help="Floor on each task weight before normalization, e.g. 0.05 — guarantees every "
                         "task contributes at least this much even if source is weak on it.")
    args = ap.parse_args()

    if not args.raw_rate and args.base_rates is None:
        print("ERROR: pass either --base-rates 'task=rate,...' or --raw-rate", file=sys.stderr)
        sys.exit(1)

    # Group by source
    by_source: Dict[str, List[Tuple[str, str, Path]]] = defaultdict(list)
    for src, task, rate_str, path in args.map:
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            sys.exit(1)
        by_source[src].append((task, rate_str, path))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for src, entries in by_source.items():
        print(f"=== {src} : {len(entries)} task(s) ===")
        # Resolve task weights
        weights: Dict[str, float] = {}
        for task, rate_str, _ in entries:
            rate = resolve_rate(rate_str, task)
            if args.raw_rate:
                w = rate
            else:
                base = args.base_rates.get(task, 0.0)
                w = max(rate - base, 0.0)
            w = max(w, args.task_equal_floor)
            weights[task] = w
            print(f"   task={task:6s} rate={rate:.4f} weight={w:.4f}")
        wsum = sum(weights.values())
        if wsum <= 0:
            print("   WARN: all task weights zero — falling back to equal weights")
            weights = {t: 1.0 for t in weights}
            wsum = float(len(weights))
        weights = {t: w / wsum for t, w in weights.items()}

        # Load + extract + normalize each task's signal
        per_task_norm: Dict[str, Dict[str, torch.Tensor]] = {}
        ref_keys = None
        for task, _, path in entries:
            print(f"   load {path.name} ({path.stat().st_size/1e9:.2f} GB)")
            ts = load_file(str(path))
            sig = extract_signal(ts, args.signal)
            del ts
            if not sig:
                print(f"   WARN: no signal '{args.signal}' tensors in {path}", file=sys.stderr)
                continue
            sig = {k: normalize_unit_mean(v) for k, v in sig.items()}
            per_task_norm[task] = sig
            if ref_keys is None:
                ref_keys = set(sig.keys())
            else:
                missing = ref_keys - set(sig.keys())
                extra = set(sig.keys()) - ref_keys
                if missing or extra:
                    print(f"   WARN task={task}: {len(missing)} missing, {len(extra)} extra keys vs ref")

        if not per_task_norm:
            print(f"   skip {src}: no usable tasks")
            continue

        # Combine: weighted sum across tasks, name-by-name
        combined: Dict[str, torch.Tensor] = {}
        # Use union of keys actually present, weighted by available task weights
        all_keys = set()
        for sig in per_task_norm.values():
            all_keys |= set(sig.keys())
        for k in sorted(all_keys):
            acc = None
            wsum_k = 0.0
            for task, sig in per_task_norm.items():
                if k not in sig:
                    continue
                w = weights[task]
                if acc is None:
                    acc = sig[k] * w
                else:
                    acc += sig[k] * w
                wsum_k += w
            if acc is not None and wsum_k > 0:
                combined[k] = (acc / wsum_k).to(torch.float32)

        out_path = args.output_dir / f"{src.replace('/', '_')}.safetensors"
        metadata = {
            "source": src,
            "signal": args.signal,
            "tasks": ",".join(t for t, _, _ in entries),
            "weights": json.dumps(weights),
            "raw_rate": str(args.raw_rate),
        }
        save_file(combined, str(out_path), metadata=metadata)
        print(f"   wrote {out_path} ({len(combined)} tensors, "
              f"{out_path.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
