#!/usr/bin/env python3
"""Wrapper around mergekit PR #682's `compute_lrp_for_model` that loads
calibration prompts from a file (one per line) instead of taking them on
the command line. Lets us pass 32+ multi-line prompts cleanly.

Run from the `fisher` conda env (CUDA torch + transformers + lxt).
"""
import argparse
import sys
from pathlib import Path


def load_prompts(cal_path: Path, n: int) -> list[str]:
    out = []
    with open(cal_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path,
                    help="Output DIRECTORY (lrp_scores.safetensors written inside)")
    ap.add_argument("--cal-data", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mergekit-dir", type=Path, default=Path("/workspace/mergekit"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.mergekit_dir))
    from lrp_computer import compute_lrp_for_model  # noqa: E402

    prompts = load_prompts(args.cal_data, args.num_samples)
    print(f"Loaded {len(prompts)} calibration prompts from {args.cal_data}", flush=True)
    if not prompts:
        sys.exit("no prompts loaded — calibration file empty?")

    args.output.mkdir(parents=True, exist_ok=True)
    compute_lrp_for_model(
        model_path=str(args.model),
        output_path=str(args.output),
        sample_prompts=prompts,
        max_length=args.max_length,
        device=args.device,
    )
    print(f"DONE → {args.output}/lrp_scores.safetensors")


if __name__ == "__main__":
    main()
