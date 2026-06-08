#!/usr/bin/env python
# patch_yarn_config.py — apply YaRN extension factor to a Gemma 4 config.json.
#
# Targets ONLY rope_parameters.full_attention (the global layers). The sliding-
# attention layers are left alone — their window (1024) is bounded regardless
# of context length, and re-scaling their rope_theta would break short-range
# behavior. Verified on 2026-05-28:
#   31B-it:     50 sliding + 10 full layers, full-attn indices [5,11,…,59]
#   26B-A4B:    25 sliding +  5 full layers, full-attn indices [5,11,17,23,29]
#
# Gemma 4 full-attn rope is:
#   {partial_rotary_factor: 0.25, rope_theta: 1000000.0, rope_type: 'proportional'}
# This script writes:
#   {partial_rotary_factor: 0.25, rope_theta: 1000000.0,
#    rope_type: 'proportional_yarn',    # NEW key — registered by proportional_yarn_rope_init.py
#    factor: <factor>,                  # YaRN scale (alias of yarn_factor in our init fn)
#    original_max_position_embeddings: <native>,
#    beta_fast: <opt>, beta_slow: <opt>,
#    attention_factor: <opt>}           # YaRN mscale = 0.1·ln(s)+1 (≈1.0693 for s=2.0)
#
# ### COUNCIL VERDICT (csl-2026-05-28-1825-5f1b, 2026-05-28) — APPLIED
# Q1a source audit confirmed transformers 5.5.0's
# `_compute_proportional_rope_parameters` (modeling_rope_utils.py:187-254)
# IGNORES yarn_factor/beta_fast/beta_slow/mscale entirely — it only reads
# rope_theta + factor + partial_rotary_factor and applies plain PI. So this
# script previously would have produced a config that silently trained on PI
# rather than YaRN.
#
# Fixes:
#  1. rope_type = 'proportional_yarn' (NOT 'proportional') — dispatches via
#     ROPE_INIT_FUNCTIONS['proportional_yarn'] which is registered by
#     ./proportional_yarn_rope_init.py. That module MUST be imported before
#     model load (see phase1_train_yarn_lora.py).
#  2. Write `attention_factor` (the JSON key transformers actually reads)
#     directly, NOT `mscale` + `mscale_all_dim`. mscale_all_dim=0.0 is falsy
#     and triggers get_mscale(factor) fallback at modeling_rope_utils.py:412
#     — silently overrides the user's mscale.
#  3. mscale default for factor=2.0 is 1.0693 (= 0.1·ln(2)+1) per YaRN §5.1.
#
# Usage:
#   patch_yarn_config.py  --src <model-dir>  --dst <out-dir> \
#                         --factor 2.0  --native 262144
#   patch_yarn_config.py  --dry-run  --src ...  --dst ...  --factor 2.0

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

DEFAULT_BETA_FAST = 32       # YaRN paper §3.3
DEFAULT_BETA_SLOW = 1


def yarn_default_attention_factor(factor: float) -> float:
    """YaRN paper §5.1: mscale = 0.1·ln(s) + 1. For s=2.0 returns ≈ 1.0693."""
    return 0.1 * math.log(float(factor)) + 1.0


def patch(cfg: dict, factor: float, native: int, *,
          beta_fast: float = DEFAULT_BETA_FAST,
          beta_slow: float = DEFAULT_BETA_SLOW,
          attention_factor: float | None = None) -> dict:
    """In-place edit of a loaded gemma4 config.json dict. Returns it."""
    tc = cfg.get("text_config", cfg)  # gemma4 nests under text_config
    # 1. Bump max_position_embeddings
    new_max = int(round(native * factor))
    tc["max_position_embeddings"] = new_max
    # 2. Patch rope_parameters.full_attention with YaRN fields, switching
    #    rope_type to our custom 'proportional_yarn' key (registered by
    #    proportional_yarn_rope_init.py — must be imported before model load).
    rp = tc.get("rope_parameters", {})
    fa = rp.get("full_attention", {})
    if not fa:
        raise SystemExit("config has no rope_parameters.full_attention — refusing to patch")
    # Sanity: must be proportional with partial_rotary_factor 0.25
    if fa.get("rope_type") != "proportional":
        print(f"WARN: rope_type was {fa.get('rope_type')!r}, expected 'proportional'", file=sys.stderr)
    if abs(float(fa.get("partial_rotary_factor", 0)) - 0.25) > 1e-6:
        print(f"WARN: partial_rotary_factor was {fa.get('partial_rotary_factor')!r}, expected 0.25", file=sys.stderr)
    # Strip any legacy yarn_* / mscale keys (lingering from earlier runs of
    # this script — they'd confuse downstream readers that look for
    # attention_factor first).
    for legacy_key in ("yarn_factor", "mscale", "mscale_all_dim"):
        fa.pop(legacy_key, None)
    if attention_factor is None:
        attention_factor = yarn_default_attention_factor(factor)
    fa.update({
        "rope_type": "proportional_yarn",           # see proportional_yarn_rope_init.py
        "factor": float(factor),                    # YaRN scale (our init fn reads this)
        "original_max_position_embeddings": int(native),
        "beta_fast": float(beta_fast),
        "beta_slow": float(beta_slow),
        "attention_factor": float(attention_factor),
    })
    rp["full_attention"] = fa
    tc["rope_parameters"] = rp
    # 3. Don't touch sliding_attention rope.
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source model dir (must contain config.json)")
    ap.add_argument("--dst", required=True, help="destination dir (will copy weights via hardlink + write patched config)")
    ap.add_argument("--factor", type=float, default=2.0, help="YaRN extension factor (default 2.0 → 256k→512k)")
    ap.add_argument("--native", type=int, default=262144, help="native max_position_embeddings (default 262144 = 256k)")
    ap.add_argument("--attention-factor", type=float, default=None,
                    help="YaRN mscale (overrides default 0.1*ln(factor)+1). "
                         "For s=2.0 the default is ≈1.0693.")
    ap.add_argument("--dry-run", action="store_true", help="print plan + diff, do not write")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    cfg_path = src / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"FATAL: {cfg_path} not found")
    cfg = json.load(cfg_path.open())

    print("=== YaRN config patch ===")
    print(f"  src: {src}")
    print(f"  dst: {dst}")
    print(f"  factor: {args.factor}")
    print(f"  native: {args.native}")
    print(f"  new max_position_embeddings: {int(round(args.native * args.factor))}")

    eff_af = (args.attention_factor
              if args.attention_factor is not None
              else yarn_default_attention_factor(args.factor))
    print(f"  attention_factor: {eff_af:.4f}  "
          f"({'explicit' if args.attention_factor is not None else 'YaRN-default'})")

    patched = patch(json.loads(json.dumps(cfg)), args.factor, args.native,
                    attention_factor=args.attention_factor)

    if args.dry_run:
        # Show only the changed fields
        import difflib
        before = json.dumps(cfg.get("text_config", {}).get("rope_parameters", {}), indent=2, sort_keys=True).splitlines()
        after = json.dumps(patched.get("text_config", {}).get("rope_parameters", {}), indent=2, sort_keys=True).splitlines()
        print("\n--- rope_parameters DIFF ---")
        for line in difflib.unified_diff(before, after, lineterm="", n=2):
            print(line)
        print("\n[dry-run] no files written. Re-run without --dry-run to execute.")
        return

    # Real write
    dst.mkdir(parents=True, exist_ok=True)
    # Hardlink all top-level FILES except config.json (we'll write the patched
    # one). Skip subdirectories — model weights/tokenizer/aux are all top-level;
    # subdirs like `.eval_results` are run cruft we don't want in the YaRN dir
    # (and hardlink_to/copy2 both fail on a directory).
    n_linked = 0
    skipped_dirs = []
    for f in src.iterdir():
        if f.name == "config.json":
            continue
        if f.is_dir():
            skipped_dirs.append(f.name)
            continue
        tgt = dst / f.name
        if tgt.exists():
            continue
        try:
            tgt.hardlink_to(f)  # py3.10+
        except OSError:
            shutil.copy2(f, tgt)
        n_linked += 1
    # Write patched config
    (dst / "config.json").write_text(json.dumps(patched, indent=2))
    print(f"  wrote {dst / 'config.json'}")
    print(f"  hardlinked {n_linked} weight/aux files")
    if skipped_dirs:
        print(f"  skipped {len(skipped_dirs)} subdir(s): {', '.join(skipped_dirs)}")
    print("[done]")


if __name__ == "__main__":
    main()
