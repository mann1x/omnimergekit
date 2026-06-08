#!/usr/bin/env python
# make_serve_config.py — derive a vLLM-servable dir from the merged 512k model by
# translating the rope config, leaving the merged weights untouched (hardlinked).
#
# WHY A SEPARATE STEP (not folded into the merge):
#   The merged dir carries the CANONICAL training rope: rope_type =
#   'proportional_yarn' — a CUSTOM transformers init (proportional_yarn_rope_init.py)
#   that composes YaRN over Gemma 4's proportional base. vLLM 0.20.2 does NOT know
#   'proportional_yarn'; its rope factory knows 'yarn'/'proportional'/'linear'/…
#   We therefore write a SERVE-ONLY config that maps full_attention.rope_type
#   'proportional_yarn' → vLLM 'yarn', carrying every YaRN field verbatim
#   (factor, original_max_position_embeddings, beta_fast, beta_slow,
#   attention_factor).
#
#   THIS MAPPING IS A HYPOTHESIS, NOT A PROVEN EQUIVALENCE. vLLM 'yarn' is YaRN
#   over the STANDARD rope base; training used YaRN over Gemma 4's proportional
#   base. For Gemma 4 the proportional base with no extra factor reduces to the
#   standard rope_theta=1e6 over the partial-rotary (0.25) dims, so the two
#   *should* coincide — but the driver MUST gate on it: serve this dir and confirm
#   the EXTENDED model's RULER score at 256k matches the BASE 256k anchor before
#   trusting 384k/512k. If the anchor diverges, re-run this script with a
#   different --rope-type (cheap: hardlinks only) and re-serve — no re-merge.
#
# Because it only hardlinks weights + writes one config.json, it is seconds-cheap
# and fully re-runnable. Also (idempotently) synthesizes preprocessor_config.json
# (vLLM 0.20.x requires it for Gemma 4) if the merged dir lacks one.
#
# Usage:
#   make_serve_config.py --merged /srv/ml/longctx/gemma-4-26B-A4B-it-512k \
#                        --out    /srv/ml/longctx/gemma-4-26B-A4B-it-512k-vllm-yarn \
#                        [--rope-type yarn] [--dry-run]

import argparse
import json
import shutil
import sys
from pathlib import Path

# rope_type values transformers writes for the trained model, and what vLLM
# understands. We only translate the CUSTOM one; a config already on a vLLM-known
# type is passed through (handy for re-derivations).
CUSTOM_ROPE = "proportional_yarn"
VLLM_KNOWN = {"yarn", "proportional", "linear", "dynamic", "llama3", "longrope", "ntk", "default"}
# YaRN fields to carry verbatim from the trained full_attention block.
YARN_FIELDS = ("factor", "original_max_position_embeddings",
               "beta_fast", "beta_slow", "attention_factor")


def fatal(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def translate_rope(cfg: dict, rope_type: str) -> tuple[dict, dict]:
    """Rewrite text_config.rope_parameters.full_attention.rope_type → rope_type,
    preserving partial_rotary_factor/rope_theta + all YaRN fields. Returns
    (patched_cfg, before_fa) for logging."""
    tc = cfg.get("text_config", cfg)
    rp = tc.get("rope_parameters")
    if not rp or "full_attention" not in rp:
        fatal("merged config has no text_config.rope_parameters.full_attention")
    fa = rp["full_attention"]
    before = dict(fa)
    src_type = fa.get("rope_type")
    if src_type == rope_type:
        print(f"  rope_type already {rope_type!r} — pass-through")
    elif src_type == CUSTOM_ROPE:
        print(f"  rope_type {src_type!r} → {rope_type!r} (carrying YaRN fields)")
    elif src_type in VLLM_KNOWN:
        print(f"  WARN: source rope_type {src_type!r} is already vLLM-known; "
              f"forcing → {rope_type!r} anyway (--rope-type)")
    else:
        print(f"  WARN: unexpected source rope_type {src_type!r}; setting → {rope_type!r}")
    # carry YaRN fields verbatim; absence of attention_factor lets vLLM derive
    # the default mscale (0.1·ln(factor)+1) — but the trained config sets it, so
    # we keep it explicit.
    new_fa = dict(fa)
    new_fa["rope_type"] = rope_type
    missing = [k for k in YARN_FIELDS if k not in new_fa and k != "attention_factor"]
    if "factor" in missing:
        fatal(f"full_attention lacks 'factor' — cannot build a {rope_type} config")
    rp["full_attention"] = new_fa
    tc["rope_parameters"] = rp
    return cfg, before


def synth_preprocessor(d: Path) -> None:
    pp = d / "preprocessor_config.json"
    if pp.exists():
        print("  preprocessor_config.json present — leaving as-is")
        return
    proc_f = d / "processor_config.json"
    if not proc_f.exists():
        print("  WARN: no preprocessor_config.json AND no processor_config.json — "
              "vLLM may fail to boot Gemma 4. Continuing (text-only RULER may still work).")
        return
    proc = json.loads(proc_f.read_text())
    fe = proc.get("feature_extractor", {}) or proc
    pp.write_text(json.dumps(fe, indent=2))
    print(f"  synth preprocessor_config.json ← processor_config.json ({len(fe)} keys)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="merged 512k dir (proportional_yarn config)")
    ap.add_argument("--out", required=True, help="serve dir (hardlinked weights + translated config)")
    ap.add_argument("--rope-type", default="yarn", help="vLLM rope_type to write (default: yarn)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    merged = Path(args.merged)
    out = Path(args.out)
    cfg_path = merged / "config.json"
    if not cfg_path.is_file():
        fatal(f"missing {cfg_path}")

    cfg = json.loads(cfg_path.read_text())
    print("=== serve-config translation ===")
    print(f"  merged : {merged}")
    print(f"  out    : {out}")
    print(f"  rope   : → {args.rope_type}")
    patched, before = translate_rope(json.loads(json.dumps(cfg)), args.rope_type)

    if args.dry_run:
        import difflib
        a = json.dumps(before, indent=2, sort_keys=True).splitlines()
        tc = patched.get("text_config", patched)
        after = json.dumps(tc["rope_parameters"]["full_attention"], indent=2, sort_keys=True).splitlines()
        print("\n--- full_attention rope DIFF ---")
        for line in difflib.unified_diff(a, after, lineterm="", n=3):
            print(line)
        print("\n[dry-run] no files written.")
        return

    out.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    for f in merged.iterdir():
        if f.is_dir():
            continue
        if f.name == "config.json":
            continue
        tgt = out / f.name
        if tgt.exists():
            continue
        try:
            tgt.hardlink_to(f)
        except OSError:
            shutil.copy2(f, tgt)
        n_linked += 1
    (out / "config.json").write_text(json.dumps(patched, indent=2))
    print(f"  wrote {out / 'config.json'}")
    print(f"  hardlinked {n_linked} file(s) from merged dir")
    synth_preprocessor(out)
    print(f"[done] serve dir ready → {out}")


if __name__ == "__main__":
    main()
