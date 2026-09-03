#!/usr/bin/env python
"""Retro-gate quant tiers that were already published without the post-push gate.

`quantize_gguf.py` gates every tier it publishes (default ON since 2026-09-03),
but tiers shipped by an earlier run — or by a ladder whose Python process had
already loaded the pre-gate source — were never certified against the bytes
that actually landed. This sweep closes that gap: it pulls each published GGUF
back from HF, runs the SAME `sanity_check()` the publish path runs, and applies
the SAME withdrawal policy on failure.

It is deliberately the same gate, imported from `quantize_gguf`, not a
reimplementation — a second scorer would certify a different thing.

Usage:
    export HF_TOKEN="${HF_TOKEN:?...}"
    python scripts/gate_published_tiers.py \
        --repo mannix/Qwen3.8-27B-Omnimerge-v6-GGUF \
        --ollama-target mannix/omnimerge-v6 \
        --scratch /mnt/sdc/omnimerge_v6/_gate_scratch

    # certify a subset, or resume after an interruption
    python scripts/gate_published_tiers.py --repo ... --only IQ2_M,IQ2_S

Disk: one tier at a time; the scratch file is removed before the next pull, so
peak usage is a single GGUF. Report is written to <scratch>/gate_report.json
after EVERY tier, so an interrupted sweep never loses its verdicts.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_quantize_gguf():
    """Import quantize_gguf as a module to reuse the canonical gate.

    It guards main() behind `if __name__ == "__main__"`, so importing runs
    only constants and defs — no side effects.
    """
    spec = importlib.util.spec_from_file_location(
        "quantize_gguf", _HERE / "quantize_gguf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_published_ggufs(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id, repo_type="model")
    return sorted(f for f in files if f.endswith(".gguf"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF GGUF repo to certify")
    ap.add_argument("--ollama-target", default=None,
                    help="ollama namespace (e.g. mannix/omnimerge-v6). Only used "
                         "to name the tag a FAILED tier needs removed by hand.")
    ap.add_argument("--scratch", required=True,
                    help="Persistent-disk scratch dir for the one-at-a-time "
                         "downloads. NEVER point this at /tmp (tmpfs).")
    ap.add_argument("--only", default=None,
                    help="Comma-separated tier names to certify (default: all)")
    ap.add_argument("--skip-base", action="store_true", default=True,
                    help="Skip F16/F32 base GGUFs (default: on)")
    ap.add_argument("--port", type=int, default=18133)
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be certified, download nothing")
    args = ap.parse_args()

    scratch = Path(args.scratch)
    if str(scratch).startswith("/tmp"):
        print("REFUSING: --scratch is under /tmp (tmpfs). Use persistent disk.",
              file=sys.stderr)
        return 2
    scratch.mkdir(parents=True, exist_ok=True)

    qg = _load_quantize_gguf()
    tools = qg.find_llama_cpp()
    if not tools.get("server"):
        print("REFUSING: llama-server not found; the gate cannot run.",
              file=sys.stderr)
        return 2

    ggufs = list_published_ggufs(args.repo)
    if args.skip_base:
        ggufs = [f for f in ggufs
                 if not Path(f).stem.upper().endswith(("F16", "F32"))]
    if args.only:
        want = {t.strip().upper() for t in args.only.split(",") if t.strip()}
        ggufs = [f for f in ggufs
                 if any(Path(f).stem.upper().endswith("-" + t) for t in want)]

    print(f"repo:   {args.repo}")
    print(f"server: {tools['server']}")
    print(f"tiers:  {len(ggufs)}")
    for f in ggufs:
        print(f"  - {f}")
    if args.dry_run:
        print("\n--dry-run: nothing downloaded.")
        return 0
    if not ggufs:
        print("\nNothing to certify.")
        return 0

    from huggingface_hub import hf_hub_download

    report_path = scratch / "gate_report.json"
    results: list[dict] = []
    failed: list[dict] = []

    for i, fname in enumerate(ggufs, 1):
        quant = Path(fname).stem.split("-")[-1]
        print(f"\n[{i}/{len(ggufs)}] {fname}", flush=True)
        t0 = time.time()
        local = None
        try:
            local = Path(hf_hub_download(
                repo_id=args.repo, filename=fname, repo_type="model",
                local_dir=str(scratch),
            ))
            if not qg.gguf_magic_ok(local):
                print(f"  {quant}: NOT A GGUF (bad magic) — failing without load",
                      flush=True)
                ok = False
            else:
                ok = qg.sanity_check(tools, local, quant, port=args.port)

            entry = {
                "tier": quant, "file": fname, "pass": bool(ok),
                "seconds": round(time.time() - t0, 1),
            }
            if not ok:
                print(f"  {quant}: SANITY FAILED — withdrawing from {args.repo}",
                      flush=True)
                entry["hf_removed"] = qg.hf_delete_quant(args.repo, local, quant)
                entry["ollama_tag"] = (
                    f"{args.ollama_target}:{qg._gguf_tag_from_filename(local)}"
                    if args.ollama_target else None)
                failed.append(entry)
            results.append(entry)
        except Exception as exc:
            print(f"  {quant}: ERROR — {exc}", flush=True)
            results.append({"tier": quant, "file": fname, "pass": None,
                            "error": str(exc)})
        finally:
            # Free the disk before the next pull. A FAILED tier's bytes are
            # still on ollama.com and reproducible from there, so unlike the
            # publish path there is nothing unique to preserve here.
            if local is not None and local.exists():
                local.unlink(missing_ok=True)
            # Checkpoint after EVERY tier — an interrupted sweep keeps its
            # verdicts rather than restarting from zero.
            report_path.write_text(json.dumps(
                {"repo": args.repo, "ollama_target": args.ollama_target,
                 "results": results}, indent=2))

    passed = sum(1 for r in results if r.get("pass") is True)
    errored = [r for r in results if r.get("pass") is None]
    print(f"\n{'='*60}")
    print(f"  certified {passed}/{len(results)}")
    if errored:
        print(f"  ERRORED (verdict unknown, NOT certified): {len(errored)}")
        for r in errored:
            print(f"    {r['tier']}: {r.get('error', '')[:120]}")
    if failed:
        print(f"\n{'!'*60}")
        print(f"  {len(failed)} PUBLISHED TIER(S) FAILED THE GATE")
        print(f"{'!'*60}")
        for r in failed:
            print(f"  {r['tier']}:")
            print(f"    HF   — removed: {', '.join(r.get('hf_removed') or []) or 'REMOVAL FAILED'}")
            if r.get("ollama_tag"):
                print(f"    OLLAMA — REMOVE THIS TAG BY HAND: {r['ollama_tag']}")
        print("\n  ollama.com has no delete API. Remove each tag above at")
        print("  https://ollama.com/<namespace>/<model> -> tag -> Delete.")
        print(f"{'!'*60}")
    print(f"  report: {report_path}")
    print(f"{'='*60}")
    return 1 if (failed or errored) else 0


if __name__ == "__main__":
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    sys.exit(main())
