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

Two modes:

  --watch (preferred while a ladder is still running) certifies each tier AS IT
  IS PUBLISHED. The window a bad tier stays live is one poll plus one gate, not
  the publisher's whole run, and the pulls spread across a runtime that is
  otherwise CPU-bound. It stops on a sentinel in the publisher's log, and gives
  up rather than wedging if the publisher dies without one.

  one-shot (default) sweeps a repo that is already finished.

Verdicts are keyed on (file, LFS sha256), never the filename alone: a ladder
REBUILDS tiers, and a name-keyed verdict would certify a tier against bytes
that have since been replaced.

Usage:
    export HF_TOKEN="${HF_TOKEN:?...}"
    python scripts/gate_published_tiers.py \
        --repo mannix/Qwen3.8-27B-Omnimerge-v6-GGUF \
        --ollama-target mannix/omnimerge-v6 \
        --scratch /mnt/sdc/omnimerge_v6/_gate_scratch

    # certify a subset, or resume after an interruption
    python scripts/gate_published_tiers.py --repo ... --only IQ2_M,IQ2_S

    # ride along with a running ladder, stopping when it signals done
    python scripts/gate_published_tiers.py --repo ... --watch \
        --watch-until-log /path/to/ladder.log --watch-until-sentinel LADDER_DONE

Disk: one tier at a time; the scratch file is removed before the next pull, so
peak usage is a single GGUF. Report is written to <scratch>/gate_report.json
after EVERY tier, so an interrupted sweep never loses its verdicts.
"""

import argparse
import importlib.util
import datetime
import json
import os
import subprocess
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


def list_published_ggufs(repo_id: str) -> dict[str, str]:
    """Map published GGUF path -> its LFS sha256 (empty string if unknown).

    The sha is the identity that matters: a ladder REBUILDS tiers, so a verdict
    keyed on the filename alone would certify a tier against bytes that have
    since been replaced. Keying on the sha re-gates a republished tier
    automatically.
    """
    from huggingface_hub import HfApi

    out: dict[str, str] = {}
    for e in HfApi().list_repo_tree(repo_id, repo_type="model", recursive=True):
        path = getattr(e, "path", "")
        if not path.endswith(".gguf"):
            continue
        lfs = getattr(e, "lfs", None)
        out[path] = (getattr(lfs, "sha256", "") or "") if lfs else ""
    return dict(sorted(out.items()))


def engine_provenance(server_bin: str) -> dict:
    """Record WHICH llama.cpp produced these verdicts.

    A verdict without its engine is not interpretable: the same file passes on a
    current build and emits whitespace on a stale one.
    """
    info = {"server": server_bin}
    try:
        info["mtime"] = datetime.date.fromtimestamp(
            Path(server_bin).stat().st_mtime).isoformat()
    except OSError:
        pass
    try:
        out = subprocess.run([server_bin, "--version"], capture_output=True,
                             text=True, timeout=30)
        for line in (out.stdout + out.stderr).splitlines():
            if "version:" in line:
                info["build"] = line.strip()
                break
    except Exception:
        pass
    return info


def select_tiers(tree: dict[str, str], skip_base: bool,
                 only: str | None) -> dict[str, str]:
    sel = dict(tree)
    if skip_base:
        sel = {f: sha for f, sha in sel.items()
               if not Path(f).stem.upper().endswith(("F16", "F32"))}
    if only:
        want = {t.strip().upper() for t in only.split(",") if t.strip()}
        sel = {f: sha for f, sha in sel.items()
               if any(Path(f).stem.upper().endswith("-" + t) for t in want)}
    return sel


def gate_one(qg, tools, args, fname: str, sha: str, scratch: Path) -> dict:
    """Pull one published tier, run the canonical gate, withdraw on failure."""
    quant = Path(fname).stem.split("-")[-1]
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    local = None
    entry = {"tier": quant, "file": fname, "sha256": sha, "pass": None}
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
        entry["pass"] = bool(ok)
        if not ok:
            print(f"  {quant}: SANITY FAILED", flush=True)
            # Report, do not withdraw. This script deleted two GOOD tiers on
            # 2026-09-03 because the llama.cpp it ran on was three months stale
            # and emitted whitespace for qwen35 i-quants that serve fine
            # elsewhere. The check cannot certify its own engine, so it does not
            # get to destroy artifacts. --withdraw-on-fail opts in.
            if args.withdraw_on_fail:
                print(f"  {quant}: --withdraw-on-fail set — removing from "
                      f"{args.repo}", flush=True)
                entry["hf_removed"] = qg.hf_delete_quant(args.repo, local, quant)
            else:
                entry["hf_removed"] = []
                print(f"  {quant}: LEFT IN PLACE on {args.repo}", flush=True)
            entry["ollama_tag"] = (
                f"{args.ollama_target}:{qg._gguf_tag_from_filename(local)}"
                if args.ollama_target else None)
    except Exception as exc:
        print(f"  {quant}: ERROR — {exc}", flush=True)
        entry["error"] = str(exc)
    finally:
        # Free the disk before the next pull. Unlike the publish path there is
        # nothing unique to preserve here — a failed tier's bytes are still on
        # ollama.com and reproducible from there.
        if local is not None and local.exists():
            local.unlink(missing_ok=True)
        entry["seconds"] = round(time.time() - t0, 1)
    return entry


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
    ap.add_argument("--withdraw-on-fail", action="store_true",
                    help="Delete a failing tier from HF. Default OFF — a failure "
                         "is reported, never auto-deleted, because this check "
                         "cannot certify that its own llama.cpp can run the "
                         "architecture and quant type under test.")
    ap.add_argument("--require-engine-newer-than", default=None,
                    help="Refuse to run unless the llama.cpp build date is at or "
                         "after this YYYY-MM-DD. A stale engine produces whitespace "
                         "for architectures it does not support, which reads as a "
                         "model failure and is not one.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be certified, download nothing")
    ap.add_argument("--watch", action="store_true",
                    help="Stay resident and gate each tier AS IT IS PUBLISHED, "
                         "instead of one sweep over a finished repo. Shortens the "
                         "window a bad tier stays live and spreads the pulls "
                         "across the publisher's runtime.")
    ap.add_argument("--watch-interval", type=int, default=120,
                    help="Seconds between repo polls in --watch mode (default 120)")
    ap.add_argument("--watch-until-log", default=None,
                    help="In --watch mode, stop once this log file contains the "
                         "sentinel (after draining whatever is still unpublished)")
    ap.add_argument("--watch-until-sentinel", default="LADDER_DONE",
                    help="Sentinel string for --watch-until-log (default LADDER_DONE)")
    ap.add_argument("--watch-idle-timeout", type=int, default=14400,
                    help="Give up in --watch mode after this many seconds with no "
                         "new tier AND no sentinel (default 4h). A publisher that "
                         "died without its sentinel must not wedge the watcher.")
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

    engine = engine_provenance(tools["server"])
    print(f"engine: {engine['server']}  build={engine.get('build', '?')}  "
          f"mtime={engine.get('mtime', '?')}")
    floor = args.require_engine_newer_than
    if floor and engine.get("mtime", "") < floor:
        print(f"REFUSING: llama-server binary dates {engine.get('mtime')}, before "
              f"the required {floor}. A stale engine emits whitespace for "
              f"architectures it does not support — that is an ENGINE failure "
              f"being read as a MODEL failure.", file=sys.stderr)
        return 2

    report_path = scratch / "gate_report.json"
    results: list[dict] = []
    failed: list[dict] = []

    # Resume: a verdict is keyed on (file, sha256), so a REBUILT tier is
    # re-gated automatically while an untouched one is not re-pulled.
    done: set[tuple[str, str]] = set()
    if report_path.exists():
        try:
            prev = json.loads(report_path.read_text())
            for r in prev.get("results", []):
                results.append(r)
                if r.get("pass") is not None:
                    done.add((r["file"], r.get("sha256", "")))
                if r.get("pass") is False:
                    failed.append(r)
            if done:
                print(f"resuming: {len(done)} tier(s) already certified in "
                      f"{report_path}")
        except Exception as exc:
            print(f"  (ignoring unreadable prior report: {exc})")

    def checkpoint():
        report_path.write_text(json.dumps(
            {"repo": args.repo, "ollama_target": args.ollama_target,
             "engine": engine, "results": results}, indent=2))

    print(f"repo:   {args.repo}")
    print(f"server: {tools['server']}")
    print(f"mode:   {'watch' if args.watch else 'one-shot sweep'}")

    tree = select_tiers(list_published_ggufs(args.repo), args.skip_base, args.only)
    pending = [(f, sha) for f, sha in tree.items() if (f, sha) not in done]
    print(f"published: {len(tree)}   pending: {len(pending)}")
    for f, _sha in tree.items():
        print(f"  - {f}")
    if args.dry_run:
        print("\n--dry-run: nothing downloaded.")
        return 0

    def sentinel_seen() -> bool:
        """True once the publisher has signalled it is finished."""
        if not args.watch_until_log:
            return False
        try:
            return args.watch_until_sentinel in Path(
                args.watch_until_log).read_text(errors="replace")
        except OSError:
            return False

    if not args.watch:
        if not pending:
            print("\nNothing to certify.")
            return 0
        for i, (fname, sha) in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] {fname}", flush=True)
            entry = gate_one(qg, tools, args, fname, sha, scratch)
            results.append(entry)
            if entry.get("pass") is False:
                failed.append(entry)
            checkpoint()
    else:
        # Gate each tier AS IT IS PUBLISHED. The window a bad tier stays live
        # is one poll interval plus one gate, not the publisher's whole run.
        n = 0
        last_progress = time.time()
        while True:
            tree = select_tiers(list_published_ggufs(args.repo),
                                args.skip_base, args.only)
            pending = [(f, sha) for f, sha in tree.items()
                       if (f, sha) not in done]
            if pending:
                last_progress = time.time()
            for fname, sha in pending:
                n += 1
                print(f"\n[{n}] {fname}", flush=True)
                entry = gate_one(qg, tools, args, fname, sha, scratch)
                results.append(entry)
                done.add((fname, sha))
                if entry.get("pass") is False:
                    failed.append(entry)
                checkpoint()

            finished = sentinel_seen()
            if finished:
                # Drain: the sentinel may land while a final tier is still
                # being uploaded, so re-poll once more before believing it.
                tree = select_tiers(list_published_ggufs(args.repo),
                                    args.skip_base, args.only)
                if not [(f, sha) for f, sha in tree.items()
                        if (f, sha) not in done]:
                    print(f"\nsentinel '{args.watch_until_sentinel}' seen and "
                          f"nothing left pending — watch complete", flush=True)
                    break
            idle = time.time() - last_progress
            if idle > args.watch_idle_timeout:
                print(f"\nWATCH GAVE UP: {idle/3600:.1f}h with no new tier and "
                      f"no '{args.watch_until_sentinel}' sentinel. The publisher "
                      f"may have died. {len(results)} tier(s) certified.",
                      flush=True)
                break
            time.sleep(args.watch_interval)

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
