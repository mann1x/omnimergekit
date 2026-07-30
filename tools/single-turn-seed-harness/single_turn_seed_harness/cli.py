"""single-turn-seed-harness CLI.

Template-agnostic: it points at ONE already-running llama-server (or any
OpenAI-compatible endpoint) and sweeps the HumanEval+/MultiPL-E prompt set x
seeds against it. The chat template under test is whichever template that server
was launched with -- run this once per server (one per template) and diff the
two summaries. The `run_seed_sweep.sh` driver does exactly that for the OLD
(google_main) vs NEW (google_20260709) Gemma-4 templates.

  single-turn-seed-harness \
      --server http://127.0.0.1:8090 --name old-gmain \
      --seeds 12 --tasks all --out results
"""
from __future__ import annotations

import argparse
import sys

from .sweep import run_sweep
from .tasks import load_tasks


def _seeds(count, seed0):
    return [seed0 + i for i in range(count)]


def main(argv=None):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        prog="single-turn-seed-harness",
        description="Single-turn HumanEval+/MultiPL-E seed sweep that grades each "
                    "response with the hammer classifier (CLEAN / RUNAWAY / "
                    "THINK_EXPLODE / CORRUPT / ABORT) and reports a per-template "
                    "fail-rate. Point --server at one llama-server per template.")
    ap.add_argument("--server", required=True,
                    help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8090")
    ap.add_argument("--name", required=True,
                    help="cohort label, e.g. old-gmain / new-g0709 "
                         "(names the JSONL + summary files)")
    ap.add_argument("--tasks", default="all",
                    help="task groups: all | he | cpp | js | 'he,cpp' | "
                         "'he:40,cpp:20,js:20' (name:cap). Default: all")
    ap.add_argument("--seeds", type=int, default=12,
                    help="seeds per problem (default 12, the June sweep count)")
    ap.add_argument("--seed0", type=int, default=2000,
                    help="first seed value (default 2000, matches hammer_raw)")
    ap.add_argument("--he-limit", type=int, default=None,
                    help="cap HumanEval+ problems (default: all 164)")
    ap.add_argument("--mpe-limit", type=int, default=None,
                    help="cap per-language MultiPL-E problems (default: 50 each)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=2400.0,
                    help="per-request timeout seconds")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel in-flight requests; keep <= server --parallel")
    ap.add_argument("--out", default="results", help="output directory")
    a = ap.parse_args(argv)

    print("loading tasks: %s" % a.tasks)
    tasks = load_tasks(a.tasks, he_limit=a.he_limit, mpe_limit=a.mpe_limit)
    by_src = {}
    for t in tasks:
        by_src[t["source"]] = by_src.get(t["source"], 0) + 1
    print("loaded %d problems: %s" % (len(tasks), by_src))
    if not tasks:
        sys.exit("no tasks loaded")

    run_sweep(a.server, a.name, tasks, _seeds(a.seeds, a.seed0), a.out,
              max_tokens=a.max_tokens, timeout=a.timeout,
              concurrency=a.concurrency)


if __name__ == "__main__":
    main()
