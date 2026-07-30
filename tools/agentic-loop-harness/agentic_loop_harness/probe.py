"""Cold-start differential probe.

The fixture-replay path (`replay.py`) replays a FROZEN mid-session conversation.
The loops users actually hit at Q6_K happen earlier: on the FIRST open-ended
coding prompt of a fresh conversation ("write a snake game", "build a solar
system"), where v7-coder ruminates / over-thinks / loops while base 128e answers
directly. This module exercises that regime: each prompt in a suite is sent as a
zero-history turn (system + user, optional tools) and scored by the SAME engine
as replay (verbatim `detect.py` + non-verbatim `softloop.py`), so a prompt is
just a one-shot fixture.

It is intentionally serving-agnostic: point `--server` at any OpenAI-compatible
endpoint. Run it once per model (v7-coder-Q6_K, 128e-Q6_K) at the SAME sampler;
`scripts/diff_divergence.py` then pairs the two result files into the
v7-fails / 128e-clean divergence set -- the diagnostic this whole harness exists
to produce.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .replay import replay_fixture

# the published "recommended" deploy sampler (model card / ollama defaults).
RECOMMENDED = [{
    "name": "recommended_t0.9",
    "params": {"temperature": 0.9, "top_k": 64, "top_p": 0.95,
               "min_p": 0.05, "repeat_penalty": 1.1},
}]


def load_suite(path):
    with open(path) as fh:
        return json.load(fh)


def build_fixture(prompt, suite):
    """A cold-start prompt -> a zero-history fixture (system + user). Returns
    (fixture, reasoning_budget) where budget is only used to inform the
    over-thinking oracle (it is a server-launch flag, not a request param)."""
    default = dict(suite.get("default") or {})
    sysmsg = prompt.get("system", suite.get("system"))
    messages = []
    if sysmsg:
        messages.append({"role": "system", "content": sysmsg})
    messages.append({"role": "user", "content": prompt["user"]})
    max_tokens = prompt.get("max_tokens", default.get("max_tokens", 32000))
    budget = prompt.get("reasoning_budget", default.get("reasoning_budget"))
    fixture = {"name": prompt["name"], "messages": messages,
               "tools": prompt.get("tools"),
               "base_params": {"max_tokens": max_tokens}}
    return fixture, budget


def run_suite(server, suite, matrix, seeds, model_name, dump_dir=None,
              timeout=1800.0, concurrency=1, log=print):
    prompts = suite["prompts"]
    log("=== probe model=%s server=%s suite=%s prompts=%d seeds=%d configs=%d ==="
        % (model_name, server, suite.get("name"), len(prompts), len(seeds),
           len(matrix)))
    out = {"model": model_name, "server": server, "suite": suite.get("name"),
           "seeds": seeds, "matrix": matrix, "prompts": []}
    for prompt in prompts:
        fixture, budget = build_fixture(prompt, suite)
        dump = None
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            dump = os.path.join(dump_dir, "%s__%s.jsonl" % (model_name,
                                                            prompt["name"]))
            open(dump, "w").close()           # truncate any prior run
        log("--- prompt=%s ---" % prompt["name"])
        configs = replay_fixture(server, fixture, matrix, seeds,
                                 timeout=timeout, concurrency=concurrency,
                                 budget_tokens=budget, dump_text=dump, log=log)
        out["prompts"].append({"name": prompt["name"], "budget": budget,
                               "dump": dump, "configs": configs})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="cold-start differential loop probe")
    ap.add_argument("--server", required=True, help="OpenAI-compatible base URL")
    ap.add_argument("--suite", required=True, help="prompt suite JSON")
    ap.add_argument("--model-name", required=True, help="tag, e.g. v7-coder-q6k")
    ap.add_argument("--out", required=True, help="result JSON path")
    ap.add_argument("--seeds", type=int, default=8, help="seed count (1..N)")
    ap.add_argument("--seed-list", help="comma-separated explicit seeds (overrides --seeds)")
    ap.add_argument("--matrix", help="sampler matrix JSON (list of {name,params}); default=recommended")
    ap.add_argument("--dump-dir", help="dir for per-prompt transcript JSONL")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    suite = load_suite(args.suite)
    matrix = json.load(open(args.matrix)) if args.matrix else RECOMMENDED
    if args.seed_list:
        seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]
    else:
        seeds = list(range(1, args.seeds + 1))

    t0 = time.time()
    out = run_suite(args.server, suite, matrix, seeds, args.model_name,
                    dump_dir=args.dump_dir, timeout=args.timeout,
                    concurrency=args.concurrency)
    out["duration_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    # compact per-prompt EXT fail-rate summary
    print("\n=== %s : EXT fail-rate per prompt (worst config) ===" % args.model_name)
    for p in out["prompts"]:
        worst = max(p["configs"], key=lambda c: c["fail_rate_ext"])
        print("  %-22s ext=%2d/%-2d strict=%2d/%-2d  (soft p=%d t=%d o=%d)"
              % (p["name"], worst["fails_ext"], worst["seeds"],
                 worst["fails"], worst["seeds"], worst["paraphrase"],
                 worst["template"], worst["overthinking"]))
    print("written:", args.out)


if __name__ == "__main__":
    main()
