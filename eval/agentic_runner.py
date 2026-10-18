#!/usr/bin/env python3
"""agentic_runner.py — omk-native runner for MULTI-TURN AGENTIC benches.

Same shape as mrcr_runner.py / nolima: omk_eval serves the model, this runs the
harness against that endpoint and writes `agentic_result.json`, which
extract_canonical_score() reads into summary.json.

TWO HARNESSES, one result schema:
  bfcl    inspect_evals/bfcl (BFCL V3 stateful multi-turn tool use). No docker.
          Deterministic state+execution scorer, no LLM judge.
  harbor  Harbor + an agent (default terminus-2) on Terminal-Bench / CompileBench
          / aider-polyglot. Docker per task. DEEP context — this is the one that
          can actually exhibit reasoning-loop degeneration.

THE HEADLINE METRIC IS DECLARED BY THE TEMPLATE, and `loop_rate` is a
first-class citizen, not a diagnostic. For a degeneration hypothesis the task
pass rate is a DOWNSTREAM PROXY that only moves when a loop is severe enough to
wreck the task; a null there says little. Templates that test degeneration
declare `scoring.metric: loop_rate`.

Loop detectors (same as eval/agentic/loop_census.py and the project's rumination
work): a 2-40 char unit repeated >=8x, a single char repeated >=41x, or a
reasoning block above --runaway-chars.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import subprocess
from pathlib import Path

RE_REPEAT = re.compile(r"(.{2,40}?)\1{7,}", re.S)
RE_RUNCHAR = re.compile(r"(.)\1{40,}", re.S)


def log(m: str) -> None:
    print(f"[agentic] {m}", flush=True)


# ---------------------------------------------------------------- bfcl ------
def run_bfcl(a) -> int:
    env = dict(os.environ)
    env["LLAMACPP_BASE_URL"] = a.base_url
    env["LLAMACPP_API_KEY"] = "none"
    env.setdefault("HF_HOME", str(Path(a.harness_env).parent / "hf"))
    cmd = [
        str(Path(a.harness_env) / "bin" / "inspect"), "eval", "inspect_evals/bfcl",
        "-T", f"categories={a.dataset}",
        "--model", f"openai-api/llamacpp/{a.name}",
        "--limit", str(a.limit),
        "--max-connections", str(a.concurrency),
        "--max-tokens", str(a.max_tokens),
        "--temperature", str(a.temperature),
        "--log-dir", str(a.out),
        "--no-fail-on-error",
    ]
    log("bfcl: " + " ".join(cmd))
    return subprocess.run(cmd, env=env).returncode


def score_bfcl(a) -> dict:
    """Score the inspect .eval log.

    MUST run in a SUBPROCESS under the harness env's own interpreter. The
    harness venv is python3.12 and ships COMPILED extensions (pydantic_core);
    sys.path-inserting it into omk's python3.11 raises
    `No module named 'pydantic_core._pydantic_core'`. Cross-version site-packages
    are not importable, only runnable.
    """
    script = r"""
import json, re, sys, statistics as st
from pathlib import Path
from inspect_ai.log import read_eval_log
out, runaway = Path(sys.argv[1]), int(sys.argv[2])
RE_REPEAT = re.compile(r"(.{2,40}?)\1{7,}", re.S)
RE_RUNCHAR = re.compile(r"(.)\1{40,}", re.S)
logs = sorted(out.glob("*.eval"))
if not logs:
    print(json.dumps({"error": "no .eval log produced"})); raise SystemExit
lg = read_eval_log(str(logs[-1]))
n=n_pass=n_loop=rep_t=run_t=away_t=turns=errs=caps=otok=itok=crtok=0
lens=[]
for s in (lg.samples or []):
    n += 1
    val = None
    for sc in (s.scores or {}).values():
        val = sc.value; break
    if (val == "C") or (val is True) or (val == 1) or (val == 1.0): n_pass += 1
    if getattr(s, "error", None) is not None: errs += 1
    looped = False
    for m in (s.messages or []):
        if getattr(m, "role", "") != "assistant": continue
        c = getattr(m, "content", None)
        if not isinstance(c, list): continue
        for p in c:
            if getattr(p, "type", "") != "reasoning": continue
            t = getattr(p, "reasoning", "") or ""
            turns += 1; lens.append(len(t))
            if RE_REPEAT.search(t): rep_t += 1; looped = True
            if RE_RUNCHAR.search(t): run_t += 1; looped = True
            if len(t) >= runaway: away_t += 1; looped = True
    if looped: n_loop += 1
    for ev in (getattr(s, "events", None) or []):
        o = getattr(ev, "output", None)
        if o is not None and getattr(o, "stop_reason", None) == "max_tokens": caps += 1
    for u in (getattr(s, "model_usage", None) or {}).values():
        otok += getattr(u, "output_tokens", 0) or 0
        itok += getattr(u, "input_tokens", 0) or 0
        crtok += getattr(u, "input_tokens_cache_read", 0) or 0
print(json.dumps({"n":n,"n_pass":n_pass,"n_loop":n_loop,"rep_t":rep_t,"run_t":run_t,
                  "away_t":away_t,"turns":turns,"lens":lens,"caps":caps,"errs":errs,
                  "otok":otok,"itok":itok,"crtok":crtok}))
"""
    py = str(Path(a.harness_env) / "bin" / "python")
    r = subprocess.run([py, "-c", script, str(a.out), str(a.runaway_chars)],
                       capture_output=True, text=True)
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return {"error": f"scorer produced no output (rc={r.returncode}): "
                         f"{(r.stderr or '')[-300:]}"}
    try:
        d = json.loads(line[-1])
    except Exception as e:
        return {"error": f"scorer output not JSON: {e}: {line[-1][:200]}"}
    if "error" in d:
        return d
    return _pack(a, d["n"], d["n_pass"], d["n_loop"], d["rep_t"], d["run_t"],
                 d["away_t"], d["turns"], d["lens"], d["caps"], d["errs"],
                 d["otok"], d["itok"], d["crtok"])


# -------------------------------------------------------------- harbor ------
def run_harbor(a) -> int:
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"
    env["OPENAI_BASE_URL"] = a.base_url
    cmd = [
        str(Path(a.harness_env) / "bin" / "harbor"), "run",
        "--dataset", a.dataset,
        "--agent", a.agent,
        "--model", f"openai/{a.name}",
        "--ak", f"api_base={a.base_url}",
        "--ak", f"temperature={a.temperature}",
        # Context summarisation COMPRESSES HISTORY — it is exactly the
        # accumulation a re-injection test measures. Off by default here.
        "--ak", f"enable_summarize={'true' if a.enable_summarize else 'false'}",
        "--ak", "proactive_summarize_threshold=0",
        "--jobs-dir", str(a.out), "--job-name", a.name,
        "-n", str(a.concurrency), "-y",
    ]
    host = a.base_url.split("//", 1)[-1].split(":", 1)[0]
    cmd += ["--allow-agent-host", host]
    if a.limit and a.limit > 0:
        cmd += ["--n-tasks", str(a.limit)]
    log("harbor: " + " ".join(cmd))
    return subprocess.run(cmd, env=env).returncode


def score_harbor(a) -> dict:
    """Parse harbor job output. Harbor writes one dir per trial; we read the
    per-trial result json for pass/fail and the agent trajectory for looping."""
    root = Path(a.out)
    results = sorted(root.glob("**/results.json")) + sorted(root.glob("**/result.json"))
    trajs = sorted(root.glob("**/*trajectory*.json")) + sorted(root.glob("**/*messages*.json"))
    n = n_pass = n_loop = rep_t = run_t = away_t = turns = 0
    lens: list[int] = []

    for rf in results:
        try:
            d = json.loads(rf.read_text())
        except Exception:
            continue
        items = d if isinstance(d, list) else d.get("results") or d.get("trials") or [d]
        for it in (items if isinstance(items, list) else [items]):
            if not isinstance(it, dict):
                continue
            n += 1
            v = it.get("resolved", it.get("passed", it.get("is_resolved")))
            if v in (True, 1, "true", "True", "resolved"):
                n_pass += 1

    for tf in trajs:
        try:
            d = json.loads(tf.read_text())
        except Exception:
            continue
        msgs = d if isinstance(d, list) else d.get("messages") or d.get("trajectory") or []
        looped = False
        for m in (msgs if isinstance(msgs, list) else []):
            if not isinstance(m, dict):
                continue
            t = m.get("reasoning") or m.get("reasoning_content") or ""
            if not t and m.get("role") == "assistant":
                c = m.get("content")
                t = c if isinstance(c, str) else ""
            if not t:
                continue
            turns += 1
            lens.append(len(t))
            if RE_REPEAT.search(t):
                rep_t += 1
                looped = True
            if RE_RUNCHAR.search(t):
                run_t += 1
                looped = True
            if len(t) >= a.runaway_chars:
                away_t += 1
                looped = True
        if looped:
            n_loop += 1

    if n == 0:
        return {"error": f"no harbor results parsed under {root}",
                "hint": "inspect the jobs dir layout and extend score_harbor()"}
    return _pack(a, n, n_pass, n_loop, rep_t, run_t, away_t, turns, lens, 0, 0, 0, 0, 0)


# ---------------------------------------------------------------- common ----
def _pack(a, n, n_pass, n_loop, rep_t, run_t, away_t, turns, lens,
          caps, errs, otok, itok, crtok) -> dict:
    p50 = int(st.median(lens)) if lens else 0
    p90 = int(sorted(lens)[int(.9 * len(lens)) - 1]) if lens else 0
    d = {
        "harness": a.harness,
        "dataset": a.dataset,
        "agent": a.agent if a.harness == "harbor" else "n/a",
        "n": n,
        "n_pass": n_pass,
        "pass_at_1": (n_pass / n) if n else None,
        # THE DEGENERATION ENDPOINT
        "loop_rate": (n_loop / n) if n else None,
        "n_looped": n_loop,
        # THREE DISTINCT detectors — do not collapse them. rep = a 2-40 char
        # unit repeated >=8x; runchar = one char repeated >=41x; runaway = a
        # reasoning block over runaway_chars. They catch different failures and
        # a sample is "looped" if ANY fires.
        "rep_turns": rep_t,
        "runchar_turns": run_t,
        "runaway_turns": away_t,
        "runaway_chars": a.runaway_chars,
        "reasoning_turns": turns,
        "reasoning_p50": p50,
        "reasoning_p90": p90,
        "reasoning_max": max(lens) if lens else 0,
        "cap_hits": caps,
        "errors": errs,
        "out_tokens": otok,
        "input_tokens": itok,
        "cache_read_tokens": crtok,
        "cache_read_frac": (crtok / (itok + crtok)) if (itok + crtok) else None,
        "enable_summarize": bool(a.enable_summarize),
        "metric": a.metric,
    }
    d["score"] = d.get(a.metric)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=("bfcl", "harbor"), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--agent", default="terminus-2")
    ap.add_argument("--runaway-chars", type=int, default=10000)
    ap.add_argument("--metric", default="pass_at_1",
                    choices=("pass_at_1", "loop_rate"))
    ap.add_argument("--enable-summarize", action="store_true",
                    help="harbor only; OFF by default because summarisation "
                         "compresses the history a re-injection test measures")
    ap.add_argument("--harness-env",
                    default=os.environ.get("OMK_AGENTIC_ENV", "/mnt/sdc/harness/venv"))
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)

    rc = 0
    if not a.score_only:
        rc = run_bfcl(a) if a.harness == "bfcl" else run_harbor(a)
        log(f"harness exit rc={rc}")

    res = score_bfcl(a) if a.harness == "bfcl" else score_harbor(a)
    res["harness_rc"] = rc
    (a.out / "agentic_result.json").write_text(json.dumps(res, indent=2))
    log(f"wrote {a.out / 'agentic_result.json'}")
    if "error" in res:
        log(f"ERROR: {res['error']}")
        return 12
    log(f"score[{res['metric']}]={res['score']}  pass_at_1={res['pass_at_1']} "
        f"loop_rate={res['loop_rate']} (n={res['n']})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
