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

from loop_detect import detect  # noqa: E402
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


def _harbor_job_result(root: Path, name: str | None = None):
    """Locate harbor's JOB-level result file.

    harbor 0.23.0 writes TWO differently-pluralised files and getting them
    confused silently double-counts every trial:
      job   -> <jobs_dir>/<job_name>/result.json   (SINGULAR, has .stats)
      trial -> <trial_dir>/results.json            (PLURAL, one TrialResult)
    The job file EMBEDS trial_results, so globbing both counts each trial
    twice. Identify the job file by its schema (.stats), never by name alone.
    """
    # Prefer THIS arm's own job dir (<jobs_dir>/<job_name>) — a paired A/B
    # design writes both arms under one --out, and scanning the parent would
    # falsely refuse as "spans 2 jobs".
    own = root / name / "result.json" if name else None
    globbed = [own] if (own and own.exists()) else sorted(root.glob("**/result.json"))
    cands = []
    for f in globbed:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and isinstance(d.get("stats"), dict):
            cands.append((f, d))
    return cands


def _harbor_trials(job_dir: Path) -> list:
    """Per-trial results. VERIFIED against a real harbor 0.23.0 run 2026-09-13:
    the trial file is `<trial_dir>/result.json` — SINGULAR — even though
    harbor's own models/trial/paths.py docstring documents it as `results.json`.
    Trust the filesystem, not the docstring: a glob for the plural finds
    nothing. Job and trial files are told apart by SCHEMA (`.stats`), not name.
    """
    out = []
    for f in sorted(job_dir.glob("*/result.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and not isinstance(d.get("stats"), dict):
            out.append(d)
    return out


def _pass_at_1(trials: list):
    """pass@1 by harbor's OWN rule (utils/pass_at_k.py), applied to the per-trial
    rewards because the job-level aggregate is frequently unavailable.

    Returns (pass_at_1, n_tasks, n_pass, reason_if_refused). Refuses — never
    returns 0.0 — when a reward is multi-key or non-binary, exactly as harbor
    does; a refusal is not a score of zero.
    """
    by_task: dict = {}
    for t in trials:
        vr = (t or {}).get("verifier_result") or {}
        rw = vr.get("rewards")
        name = t.get("task_name") or t.get("trial_name") or "?"
        if rw is None:
            by_task.setdefault(name, []).append(0)
            continue
        if len(rw) != 1:
            return None, 0, 0, f"task {name!r} has {len(rw)} reward keys ({sorted(rw)})"
        v = next(iter(rw.values()))
        if not isinstance(v, (int, float)) or v not in (0, 1):
            return None, 0, 0, f"task {name!r} has non-binary reward {v!r}"
        by_task.setdefault(name, []).append(int(v))
    if not by_task:
        return None, 0, 0, "no trials carried a verifier_result"
    per = [sum(v) / len(v) for v in by_task.values()]
    n_pass = sum(1 for v in by_task.values() if sum(v) > 0)
    return sum(per) / len(per), len(by_task), n_pass, None


def score_harbor(a) -> dict:
    """Parse a harbor job.

    Scoring comes from the PER-TRIAL result.json files. The job-level
    `stats.evals[key].pass_at_k` is preferred when present, but on a real
    completed run (2026-09-13) it was `{}` because harbor writes its final job
    file with `exclude_trial_results=True`, so its own pass@k aggregation ran
    over an empty list. Treating that empty dict as 0.0 would have reported a
    2/2 PERFECT run as 0% — so an absent aggregate falls back to the per-trial
    rewards, and a present one is used as a cross-check.
    """
    root = Path(a.out)
    cands = _harbor_job_result(root, getattr(a, "name", None))
    if not cands:
        return {"error": f"no harbor job result.json (with .stats) under {root}",
                "hint": "harbor writes it to <jobs_dir>/<job_name>/result.json"}
    if len(cands) > 1:
        return {"error": f"{len(cands)} job result.json files under {root}",
                "hint": "point --out at ONE job dir; refusing to merge jobs"}
    jf, job = cands[0]
    job_dir = jf.parent

    stats = job.get("stats") or {}
    evals = stats.get("evals") or {}
    if len(evals) > 1:
        return {"error": f"harbor job spans {len(evals)} agent/model/dataset "
                         f"cells: {sorted(evals)}; refusing to merge bases"}
    ekey, ev = (next(iter(evals.items())) if evals else ("<none>", {}))

    trials = job.get("trial_results") or _harbor_trials(job_dir)
    p1, n, n_pass, refused = _pass_at_1(trials)
    if p1 is None:
        return {"error": f"cannot score harbor job: {refused}",
                "hint": "non-binary/multi-key rewards are a REFUSAL, not a 0.0",
                "eval_key": ekey, "job_result": str(jf)}

    n_err = (ev.get("n_errors") or 0) or (stats.get("n_errored_trials") or 0)
    n_exc = sum(1 for t in trials if (t or {}).get("exception_info"))

    # ---- loop census over agent trajectories -------------------------------
    trajs = [f for f in sorted(job_dir.glob("*/agent/trajectory.json"))]
    if not trajs:
        trajs = [f for f in sorted(root.glob("**/trajectory.json"))
                 if "summarization" not in f.name]
    n_loop = rep_t = run_t = away_t = sent_t = turns = 0
    lens: list[int] = []
    for tf in trajs:
        try:
            d = json.loads(tf.read_text())
        except Exception:
            continue
        looped = False
        for t in harbor_reasoning(d):
            det = detect(t, a.runaway_chars)
            turns += 1
            lens.append(det["chars"])
            rep_t += det["rep"]
            run_t += det["runchar"]
            away_t += det["runaway"]
            sent_t += det["sentence"]
            looped = looped or det["looped"]
        if looped:
            n_loop += 1

    d = _pack(a, n, n_pass, n_loop, rep_t, run_t, away_t, turns, lens,
              0, n_err or n_exc, 0, 0, 0)
    d["eval_key"] = ekey
    d["pass_at_1"] = p1
    d["sentence_turns"] = sent_t
    d["n_trials"] = len(trials)
    d["n_exceptions"] = n_exc
    d["n_trajectories"] = len(trajs)
    d["job_result"] = str(jf)
    d["score_source"] = "per-trial rewards"
    agg = (ev.get("pass_at_k") or {})
    agg1 = agg.get("1", agg.get(1))
    d["pass_at_k_job"] = agg or None
    if agg1 is not None:
        d["score_source"] = "job aggregate (cross-checked vs per-trial)"
        if abs(agg1 - p1) > 1e-9:
            d["warn_pass_mismatch"] = (f"harbor aggregate pass@1={agg1} vs "
                                       f"per-trial {p1}; schema may have moved")
    if not trajs:
        d["loop_rate"] = None
        d["loop_census_error"] = ("no trajectory.json parsed; loop_rate "
                                  "withheld rather than reported as 0.0")
    return d


def harbor_reasoning(traj):
    """Yield ONLY the model's reasoning channel from a terminus-2 trajectory.

    Structure (verified 2026-09-13): {"steps":[{"reasoning_content": str,
    "message": ..., "observation": ...}]}. We scan `reasoning_content` ONLY.

    WHY NOT `message`/`observation`: the re-injection hypothesis is about the
    THINKING channel looping. `message` holds the agent's actions — an agent
    that rewrites the same file three times legitimately repeats its own code
    lines, and `observation` is raw terminal output. Scanning them made the
    sentence detector fire on `sample_envelope <- function(...)` x3 and the
    char detector fire on indentation. Those are not loops. Measuring the
    channel under test is both cleaner and the actual hypothesis.
    """
    if isinstance(traj, dict):
        steps = traj.get("steps")
        if isinstance(steps, list):
            for st_ in steps:
                if not isinstance(st_, dict):
                    continue
                r = st_.get("reasoning_content") or st_.get("reasoning")
                if isinstance(r, str) and r.strip():
                    yield r
            return
    # unknown shape: yield nothing rather than guess. A loop_rate over zero
    # parsed blocks is WITHHELD by the caller, never reported as 0.0.
    return


def _walk_texts(obj, _depth=0):
    """Yield assistant/model text + reasoning from a harbor trajectory of
    unknown nesting. Keys chosen from harbor's Step/AgentContext shapes."""
    if _depth > 12:
        return
    if isinstance(obj, dict):
        for k in ("reasoning", "reasoning_content", "thinking"):
            v = obj.get(k)
            if isinstance(v, str) and v:
                yield v
        if obj.get("role") in ("assistant", "model") or "response" in obj:
            v = obj.get("content", obj.get("response"))
            if isinstance(v, str) and v:
                yield v
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, dict) and isinstance(it.get("text"), str):
                        yield it["text"]
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from _walk_texts(v, _depth + 1)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, (dict, list)):
                yield from _walk_texts(it, _depth + 1)


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
