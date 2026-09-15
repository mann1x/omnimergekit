#!/usr/bin/env python3
"""extract_polyglot_traces.py — Tier-B PASS-trace JSON from harbor AGENTIC runs.

Companion to `extract_pass_traces.py` (which mines single-turn lm-eval bench
samples). This one mines MULTI-TURN agentic coding trajectories produced by
`harbor run --agent terminus-2` over the aider-polyglot dataset, and emits the
same Tier-B JSON envelope consumed by
`expert_neuron_analysis_v5_targeted.py --tier-b-json`.

The difference is the trace shape. A bench trace is `prompt` + `completion`; an
agentic trace is a full `messages` list (user -> assistant(tool_calls) -> tool
-> assistant -> ...). The analysis script's T203 patch renders `messages`
through the chat template in one pass when the key is present, so the tool-result
turns — where an agentic coder's real routing signal lives — survive into the
forward pass instead of being flattened away.

INPUT  (per harbor trial dir, under --jobs-dir):
    result.json               task_name, verifier_result
    verifier/reward.txt       binary reward ("1"/"0")
    agent/trajectory.json     {"steps": [{step_id, source, message,
                                          reasoning_content, tool_calls,
                                          observation, ...}]}

OUTPUT: {"metadata": {...}, "traces": [{bench, task_id, set, weight, messages,
                                        ...}]}

CLASS KEYS — one per language by default, so the analysis emits
`targeted_polyglot_rust`, `targeted_polyglot_go`, ... which
`generate_drop_map_v5fk.py` consumes via `--classes/--class-weights` with no
further changes.

SET SEMANTICS — identical to extract_pass_traces.py, and for the same reason
(the file is re-runnable; Set-C rows migrate to A/B as a candidate is measured):
    A = teacher PASS and candidate PASS       (positive control; weight 1.0)
    B = teacher PASS and candidate FAIL       (regression set;   weight 3.0)
    C = teacher PASS and candidate unmeasured (neutral;          weight 1.0)
With only a teacher run supplied, every trace is Set-C at weight 1.0. That is
expected on a first pass and is NOT a silent downgrade — it is reported.

TWO SHAPE FACTS about terminus-2 output, both load-bearing:

  1. A step may carry N tool_calls but exactly ONE `observation`: terminus-2
     batches keystrokes into a single terminal and returns the merged pane.
     We therefore emit ONE `role: tool` message per step carrying that merged
     observation, not one per tool_call. The rendered N-calls/1-response pair is
     structurally lopsided for an API exchange but is exactly what the model
     saw, and the chat template does not validate the counts.

  2. `reasoning_content` is a SEPARATE field from `message`. Gemma-4's chat
     template runs `strip_thinking()` unconditionally over model-turn content,
     so channel-wrapped reasoning would be dropped before it ever reaches the
     forward pass. `--reasoning` therefore selects between:
       strip   (DEFAULT) content = message only. Profiles the ANSWER channel,
               which is what the existing Tier-B bench traces profile — keeps
               this tier commensurable with the published v5 maps.
       include content = reasoning + message as plain visible text. Profiles
               reasoning routing too. This deliberately places reasoning in the
               answer channel so its tokens are actually forward-passed; it is a
               PROFILING choice and not a claim about served context.
     Whichever is used is recorded in the output metadata.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

SET_WEIGHTS = {"A": 1.0, "B": 3.0, "C": 1.0}


def _truncate(text: str, limit: int) -> str:
    """Keep head + tail. Terminal dumps are long and the ERROR is usually at the
    end, so a plain head-truncation would systematically discard the signal."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return (text[:head] + f"\n...[{len(text) - limit} chars elided]...\n"
            + text[-tail:])


def _obs_text(observation) -> str:
    """terminus-2 writes {"results": [{"content": ...}, ...]}; older/other agents
    may write a bare string. Both are accepted; anything else is stringified
    rather than silently dropped."""
    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    if isinstance(observation, dict) and isinstance(observation.get("results"), list):
        return "\n".join(str(r.get("content", "")) for r in observation["results"])
    return json.dumps(observation)


def steps_to_messages(steps: list, reasoning: str, max_obs_chars: int,
                      max_steps: int) -> list:
    """Render harbor trajectory steps into a chat `messages` list."""
    msgs: list = []
    for st in steps[:max_steps] if max_steps > 0 else steps:
        src = st.get("source")
        if src == "user":
            content = st.get("message") or ""
            if content:
                msgs.append({"role": "user", "content": content})
            continue
        if src != "agent":
            continue

        text = st.get("message") or ""
        if reasoning == "include":
            rc = st.get("reasoning_content") or ""
            if rc:
                text = (rc + "\n\n" + text) if text else rc

        tool_calls = []
        for tc in (st.get("tool_calls") or []):
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": tc.get("function_name") or tc.get("name") or "tool",
                    "arguments": tc.get("arguments", {}),
                },
            })

        # An assistant turn with neither text nor calls carries no routing we
        # can attribute; skipping it keeps the transcript well-formed.
        if not text and not tool_calls:
            continue
        am: dict = {"role": "assistant", "content": text}
        if tool_calls:
            am["tool_calls"] = tool_calls
        msgs.append(am)

        if tool_calls:
            obs = _truncate(_obs_text(st.get("observation")), max_obs_chars)
            msgs.append({
                "role": "tool",
                "name": tool_calls[0]["function"]["name"],
                "content": obs,
            })
    return msgs


def _reward(trial: Path) -> float | None:
    rf = trial / "verifier" / "reward.txt"
    if rf.exists():
        try:
            return float(rf.read_text().strip())
        except ValueError:
            return None
    res = trial / "result.json"
    if res.exists():
        try:
            d = json.loads(res.read_text())
        except json.JSONDecodeError:
            return None
        vr = (d.get("verifier_result") or {})
        rw = vr.get("rewards")
        if isinstance(rw, dict) and len(rw) == 1:
            return float(next(iter(rw.values())))
        if isinstance(rw, (int, float)):
            return float(rw)
    return None


def scan_jobs(jobs_dir: Path) -> dict:
    """trial task_name -> {"reward": float|None, "dir": Path}. Task name comes
    from result.json, never from the directory name (harbor appends a random
    suffix, e.g. `polyglot_rust_accumulate__8L6VBK7`)."""
    out: dict = {}
    for res in sorted(jobs_dir.glob("*/result.json")):
        trial = res.parent
        try:
            d = json.loads(res.read_text())
        except json.JSONDecodeError:
            print(f"  WARN unreadable result.json: {res}")
            continue
        name = d.get("task_name")
        if not name:
            print(f"  WARN result.json has no task_name: {res}")
            continue
        out[name] = {"reward": _reward(trial), "dir": trial}
    return out


def split_task(name: str) -> tuple[str, str]:
    """`polyglot_rust_accumulate` -> ("rust", "accumulate")."""
    parts = name.split("_", 2)
    if len(parts) == 3 and parts[0] == "polyglot":
        return parts[1], parts[2]
    return "unknown", name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", required=True,
                    help="harbor job dir of the TEACHER run (128e). Contains "
                         "one subdir per trial.")
    ap.add_argument("--candidate-jobs-dir", default=None,
                    help="OPTIONAL harbor job dir of the candidate (pruned) "
                         "model. Supplying it promotes traces out of Set-C into "
                         "Set-A/Set-B, which is where the 3x regression weight "
                         "lives. Omit on the first pass.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--class-by", choices=["language", "task"], default="language",
                    help="language (default) -> bench key `polyglot_<lang>`; "
                         "task -> one class per exercise (only sane with many "
                         "traces per exercise).")
    ap.add_argument("--reasoning", choices=["strip", "include"], default="strip",
                    help="see module docstring. Default `strip` keeps this tier "
                         "commensurable with the published answer-channel maps.")
    ap.add_argument("--max-obs-chars", type=int, default=4000,
                    help="per-observation truncation, head+tail (0 = no limit). "
                         "Terminal dumps dominate token count; the error text is "
                         "at the tail and is preserved.")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap steps per trajectory (0 = all).")
    ap.add_argument("--min-messages", type=int, default=3,
                    help="drop degenerate trajectories shorter than this.")
    ap.add_argument("--include-fail", action="store_true",
                    help="ALSO emit teacher-FAIL trajectories. Off by default: "
                         "Tier-B is a PASS-trace replay by construction — a "
                         "failing trajectory teaches the map the routing of a "
                         "wrong answer.")
    args = ap.parse_args()

    jobs = Path(args.jobs_dir)
    if not jobs.is_dir():
        raise SystemExit(f"--jobs-dir does not exist: {jobs}")
    print(f"=== extract_polyglot_traces — teacher={jobs} ===", flush=True)
    teacher = scan_jobs(jobs)
    print(f"  {len(teacher)} teacher trials", flush=True)

    cand: dict = {}
    if args.candidate_jobs_dir:
        cp = Path(args.candidate_jobs_dir)
        if not cp.is_dir():
            raise SystemExit(f"--candidate-jobs-dir does not exist: {cp}")
        cand = scan_jobs(cp)
        print(f"  {len(cand)} candidate trials", flush=True)

    traces = []
    skipped = Counter()
    for name, info in sorted(teacher.items()):
        rw = info["reward"]
        if rw is None:
            skipped["no_reward"] += 1
            continue
        passed = rw > 0
        if not passed and not args.include_fail:
            skipped["teacher_fail"] += 1
            continue

        traj = info["dir"] / "agent" / "trajectory.json"
        if not traj.exists():
            skipped["no_trajectory"] += 1
            continue
        try:
            steps = json.loads(traj.read_text()).get("steps") or []
        except json.JSONDecodeError:
            skipped["bad_trajectory"] += 1
            continue

        msgs = steps_to_messages(steps, args.reasoning, args.max_obs_chars,
                                 args.max_steps)
        if len(msgs) < args.min_messages:
            skipped["too_short"] += 1
            continue

        lang, exercise = split_task(name)
        bench = f"polyglot_{lang}" if args.class_by == "language" else f"polyglot_{lang}_{exercise}"

        if not cand:
            s = "C"
        elif name not in cand or cand[name]["reward"] is None:
            s = "C"
        else:
            s = "A" if cand[name]["reward"] > 0 else "B"

        traces.append({
            "bench": bench,
            "task_id": exercise,
            "language": lang,
            "set": s,
            "weight": SET_WEIGHTS[s],
            "messages": msgs,
            # Kept EMPTY on purpose: the T203 analysis path uses `messages` and
            # ignores these, but the keys are present so any consumer written
            # against the single-turn schema fails loudly rather than silently
            # replaying a truncated trace.
            "prompt": "",
            "completion": "",
            "n_steps": len(steps),
            "n_messages": len(msgs),
            "teacher_reward": rw,
            "trial_dir": str(info["dir"]),
        })

    set_counts = Counter(t["set"] for t in traces)
    bench_counts = Counter(t["bench"] for t in traces)
    out = {
        "metadata": {
            "source": "harbor/aider-polyglot",
            "teacher_jobs_dir": str(jobs),
            "candidate_jobs_dir": args.candidate_jobs_dir,
            "class_by": args.class_by,
            "reasoning": args.reasoning,
            "max_obs_chars": args.max_obs_chars,
            "max_steps": args.max_steps,
            "include_fail": args.include_fail,
            "trace_count": len(traces),
            "set_counts": dict(set_counts),
            "set_weights": SET_WEIGHTS,
            "bench_counts": dict(bench_counts),
            "skipped": dict(skipped),
            "trace_shape": "messages",
        },
        "traces": traces,
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f)

    print(f"\n  traces:   {len(traces)}")
    print(f"  sets:     {dict(set_counts)}")
    print(f"  classes:  {dict(bench_counts)}")
    print(f"  skipped:  {dict(skipped)}")
    if traces and not cand:
        print("  NOTE: no --candidate-jobs-dir, so every trace is Set-C at "
              "weight 1.0; the 3x Set-B regression weight is UNUSED. Re-run "
              "this extractor once a candidate has been measured.")
    if not traces:
        raise SystemExit("REFUSING to write a useful-looking empty map: 0 traces "
                         f"survived (skipped={dict(skipped)})")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
