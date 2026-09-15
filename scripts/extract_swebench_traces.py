#!/usr/bin/env python3
"""extract_swebench_traces.py — Tier-B trace JSON from mini-swe-agent SWE-bench runs.

Sibling of `extract_polyglot_traces.py` (harbor) and `extract_pass_traces.py`
(single-turn lm-eval benches). Emits the same envelope consumed by
`expert_neuron_analysis_v5_targeted.py --tier-b-json`, using the T203 `messages`
key so the multi-turn agentic context survives into the forward pass.

WHY SWE-BENCH AND NOT AIDER-POLYGLOT: polyglot mounts only the stub workspace,
so the agent can never run the graded tests and there is no reproduce -> fix ->
re-test loop; the trajectories are dominated by filesystem search. SWE-bench
ships the repo's own suite in-tree at base_commit (the test_patch modifies
EXISTING test files), so the loop is real. See
memory/feedback_aider_polyglot_withholds_tests_from_the_agent.md.

INPUT
  <run_dir>/<instance_id>/<instance_id>.traj.json
      {"info": {...}, "messages": [{role, content, tool_calls?}], "instance_id"}
  <report.json>  from `swebench.harness.run_evaluation` — supplies the PASS
      labels. Tier-B is a PASS-trace replay; without the report there are no
      labels and this script refuses to guess.

TWO NORMALISATIONS, both load-bearing:

  1. tool_call `arguments` arrive as a JSON **string**. Gemma-4's chat template
     renders a string verbatim inside braces -> `call:bash{{"command":"ls"}}`,
     but renders a dict in the canonical form -> `call:bash{command:<|"|>ls<|"|>}`
     using the special quote tokens. Replaying the string form would profile
     routing over a token sequence the model NEVER emitted. We parse to dict and
     GATE on the result.

  2. mini-swe-agent appends a synthetic `{"role": "exit"}` message. It is not a
     chat role; the template would either drop or mis-render it. Removed.

REASONING IS DROPPED FROM THE REPLAYED TRACE, deliberately. mini-swe-agent DOES
persist `reasoning_content` per message, but the published Tier-B profiles the
ANSWER channel (it replays saved answer completions), and Gemma-4's template
strips model-turn thinking anyway. Keeping the tiers commensurable matters more
than the extra tokens. The field is still READ for one purpose: detecting which
traces contain a thinking-budget cut, since the nudge is injected into the
thinking channel and never appears in `content`.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

SET_WEIGHTS = {"A": 1.0, "B": 3.0, "C": 1.0}

# Proxy only. Measured on a real agentic trace: 17,105 template-rendered tokens.
# Used solely to EXCLUDE pathological traces, never to price a run -- for that,
# render through the tokenizer.
CHARS_PER_TOKEN = 3.5

# Emitted by llama-server when a generation exhausts --reasoning-budget. Its
# presence means the model was cut off mid-thought and nudged; the trace is
# still valid (it passed), but the count is reported so a cohort dominated by
# capped traces is never mistaken for a clean one.
BUDGET_MARKER = "I have used my thinking budget"

# log_parser -> coarse language family. The dataset has no `language` column;
# the parser is the toolchain and is the honest proxy for it.
FAMILY_BY_PARSER = {
    "parse_log_cargo": "rust",
    "parse_log_gotest": "go",
    "parse_log_phpunit": "php",
    "parse_log_jest": "javascript",
    "parse_log_karma": "javascript",
    "parse_log_vitest": "javascript",
    "parse_log_tap": "javascript",
    "parse_log_immutable_js": "javascript",
    "parse_log_maven": "java",
    "parse_log_gradle_custom": "java",
    "parse_log_ant": "java",
    "parse_log_googletest": "cpp_c",
    "parse_log_redis": "cpp_c",
    "parse_log_jq": "cpp_c",
    "parse_log_micropython_test": "cpp_c",
    "parse_log_rspec_transformed_json": "ruby",
    "parse_log_ruby_unit": "ruby",
    "parse_log_jekyll": "ruby",
    "parse_log_doctest": "other",
}


def normalise_tool_calls(tcs: list) -> list:
    """JSON-string arguments -> dict. See normalisation (1) in the docstring."""
    out = []
    for tc in tcs or []:
        fn = dict(tc.get("function") or {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.loads(args)
            except json.JSONDecodeError:
                # A non-JSON argument string is not renderable in canonical
                # form. Keep it, but the caller's gate will flag the trace.
                fn["arguments"] = {"command": args}
        out.append({"id": tc.get("id", ""), "type": tc.get("type", "function"),
                    "function": fn})
    return out


def clean_messages(msgs: list, max_obs_chars: int, reasoning: str) -> tuple[list, int]:
    """Drop the synthetic `exit` turn, normalise tool calls, cap observations,
    and (by default) CARRY REASONING through as `reasoning_content`."""
    out = []
    n_calls = 0
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "exit":
            continue
        if role == "tool":
            c = str(m.get("content") or "")
            if 0 < max_obs_chars < len(c):
                head = max_obs_chars // 3
                c = (c[:head] + f"\n...[{len(c) - max_obs_chars} chars elided]...\n"
                     + c[-(max_obs_chars - head):])
            out.append({"role": "tool", "name": m.get("name", "bash"), "content": c})
            continue
        nm: dict = {"role": role, "content": m.get("content") or ""}
        if reasoning == "include" and role == "assistant":
            rc = m.get("reasoning_content") or m.get("reasoning") or ""
            if rc:
                nm["reasoning_content"] = rc
        if m.get("tool_calls"):
            nm["tool_calls"] = normalise_tool_calls(m["tool_calls"])
            n_calls += len(nm["tool_calls"])
        out.append(nm)
    return out, n_calls


def load_report(path: Path) -> set:
    """Resolved instance ids from a swebench run_evaluation report."""
    d = json.loads(path.read_text())
    for key in ("resolved_ids", "resolved", "resolved_instances"):
        if isinstance(d.get(key), list):
            return set(d[key])
    # Some report shapes are {instance_id: {"resolved": bool}}
    out = set()
    for k, v in d.items():
        if isinstance(v, dict) and v.get("resolved") is True:
            out.add(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="mini-swe-agent output dir")
    ap.add_argument("--report", required=True,
                    help="swebench run_evaluation report json (PASS labels). "
                         "Tier-B replays PASSES; without labels this refuses.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="swe-bench/SWE-Bench_Multilingual")
    ap.add_argument("--split", default="test")
    ap.add_argument("--class-by", choices=["family", "flat"], default="family",
                    help="family (default) -> swe_<language>; flat -> one class")
    ap.add_argument("--max-obs-chars", type=int, default=4000,
                    help="per-observation cap, head+tail (0 = uncapped). Test "
                         "output is long and the failure is at the tail.")
    ap.add_argument("--reasoning", choices=["include", "strip"], default="include",
                    help="include (DEFAULT): carry `reasoning_content` on assistant "
                         "turns. For AGENTIC CODING the overwhelming majority of "
                         "expert usage is in the thinking channel, so stripping it "
                         "throws away the signal the map exists to capture. NOTE: "
                         "this only reaches the forward pass if the analysis renders "
                         "with the GENERATION-TIME template and preserve_thinking=True "
                         "-- the stock model-dir template has no reasoning support and "
                         "drops the field silently.")
    ap.add_argument("--min-messages", type=int, default=6)
    ap.add_argument("--max-messages", type=int, default=600,
                    help="EXCLUDE traces with more turns than this (degenerate "
                         "step-limit grinders).")
    ap.add_argument("--max-trace-tokens", type=int, default=60000,
                    help="EXCLUDE traces longer than this (estimated at "
                         f"{CHARS_PER_TOKEN} chars/token). A single pathological "
                         "trace can cost more replay time than the rest of the "
                         "cohort combined. 0 disables.")
    ap.add_argument("--keep-noncanonical", action="store_true",
                    help="keep traces whose tool arguments will not render in "
                         "canonical form. Off by default: replaying them "
                         "profiles a token sequence the model never emitted.")
    ap.add_argument("--exclude-capped", action="store_true",
                    help="also EXCLUDE traces containing a thinking-budget cut "
                         "(reported either way).")
    ap.add_argument("--include-fail", action="store_true",
                    help="also emit unresolved trajectories (NOT Tier-B semantics)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    resolved = load_report(Path(args.report))
    print(f"=== extract_swebench_traces — {run_dir} ===")
    print(f"  resolved instances in report: {len(resolved)}")

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    parser_by_id = {r["instance_id"]: r["log_parser"] for r in ds}

    traces = []
    skipped = Counter()
    noncanonical = []
    n_capped = 0
    for tj in sorted(glob.glob(str(run_dir / "*" / "*.traj.json"))):
        d = json.loads(Path(tj).read_text())
        iid = d.get("instance_id") or os.path.basename(os.path.dirname(tj))
        passed = iid in resolved
        if not passed and not args.include_fail:
            skipped["unresolved"] += 1
            continue

        msgs, n_calls = clean_messages(d.get("messages") or [], args.max_obs_chars,
                                       args.reasoning)
        if len(msgs) < args.min_messages:
            skipped["too_short"] += 1
            continue
        if n_calls == 0:
            # A trajectory with no tool calls is not an agentic trace.
            skipped["no_tool_calls"] += 1
            continue

        # EXCLUSION 1 — non-canonical tool arguments. Replaying these profiles a
        # token sequence the model never produced.
        bad_args = any(not isinstance(tc["function"].get("arguments"), dict)
                       for m in msgs for tc in (m.get("tool_calls") or []))
        if bad_args:
            noncanonical.append(iid)
            if not args.keep_noncanonical:
                skipped["noncanonical_tool_args"] += 1
                continue

        # EXCLUSION 2 — degenerate turn count.
        if args.max_messages and len(msgs) > args.max_messages:
            skipped["too_many_messages"] += 1
            continue

        # EXCLUSION 3 — pathologically long trace. One of these can cost more
        # replay time than the whole rest of the cohort.
        chars = sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or []))
                    for m in msgs)
        est_tok = int(chars / CHARS_PER_TOKEN)
        if args.max_trace_tokens and est_tok > args.max_trace_tokens:
            skipped["too_long"] += 1
            continue

        # The budget nudge is injected BEFORE the end-of-thinking tag, so with
        # --reasoning-format deepseek it lands in `reasoning_content`, never in
        # `content`. Searching only `content` silently reports zero capped
        # traces. Search the RAW message (pre-clean) across both fields.
        capped = any(BUDGET_MARKER in str(m.get("content") or "")
                     or BUDGET_MARKER in str(m.get("reasoning_content") or "")
                     for m in (d.get("messages") or []) if isinstance(m, dict))
        if capped:
            n_capped += 1
            if args.exclude_capped:
                skipped["budget_capped"] += 1
                continue

        fam = FAMILY_BY_PARSER.get(parser_by_id.get(iid, ""), "other")
        bench = f"swe_{fam}" if args.class_by == "family" else "swe_agentic"
        traces.append({
            "bench": bench,
            "task_id": iid,
            "language": fam,
            "set": "C",
            "weight": SET_WEIGHTS["C"],
            "messages": msgs,
            "prompt": "",
            "completion": "",
            "n_messages": len(msgs),
            "n_tool_calls": n_calls,
            "est_tokens": est_tok,
            "n_reasoning_turns": sum(1 for m in msgs if m.get("reasoning_content")),
            "reasoning_chars": sum(len(m.get("reasoning_content") or "") for m in msgs),
            "budget_capped": capped,
            "resolved": passed,
            "api_calls": (d.get("info", {}).get("model_stats", {}) or {}).get("api_calls"),
        })

    if noncanonical:
        print(f"  WARNING: {len(set(noncanonical))} instance(s) have non-dict tool "
              f"arguments and would render non-canonically: {sorted(set(noncanonical))[:5]}")

    set_counts = Counter(t["set"] for t in traces)
    bench_counts = Counter(t["bench"] for t in traces)
    out = {
        "metadata": {
            "source": "mini-swe-agent/SWE-Bench_Multilingual",
            "run_dir": str(run_dir),
            "report": args.report,
            "dataset": args.dataset,
            "class_by": args.class_by,
            "max_obs_chars": args.max_obs_chars,
            "include_fail": args.include_fail,
            "trace_count": len(traces),
            "set_counts": dict(set_counts),
            "set_weights": SET_WEIGHTS,
            "bench_counts": dict(bench_counts),
            "skipped": dict(skipped),
            "trace_shape": "messages",
            "tool_args_normalised": True,
            "reasoning": args.reasoning,
            "max_trace_tokens": args.max_trace_tokens,
            "max_messages": args.max_messages,
            "budget_capped_kept": n_capped,
            "est_tokens_total": sum(t["est_tokens"] for t in traces),
        },
        "traces": traces,
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f)

    print(f"\n  traces:  {len(traces)}")
    print(f"  classes: {dict(bench_counts)}")
    print(f"  skipped: {dict(skipped)}")
    if traces:
        et = sorted(t["est_tokens"] for t in traces)
        print(f"  est tokens: p50={et[len(et)//2]:,} max={et[-1]:,} "
              f"TOTAL={sum(et):,} (proxy at {CHARS_PER_TOKEN} chars/tok)")
        print(f"  traces containing a thinking-budget cut: {n_capped}/{len(traces)}")
        rc = sum(t["reasoning_chars"] for t in traces)
        tc = sum(t["est_tokens"] for t in traces) * CHARS_PER_TOKEN
        print(f"  reasoning: {sum(t['n_reasoning_turns'] for t in traces)} turns, "
              f"{rc:,} chars ({100.0 * rc / max(tc, 1):.0f}% of trace text)")
    if not traces:
        raise SystemExit("REFUSING to write an empty-but-valid-looking map: 0 traces "
                         f"survived (skipped={dict(skipped)})")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
