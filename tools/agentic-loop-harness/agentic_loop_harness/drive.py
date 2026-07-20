"""Closed-loop multi-turn agentic driver (H1).

The single-turn probe (`probe.py`) cannot reach the loops users actually report.
Those happen on the SECOND turn of a live agentic session -- after the model
emits a tool call, the tool runs, and its result comes back -- where the model
either ruminates ("looped during reasoning on the second attempt") or re-emits
the same tool call over and over ("looped on calling a tool"). That regime also
is exactly where the chat-template thinking-reinject path fires (a prior
assistant turn carrying `tool_calls` + reasoning before the last user message).

This module drives the model through that live session: it sends a real tool set,
EXECUTES the tool calls the model emits against a per-run sandbox, feeds the
results back as `role:tool` messages, and continues turn by turn until the model
terminates cleanly, loops, thrashes a tool, or hits a turn cap. Each turn is
scored with the SAME oracles as the rest of the harness (verbatim `detect.py` +
non-verbatim `softloop.py`); on top of that it scores three CROSS-TURN failures
single-turn scoring is structurally blind to:

  * tool_thrash        -- the same (tool, args) emitted in >= THRASH_REPEAT turns.
  * cross_turn_repeat  -- assistant content/thinking near-identical across turns.
  * no_terminate       -- hit the turn cap without ever returning a clean final
                          answer (content + finish=stop + no tool_calls).

It is serving-agnostic (talks OpenAI `/v1/chat/completions` via replay.chat) and
writes the SAME result schema as `probe.py`, so `scripts/diff_divergence.py`
pairs a v7 run against a 128e run into the v7-fails / 128e-clean divergence set
with no changes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from .detect import detect_turn_loop
from .replay import chat
from .softloop import assess_extra

# --- cross-turn thresholds (named, so a report can cite them) -----------------
THRASH_REPEAT = 3        # same (tool,args) in >=N turns -> tool_thrash
CROSS_SIM = 0.92         # assistant text Jaccard(5gram) >= this across turns -> repeat
CROSS_MIN_CHARS = 80     # ignore tiny assistant turns for cross-turn repeat
DEFAULT_MAX_TURNS = 8

# the published "recommended" deploy sampler (model card / ollama defaults).
RECOMMENDED = [{
    "name": "recommended_t0.9",
    "params": {"temperature": 0.9, "top_k": 64, "top_p": 0.95,
               "min_p": 0.05, "repeat_penalty": 1.1},
}]


# ---------------------------------------------------------------------------
# Tool executor: a minimal but realistic coding-agent sandbox. Just enough to
# carry the reported tasks ("create solar-system-v4.html", "list then write")
# through a genuine multi-turn tool sequence. Pure stdlib, no shell execution.
# ---------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List the files and subdirectories in a directory of the project.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string",
                     "description": "Directory path relative to the project root. Use '.' for the root."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read and return the full text contents of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the project root."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given contents. Use this to write code.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the project root."},
            "content": {"type": "string", "description": "The full file contents to write."}},
            "required": ["path", "content"]}}},
]


class ToolExecutor:
    """Executes the harness tool set against a sandbox directory. Every path is
    confined to the sandbox; traversal is rejected (returns an error result, the
    way a real agent runtime would, rather than raising)."""

    def __init__(self, sandbox):
        self.sandbox = os.path.abspath(sandbox)
        os.makedirs(self.sandbox, exist_ok=True)

    def _resolve(self, path):
        p = os.path.abspath(os.path.join(self.sandbox, path or "."))
        if p != self.sandbox and not p.startswith(self.sandbox + os.sep):
            raise ValueError("path escapes sandbox: %r" % path)
        return p

    def run(self, name, args):
        """Return (result_dict, ok_bool). Never raises -- a tool error is data."""
        try:
            if name == "list_directory":
                p = self._resolve(args.get("path", "."))
                if not os.path.isdir(p):
                    return {"error": "not a directory: %s" % args.get("path")}, False
                return {"entries": sorted(os.listdir(p))}, True
            if name == "read_file":
                p = self._resolve(args["path"])
                if not os.path.isfile(p):
                    return {"error": "no such file: %s" % args.get("path")}, False
                with open(p, encoding="utf-8", errors="replace") as fh:
                    return {"content": fh.read()}, True
            if name == "write_file":
                p = self._resolve(args["path"])
                os.makedirs(os.path.dirname(p) or self.sandbox, exist_ok=True)
                data = args.get("content", "")
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(data)
                return {"ok": True, "path": args["path"], "bytes": len(data)}, True
            return {"error": "unknown tool: %s" % name}, False
        except KeyError as e:
            return {"error": "missing argument: %s" % e}, False
        except Exception as e:  # noqa: BLE001 -- surface as a tool error, faithfully
            return {"error": "%s: %s" % (type(e).__name__, e)}, False


# ---------------------------------------------------------------------------
# cross-turn helpers
# ---------------------------------------------------------------------------
def _shingles(text, n=5):
    toks = re.findall(r"\w+", (text or "").lower())
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a) + len(b) - inter)


def _norm_args(arguments):
    """Normalise a tool-call arguments JSON string for thrash comparison. Falls
    back to the raw string when it is not parseable (a degenerate partial call)."""
    try:
        obj = json.loads(arguments or "{}")
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return (arguments or "").strip()


# ---------------------------------------------------------------------------
# one live multi-turn session
# ---------------------------------------------------------------------------
def drive_session(server, task, params, seed, max_turns, budget_tokens,
                  sandbox, timeout=1800.0, dump_run=None, log=print):
    """Drive one task to completion / failure. Returns a per-session result dict."""
    ex = ToolExecutor(sandbox)
    # pre-seed any files the task needs to exist (read_modify-style tasks).
    for fn, body in (task.get("seed_files") or {}).items():
        ex.run("write_file", {"path": fn, "content": body})

    sysmsg = task.get("system")
    messages = []
    if sysmsg:
        messages.append({"role": "system", "content": sysmsg})
    messages.append({"role": "user", "content": task["user"]})

    base = {"max_tokens": task.get("max_tokens", 32000), "tool_choice": "auto",
            "stream_options": {"include_usage": True}}
    base.update(params)
    base["seed"] = seed

    tool_sig_turns = {}       # (name,nargs) -> set(turn idx)
    assist_shingles = []      # per-turn assistant shingles for cross-turn repeat
    turns = []
    terminal = False
    any_loop = any_runaway = any_soft = False
    cross_turn_repeat = False
    sum_para = sum_tmpl = sum_over = 0

    for ti in range(max_turns):
        try:
            out = chat(server, messages, TOOLS, base, timeout, stream=True)
        except Exception as e:  # noqa: BLE001
            log("    [seed=%d turn=%d] ERROR %s" % (seed, ti, e))
            turns.append({"turn": ti, "error": str(e)})
            return _session_result(seed, turns, fail=True, fail_ext=True,
                                   reason="error", terminal=False,
                                   sums=(sum_para, sum_tmpl, sum_over),
                                   flags=dict(loop=any_loop, runaway=any_runaway,
                                              soft=any_soft, thrash=False,
                                              cross=False, no_term=False),
                                   dump_run=dump_run, task=task["name"])

        content = out["content"]
        tcs = out.get("tool_calls") or []
        # Score the model's NATURAL-LANGUAGE answer + reasoning ONLY. Tool-call
        # arguments are code/data (e.g. a whole HTML file passed to write_file)
        # and must never be fed to the loop/template/paraphrase oracles -- that
        # false-positives on a clean write_file payload exactly the way scanning
        # the answer-channel HTML did in the single-turn probe. Tool-level
        # repetition is caught structurally by tool_thrash below, not by the
        # prose oracles.
        v = detect_turn_loop(content, out["reasoning_content"])
        s = assess_extra(content, out["reasoning_content"], out["finish_reason"],
                         out.get("completion_tokens"), budget_tokens,
                         has_tool=bool(tcs))
        runaway = out["finish_reason"] not in ("stop", "tool_calls")
        any_loop = any_loop or bool(v["is_loop"])
        any_runaway = any_runaway or runaway
        any_soft = any_soft or bool(s["soft_fail"])
        sum_para += int(s["paraphrase_loop"])
        sum_tmpl += int(s["template_loop"])
        sum_over += int(s["overthinking"])

        # cross-turn repeat (only on substantive assistant text)
        if len(content) >= CROSS_MIN_CHARS:
            sh = _shingles(content)
            for prev in assist_shingles:
                if _jaccard(sh, prev) >= CROSS_SIM:
                    cross_turn_repeat = True
                    break
            assist_shingles.append(sh)

        # record tool signatures for thrash detection: BOTH the full-args
        # signature AND a (tool, target-path) signature. The latter catches the
        # real field loop -- the model calling write_file on the SAME path every
        # turn with slightly churning content (the args differ byte-wise so the
        # full-args sig misses it, yet it is exactly "looped on calling a tool").
        for tc in tcs:
            tool_sig_turns.setdefault((tc["name"], _norm_args(tc["arguments"])),
                                      set()).add(ti)
            try:
                pth = json.loads(tc["arguments"] or "{}").get("path")
            except Exception:
                pth = None
            if pth:
                tool_sig_turns.setdefault((tc["name"], "path:%s" % pth),
                                          set()).add(ti)

        turns.append({"turn": ti, "finish": out["finish_reason"],
                      "n_tool_calls": len(tcs),
                      "tools": [tc["name"] for tc in tcs],
                      "think_len": v["thinking_len"], "ans_len": len(content),
                      "is_loop": bool(v["is_loop"]), "runaway": runaway,
                      "soft_fail": bool(s["soft_fail"]),
                      "overthinking": bool(s["overthinking"])})
        log("    [seed=%d turn=%d] tools=%s think=%d ans=%d finish=%s loop=%s soft=%s"
            % (seed, ti, [tc["name"] for tc in tcs] or "-", v["thinking_len"],
               len(content), out["finish_reason"], v["is_loop"], s["soft_fail"]))

        if runaway:
            break
        if not tcs:
            # terminal: a final answer with no further tool calls
            terminal = (out["finish_reason"] == "stop")
            break

        # execute tool calls, append assistant + tool messages, continue
        messages.append({"role": "assistant", "content": content or "",
                         "reasoning_content": out["reasoning_content"],
                         "tool_calls": [{"id": tc["id"], "type": "function",
                                         "function": {"name": tc["name"],
                                                      "arguments": tc["arguments"]}}
                                        for tc in tcs]})
        for tc in tcs:
            try:
                cargs = json.loads(tc["arguments"] or "{}")
            except Exception:
                cargs = {}
            res, _ok = ex.run(tc["name"], cargs)
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(res)[:8000]})

    thrash = any(len(tset) >= THRASH_REPEAT for tset in tool_sig_turns.values())
    no_terminate = (not terminal) and (not any_runaway)
    # strict / structural failure (verbatim-comparable to the rest of the harness)
    is_fail = bool(any_loop or any_runaway or thrash or no_terminate)
    soft_fail = bool(any_soft or cross_turn_repeat)
    is_fail_ext = bool(is_fail or soft_fail)
    reason = _first_reason(any_loop, any_runaway, thrash, no_terminate,
                           cross_turn_repeat, any_soft)
    return _session_result(seed, turns, fail=is_fail, fail_ext=is_fail_ext,
                           reason=reason, terminal=terminal,
                           sums=(sum_para, sum_tmpl, sum_over),
                           flags=dict(loop=any_loop, runaway=any_runaway,
                                      soft=any_soft, thrash=thrash,
                                      cross=cross_turn_repeat, no_term=no_terminate),
                           dump_run=dump_run, task=task["name"], messages=messages)


def _first_reason(loop, runaway, thrash, no_term, cross, soft):
    for ok, name in ((loop, "turn_loop"), (runaway, "runaway"),
                     (thrash, "tool_thrash"), (no_term, "no_terminate"),
                     (cross, "cross_turn_repeat"), (soft, "soft")):
        if ok:
            return name
    return "clean"


def _session_result(seed, turns, fail, fail_ext, reason, terminal, sums, flags,
                    dump_run, task, messages=None):
    para, tmpl, over = sums
    rec = {"seed": seed, "n_turns": len(turns), "terminal": terminal,
           "reason": reason, "is_fail": fail, "is_fail_ext": fail_ext,
           "is_loop": flags["loop"], "runaway": flags["runaway"],
           "soft_fail": flags["soft"], "tool_thrash": flags["thrash"],
           "cross_turn_repeat": flags["cross"], "no_terminate": flags["no_term"],
           "paraphrase_loop": para > 0, "template_loop": tmpl > 0,
           "overthinking": over > 0, "turns": turns}
    if dump_run:
        d = dict(rec)
        if messages is not None:
            d["messages"] = messages
        dump_run(d)
    return rec


# ---------------------------------------------------------------------------
# task suite runner -> probe-compatible result JSON
# ---------------------------------------------------------------------------
def run_tasks(server, suite, matrix, seeds, model_name, sandbox_root,
              max_turns=DEFAULT_MAX_TURNS, dump_dir=None, timeout=1800.0,
              log=print):
    tasks = suite["tasks"]
    log("=== drive model=%s server=%s suite=%s tasks=%d seeds=%d configs=%d turns<=%d ==="
        % (model_name, server, suite.get("name"), len(tasks), len(seeds),
           len(matrix), max_turns))
    out = {"model": model_name, "server": server, "suite": suite.get("name"),
           "mode": "drive", "seeds": seeds, "matrix": matrix,
           "max_turns": max_turns, "prompts": []}
    for task in tasks:
        budget = task.get("reasoning_budget", suite.get("default", {}).get("reasoning_budget"))
        dump_path = None
        dump_run = None
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, "%s__%s.jsonl" % (model_name, task["name"]))
            open(dump_path, "w").close()

            def dump_run(rec, _p=dump_path):
                with open(_p, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
        log("--- task=%s ---" % task["name"])
        configs = []
        for cfg in matrix:
            name, sp = cfg["name"], dict(cfg.get("params") or {})
            runs = []
            for sd in seeds:
                sbx = os.path.join(sandbox_root, "%s__%s__%s__s%d"
                                   % (model_name, task["name"], name, sd))
                runs.append(drive_session(server, task, sp, sd, max_turns, budget,
                                          sbx, timeout=timeout, dump_run=dump_run,
                                          log=log))
            n = len(seeds)
            agg = _aggregate(name, sp, runs, n)
            configs.append(agg)
            log("  ==> %-22s strict=%d/%d=%.0f%%  EXT=%d/%d=%.0f%%"
                "  (loop=%d runaway=%d thrash=%d noterm=%d cross=%d over=%d)"
                % (name, agg["fails"], n, 100 * agg["fail_rate"],
                   agg["fails_ext"], n, 100 * agg["fail_rate_ext"],
                   agg["loops"], agg["runaways"], agg["tool_thrash"],
                   agg["no_terminate"], agg["cross_turn_repeat"], agg["overthinking"]))
        out["prompts"].append({"name": task["name"], "budget": budget,
                               "dump": dump_path, "configs": configs})
    return out


def _aggregate(name, sp, runs, n):
    def c(k):
        return sum(int(r.get(k, False)) for r in runs)
    fails = c("is_fail")
    fails_ext = c("is_fail_ext")
    return {"config": name, "params": sp, "seeds": n,
            "fails": fails, "fail_rate": fails / max(1, n),
            "fails_ext": fails_ext, "fail_rate_ext": fails_ext / max(1, n),
            "loops": c("is_loop"), "loop_rate": c("is_loop") / max(1, n),
            "runaways": c("runaway"), "runaway_rate": c("runaway") / max(1, n),
            "tool_thrash": c("tool_thrash"), "no_terminate": c("no_terminate"),
            "cross_turn_repeat": c("cross_turn_repeat"), "soft_fails": c("soft_fail"),
            # keys diff_divergence.py reads for its A[para/tmpl/over] column:
            "paraphrase": c("paraphrase_loop"), "template": c("template_loop"),
            "overthinking": c("overthinking"), "runs": runs}


def main(argv=None):
    ap = argparse.ArgumentParser(description="closed-loop multi-turn agentic driver (H1)")
    ap.add_argument("--server", required=True, help="OpenAI-compatible base URL")
    ap.add_argument("--suite", required=True, help="task suite JSON")
    ap.add_argument("--model-name", required=True, help="tag, e.g. v7-coder-q6k")
    ap.add_argument("--out", required=True, help="result JSON path")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-list", help="comma-separated explicit seeds (overrides --seeds)")
    ap.add_argument("--matrix", help="sampler matrix JSON; default=recommended")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--sandbox-root", required=True, help="dir for per-session sandboxes")
    ap.add_argument("--dump-dir", help="dir for per-task session JSONL")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    with open(args.suite) as fh:
        suite = json.load(fh)
    matrix = json.load(open(args.matrix)) if args.matrix else RECOMMENDED
    if args.seed_list:
        seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]
    else:
        seeds = list(range(1, args.seeds + 1))

    t0 = time.time()
    out = run_tasks(args.server, suite, matrix, seeds, args.model_name,
                    args.sandbox_root, max_turns=args.max_turns,
                    dump_dir=args.dump_dir, timeout=args.timeout)
    out["duration_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n=== %s : multi-turn EXT fail-rate per task (worst config) ===" % args.model_name)
    for p in out["prompts"]:
        worst = max(p["configs"], key=lambda c: c["fail_rate_ext"])
        print("  %-22s ext=%d/%-2d strict=%d/%-2d  (loop=%d thrash=%d noterm=%d cross=%d over=%d)"
              % (p["name"], worst["fails_ext"], worst["seeds"], worst["fails"],
                 worst["seeds"], worst["loops"], worst["tool_thrash"],
                 worst["no_terminate"], worst["cross_turn_repeat"], worst["overthinking"]))
    print("written:", args.out)


if __name__ == "__main__":
    main()
