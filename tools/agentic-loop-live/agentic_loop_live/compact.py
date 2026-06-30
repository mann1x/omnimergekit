#!/usr/bin/env python3
"""compact.py — turn one driven agent session's wire log into a compact, queryable
record: meta.json + summary.md, and append a line to <out>/sessions/INDEX.jsonl.

Reads <session>/wirelog/session-*.jsonl (the proxy capture: paired request/response
records keyed by rid), classifies each model turn, detects agentic tool-repeat loops,
and rolls up a session verdict.

Verdict taxonomy:
  LOOP set = {DEGENERATE, TOOL_LOOP} ONLY.
    DEGENERATE  = >=1 turn is RUNAWAY (finish=length & completion_tokens>RUNAWAY_MIN),
                  THINK_EXPLODE (content|reasoning > THINK_CAP chars), or
                  CORRUPT (control-token leak into content/reasoning).
    TOOL_LOOP   = >=4 consecutive turns issuing the SAME (tool, FULL-args) signature.
  NOT loops: TIMEOUT (slow decode hit the per-turn wall), CONTEXT_EXHAUSTED (accreted
  prompt filled the window), SERVER_DOWN (upstream crashed -> 0 responses), COMPLETED,
  NO_TURNS.
"""
import argparse, ast, glob, hashlib, json, os, re, sys, time

CORRUPT = [r"<\|channel", r"<\|\"\|>", r"<tool_call\|>", r"<\|tool_response",
           r"<\|tool_call", r"<unused\d", r"<\|message", r"<\|constrain",
           r"<end_of_turn\|>", r"<start_of_turn\|>", r"\}<tool_call"]
THINK_CAP = 20000
# In this harness every finish_reason=length turn fills exactly to the context
# window (observed: prompt_tokens + completion_tokens == n_ctx, always). So
# completion_tokens == the room the model had and burned. A real runaway burns
# thousands of tokens without terminating; a context-EXHAUSTED turn had almost no
# room because accreted multi-turn history already filled the prompt (legit turns
# that actually stop reach ~5-6k via tool_calls/stop, so >4k with finish=length and
# no terminator is genuine rumination). Below the threshold = context-bound, NOT a loop.
RUNAWAY_MIN_COMPLETION = 4096  # completion tokens; below this a length-stop is context-bound
CTX_EXHAUST_CHARS = 400        # fallback when usage tokens are unavailable


def _usage(r):
    """Return (completion_tokens, prompt_tokens) from a response record, or (None, None)."""
    u = r.get("usage")
    if isinstance(u, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                u = parse(u)
                break
            except Exception:
                u = None
    if isinstance(u, dict):
        return u.get("completion_tokens"), u.get("prompt_tokens")
    return None, None


def classify_resp(r):
    """Per-turn verdict from a response record."""
    if r.get("http") and r["http"] != 200:
        return "HTTP_%s" % r["http"]
    cont = r.get("content") or ""
    reas = r.get("reasoning_content") or ""
    for blob in (cont, reas):
        for pat in CORRUPT:
            if re.search(pat, blob):
                return "CORRUPT"
    # A genuine within-turn explosion is a loop regardless of how it stopped.
    if len(reas) > THINK_CAP or len(cont) > THINK_CAP:
        return "THINK_EXPLODE"
    fin = r.get("finish_reason")
    if fin == "length":
        comp, _ = _usage(r)
        if comp is not None:
            # large generation that hit the cap = real runaway; tiny = context-bound
            return "RUNAWAY" if comp > RUNAWAY_MIN_COMPLETION else "CONTEXT_EXHAUSTED"
        # no usage: fall back to emitted-char heuristic
        if (len(cont) + len(reas)) <= CTX_EXHAUST_CHARS:
            return "CONTEXT_EXHAUSTED"
        return "RUNAWAY"
    if fin is None:
        return "ABORT"
    return "CLEAN"


def load_turns(wirelog_dir):
    files = sorted(glob.glob(os.path.join(wirelog_dir, "session-*.jsonl")))
    reqs, resps = {}, {}
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                rid = o.get("rid")
                if rid is None:
                    continue
                if o.get("dir") == "request":
                    reqs[rid] = o
                elif o.get("dir") == "response":
                    resps[rid] = o
    turns = []
    for rid in sorted(set(reqs) | set(resps)):
        rq = reqs.get(rid, {}).get("req", {})
        rp = resps.get(rid, {})
        tcs = rp.get("tool_calls") or []
        turns.append({
            "rid": rid,
            "n_messages": rq.get("n_messages"),
            "tools_avail": len(rq.get("tools") or []),
            "tool_choice": rq.get("tool_choice"),
            "finish_reason": rp.get("finish_reason"),
            "c": len(rp.get("content") or ""),
            "r": len(rp.get("reasoning_content") or ""),
            "n_tool_calls": len(tcs),
            # `args` is truncated to 200 chars for display/storage economy, but the loop
            # signature MUST use the FULL args -- two distinct calls sharing a 200-char
            # boilerplate prefix (e.g. iterative full-file `write`s that all start
            # `import curses\n...`) would otherwise collapse into a fake repeat-run and
            # mis-classify the session TOOL_LOOP. `args_sig` hashes name+full-args so the
            # detector compares the whole payload while meta.json stays small.
            "tool_calls": [{"name": t.get("name"),
                            "args": (t.get("arguments") or "")[:200],
                            "args_sig": hashlib.sha1(
                                ((t.get("name") or "") + "\x00" + (t.get("arguments") or "")
                                 ).encode("utf-8", "replace")).hexdigest()}
                           for t in tcs],
            "gen_secs": rp.get("gen_secs"),
            "completion_tokens": _usage(rp)[0],
            "prompt_tokens": _usage(rp)[1],
            "verdict": classify_resp(rp),
            "_reas_head": (rp.get("reasoning_content") or "")[:500],
            "_cont_head": (rp.get("content") or "")[:300],
        })
    return turns


def detect_tool_loop(turns, run=4):
    """>=run consecutive assistant turns issuing the SAME (tool,args) signature."""
    sigs = []
    for t in turns:
        if t["tool_calls"]:
            tc = t["tool_calls"][0]
            # compare on the FULL-args hash, not the 200-char display field, so distinct
            # calls with a shared boilerplate prefix aren't fused into a fake repeat-run.
            sig = tc.get("args_sig")
            if sig is None:  # back-compat for turns built before args_sig existed
                sig = (tc["name"], tc["args"])
            sigs.append((t["rid"], sig))
        else:
            sigs.append((t["rid"], None))
    best = 0
    cur = 0
    prev = object()
    for rid, s in sigs:
        if s is not None and s == prev:
            cur += 1
        else:
            cur = 1 if s is not None else 0
            prev = s
        best = max(best, cur)
    return best


def detect_server_down(sdir):
    """Upstream-down session: requests were sent but NO response ever came back
    (server crashed -> connection refused), so the per-rid 'turns' are request-only
    ABORTs that reflect infra failure, not model behaviour. Without this, such a
    session looks identical to a slow-killed turn and is mislabeled TIMEOUT, polluting
    the loop table.

    Signal: response records in the wire log (n_resp==0 with requests present = nothing
    ever returned), corroborated by 'Connection refused' in the per-session proxy.log."""
    n_req = n_resp = 0
    for fp in glob.glob(os.path.join(sdir, "wirelog", "session-*.jsonl")):
        with open(fp, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if '"dir": "request"' in line:
                    n_req += 1
                elif '"dir": "response"' in line:
                    n_resp += 1
    n_refused = 0
    pl = os.path.join(sdir, "proxy.log")
    if os.path.isfile(pl):
        try:
            n_refused = open(pl, encoding="utf-8", errors="ignore").read().count("Connection refused")
        except Exception:
            pass
    return {"n_requests": n_req, "n_responses": n_resp, "n_conn_refused": n_refused,
            "server_down": (n_req > 0 and n_resp == 0)}


def compact_session(sdir, model_label, model_port, task_id, task_prompt="",
                    rc=0, wall=0, timeout=0, index_path=None):
    """Classify one session dir and write meta.json + summary.md (+ append INDEX). Returns meta."""
    sdir = sdir.rstrip("/")
    sid = os.path.basename(sdir)
    if index_path is None:
        # default: <out>/sessions/INDEX.jsonl  where <out> is the parent of sessions/
        index_path = os.path.join(os.path.dirname(os.path.dirname(sdir)), "sessions", "INDEX.jsonl")

    turns = load_turns(os.path.join(sdir, "wirelog"))

    # upstream sampler/ctx at run time (recorded by the orchestrator into server_props.json)
    sampler, n_ctx = {}, None
    try:
        props = json.load(open(os.path.join(sdir, "server_props.json")))
        p = props.get("default_generation_settings", {})
        gp = p.get("params", {})
        sampler = {k: gp.get(k) for k in
                   ("temperature", "top_k", "top_p", "min_p", "repeat_penalty")}
        n_ctx = p.get("n_ctx")
    except Exception:
        pass

    vc = {}
    for t in turns:
        vc[t["verdict"]] = vc.get(t["verdict"], 0) + 1
    # A real model loop needs POSITIVE content evidence. ABORT (finish_reason=None,
    # partial < THINK_CAP) is a streamed response cut off by the per-turn wall-kill --
    # infra, not a loop. A genuinely-runaway abort has huge text and is already caught
    # as THINK_EXPLODE above, so the surviving ABORTs are never loops.
    degen = [t for t in turns if t["verdict"] in ("RUNAWAY", "THINK_EXPLODE", "CORRUPT")]
    ctx_exhausted = [t for t in turns if t["verdict"] == "CONTEXT_EXHAUSTED"]
    max_r = max([t["r"] for t in turns], default=0)
    tool_loop = detect_tool_loop(turns)
    killed = (rc == 137) or (timeout and wall >= timeout)
    sd = detect_server_down(sdir)

    # session verdict. CONTEXT_EXHAUSTED is NOT a loop and takes precedence over
    # killed/TIMEOUT (a wall-kill once the window is full is a downstream symptom of the
    # exhaustion). Real loops (degen / tool-repeat) still win.
    if degen:
        verdict = "DEGENERATE"
    elif tool_loop >= 4:
        verdict = "TOOL_LOOP"
    elif sd["server_down"]:
        verdict = "SERVER_DOWN"
    elif ctx_exhausted:
        verdict = "CONTEXT_EXHAUSTED"
    elif killed:
        verdict = "TIMEOUT"
    elif not turns:
        verdict = "NO_TURNS"
    else:
        verdict = "COMPLETED"

    # artifacts the agent created in its blank root (excluding the config we seeded)
    artifacts = []
    rootdir = os.path.join(sdir, "root")
    for dp, _, fns in os.walk(rootdir):
        for fn in fns:
            if fn == "opencode.json" and dp == rootdir:
                continue
            rel = os.path.relpath(os.path.join(dp, fn), rootdir)
            try:
                sz = os.path.getsize(os.path.join(dp, fn))
            except OSError:
                sz = -1
            artifacts.append({"path": rel, "bytes": sz})

    meta = {
        "session_id": sid,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_label": model_label,
        "model_port": int(model_port),
        "task_id": task_id,
        "task_prompt": task_prompt,
        "sampler": sampler,
        "n_ctx": n_ctx,
        "opencode_rc": rc,
        "wall_secs": wall,
        "timeout_secs": timeout,
        "killed_on_timeout": bool(killed),
        "n_turns": len(turns),
        "verdict_counts": vc,
        "n_degenerate_turns": len(degen),
        "max_tool_repeat_run": tool_loop,
        "max_reasoning_chars": max_r,
        "n_requests": sd["n_requests"],
        "n_responses": sd["n_responses"],
        "n_conn_refused": sd["n_conn_refused"],
        "server_down": sd["server_down"],
        "verdict": verdict,
        "artifacts": artifacts,
        "turns": turns,
    }
    with open(os.path.join(sdir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # --- human-readable summary.md ---
    L = []
    L.append("# Session %s\n" % sid)
    L.append("- **model**: `%s` (upstream :%s)" % (model_label, model_port))
    L.append("- **sampler**: temp=%s top_k=%s top_p=%s min_p=%s rep_pen=%s · n_ctx=%s" % (
        sampler.get("temperature"), sampler.get("top_k"), sampler.get("top_p"),
        sampler.get("min_p"), sampler.get("repeat_penalty"), n_ctx))
    L.append("- **task** `%s`: %s" % (task_id, (task_prompt or "")[:300]))
    L.append("- **wall**: %ss (timeout %ss, rc=%s%s)" % (
        wall, timeout, rc, ", KILLED" if killed else ""))
    L.append("- **VERDICT: %s** — %d turns, %d degenerate, max_tool_repeat=%d, max_reasoning=%d chars" % (
        verdict, len(turns), len(degen), tool_loop, max_r))
    L.append("- **verdict counts**: %s" % (", ".join("%s=%d" % kv for kv in sorted(vc.items())) or "none"))
    L.append("")
    L.append("## Turns")
    L.append("| rid | n_msg | tools? | finish | c | r | tool_calls | secs | verdict |")
    L.append("|----:|------:|-------:|--------|---:|---:|-----------|-----:|---------|")
    for t in turns:
        tcn = ",".join(tc["name"] or "?" for tc in t["tool_calls"]) or "-"
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            t["rid"], t["n_messages"], t["tools_avail"], t["finish_reason"],
            t["c"], t["r"], tcn, t["gen_secs"], t["verdict"]))
    if degen:
        L.append("")
        L.append("## Degenerate turns (reasoning head)")
        for t in degen:
            L.append("\n**rid %s — %s** (r=%d c=%d fin=%s):\n" % (
                t["rid"], t["verdict"], t["r"], t["c"], t["finish_reason"]))
            L.append("```\n%s\n```" % (t["_reas_head"] or t["_cont_head"]))
    L.append("")
    L.append("## Artifacts (files created in blank root)")
    if artifacts:
        for a in artifacts:
            L.append("- `%s` (%d bytes)" % (a["path"], a["bytes"]))
    else:
        L.append("- _(none)_")
    L.append("")
    with open(os.path.join(sdir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    # --- append INDEX.jsonl ---
    idx = {
        "session_id": sid, "model_label": model_label, "model_port": int(model_port),
        "task_id": task_id, "verdict": verdict, "n_turns": len(turns),
        "n_degenerate_turns": len(degen), "max_tool_repeat_run": tool_loop,
        "max_reasoning_chars": max_r, "wall_secs": wall, "rc": rc,
        "killed_on_timeout": bool(killed), "sampler": sampler, "n_ctx": n_ctx,
        "n_responses": sd["n_responses"], "n_conn_refused": sd["n_conn_refused"],
        "server_down": sd["server_down"], "n_artifacts": len(artifacts),
    }
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(idx, ensure_ascii=False) + "\n")

    print("[compact] %s -> verdict=%s turns=%d degen=%d tool_repeat=%d wall=%ss" % (
        sid, verdict, len(turns), len(degen), tool_loop, wall))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic_loop_live compact")
    ap.add_argument("--session", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--model-port", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--task-prompt", default="")
    ap.add_argument("--rc", type=int, default=0)
    ap.add_argument("--wall", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--index", default=None, help="path to INDEX.jsonl (default <out>/sessions/INDEX.jsonl)")
    args = ap.parse_args(argv)
    compact_session(args.session, args.model_label, args.model_port, args.task_id,
                    args.task_prompt, args.rc, args.wall, args.timeout, args.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
