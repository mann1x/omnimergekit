#!/usr/bin/env python3
"""compact_session.py — turn one driven opencode session's wire log into a compact,
queryable record: meta.json + summary.md, and append a line to sessions/INDEX.jsonl.

Reads <session>/wirelog/session-*.jsonl (the wire_proxy capture: paired request/response
records keyed by rid), classifies each model turn the same way as hammer_raw.py, detects
agentic tool-repeat loops, and rolls up a session verdict.
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
            # signature MUST use the FULL args — two distinct calls sharing a 200-char
            # boilerplate prefix (e.g. iterative full-file `write`s that all start
            # `import curses\n…`) would otherwise collapse into a fake repeat-run and
            # mis-classify the session TOOL_LOOP. `args_sig` hashes name+full-args so the
            # detector compares the whole payload while meta.json stays small. (2026-06-28
            # faithfulness audit: 1 false dense-31B TOOL_LOOP from this exact collision.)
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
    span = []
    for rid, s in sigs:
        if s is not None and s == prev:
            cur += 1
            span.append(rid)
        else:
            cur = 1 if s is not None else 0
            span = [rid] if s is not None else []
            prev = s
        best = max(best, cur)
    return best


def detect_server_down(sdir):
    """Upstream-down session: requests were sent but NO response ever came back
    (llama-server/llamafile crashed → connection refused), so the per-rid 'turns'
    are request-only ABORTs that reflect infra failure, not model behaviour. Without
    this, such a session looks identical to a slow-killed turn and is mislabeled
    TIMEOUT, polluting the loop table. RCA: opencoti llamafile OOM 2026-06-27.

    Signal: response records in the wire log (definitive — n_resp==0 with requests
    present = nothing ever returned), corroborated by 'Connection refused' in the
    per-session proxy.log. A session with SOME valid responses then a mid-run crash
    (n_resp>0) keeps its content-based verdict but records n_conn_refused for audit."""
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


def count_user_turns(sdir):
    """How many USER turns actually reached the model — the answer to 'did the full
    adversarial script run, or did the session die early?'

    `n_turns` elsewhere counts HTTP model calls (an agentic turn issues many), so it
    cannot answer this. The last request carries the whole accreted conversation, so the
    count of role=="user" messages in it IS the number of delivered user turns. Falls back
    to the driver's own '--- TURN n ---' markers in opencode.log."""
    last = None
    for fp in sorted(glob.glob(os.path.join(sdir, "wirelog", "session-*.jsonl"))):
        with open(fp, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if '"dir": "request"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                msgs = (r.get("req") or {}).get("messages")
                if msgs:
                    last = msgs
    if last:
        return sum(1 for m in last if m.get("role") == "user")
    ol = os.path.join(sdir, "opencode.log")
    if os.path.isfile(ol):
        try:
            return open(ol, encoding="utf-8", errors="ignore").read().count("\n--- TURN ")
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--model-port", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--task-prompt", default="")
    ap.add_argument("--rc", type=int, default=0)
    ap.add_argument("--wall", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=0,
                    help="PER-TURN budget (driver PER_TURN_TIMEOUT). Informational here: it "
                         "must NEVER be compared against --wall, which is the session total.")
    ap.add_argument("--session-timeout", type=int, default=0,
                    help="genuine WHOLE-SESSION wall budget (0 = disabled). Only this may be "
                         "compared against --wall.")
    args = ap.parse_args()

    sdir = args.session.rstrip("/")
    sid = os.path.basename(sdir)
    root = os.path.dirname(os.path.dirname(sdir))  # opencode_capture/

    turns = load_turns(os.path.join(sdir, "wirelog"))

    # upstream sampler/ctx at run time
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

    # roll-up
    vc = {}
    for t in turns:
        vc[t["verdict"]] = vc.get(t["verdict"], 0) + 1
    # A real model loop needs POSITIVE content evidence. ABORT (finish_reason=None,
    # partial < THINK_CAP) is a streamed response cut off by the per-turn wall-kill —
    # infra, not a loop. A genuinely-runaway abort has huge text and is already caught
    # as THINK_EXPLODE above, so the surviving ABORTs are never loops.
    degen = [t for t in turns if t["verdict"] in ("RUNAWAY", "THINK_EXPLODE", "CORRUPT")]
    ctx_exhausted = [t for t in turns if t["verdict"] == "CONTEXT_EXHAUSTED"]
    max_r = max([t["r"] for t in turns], default=0)
    tool_loop = detect_tool_loop(turns)
    # A TIMEOUT verdict requires that a turn was ACTUALLY killed. The driver sets rc=137
    # only when `timeout` fired on some turn. The old form also OR'd in
    # `args.wall >= args.timeout`, comparing the WHOLE-SESSION wall against the PER-TURN
    # budget -- a unit mismatch that stamped every merely-slow model TIMEOUT even though
    # no turn was killed, all 8 user turns ran, and every model call was CLEAN.
    # Measured 2026-07-25: 113 of 132 TIMEOUT verdicts were this artifact (113/113 of them
    # with zero degenerate turns and 100% CLEAN calls); it flipped the dense gemma-31b
    # control from 0% to 60-100% not-a-loop across three independent backends.
    killed = (args.rc == 137)
    # Ran longer in TOTAL than one turn's budget. INFORMATIONAL ONLY -- recorded for audit,
    # never allowed to decide a verdict. This is the flag the old bug conflated with a loop.
    slow = bool(args.timeout and args.wall >= args.timeout)
    # A real whole-session budget, opt-in. Distinct verdict so it can never masquerade
    # as model looping.
    over_session_budget = bool(args.session_timeout and args.wall >= args.session_timeout)
    n_user_turns = count_user_turns(sdir)
    sd = detect_server_down(sdir)

    # session verdict
    # NB: CONTEXT_EXHAUSTED is NOT a loop — the prompt (accreted multi-turn history)
    # filled the context window. It takes precedence over `killed`/TIMEOUT, because a
    # wall-kill that happens once the window is full is a downstream symptom of the
    # exhaustion, not a within-turn loop. Real loops (degen / tool-repeat) still win.
    if degen:
        verdict = "DEGENERATE"          # at least one looping/runaway/corrupt model turn
    elif tool_loop >= 4:
        verdict = "TOOL_LOOP"           # agent stuck repeating same tool call
    elif sd["server_down"]:
        verdict = "SERVER_DOWN"         # upstream crashed (0 responses) — INVALID, NOT a loop
    elif ctx_exhausted:
        verdict = "CONTEXT_EXHAUSTED"   # ran out of context window — NOT a loop
    elif killed:
        verdict = "TIMEOUT"             # a turn was ACTUALLY killed by the per-turn timeout
    elif over_session_budget:
        verdict = "SESSION_BUDGET"      # hit an explicit whole-session budget — NOT a loop
    elif not turns:
        verdict = "NO_TURNS"            # opencode failed before any model call
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
        "model_label": args.model_label,
        "model_port": int(args.model_port),
        "task_id": args.task_id,
        "task_prompt": args.task_prompt,
        "sampler": sampler,
        "n_ctx": n_ctx,
        "opencode_rc": args.rc,
        "wall_secs": args.wall,
        "timeout_secs": args.timeout,
        "session_timeout_secs": args.session_timeout,
        "killed_on_timeout": bool(killed),
        "ran_long": slow,                 # wall >= per-turn budget: informational, NOT a failure
        "over_session_budget": over_session_budget,
        "n_user_turns": n_user_turns,     # user turns actually delivered (script completion)
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
    L.append("- **model**: `%s` (upstream :%s)" % (args.model_label, args.model_port))
    L.append("- **sampler**: temp=%s top_k=%s top_p=%s min_p=%s rep_pen=%s · n_ctx=%s" % (
        sampler.get("temperature"), sampler.get("top_k"), sampler.get("top_p"),
        sampler.get("min_p"), sampler.get("repeat_penalty"), n_ctx))
    L.append("- **task** `%s`: %s" % (args.task_id, (args.task_prompt or "")[:300]))
    L.append("- **wall**: %ss (per-turn budget %ss, rc=%s%s%s)" % (
        args.wall, args.timeout, args.rc,
        ", KILLED" if killed else "",
        ", ran-long (session total > one turn's budget — NOT a failure)"
        if slow and not killed else ""))
    L.append("- **user turns delivered**: %s (full script = 8)" % n_user_turns)
    L.append("- **VERDICT: %s** — %d model calls, %d degenerate, max_tool_repeat=%d, max_reasoning=%d chars" % (
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
        "session_id": sid, "model_label": args.model_label, "model_port": int(args.model_port),
        "task_id": args.task_id, "verdict": verdict, "n_turns": len(turns),
        "n_degenerate_turns": len(degen), "max_tool_repeat_run": tool_loop,
        "max_reasoning_chars": max_r, "wall_secs": args.wall, "rc": args.rc,
        "timeout_secs": args.timeout, "ran_long": slow,
        "over_session_budget": over_session_budget, "n_user_turns": n_user_turns,
        "killed_on_timeout": bool(killed), "sampler": sampler, "n_ctx": n_ctx,
        "n_responses": sd["n_responses"], "n_conn_refused": sd["n_conn_refused"],
        "server_down": sd["server_down"], "n_artifacts": len(artifacts),
    }
    with open(os.path.join(root, "sessions", "INDEX.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(idx, ensure_ascii=False) + "\n")

    # EVERY captured run saves a readable conversation.md alongside the raw wire log —
    # a verdict is not auditable without the exchange that produced it.
    try:
        import dump_transcript
        tpath, nmsg, nuser = dump_transcript.render(sdir)
        print("[compact] conversation saved: %s (%d messages, %d user turns)"
              % (tpath, nmsg, nuser))
    except Exception as e:                       # never let this break a capture
        print("[compact] WARNING: conversation.md not written (%s: %s)"
              % (type(e).__name__, e))

    print("[compact] %s -> verdict=%s turns=%d degen=%d tool_repeat=%d wall=%ss" % (
        sid, verdict, len(turns), len(degen), tool_loop, args.wall))


if __name__ == "__main__":
    sys.exit(main())
