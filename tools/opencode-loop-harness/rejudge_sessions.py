#!/usr/bin/env python3
"""rejudge_sessions.py — re-classify EVERY captured opencode session with the patched
compact_session detector (CONTEXT_EXHAUSTED split out of RUNAWAY/TIMEOUT), WITHOUT
re-running any model. Reads each session's wirelog + meta.json, recomputes the verdict,
optionally rewrites meta.json + rebuilds sessions/INDEX.jsonl, and prints an old->new
per-model table flagging which "loops" were context-exhaustion false positives.

  python rejudge_sessions.py            # dry-run: print table only
  python rejudge_sessions.py --write    # rewrite meta.json + INDEX.jsonl (backs up INDEX)
"""
import argparse, glob, json, os, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compact_session import load_turns, detect_tool_loop, detect_server_down  # patched detector

ROOT = os.environ.get("OPENCODE_CAPTURE_ROOT", "/mnt/sdc/ml/opencode_capture")
SESS = os.path.join(ROOT, "sessions")
# A loop is ONLY a model/agent loop. TIMEOUT (slow), CONTEXT_EXHAUSTED (prompt filled)
# and SERVER_DOWN (upstream crashed) are explicitly NOT loops.
LOOP = {"DEGENERATE", "TOOL_LOOP"}


def rollup(turns, killed, sdir):
    degen = [t for t in turns if t["verdict"] in ("RUNAWAY", "THINK_EXPLODE", "CORRUPT")]
    ctx = [t for t in turns if t["verdict"] == "CONTEXT_EXHAUSTED"]
    tool_loop = detect_tool_loop(turns)
    sd = detect_server_down(sdir)
    if degen:
        v = "DEGENERATE"
    elif tool_loop >= 4:
        v = "TOOL_LOOP"
    elif sd["server_down"]:
        v = "SERVER_DOWN"          # upstream crashed (0 responses) — INVALID, NOT a loop
    elif ctx:
        v = "CONTEXT_EXHAUSTED"
    elif killed:
        v = "TIMEOUT"
    elif not turns:
        v = "NO_TURNS"
    else:
        v = "COMPLETED"
    return v, degen, ctx, tool_loop, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows = []          # (session_id, model, old_v, new_v, idx_dict)
    for sd in sorted(glob.glob(os.path.join(SESS, "*"))):
        meta_p = os.path.join(sd, "meta.json")
        wl = os.path.join(sd, "wirelog")
        if not (os.path.isfile(meta_p) and os.path.isdir(wl)):
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        turns = load_turns(wl)
        killed = bool(meta.get("killed_on_timeout"))
        new_v, degen, ctx, tool_loop, sdinfo = rollup(turns, killed, sd)
        old_v = meta.get("verdict")
        vc = collections.Counter(t["verdict"] for t in turns)

        if args.write:
            meta["verdict"] = new_v
            meta["verdict_counts"] = dict(vc)
            meta["n_degenerate_turns"] = len(degen)
            meta["n_context_exhausted_turns"] = len(ctx)
            meta["max_tool_repeat_run"] = tool_loop
            meta["n_requests"] = sdinfo["n_requests"]
            meta["n_responses"] = sdinfo["n_responses"]
            meta["n_conn_refused"] = sdinfo["n_conn_refused"]
            meta["server_down"] = sdinfo["server_down"]
            meta["turns"] = turns
            meta["rejudged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        rows.append((meta["session_id"], meta.get("model_label", "?"), old_v, new_v, {
            "session_id": meta["session_id"], "model_label": meta.get("model_label"),
            "model_port": meta.get("model_port"), "task_id": meta.get("task_id"),
            "verdict": new_v, "n_turns": len(turns), "n_degenerate_turns": len(degen),
            "n_context_exhausted_turns": len(ctx), "max_tool_repeat_run": tool_loop,
            "max_reasoning_chars": meta.get("max_reasoning_chars"),
            "wall_secs": meta.get("wall_secs"), "rc": meta.get("opencode_rc"),
            "killed_on_timeout": killed, "sampler": meta.get("sampler"),
            "n_ctx": meta.get("n_ctx"), "n_artifacts": len(meta.get("artifacts") or []),
            "n_responses": sdinfo["n_responses"], "n_conn_refused": sdinfo["n_conn_refused"],
            "server_down": sdinfo["server_down"],
        }))

    if args.write:
        idx_p = os.path.join(SESS, "INDEX.jsonl")
        if os.path.isfile(idx_p):
            os.replace(idx_p, idx_p + ".pre_rejudge.bak")
        with open(idx_p, "w", encoding="utf-8") as fh:
            for _, _, _, _, idx in rows:
                fh.write(json.dumps(idx, ensure_ascii=False) + "\n")

    # ---- flips ----
    flips = [r for r in rows if r[2] != r[3]]
    print("=== FLIPS (old -> new) : %d of %d sessions ===" % (len(flips), len(rows)))
    for sid, model, ov, nv, _ in flips:
        print("  %-46s %-26s %-16s -> %s" % (sid, model, ov, nv))

    # ---- per-model old vs new loop% ----
    by = collections.defaultdict(lambda: {"n": 0, "old": 0, "new": 0,
                                          "ctx": 0, "olds": [], "news": []})
    for sid, model, ov, nv, _ in rows:
        d = by[model]
        d["n"] += 1
        d["old"] += 1 if ov in LOOP else 0
        d["new"] += 1 if nv in LOOP else 0
        d["ctx"] += 1 if nv == "CONTEXT_EXHAUSTED" else 0
        d["olds"].append(ov[:4]); d["news"].append(nv[:4])
    print("\n=== PER-MODEL  loop%% OLD -> NEW (loop = DEGENERATE/TOOL_LOOP/TIMEOUT) ===")
    print("%-28s %4s  %-14s  %-14s  %s" % ("model", "n", "OLD loop", "NEW loop", "ctx_exh"))
    for model in sorted(by):
        d = by[model]
        op = 100 * d["old"] / d["n"]; npc = 100 * d["new"] / d["n"]
        print("%-28s %4d  %2d/%-2d=%5.1f%%   %2d/%-2d=%5.1f%%   %d" % (
            model, d["n"], d["old"], d["n"], op, d["new"], d["n"], npc, d["ctx"]))
    print("\n=== verdict strings (NEW) ===")
    for model in sorted(by):
        print("  %-26s %s" % (model, ",".join(by[model]["news"])))


if __name__ == "__main__":
    main()
