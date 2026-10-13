#!/usr/bin/env python3
"""manic_to_tierb.py — reconstruct manic-harness runs into the Tier-B trace schema.

run.jsonl(.gz) is a TOKEN-LEVEL event stream, not a message list. Reconstruction:

  iteration_start / iteration_end     delimit one assistant turn
  content_end   contentType=reasoning -> assistant.reasoning_content
  content_end   contentType=text      -> assistant.content
  content_start contentType=tool      -> assistant.tool_calls  (carries `input` = the
                                         ARGUMENTS, i.e. the model's own output)
  content_end   contentType=tool      -> a following {role:"tool"} message (the RESULT)

content_start for reasoning/text are streaming deltas and are ignored; the matching
content_end carries the complete block. Only content_start[tool] is used, because the
tool arguments appear nowhere else.

Emits the same shape Tier-B already consumes (role / content / reasoning_content /
tool_calls / name), so the existing chat template renders it unchanged.
"""
from __future__ import annotations
import argparse, gzip, json, os, re, glob, collections

H = "/srv/dev-disk-by-uuid-92295e2c-12bd-4d15-a50c-1d80e1a33ee8/spool/manic-harness/runs-native"


def read_events(d):
    for name in ("run.jsonl", "run.jsonl.gz"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        op = gzip.open(p, "rt") if name.endswith(".gz") else open(p)
        with op as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
        return


def build_messages(d):
    """Fold the event stream into messages. Returns (messages, finish_reason)."""
    msgs = []
    prompt = os.path.join(d, "prompt.txt")
    if os.path.exists(prompt):
        txt = open(prompt).read().strip()
        if txt:
            msgs.append({"role": "user", "content": txt})

    cur_think, cur_text, cur_calls, pending_tool = [], [], [], []
    finish = None

    def flush():
        """Close the open assistant turn, then emit its tool results."""
        nonlocal cur_think, cur_text, cur_calls, pending_tool
        if cur_think or cur_text or cur_calls:
            # Heterogeneous by design: content/reasoning_content are str, tool_calls a
            # list. Annotate, or the dict is inferred dict[str, str] from its initializer.
            a: dict[str, object] = {"role": "assistant", "content": "\n".join(cur_text)}
            if cur_think:
                a["reasoning_content"] = "\n".join(cur_think)
            if cur_calls:
                a["tool_calls"] = cur_calls
            msgs.append(a)
        for t in pending_tool:
            msgs.append(t)
        cur_think, cur_text, cur_calls, pending_tool = [], [], [], []

    for rec in read_events(d):
        if rec.get("type") == "run_result":
            finish = rec.get("finishReason")
            continue
        if rec.get("type") != "agent_event":
            continue
        e = rec.get("event")
        if not isinstance(e, dict):
            continue
        t = e.get("type")
        if t == "iteration_end":
            flush()
        elif t == "content_end":
            ct = e.get("contentType")
            if ct == "reasoning":
                cur_think.append(e.get("reasoning", "") or "")
            elif ct == "text":
                cur_text.append(e.get("text", "") or "")
            elif ct == "tool":
                out = e.get("output")
                pending_tool.append({
                    "role": "tool", "name": e.get("toolName", "tool"),
                    "content": out if isinstance(out, str) else json.dumps(out)})
        elif t == "content_start" and e.get("contentType") == "tool":
            # `arguments` must be a MAPPING, not a JSON string: the Gemma-4 template
            # raises "must be a JSON object (mapping), not a string", and the SWE
            # Tier-B traces this corpus sits beside carry a dict here too. Match the
            # shipped artifact, not the OpenAI wire convention.
            cur_calls.append({
                "id": e.get("toolCallId", ""), "type": "function",
                "function": {"name": e.get("toolName", ""),
                             "arguments": e.get("input", {}) or {}}})
    flush()
    return msgs, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-contains", default="omnimerge-v4")
    ap.add_argument("--verdict", default="FIXED")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    traces, stats = [], collections.Counter()
    for d in sorted(glob.glob(H + "/*/")):
        b, e = os.path.join(d, "build.txt"), os.path.join(d, "exit.txt")
        if not (os.path.exists(b) and os.path.exists(e)):
            continue
        bt = open(b).read()
        m = re.search(r"model=(\S+)", bt)
        model = m.group(1) if m else "?"
        if args.model_contains and args.model_contains not in model:
            continue
        v = re.search(r"verdict=(\S+)", open(e).read())
        verdict = v.group(1) if v else None
        if args.verdict and verdict != args.verdict:
            continue
        msgs, finish = build_messages(d)
        asst = [x for x in msgs if x["role"] == "assistant"]
        if not asst:
            stats["no_assistant"] += 1
            continue
        # the DELIVERY turn: final assistant turn with no tool call
        delivers = bool(asst) and "tool_calls" not in asst[-1] and bool(asst[-1].get("content", "").strip())
        stats["delivers" if delivers else "no_delivery"] += 1
        traces.append({
            "task_id": os.path.basename(d.rstrip("/")), "model": model,
            "verdict": verdict, "finish_reason": finish, "delivers": delivers,
            "n_assistant": len(asst), "messages": msgs, "n_messages": len(msgs)})
        if args.limit and len(traces) >= args.limit:
            break

    json.dump({"traces": traces}, open(args.out, "w"))
    print("wrote %s: %d traces  %s" % (args.out, len(traces), dict(stats)))
    for t in traces[:10]:
        print("  %-24s %-34s %-9s deliver=%-5s asst=%d msgs=%d"
              % (t["task_id"], t["model"][:34], t["finish_reason"], t["delivers"],
                 t["n_assistant"], t["n_messages"]))


if __name__ == "__main__":
    main()
