#!/usr/bin/env python3
"""trim_traces.py — shrink agentic traces WITHOUT falsifying what the model emitted.

A manic v4 episode is ~49k tokens at p50 and 445k at max, so a teacher-forcing probe
either OOMs or left-truncates to a window that begins mid-stream. Left-truncation is the
worse failure: it silently changes the question from "does the teacher agree this episode
ends here" to "...given an arbitrary tail of it".

Measured composition of the 71-episode v4 corpus (17.6 M chars):

    reasoning     58.0%   (99.9% of it on turns that are NOT the delivery turn)
    tool_result   34.1%   (read_files alone is 67% of that, mean 6.2 kB)
    tool_args      4.7%
    content        3.1%   <- the delivery text itself

THE ONE RULE: never trim what the MODEL produced at or before a measured position.
reasoning_content, content and tool_calls[].arguments are the model's own output; cutting
them rewrites the very sequence whose continuation we are scoring. Only two things may be
touched, and both because the real system already does it:

  * PRIOR-TURN THINKING -- the Gemma-4 template gates history reasoning behind
    `preserve_thinking`, whose default is false, and its own comment records that
    replaying it "drives verbatim repetition (the v7-coder loop)". A served model never
    sees it. Dropping it here makes the artifact agree with the render, instead of
    relying on a flag at every call site. The CURRENT turn's thinking is always kept.

  * TOOL RESULTS -- environment input, not model output. The manic harness already
    truncates its own results (95 instances carry "[command truncated: N more chars;
    full command is in the tool call input]"), so a truncated result is in-distribution
    for this agent. The marker below matches that shape.

Head+tail beats head-only for a file read: the head carries imports and signatures, the
tail carries whatever the agent scrolled to. A middle elision keeps both ends addressable.
"""
from __future__ import annotations

import argparse
import json

MARKER = " ... [output truncated: %d more chars; full output was in the tool result] ... "


def cap(text: str, limit: int, head_frac: float) -> str:
    """Head+tail elision with the harness's own marker shape."""
    if len(text) <= limit:
        return text
    drop = len(text) - limit
    mark = MARKER % drop
    room = max(limit - len(mark), 200)
    head = int(room * head_frac)
    tail = room - head
    return text[:head] + mark + (text[-tail:] if tail > 0 else "")


def trim_trace(tr: dict, cap_result: int, head_frac: float,
               keep_last_k_thinking: int, cap_args: int) -> dict:
    msgs = tr.get("messages") or []
    asst_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    # keep thinking only on the last k assistant turns; k<0 disables the drop entirely
    keep = set(asst_idx[-keep_last_k_thinking:]) if keep_last_k_thinking > 0 else set()
    if keep_last_k_thinking < 0:
        keep = set(asst_idx)

    out = []
    for i, m in enumerate(msgs):
        m = dict(m)
        if m.get("role") == "assistant":
            if i not in keep:
                m.pop("reasoning_content", None)
                m.pop("reasoning", None)
            if cap_args and m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    tc = json.loads(json.dumps(tc))
                    fn = tc.get("function") or {}
                    a = fn.get("arguments")
                    # arguments is a MAPPING here (bug-364); cap only long string values,
                    # never the key set -- a dropped key is a malformed call, not a
                    # shorter one.
                    if isinstance(a, dict):
                        fn["arguments"] = {
                            k: (cap(v, cap_args, head_frac) if isinstance(v, str) else v)
                            for k, v in a.items()}
                    calls.append(tc)
                m["tool_calls"] = calls
        elif m.get("role") == "tool" and cap_result:
            m["content"] = cap(m.get("content") or "", cap_result, head_frac)
        out.append(m)
    tr = dict(tr)
    tr["messages"] = out
    return tr


def chars(tr) -> int:
    n = 0
    for m in tr.get("messages") or []:
        n += len(m.get("content") or "") + len(m.get("reasoning_content") or "")
        for tc in m.get("tool_calls") or []:
            n += len(json.dumps((tc.get("function") or {}).get("arguments") or {}))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-tool-result", type=int, default=2500,
                    help="0 disables. p50 result is 1077 chars, so 2500 leaves the "
                         "median untouched and only bites the file dumps.")
    ap.add_argument("--cap-tool-args", type=int, default=0,
                    help="0 disables (default). These are MODEL output -- only cap them "
                         "if an episode still does not fit, and say so when reporting.")
    ap.add_argument("--head-frac", type=float, default=0.75)
    ap.add_argument("--keep-last-k-thinking", type=int, default=1,
                    help="Keep reasoning_content on the last K assistant turns. "
                         "-1 keeps all (no drop); 0 drops all.")
    ap.add_argument("--model", default=None,
                    help="Tokenizer dir. If given, report TOKEN deltas, not chars.")
    ap.add_argument("--template", default=None)
    args = ap.parse_args()

    data = json.load(open(args.inp))
    traces = data["traces"]
    trimmed = [trim_trace(t, args.cap_tool_result, args.head_frac,
                          args.keep_last_k_thinking, args.cap_tool_args)
               for t in traces]

    before = [chars(t) for t in traces]
    after = [chars(t) for t in trimmed]
    print("%d traces  %.2f M -> %.2f M chars  (-%.1f%%)"
          % (len(traces), sum(before) / 1e6, sum(after) / 1e6,
             100 * (1 - sum(after) / max(sum(before), 1))))

    if args.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        template = open(args.template).read() if args.template else None
        def ntok(t, preserve):
            ids = tok.apply_chat_template(
                t["messages"], chat_template=template, add_generation_prompt=False,
                preserve_thinking=preserve, enable_thinking=True, tokenize=True)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return len(ids)
        # preserve=False is the SERVED regime; preserve=True is what Tier-B renders with.
        for label, src, pres in (("orig  preserve_thinking=True ", traces, True),
                                 ("orig  preserve_thinking=False", traces, False),
                                 ("trim  preserve_thinking=False", trimmed, False)):
            v = sorted(ntok(t, pres) for t in src)
            n = len(v)
            print("  %s  p50=%7d p90=%7d max=%7d   fits<=16384: %d/%d  <=32768: %d/%d"
                  % (label, v[n // 2], v[int(.9 * n)], v[-1],
                     sum(1 for x in v if x <= 16384), n,
                     sum(1 for x in v if x <= 32768), n))

    json.dump({"traces": trimmed,
               "trim": {"cap_tool_result": args.cap_tool_result,
                        "cap_tool_args": args.cap_tool_args,
                        "head_frac": args.head_frac,
                        "keep_last_k_thinking": args.keep_last_k_thinking,
                        "source": args.inp}}, open(args.out, "w"))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
