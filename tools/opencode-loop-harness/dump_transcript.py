#!/usr/bin/env python3
"""dump_transcript.py — write a human-readable conversation.md for a captured session.

The wire log already contains the complete conversation (the LAST request carries the
whole accreted message array: system + every user turn + every assistant turn + every tool
result), but it is raw JSONL. This renders it so a conversation can actually be read and
audited without a JSON parser.

Called automatically at the end of compact_session.py, so EVERY test run saves its
conversation. Also usable standalone:

    python dump_transcript.py --session sessions/<sid>
"""
import argparse
import glob
import json
import os

MAXCHARS = 4000  # per message body; full fidelity stays in the wirelog + raw/


def _load(sdir):
    """Return (final_message_array, [response_records]) from the wire log."""
    last_msgs, resps = None, []
    for fp in sorted(glob.glob(os.path.join(sdir, "wirelog", "session-*.jsonl"))):
        with open(fp, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("dir") == "request":
                    m = (r.get("req") or {}).get("messages")
                    if m:
                        last_msgs = m
                elif r.get("dir") == "response":
                    resps.append(r)
    return last_msgs or [], resps


def _clip(s):
    s = "" if s is None else str(s)
    return s if len(s) <= MAXCHARS else s[:MAXCHARS] + "\n...[clipped %d chars — full text in wirelog]" % (len(s) - MAXCHARS)


def render(sdir):
    msgs, resps = _load(sdir)
    sid = os.path.basename(sdir.rstrip("/"))
    L = ["# Conversation — %s\n" % sid,
         "Rendered from the wire log. The final request carries the whole accreted",
         "conversation, so this is the full exchange as the model saw it.\n",
         "- messages in final request: **%d**" % len(msgs),
         "- model responses captured: **%d**\n" % len(resps)]

    user_n = 0
    for m in msgs:
        role = m.get("role")
        if role == "user":
            user_n += 1
            L.append("\n---\n\n## USER turn %d\n" % user_n)
            L.append(_clip(m.get("content")))
        elif role == "system":
            L.append("\n---\n\n## SYSTEM\n")
            L.append("```\n%s\n```" % _clip(m.get("content")))
        elif role == "assistant":
            tc = m.get("tool_calls") or []
            L.append("\n### assistant")
            c = m.get("content")
            if c:
                L.append("\n%s" % _clip(c))
            for t in tc:
                fn = (t.get("function") or {})
                L.append("\n**tool_call** `%s`\n" % fn.get("name"))
                L.append("```json\n%s\n```" % _clip(fn.get("arguments")))
        elif role == "tool":
            L.append("\n**tool result**\n")
            L.append("```\n%s\n```" % _clip(m.get("content")))

    # The last assistant reply of the last turn is only in the response records
    # (it never becomes part of a subsequent request), so append it explicitly.
    if resps:
        r = resps[-1]
        L.append("\n---\n\n## FINAL model response (rid %s)\n" % r.get("rid"))
        L.append("- finish_reason: `%s`  gen_secs: `%s`  usage: `%s`"
                 % (r.get("finish_reason"), r.get("gen_secs"), r.get("usage")))
        if r.get("reasoning_content"):
            L.append("\n**reasoning**\n\n```\n%s\n```" % _clip(r.get("reasoning_content")))
        if r.get("content"):
            L.append("\n**content**\n\n%s" % _clip(r.get("content")))
        if r.get("tool_calls"):
            L.append("\n**tool_calls**\n\n```json\n%s\n```"
                     % _clip(json.dumps(r.get("tool_calls"), ensure_ascii=False)))

    out = os.path.join(sdir, "conversation.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return out, len(msgs), user_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    a = ap.parse_args()
    out, n, u = render(a.session.rstrip("/"))
    print("[transcript] %s (%d messages, %d user turns)" % (out, n, u))


if __name__ == "__main__":
    main()
