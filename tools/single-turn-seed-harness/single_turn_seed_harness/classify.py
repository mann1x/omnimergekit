"""Response classifier -- COPIED VERBATIM from
`/mnt/sdc/ml/opencode_capture/hammer_raw.py` (bs2) so the single-turn seed
sweep grades responses with byte-identical logic to the multi-turn wire hammer.

Verdict precedence (highest first):
    CORRUPT       -- HTTP error, PEG/parse failure, or a special-token leak
                     (`<|channel`, `<tool_call|>`, `<unused..>`, ...) in the
                     content OR reasoning channel.
    RUNAWAY       -- finish_reason == "length": the generation hit the token
                     cap and never terminated.
    THINK_EXPLODE -- finished, but reasoning OR content exceeded 20,000 chars
                     in a single turn (a within-turn ruminating loop).
    ABORT         -- finish_reason is None (server aborted / empty).
    CLEAN         -- everything else.

Do NOT "improve" this file -- it must stay identical to hammer_raw.classify so
the single-turn and multi-turn numbers are directly comparable.
"""
import json
import re

CORRUPT = [r"<\|channel", r"<\|\"\|>", r"<tool_call\|>", r"<\|tool_response",
           r"<\|tool_call", r"<unused\d", r"<\|message", r"<\|constrain",
           r"<end_of_turn\|>", r"<start_of_turn\|>", r"\}<tool_call"]
PARSE_FAIL = ["unparsed", "not in expected format", "peg-gemma4"]


def classify(status, raw, max_tokens):
    flags = []
    try:
        obj = json.loads(raw)
    except Exception:
        return "BADJSON", ["non-json body: %r" % raw[:120]], {}
    if status != 200:
        flags.append("HTTP_%s" % status)
    if any(m in raw.lower() for m in PARSE_FAIL):
        flags.append("PARSE_FAIL")
    fin = c = r = None
    ntools = 0
    for ch in obj.get("choices", []):
        fin = ch.get("finish_reason")
        msg = ch.get("message") or {}
        cont = msg.get("content") or ""
        reas = msg.get("reasoning_content") or ""
        ntools = len(msg.get("tool_calls") or [])
        c, r = len(cont), len(reas)
        for blob, lab in ((cont, "content"), (reas, "reasoning")):
            for pat in CORRUPT:
                m = re.search(pat, blob)
                if m:
                    flags.append("SPECIAL_TOKEN[%s]:%r" % (lab, blob[max(0, m.start()-20):m.start()+30].replace("\n", " ")))
                    break
    usage = obj.get("usage") or {}
    ct = usage.get("completion_tokens")
    # verdict precedence
    corrupt = any(f.startswith(("HTTP_", "PARSE_FAIL", "SPECIAL_TOKEN")) for f in flags)
    if corrupt:
        v = "CORRUPT"
    elif fin == "length":
        v = "RUNAWAY"          # hit the token cap = non-terminating generation
    elif (r or 0) > 20000 or (c or 0) > 20000:
        v = "THINK_EXPLODE"    # finished but emitted a huge ruminating block
    elif fin is None:
        v = "ABORT"
    else:
        v = "CLEAN"
    return v, flags, {"fin": fin, "c": c, "r": r, "ct": ct, "tools": ntools}


# verdicts that count against the model in the fail-rate
FAIL_VERDICTS = ("RUNAWAY", "THINK_EXPLODE", "CORRUPT")
# every verdict classify() can emit, in a stable report order
ALL_VERDICTS = ("CLEAN", "RUNAWAY", "THINK_EXPLODE", "CORRUPT", "ABORT", "BADJSON")
