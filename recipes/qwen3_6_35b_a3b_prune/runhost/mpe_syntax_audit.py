#!/usr/bin/env python
"""WHY does armI emit unparseable code? Classify every MultiPL-E SyntaxError.

armI's whole MPE regression is +12 SyntaxError in rs and +12 in js, with AssertionError flat.
So the model is not solving worse -- it is EMITTING worse. MPE runs raw-completion mode with
stop_tokens ["\\n}", "<file_sep>"], so the candidate mechanisms are distinguishable from the
completion text itself:

  cap_hit      -- ran to the generation ceiling, cut mid-code
  control_tok  -- leaked a chat/control token into a raw-completion stream (the v7 EOG class)
  fence        -- emitted markdown ``` inside what must be bare code
  prose        -- natural-language lead-in before any code
  empty        -- nothing usable
  brace_imbal  -- terminator/brace accounting wrong (stop-token interaction)
  other        -- unclassified; printed verbatim so it cannot hide

Compares armI against armD on the SAME problems so the EXTRA failures are isolated, not just
the total. A pattern present in both arms is the bench; a pattern only armI has is the recipe.
"""
import collections
import glob
import json
import os
import re
import sys

R = "/srv/ml/eval_results/ream_arms/multipl_e_100"
LANGS = {"humaneval-rs": "rs", "humaneval-java": "java", "humaneval-js": "js"}

# control / chat tokens that must NEVER appear in a raw code completion
CTRL = [
    "<turn|", "|turn>", "<|", "|>", "<end_of_turn>", "<start_of_turn>",
    "<think>", "</think>", "<file_sep>", "<|endoftext|>", "<eos>", "<bos>",
    "<｜", "＜", "<|im_start|>", "<|im_end|>",
]
PROSE = re.compile(r"^\s*(here|this|the|to |we |i |sure|okay|let'?s|note|first|"
                   r"below|explanation|solution:)", re.I)


def samples(arm):
    p = os.path.join(R, arm, "mpe_result.samples.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            tid = d.get("task_id") or d.get("doc_id")
            if tid:
                out[tid] = d.get("completion") or ""
    return out


def statuses(arm):
    """-> {(lang, problem): status}"""
    out = {}
    for ldir, lang in LANGS.items():
        for p in glob.glob(os.path.join(R, arm, "results", ldir, "*.results.json")):
            if p.endswith("_summary.json"):
                continue
            try:
                d = json.load(open(p))
            except Exception:
                continue
            res = d.get("results") or []
            if not res:
                continue
            out[(lang, d.get("name"))] = res[0].get("status", "?")
    return out


def classify(c, maxlen):
    if not c or not c.strip():
        return "empty"
    hits = [t for t in CTRL if t in c]
    if hits:
        return "control_tok:" + hits[0]
    if "```" in c:
        return "fence"
    if len(c) >= maxlen * 0.98:
        return "cap_hit"
    if PROSE.match(c):
        return "prose"
    if c.count("{") != c.count("}") or c.count("(") != c.count(")"):
        return "brace_imbal"
    return "other"


def main():
    arms = sys.argv[1:] or ["hybrid_p12_ourssal_reapfloor", "reamD_ourssal_nomerge"]
    S, ST = {}, {}
    for a in arms:
        S[a], ST[a] = samples(a), statuses(a)
        if S[a]:
            mx = max(len(v) for v in S[a].values())
            print("%-34s samples=%d  max_completion_chars=%d" % (a, len(S[a]), mx))
    maxlen = max((max((len(v) for v in S[a].values()), default=0) for a in arms), default=1)

    for a in arms:
        print("\n===== %s : SyntaxError breakdown" % a)
        for lang in ("rs", "java", "js"):
            bad = [(l, n) for (l, n), st in ST[a].items()
                   if l == lang and st == "SyntaxError"]
            c = collections.Counter()
            for l, n in bad:
                comp = S[a].get("%s::%s" % (l, n), "")
                c[classify(comp, maxlen)] += 1
            print("  %-5s n_syntaxerr=%-3d %s" % (lang, len(bad), dict(c.most_common())))

    if len(arms) == 2:
        A, B = arms
        print("\n===== EXTRA failures in %s vs %s (same problem, was OK, now not)" % (A, B))
        for lang in ("rs", "java", "js"):
            extra = [n for (l, n), st in ST[A].items()
                     if l == lang and st != "OK" and ST[B].get((l, n)) == "OK"]
            fixed = [n for (l, n), st in ST[B].items()
                     if l == lang and st != "OK" and ST[A].get((l, n)) == "OK"]
            c = collections.Counter()
            for n in extra:
                c[classify(S[A].get("%s::%s" % (lang, n), ""), maxlen)] += 1
            print("  %-5s newly-broken=%-3d newly-fixed=%-3d  %s"
                  % (lang, len(extra), len(fixed), dict(c.most_common())))
            for n in extra[:3]:
                comp = S[A].get("%s::%s" % (lang, n), "")
                print("      -- %s [%s]" % (n, ST[A][(lang, n)]))
                print("         %r" % comp[:260])


if __name__ == "__main__":
    main()
