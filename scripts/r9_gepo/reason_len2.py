"""Reasoning-channel length across armJ/gepo1/gepo2.

WHY: on thinking benches lm-eval's `resps` holds CONTENT ONLY -- the thinking
channel is stripped into a separate chars-only sidecar (reasoning_log.jsonl,
bug-635). Measuring length off `resps` therefore measures the ANSWER, not the
reasoning, and cannot answer "did brevity training shorten the thinking?".
That is the channel GEPO was optimising, so it is the only one whose movement
means anything here.

Chars, not tokens -- the sidecar stores no token counts (bug-638). Ratios
within a bench are still readable because chars/token is roughly constant
across arms of the SAME family on the SAME bench; the ratio is the claim, the
absolute number is not.
"""
import json
import os

RES = "/srv/ml/eval_results/qwen_suite"
ARMS = [("armJ", "qwenhybridp24_q6k"),
        ("gepo1", "qwena3bgepo1_q6k"),
        ("gepo2", "qwena3bgepo2_q6k")]
BENCHES = ["gpqa_diamond_full", "humaneval_full_think",
           "humanevalplus_full_think", "ifeval_100", "arc_challenge_100",
           "math500_100_qwen", "aime_30_qwen", "gsm8k_100_boxed"]

FIRST = True


def load(bench, cell):
    p = os.path.join(RES, bench, cell, "reasoning_log.jsonl")
    if not os.path.isfile(p):
        return None
    global FIRST
    out = {}
    for line in open(p, errors="ignore"):
        if not line.strip():
            continue
        r = json.loads(line)
        if FIRST:
            print(f"  [sidecar keys] {sorted(r.keys())}")
            FIRST = False
        key = r.get("idx", r.get("doc_id"))
        n = None
        for k in ("reasoning_chars", "think_chars", "reasoning_len", "chars"):
            if k in r and isinstance(r[k], (int, float)):
                n = int(r[k]); break
        if n is None:
            t = r.get("reasoning_content") or r.get("reasoning") or ""
            n = len(t)
        out[key] = n
    return out or None


print("=== reasoning-channel p50 chars (the channel GEPO optimised) ===")
print(f"{'bench':<26} {'n':>4} {'armJ':>9} {'gepo1':>9} {'gepo2':>9} "
      f"{'g1/aJ':>7} {'g2/aJ':>7}")
print("-" * 78)
for b in BENCHES:
    arms = {t: load(b, c) for t, c in ARMS}
    if any(v is None for v in arms.values()):
        print(f"{b:<26} (no sidecar for {[t for t,v in arms.items() if v is None]})")
        continue
    common = sorted(set(arms["armJ"]) & set(arms["gepo1"]) & set(arms["gepo2"]),
                    key=str)
    if not common:
        print(f"{b:<26} (no common keys)")
        continue

    def p50(t):
        v = sorted(arms[t][k] for k in common)
        return v[len(v) // 2]
    a, g1, g2 = p50("armJ"), p50("gepo1"), p50("gepo2")
    print(f"{b:<26} {len(common):>4} {a:>9} {g1:>9} {g2:>9} "
          f"{(g1/a if a else 0):>6.2f}x {(g2/a if a else 0):>6.2f}x")
