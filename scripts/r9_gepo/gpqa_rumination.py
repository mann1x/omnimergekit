"""GPQA correct-vs-wrong output length on the TRUE total (reasoning + content).

Supersedes the content-only version, which saw ~17% of the output and therefore could not
see rumination at all -- rumination lives in the thinking channel.
Join key: reasoning_log idx <-> samples doc_id.
"""
import glob, json
R = "/srv/ml/eval_results/qwen_suite/gpqa_diamond_full"
ARMS = [("armJ", "qwenhybridp24_q6k"), ("gepo1", "qwena3bgepo1_q6k"), ("gepo2", "qwena3bgepo2_q6k")]
FILT = "flexible-extract"


def med(x):
    x = sorted(x)
    return x[len(x) // 2] if x else 0


print("%-7s %22s %22s %9s" % ("arm", "CORRECT total_chars", "WRONG total_chars", "ratio"))
print("%-7s %10s %11s %10s %11s %9s" % ("", "n  p50", "mean", "n  p50", "mean", "W/C mean"))
print("-" * 66)
for tag, name in ARMS:
    ln = {}
    for line in open(R + "/" + name + "/reasoning_log.jsonl"):
        if not line.strip():
            continue
        d = json.loads(line)
        ln[d["idx"]] = (d.get("reasoning_chars") or 0) + (d.get("content_chars") or 0)
    f = sorted(glob.glob(R + "/" + name + "/lm_eval_out/**/samples_*.jsonl", recursive=True))[-1]
    ok, no = [], []
    for line in open(f):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("filter") != FILT:
            continue
        t = ln.get(d["doc_id"])
        if t is None:
            continue
        (ok if d.get("exact_match") else no).append(t)
    print("%-7s %4d %6d %11.0f %4d %6d %11.0f %9.2f"
          % (tag, len(ok), med(ok), sum(ok) / len(ok), len(no), med(no), sum(no) / len(no),
             (sum(no) / len(no)) / (sum(ok) / len(ok))))
