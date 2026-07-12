#!/usr/bin/env python3
"""Read the v8 fkbroad-soft2 full 9-bench summaries (+ MPE from tradecheck).
Prints bench, .score, metric, filter, sampler provenance. GPQA .score is already
flexible-extract per omk's summarizer."""
import json
import glob
import os

BASE = "/srv/ml/v8_q6k_full/eval_results_llama_suite/v8_fkbroad_soft2_q6k"
MPE = "/srv/ml/agentic_loop/results/fkbroad_soft2_tradecheck/multipl_e_100/fkbroad-soft2-imatq6/summary.json"

ORDER = ["gpqa_diamond_full", "gsm8k_100", "math500_100", "aime_30",
         "arc_challenge_full", "ifeval_100", "humaneval_full",
         "humanevalplus_full", "lcb_medium_55_v4"]


def readone(path):
    try:
        j = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        return None, "ERR %s" % e
    samp = j.get("sampler")
    sname = samp.get("name") if isinstance(samp, dict) else samp
    return j, "score=%s metric=%s filter=%s sampler=%s" % (
        j.get("score"), j.get("metric"), j.get("filter"), sname)


def main():
    print("== v8 fkbroad-soft2 full 9-bench (Q6_K, llama.cpp, greedy) ==")
    for b in ORDER:
        sp = glob.glob(os.path.join(BASE, b, "*", "summary.json"))
        if not sp:
            print("  %-24s (no summary.json)" % b)
            continue
        _, line = readone(sp[0])
        print("  %-24s %s" % (b, line))
    if os.path.exists(MPE):
        _, line = readone(MPE)
        print("  %-24s %s" % ("multipl_e_100(trade)", line))


if __name__ == "__main__":
    main()
