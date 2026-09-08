import json, os
ROOT = "/srv/ml/eval_results_routing/qwen_suite"
BANK = "/srv/ml/eval_results/qwen_suite"
for bench in ("ifeval_100", "lcb_v6_77q"):
    print("=== %s" % bench)
    for cell in ("armJ_t8_a", "armJ_t10_a"):
        f = os.path.join(ROOT, bench, cell, "summary.json")
        if not os.path.exists(f):
            print("   %-11s MISSING" % cell); continue
        d = json.load(open(f))
        ts = d.get("token_stats") or {}
        print("   %-11s score=%-8s sampler=%-12s metric=%s filter=%s" % (
            cell, round(d["score"], 4), (d.get("sampler") or {}).get("name"),
            d.get("metric"), d.get("filter")))
        print("               caps=%s" % (d.get("generation_caps"),))
        print("               tok p50=%s p95=%s max=%s  warnings=%s" % (
            ts.get("p50"), ts.get("p95"), ts.get("max"), d.get("sanity_warnings")))
    # banked same-model cell for basis comparison
    fb = os.path.join(BANK, bench, "qwenhybridp24_q6k", "summary.json")
    if os.path.exists(fb):
        db = json.load(open(fb))
        print("   %-11s score=%-8s  (BANKED armJ t8, different run root/date)" % (
            "banked_t8", round(db["score"], 4)))
