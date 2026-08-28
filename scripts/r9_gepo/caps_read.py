import json, os
ROOT = "/srv/ml/eval_results/ream_arms/multipl_e_100"
CELLS = [("armJ pre-b604 (VOID)", "hybrid_p24_ourssal_reapfloor"),
         ("armJ post-b604",       "hybrid_p24_ourssal_reapfloor_b604"),
         ("gepo1 post-b604",      "a3b_gepo1")]
for tag, cell in CELLS:
    p = os.path.join(ROOT, cell, "summary.json")
    d = json.load(open(p))
    g = d.get("generation_caps") or {}
    t = d.get("token_stats") or {}
    c = t.get("completion_tokens") or {}
    print("=== %s   score=%s" % (tag, d.get("score")))
    print("    n=%s  p50=%s p90=%s max_completion=%s" % (
        t.get("n"), c.get("p50"), c.get("p90"), g.get("max_completion_tokens")))
    for k in ("status", "reasoning_regime", "max_gen_toks", "thinking_token_budget",
              "answer_allowance", "ceiling_hit", "tolerance_tokens", "ties_at_max",
              "capped_total", "capped_pct", "capped_correct", "capped_scored",
              "verdict", "score_upper_bound_note"):
        if k in g:
            print("    %-24s %s" % (k, g[k]))
    print()
