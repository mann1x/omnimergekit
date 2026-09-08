import json, os
C = {"banked_t8":"/srv/ml/eval_results/qwen_suite/lcb_v6_77q/qwenhybridp24_q6k",
     "fresh_t8":"/srv/ml/eval_results_routing/qwen_suite/lcb_v6_77q/armJ_t8_a",
     "fresh_t10":"/srv/ml/eval_results_routing/qwen_suite/lcb_v6_77q/armJ_t10_a"}
P = {}
for n, d in C.items():
    s = json.load(open(os.path.join(d, "summary.json")))
    P[n] = {json.loads(l)["task_id"]: bool(json.loads(l)["passed"])
            for l in open(s["samples_file"]) if l.strip()}
ids = sorted(set(P["banked_t8"]) & set(P["fresh_t8"]) & set(P["fresh_t10"]))
A = {i for i in ids if P["banked_t8"][i] != P["fresh_t8"][i]}   # same-config churn
B = {i for i in ids if P["fresh_t8"][i]  != P["fresh_t10"][i]}  # routing contrast
print("problems paired across all 3 cells : %d" % len(ids))
print("flip under SAME config (repeat)    : %d" % len(A))
print("flip under routing change          : %d" % len(B))
print("flip in BOTH (chronically unstable): %d" % len(A & B))
print("EVER flip (union)                  : %d  (%.0f%% of bench)" % (len(A | B), 100*len(A | B)/len(ids)))
print("STABLE in all three cells          : %d" % (len(ids) - len(A | B)))
st = [i for i in ids if i not in (A | B)]
for n in ("banked_t8", "fresh_t8", "fresh_t10"):
    k = sum(P[n][i] for i in st)
    print("   %-10s on the %d stable problems: %d pass (%.4f)" % (n, len(st), k, k/len(st)))
