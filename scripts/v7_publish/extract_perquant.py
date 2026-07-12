#!/usr/bin/env python3
"""Per-quant HE+/MPE score + completion length (tokens p50/p90/max, chars p50/max)
across all v7 quant eval roots. Emits one record per (model, tier)."""
import json, glob, os

ROOTS = [
    "/srv/ml/eval_results_v7_quant_sweep",
    "/srv/ml/eval_results_cd_recovery",
    "/srv/ml/eval_results_coderx_tiers",
    "/srv/ml/eval_results_qat_investig",
    "/srv/ml/eval_results_v7_qat",
]
# served-name -> (model, tier).  model in {coder,coderx}
def classify(served):
    s = served
    model = None
    if s.startswith("v7coderx-"): model, tier = "coderx", s[len("v7coderx-"):]
    elif s.startswith("v7coder-"): model, tier = "coder", s[len("v7coder-"):]
    elif s.startswith("cx-"):      model, tier = "coderx", s[len("cx-"):]
    elif s.startswith("vancd-"):   model, tier = "coder", "CD-"+s[len("vancd-"):]
    elif s == "qatboth-CD-Q4_K_M": model, tier = "coder", "CD-qat-Q4_K_M"
    elif s == "qatboth-CD-Q2_K":   model, tier = "coder", "CD-qat-Q2_K"
    elif s == "qatboth-Q2_K":      model, tier = "coder", "qat-Q2_K"
    elif s == "qatboth-Q3_K_M":    model, tier = "coder", "qat-Q3_K_M"
    elif "coderx-it-qat-Q4_0" in s and "noshared" not in s: model, tier = "coderx", "qat-Q4_0"
    elif "coder-it-qat-Q4_0" in s and "noshared" not in s:  model, tier = "coder", "qat-Q4_0"
    elif "noshared" in s: model, tier = ("coderx" if "coderx" in s else "coder"), "qat-noshared-Q4_0"
    else: return None, None
    return model, tier

def trip(ts, key, ks=("p50","p90","max")):
    s = (ts or {}).get(key) or {}
    return [s.get(k) for k in ks]

rec = {}  # (model,tier) -> {hep:{score,tok,chr}, mpe:{...}}
for root in ROOTS:
    for bench, short in (("humanevalplus_full","hep"),("multipl_e_100","mpe")):
        for sj in glob.glob(os.path.join(root, bench, "*", "summary.json")):
            served = os.path.basename(os.path.dirname(sj))
            model, tier = classify(served)
            if model is None: continue
            try: d = json.load(open(sj))
            except Exception: continue
            ts = d.get("token_stats") or {}
            key = (model, tier)
            rec.setdefault(key, {})
            rec[key][short] = {
                "score": d.get("score"),
                "tok": trip(ts, "completion_tokens"),
                "chr": trip(ts, "completion_chars", ("p50","max")),
                "served": served, "root": root.replace("/srv/ml/",""),
            }
out = {f"{m}|{t}": v for (m,t),v in sorted(rec.items())}
print(json.dumps(out, indent=1))
