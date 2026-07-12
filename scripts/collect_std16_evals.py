#!/usr/bin/env python3
# collect_std16_evals.py — robust HE+/MPE collector + HE+ NA diagnostic for the STD16
# (v7-coder force-keep) deploy-sampler quant re-eval. Reads summary.json (.score) only;
# when HE+ score is missing it reports the summary keys + whether results_*.json /
# samples_*.jsonl exist so we can tell a collection-glob miss from a real scorer-None.
import glob
import json
import os

WORK = "/mnt/sdc/ml/std16_gate/he_mpe"
TIERS = "Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q4_K_L Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_L Q3_K_M CD-Q2_K".split()


def all_summ(d):
    return sorted(glob.glob(os.path.join(d, "**", "summary.json"), recursive=True), key=len)


def best_summary(tier, pat):
    ds = glob.glob(os.path.join(WORK, "results", tier, pat))
    chosen = None
    for d in ds:
        for f in all_summ(d):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            if isinstance(j, dict) and j.get("score") is not None:
                return j, f
            if chosen is None:
                chosen = (j, f)
    return chosen if chosen else (None, None)


def pct(j):
    return ("%.2f" % (j["score"] * 100)) if (j and j.get("score") is not None) else "NA"


def prov(j):
    if not j:
        return ""
    m = j.get("metric") or ""
    fl = j.get("filter") or ""
    return m + ("/" + fl if fl else "")


def sampler(j):
    s = (j or {}).get("sampler") or {}
    return s.get("name", "?")


print("TIER\tHE+\tHE+prov\tMPE\tMPEprov\tsampler")
for T in TIERS:
    he, hep = best_summary(T, "humanevalplus_full_minprep*")
    mpe, mpep = best_summary(T, "multipl_e_100_minprep*")
    smp = sampler(he) if he else sampler(mpe)
    print(f"{T}\t{pct(he)}\t{prov(he)}\t{pct(mpe)}\t{prov(mpe)}\t{smp}")

print("\n=== HE+ diagnostics (tiers where HE+ score is NA) ===")
for T in TIERS:
    he, hep = best_summary(T, "humanevalplus_full_minprep*")
    if he is not None and he.get("score") is not None:
        continue
    print(f"[{T}] summary_keys={list(he.keys()) if he else 'NO_SUMMARY_JSON'}  path={hep}")
    for d in glob.glob(os.path.join(WORK, "results", T, "humanevalplus_full_minprep*")):
        rj = glob.glob(os.path.join(d, "**", "results*.json"), recursive=True)
        sj = glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True)
        print(f"    dir={os.path.basename(d)}  results_json={len(rj)}  samples_jsonl={len(sj)}")
        # peek into the first results json for a pass@1 / score hint
        for f in sorted(rj, key=len)[:1]:
            try:
                j = json.load(open(f))
                res = j.get("results", j)
                print(f"      results_top_keys={list(res.keys())[:6] if isinstance(res, dict) else type(res)}")
            except Exception as e:
                print(f"      (results parse fail: {e})")
