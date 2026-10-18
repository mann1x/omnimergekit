#!/usr/bin/env python3
# Clean summary reader: walk <root>/<bench>/<served>/summary.json and dump
# score + provenance + key token_stats. File-based (no nested-quote-over-ssh).
# Usage: _rd_v8_bench.py <results_root> [<results_root> ...]
import glob
import json
import os
import sys

def fmt(v):
    return "None" if v is None else (("%.4f" % v) if isinstance(v, float) else str(v))

for root in sys.argv[1:]:
    print("##### ROOT:", root)
    summ_paths = sorted(glob.glob(os.path.join(root, "*", "*", "summary.json")))
    if not summ_paths:
        print("  (no summaries)")
        continue
    for summ in summ_paths:
        rel = os.path.relpath(summ, root)
        bench = rel.split(os.sep)[0]
        try:
            d = json.load(open(summ))
        except Exception as e:
            print("  %-26s ERR %s" % (bench, e))
            continue
        sc = d.get("score")
        metric = d.get("metric")
        filt = d.get("filter")
        sampler = d.get("sampler")
        ts = d.get("token_stats") or {}
        fr = ts.get("finish_reasons") or {}
        cc = ts.get("completion_chars") or {}
        ct = ts.get("completion_tokens") or {}
        n = ts.get("n") or d.get("n")
        print("  %-26s score=%-8s metric=%s/%s n=%s" % (bench, fmt(sc), metric, filt, n))
        print("       sampler=%s" % (sampler if sampler else "{}"))
        print("       chars p50=%s p90=%s max=%s | toks p50=%s p90=%s | finish=%s" % (
            fmt(cc.get("p50")), fmt(cc.get("p90")), fmt(cc.get("max")),
            fmt(ct.get("p50")), fmt(ct.get("p90")), fr))
