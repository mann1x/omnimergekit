#!/usr/bin/env python
"""Pair two cold-start probe runs into the divergence diagnostic.

Usage:
    diff_divergence.py A.json B.json [--a-fail 0.25] [--b-clean 0.10]

A = the model under suspicion (v7-coder-Q6_K), B = the control (128e-Q6_K). For
every (prompt, sampler-config) it reports the EXT fail-rate of each model and the
divergence (A - B). A row is flagged DIVERGENT when A fails often (>= --a-fail)
while B stays clean (<= --b-clean) -- those prompts are the v7-only failures, and
their transcript dumps are the evidence for the model fix. Strict (verbatim)
rates are shown too, so you can see what the OLD harness would have reported
(typically 0 on these soft loops).

Reads only the JSON written by `probe.py`; no engine import, runs anywhere.
"""
import argparse
import json


def index(run):
    """{(prompt, config) -> cell} from a probe result JSON."""
    out = {}
    dumps = {}
    for p in run.get("prompts", []):
        dumps[p["name"]] = p.get("dump")
        for c in p.get("configs", []):
            out[(p["name"], c["config"])] = c
    return out, dumps


def main(argv=None):
    ap = argparse.ArgumentParser(description="pair two probe runs into divergence")
    ap.add_argument("a_json", help="model under test (e.g. v7-coder-q6k)")
    ap.add_argument("b_json", help="control (e.g. 128e-q6k)")
    ap.add_argument("--a-fail", type=float, default=0.25,
                    help="A EXT fail-rate at/above this counts as 'A fails'")
    ap.add_argument("--b-clean", type=float, default=0.10,
                    help="B EXT fail-rate at/below this counts as 'B clean'")
    args = ap.parse_args(argv)

    A = json.load(open(args.a_json))
    B = json.load(open(args.b_json))
    ai, adumps = index(A)
    bi, _ = index(B)
    an, bn = A.get("model", "A"), B.get("model", "B")

    rows = []
    for key, ac in ai.items():
        bc = bi.get(key)
        if not bc:
            continue
        prompt, cfg = key
        a_ext, b_ext = ac["fail_rate_ext"], bc["fail_rate_ext"]
        rows.append({
            "prompt": prompt, "config": cfg,
            "a_ext": a_ext, "b_ext": b_ext, "div": a_ext - b_ext,
            "a_strict": ac["fail_rate"], "b_strict": bc["fail_rate"],
            "a_break": (ac["paraphrase"], ac["template"], ac["overthinking"]),
            "seeds": ac["seeds"], "dump": adumps.get(prompt),
        })
    rows.sort(key=lambda r: r["div"], reverse=True)

    print("differential: A=%s  vs  B=%s   (EXT = verbatim OR soft)" % (an, bn))
    print("%-22s %-16s %8s %8s %7s %-14s" %
          ("prompt", "config", "A_ext", "B_ext", "DIV", "A[para/tmpl/over]"))
    print("-" * 88)
    divergent = []
    for r in rows:
        flag = ""
        if r["a_ext"] >= args.a_fail and r["b_ext"] <= args.b_clean:
            flag = "  <== DIVERGENT (v7-only)"
            divergent.append(r)
        print("%-22s %-16s %7.0f%% %7.0f%% %+6.0f%% %-14s%s" % (
            r["prompt"][:22], r["config"][:16], 100 * r["a_ext"],
            100 * r["b_ext"], 100 * r["div"], "%d/%d/%d" % r["a_break"], flag))

    print("\n%d divergent cell(s): %s fails while %s stays clean." %
          (len(divergent), an, bn))
    if divergent:
        print("Inspect transcripts (the diagnostic evidence):")
        seen = set()
        for r in divergent:
            if r["dump"] and r["dump"] not in seen:
                seen.add(r["dump"])
                print("  %-22s %s" % (r["prompt"], r["dump"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
