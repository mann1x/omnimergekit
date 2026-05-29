#!/usr/bin/env python3
"""
Comprehensive analysis of track results across the cohort.
Reports: score, length distributions, rumination/looping/under-thinking/over-thinking signals.
Author: claude opus 4.7  2026-05-29
"""
import json, sys, os, re
from pathlib import Path
from collections import defaultdict, Counter
from statistics import median, quantiles

BM = Path("/srv/ml")
COHORT = [
    # (display_name, served_name, base_results_dir)
    ("A2 base",              "a2-62e-fc15_25-p8-s1_0p1_20",          "eval_results_a2_21q_validation"),
    ("A2_EAC (wiki+calib)",  "a2eac-62e-fc15_25-p8-s1_0p1_20",       "eval_results_a2_eac_21q_validation"),
    ("A2_KDONLY (code-bias)","a2kdonly-62e-fc15_25-p8-s1_0p1_20",    "eval_results_a2_kdonly_21q_validation"),
    ("A2_RKD (EAC+KD)",      "a2rkd-62e-fc15_25-p8-s1_0p1_20",       "eval_results_a2_rkd_21q_validation"),
    ("pes1_10",              "pes1_10-62e-fc15_25-p8",               "eval_results_pes1_10_21q_validation"),
]
# Then the 9 from corpora_21q dir
for c in ("calibonly","9bench","ifheavy"):
    for m in ("eac","rkd","kdonly"):
        COHORT.append((f"A2_{m.upper()}_{c}", f"a2{m}_{c}-62e-fc15_25-p8-s1_0p1_20", "eval_results_corpora_21q"))

BENCHES = {
    "humanevalplus_rum3":  {"budget": 16384, "n": 3},
    "ifeval_rum3":         {"budget": 4096,  "n": 3},
    "multipl_e_rum15":     {"budget": 4096,  "n": 15},
    "humanevalplus_full":  {"budget": 16384, "n": 164},
    "ifeval_100":          {"budget": 4096,  "n": 100},
    "multipl_e_100":       {"budget": 4096,  "n": 100},
}

def find_samples(served, bench, base_dir):
    """Find samples_*.jsonl for this (served, bench) combo, looking in 21q OR tracks_2_3 dirs."""
    candidates = [
        BM / base_dir / bench / served / "lm_eval_out" / served,
        BM / "eval_results_tracks_2_3" / bench / served / "lm_eval_out" / served,
        BM / "eval_results_corpora_21q" / bench / served / "lm_eval_out" / served,
    ]
    for c in candidates:
        if c.exists():
            files = sorted(c.glob("samples_*.jsonl"), reverse=True)
            if files:
                return files[0]
    return None

def find_summary(served, bench, base_dir):
    candidates = [
        BM / base_dir / bench / served / "summary.json",
        BM / "eval_results_tracks_2_3" / bench / served / "summary.json",
        BM / "eval_results_corpora_21q" / bench / served / "summary.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def detect_repetition(text, window=30, min_repeats=4):
    """Return True if any window-char substring appears ≥ min_repeats times."""
    if len(text) < window * min_repeats: return False
    counts = Counter()
    for i in range(0, len(text) - window + 1, max(1, window // 3)):
        counts[text[i:i+window]] += 1
    return any(v >= min_repeats for v in counts.values())

def pcts(values):
    if not values: return (0, 0, 0, 0)
    if len(values) == 1: return (values[0],)*4
    qs = sorted(values)
    n = len(qs)
    return (qs[max(0, n*1//10 - 1)], qs[n//2], qs[min(n-1, n*9//10)], qs[-1])

def analyze_bench(served, bench, base_dir):
    """Return dict of metrics for this (served, bench)."""
    summary_p = find_summary(served, bench, base_dir)
    samples_p = find_samples(served, bench, base_dir)
    out = {"bench": bench, "have_summary": False, "have_samples": False}
    if summary_p:
        s = json.loads(summary_p.read_text())
        out["have_summary"] = True
        out["score"] = s.get("score")
        out["metric"] = s.get("metric", "?")
        ct = s.get("token_stats", {}).get("completion_tokens", {})
        out["compl_tok_p10"] = ct.get("p10")
        out["compl_tok_p50"] = ct.get("p50")
        out["compl_tok_p90"] = ct.get("p90")
        out["compl_tok_max"] = ct.get("max")
        # Thinking estimate (may be zero if reasoning stripped)
        tt = s.get("token_stats", {}).get("thinking_tokens_est", {})
        out["thinking_ratio"] = tt.get("ratio_of_completion", 0.0)
    if samples_p:
        out["have_samples"] = True
        chars = []
        n_loop = 0
        n_empty = 0
        n_short = 0
        budget = BENCHES[bench]["budget"]
        n_sat = 0
        for line in samples_p.open():
            try:
                d = json.loads(line)
            except: continue
            r = d.get("resps", [[]])[0]
            if isinstance(r, list) and r: r = r[0]
            if not isinstance(r, str): continue
            chars.append(len(r))
            if len(r) == 0: n_empty += 1
            if 0 < len(r) < 50: n_short += 1
            if detect_repetition(r): n_loop += 1
            # Check saturation: if completion_tokens for this sample is near budget
        out["n_samples"] = len(chars)
        if chars:
            p10, p50, p90, mx = pcts(chars)
            out["chars_p10"] = p10
            out["chars_p50"] = p50
            out["chars_p90"] = p90
            out["chars_max"] = mx
            out["n_loop"] = n_loop
            out["n_empty"] = n_empty
            out["n_short"] = n_short
            # Saturation = compl_tok_max near budget
            if out.get("compl_tok_max", 0) >= 0.95 * budget:
                out["saturation"] = True
            else:
                out["saturation"] = False
    return out


def find_reasoning_log(served, bench, base_dir):
    """Find reasoning_log.jsonl written by patched openai_completions.py."""
    for d in [BM / base_dir / bench / served,
              BM / "eval_results_tracks_2_3" / bench / served,
              BM / "eval_results_corpora_21q" / bench / served]:
        rl = d / "reasoning_log.jsonl"
        if rl.exists():
            return rl
    return None

def parse_reasoning_log(rl_path):
    # REASONING_LOG_PARSE
    """Returns dict of stats from reasoning_log.jsonl."""
    if rl_path is None or not rl_path.exists():
        return None
    content_chars = []
    reasoning_chars = []
    with rl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                content_chars.append(d.get("content_chars", 0))
                reasoning_chars.append(d.get("reasoning_chars", 0))
            except: pass
    if not content_chars: return None
    return {
        "n": len(content_chars),
        "content_p50": sorted(content_chars)[len(content_chars)//2],
        "reasoning_p50": sorted(reasoning_chars)[len(reasoning_chars)//2],
        "reasoning_p90": sorted(reasoning_chars)[min(len(reasoning_chars)-1, len(reasoning_chars)*9//10)],
        "reasoning_max": max(reasoning_chars),
        "reasoning_sum": sum(reasoning_chars),
        "content_sum": sum(content_chars),
        "thinking_ratio": sum(reasoning_chars) / max(1, sum(content_chars) + sum(reasoning_chars)),
        "n_overthink": sum(1 for r in reasoning_chars if r > 30000),  # >7.5k tok thinking
        "n_underthink": sum(1 for r,c in zip(reasoning_chars, content_chars) if r < 100 and c > 100),  # short reason, long content
    }

def main():
    print(f"# Comprehensive cohort report")
    print(f"_Generated: {os.popen('date -Iseconds').read().strip()}_")
    print()
    print("## Legend")
    print("- **score**: omk canonical score (.score in summary.json)")
    print("- **C-tok**: completion tokens p50/p90/max (max-budget saturation > 0.95 indicates rumination/over-thinking)")
    print("- **chars**: response content char length distribution (post reasoning-strip)")
    print("- **loop%**: fraction of samples with ≥4 repetitions of a 30-char window (signal for looping)")
    print("- **empty%**: fraction with zero-length content (under-thinking or generation failure)")
    print("- **short%**: fraction with content<50 chars (under-thinking)")
    print("- **sat**: did ANY sample hit ≥95% of token budget? (rumination ceiling)")
    print()
    by_model = defaultdict(dict)
    for disp, served, base_dir in COHORT:
        for bench in BENCHES:
            a = analyze_bench(served, bench, base_dir)
            if a.get("have_summary") or a.get("have_samples"):
                by_model[disp][bench] = a
    # served+base_dir maps for reasoning log lookup
    served_from_disp = {disp: served for disp, served, _ in COHORT}
    base_dir_from_disp = {disp: base_dir for disp, _, base_dir in COHORT}
    for bench in BENCHES:
        print(f"## {bench}")
        rows = []
        for disp in by_model:
            d = by_model[disp].get(bench)
            if d is None: continue
            score = d.get("score", "-")
            score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
            ct50 = d.get("compl_tok_p50", "-")
            ct90 = d.get("compl_tok_p90", "-")
            ctmx = d.get("compl_tok_max", "-")
            sat = "YES" if d.get("saturation") else "no"
            n = d.get("n_samples", 0)
            n_loop = d.get("n_loop", 0)
            n_empty = d.get("n_empty", 0)
            n_short = d.get("n_short", 0)
            loop_pct = f"{100*n_loop/n:.0f}%" if n else "-"
            empty_pct = f"{100*n_empty/n:.0f}%" if n else "-"
            short_pct = f"{100*n_short/n:.0f}%" if n else "-"
            chars50 = d.get("chars_p50", "-")
            chars90 = d.get("chars_p90", "-")
            rl = find_reasoning_log(served_from_disp.get(disp, ""), bench, base_dir_from_disp.get(disp, ""))
            rstats = parse_reasoning_log(rl)
            think_p50 = str(rstats["reasoning_p50"]) if rstats else "-"
            think_pct = f"{100*rstats[thinking_ratio]:.0f}%" if rstats else "-"
            rows.append([disp, score_s, str(ct50), str(ct90), str(ctmx), sat, str(chars50), str(chars90), loop_pct, empty_pct, short_pct, think_p50, think_pct])
        if not rows: print("_no data yet_"); print(); continue
        widths = [max(len(str(r[i])) for r in rows + [["model","score","C-tok-p50","C-tok-p90","C-tok-max","sat","char-p50","char-p90","loop%","empty%","short%","think-p50","think%"]]) for i in range(13)]
        hdr = ["model","score","C-tok-p50","C-tok-p90","C-tok-max","sat","char-p50","char-p90","loop%","empty%","short%","think-p50","think%"]
        print("| " + " | ".join(h.ljust(widths[i]) for i,h in enumerate(hdr)) + " |")
        print("|" + "|".join("-"*(widths[i]+2) for i in range(13)) + "|")
        for r in rows:
            print("| " + " | ".join(str(c).ljust(widths[i]) for i,c in enumerate(r)) + " |")
        print()

if __name__ == "__main__":
    main()
