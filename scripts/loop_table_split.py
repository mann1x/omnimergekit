"""T223 loop table with LOOPS and RUNAWAYS split.

`fails = loop OR runaway` conflates two different failures: a genuine repetition and a
generation that merely hit the token cap (finish=None). STD27 for example is fails=1/2 with
loops=0/0 -- entirely runaways -- which the conflated column reads as 'worse than STD16'.
"""
import json, os, sys

AL = "/srv/ml/agentic_loop/results"
ROWS = [
    ("CTRL",      "fkbroad: 30 base pins (NOT zero)", "t223_CTRL_Q40_minp48.json"),
    ("STD8",      "fkbroad +8 pins (38 total)",    "t223_STD8_Q40_minp48.json"),
    ("STD16",     "fkbroad +16 (46 tot) = PUBLISHED", "t223_STD16_Q40_minp48.json"),
    ("STD27",     "fkbroad +27 (57 total)",        "t223_STD27_Q40_minp48.json"),
    ("ADD16",     "STD16 + DERN",                "t223_ADD16_Q40_minp48.json"),
    ("V8",        "DERN, no force-keep",         "t223_V8_Q40_minp48.json"),
    ("V9AGENTIC", "agentic map; t106 fk 120 req/2 real", "t223_V9AGENTIC_Q40_minp48.json"),
    ("V9AGENTIC", "same map, v9 sampler",        "t223_V9AGENTIC_v9samp_minp48.json"),
]

def cell(r):
    """Three-way partition. loops and runaways OVERLAP -- a run that repeats until it
    hits the cap is counted in BOTH of the harness's counters, so `loops` and `runaways`
    are not a partition of `fails` and must not be printed as two columns. Measured:
    CTRL t0.9 has 31 loops + 30 runaways but only 33 fails -- 28 runs are both."""
    runs = r.get("runs", [])
    lo = sum(1 for x in runs if x.get("is_loop") and not x.get("runaway"))
    bo = sum(1 for x in runs if x.get("is_loop") and x.get("runaway"))
    ro = sum(1 for x in runs if x.get("runaway") and not x.get("is_loop"))
    return "%2d/%2d/%2d" % (lo, bo, ro)

print("no-imat Q4_0, 48 seeds (1000-1047), b9700, solar_build_start")
print("cells show  LOOP-ONLY / BOTH / RUNAWAY-ONLY (the conflated 'fails' hides the overlap)\nThe fk ladder is STRICTLY NESTED: fkbroad 30 pins < fk8 38 < fk16 46 < fk27 57.\nSTD8/16/27 name pins ADDED over fkbroad, not total. CTRL is NOT a zero-force-keep\ncontrol -- it already carries 30 pins; the true no-force-keep map is C6v3lcb/ml1.\nAll pins are id50 (per-step stop); overlap with the v9 t106 set is 0 for all four.\n")
hdr = "%-11s %-38s %12s %12s" % ("cell", "recipe", "t0.9 L/B/R", "t0.8 L/B/R")
print(hdr); print("-" * len(hdr))
for nm, desc, fn in ROWS:
    p = os.path.join(AL, fn)
    if not os.path.exists(p):
        print("%-11s %-38s %12s" % (nm, desc, "(pending)")); continue
    d = json.load(open(p))
    got = {}
    for r in d.get("results", []):
        c = str(r.get("config", ""))
        key = "t0.9" if "0.9" in c else ("t0.8" if "0.8" in c else c)
        got[key] = cell(r)
    print("%-11s %-38s %12s %12s" % (nm, desc, got.get("t0.9", "-"), got.get("t0.8", "-")))
    for k, v in got.items():
        if k not in ("t0.9", "t0.8"):
            print("%-11s %-38s   [%s] %s" % ("", "", k, v))
print("\nL/B/R = loop-only / BOTH / runaway-only.  L+B+R = fails exactly.")
print("  loop    = detect_loop fired on the thinking or answer channel")
print("  runaway = finish_reason not in (stop, tool_calls), i.e. hit the 16384 cap")
print("  BOTH dominates every bad cell: a runaway is usually a loop that ran to the cap,")
print("  NOT an independent failure. STD27 is the exception -- its failures are")
print("  runaway-only (0 loops), a different failure shape from CTRL's repetition.")
print("COLUMN CAVEAT: the t0.8 column is NOT one sampler. Every row but the last used\n  min_p 0.05 / repeat_penalty 1.1; the v9-sampler row used min_p 0.08 /\n  repeat_penalty 1.12 / repeat_last_n 96. Do not read that column straight down.")
print("NOTE: harness reliability is disputed by the owner — read as comparison, not verdict.")
