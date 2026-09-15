"""Split the SWE-bench Tier-B pool into disjoint 1-per-language q-sets.

DESIGN (owner's convergence plan): build set A = 1 trace/language, set B = 1 DIFFERENT
trace/language, map each, and compare. If the experts barely move, 2q/language is enough;
if identical, 1q; if they diverge, keep adding sets until the movement flattens.

Selection must be deterministic and must not bias a set toward big or small traces --
otherwise "the map moved" could just mean "set B had more tokens". A plain round-robin (largest
always to set A) is NOT good enough -- it gave A 320k tokens against B's 186k, a 1.7x
imbalance that would let "the map moved" mean nothing more than "A saw more tokens". So
each language's traces are assigned largest-first to whichever set is currently LIGHTEST,
which balances the per-set token budget across languages.

rust has the fewest traces and therefore caps the number of FULL sets. Sets beyond that
are still emitted but are short a language, which the owner has accepted ("if we miss 1
rust for 5q, it's not an issue") -- it is printed loudly rather than hidden.
"""
import json
import sys
from collections import defaultdict

POOL = "/mnt/sdc/agentbench/tierb/tierb_swebench_pool.json"
OUTDIR = "/mnt/sdc/agentbench/tierb"
NSETS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
# Sets already built keep their traces; the remainder is dealt into the new sets. This
# exists so extending the experiment (1q -> 2q) does not invalidate maps already computed:
# re-dealing from scratch would reshuffle A and B and waste their GPU time.
EXCLUDE = set()
START = 0
# PER_LANG > 1 builds Nq sets (N traces per language per set). The convergence curve
# needs a 3q point, and only the languages with >= 2*3 PASS traces can supply one, so
# --langs restricts the basis explicitly rather than silently emitting short sets.
PER_LANG = 1
ONLY = None
for a in sys.argv[2:]:
    if a.startswith("--exclude="):
        EXCLUDE |= {l.strip() for l in open(a.split("=", 1)[1]) if l.strip()}
    elif a.startswith("--start-letter="):
        START = ord(a.split("=", 1)[1]) - ord("A")
    elif a.startswith("--per-lang="):
        PER_LANG = int(a.split("=", 1)[1])
    elif a.startswith("--langs="):
        ONLY = {x.strip() for x in a.split("=", 1)[1].split(",") if x.strip()}

d = json.load(open(POOL))
meta, traces = d["metadata"], d["traces"]

bylang = defaultdict(list)
n_excl = 0
for t in traces:
    if t["task_id"] in EXCLUDE:
        n_excl += 1
        continue
    if ONLY is not None and t["bench"] not in ONLY:
        continue
    bylang[t["bench"]].append(t)
if EXCLUDE:
    print("excluded %d already-assigned traces" % n_excl)
for lang in bylang:
    bylang[lang].sort(key=lambda t: (-t["est_tokens"], t["task_id"]))

langs = sorted(bylang)
if ONLY is not None:
    missing_req = sorted(ONLY - set(langs))
    if missing_req:
        raise SystemExit("FATAL: --langs asked for %s, which have no traces left" % missing_req)
cap = min(len(bylang[l]) // PER_LANG for l in langs)
print("languages: %d   traces/language: %s"
      % (len(langs), {l: len(bylang[l]) for l in langs}))
print("q per set (--per-lang): %d" % PER_LANG)
print("full sets possible (limited by the scarcest language): %d" % cap)
if NSETS > cap:
    print("NOTE: %d sets requested but only %d can be FULL — later sets will be short."
          % (NSETS, cap))
print()

# Balancing draft: languages with the widest spread are placed first (they have the
# most power to unbalance), and within a language the largest trace goes to whichever
# set is lightest so far.
sets = [[] for _ in range(NSETS)]
load = [0] * NSETS
NEED = PER_LANG * NSETS
order = sorted(langs, key=lambda l: -(bylang[l][0]["est_tokens"]
                                      - bylang[l][min(NEED, len(bylang[l])) - 1]["est_tokens"]))
for lang in order:
    # A language's traces MUST go to DISTINCT sets — the whole design is "1 per
    # language per set". A plain "assign to the lightest set" loop can put both of a
    # language's traces in the same set (observed: set C got 2x go and 0x ruby), which
    # silently destroys the invariant. So pair the largest remaining trace with the
    # lightest remaining set, each set used at most once per language.
    cnt = [0] * NSETS
    for t in bylang[lang][:NEED]:
        avail = [j for j in range(NSETS) if cnt[j] < PER_LANG]
        if not avail:
            break
        i = min(avail, key=lambda j: (load[j], j))
        sets[i].append(t)
        load[i] += t["est_tokens"]
        cnt[i] += 1
for S in sets:
    S.sort(key=lambda t: t["bench"])

spread = (max(load) - min(load)) / max(max(load), 1)
print("token balance across sets: %s   spread=%.1f%%"
      % ([f"{x:,}" for x in load], 100 * spread))
if spread > 0.15:
    print("WARNING: >15%% imbalance — a map difference could be a token-budget artifact.")
print()

for i, S in enumerate(sets):
    tag = chr(ord("A") + START + i)
    out = {"metadata": dict(meta), "traces": S}
    out["metadata"]["qset"] = tag
    out["metadata"]["trace_count"] = len(S)
    out["metadata"]["set_counts"] = {"C": len(S)}
    bc = defaultdict(int)
    for t in S:
        bc[t["bench"]] += 1
    out["metadata"]["bench_counts"] = dict(bc)
    out["metadata"]["per_lang"] = PER_LANG
    p = "%s/qset_%s.json" % (OUTDIR, tag)
    with open(p, "w") as fh:
        json.dump(out, fh)
    missing = sorted(set(langs) - {t["bench"] for t in S})
    print("set %s: %d traces, %s est tokens%s"
          % (tag, len(S), f"{sum(t['est_tokens'] for t in S):,}",
             "   SHORT: " + ", ".join(missing) if missing else ""))
    for t in S:
        print("    %-16s %-34s %7s tok  capped=%s  arm=%s"
              % (t["bench"], t["task_id"][:34], f"{t['est_tokens']:,}",
                 t["budget_capped"], t["arm"]))
    print("    -> %s" % p)
    print()

for i, S in enumerate(sets):
    c = defaultdict(int)
    for t in S:
        c[t["bench"]] += 1
    bad = {k: v for k, v in c.items() if v != PER_LANG}
    if bad:
        raise SystemExit("FATAL: set %s has %s but every language must appear exactly "
                         "%d time(s) — otherwise it is not a %dq set and the comparison "
                         "basis is uneven." % (chr(ord("A") + START + i), bad, PER_LANG, PER_LANG))
print("exactly-%d-per-language invariant: PASS" % PER_LANG)

allsel = [t["task_id"] for S in sets for t in S]
assert len(allsel) == len(set(allsel)), "FATAL: a trace was dealt into two sets"
print("disjointness: PASS — %d traces dealt, none repeated" % len(allsel))
