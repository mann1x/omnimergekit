#!/usr/bin/env python3
"""agentic_eog_mass_diag.py — T202.3 completeness diagnostic.

The top-K pin pre-check counts how many of the per-layer top terminator experts
fall in the dropped set. That is a thin proxy. This script measures the *mass*:
of all emit-position routing events on the 128e teacher, what fraction landed on
experts the C6v3lcb prune KEEPS vs DROPS, per layer and overall — plus a
competence-weighted version (wnorm * tc) and the highest-competence dropped
experts. If kept experts capture ~all the emit-position routing mass, the
"prune discarded the stop-capability" RCA is falsified by mass, not just by a
top-K count.

Usage: agentic_eog_mass_diag.py <eog_map.json> <drop_map.json>
"""
import json
import sys

eog = json.load(open(sys.argv[1]))["categories"]["agentic_eog"]
drop = json.load(open(sys.argv[2]))
dropped = {int(k): set(int(x) for x in v) for k, v in drop.items()}

hdr = "{:>3} {:>9} {:>7} {:>7} {:>13} {:>14} {:>11}".format(
    "L", "emitMass", "%kept", "%drop", "compMass%kept", "#dropRouted", "maxDropRMS")
print(hdr)

tot_mass = tot_kept = tot_comp = tot_compk = 0.0
worst = []
for li in range(30):
    rows = eog[str(li)]
    dset = dropped[li]
    mass_total = sum(int(r["tc"]) for r in rows)
    mass_kept = sum(int(r["tc"]) for r in rows if int(r["id"]) not in dset)
    mass_drop = mass_total - mass_kept
    comp_total = sum(float(r.get("wnorm", 0.0)) * int(r["tc"]) for r in rows)
    comp_kept = sum(float(r.get("wnorm", 0.0)) * int(r["tc"])
                    for r in rows if int(r["id"]) not in dset)
    dropped_routed = [(int(r["id"]), float(r.get("wnorm", 0.0)), int(r["tc"]))
                      for r in rows if int(r["id"]) in dset and int(r["tc"]) > 0]
    max_drop_rms = max((w for _, w, _ in dropped_routed), default=0.0)
    pk = 100 * mass_kept / mass_total if mass_total else 0
    pd = 100 * mass_drop / mass_total if mass_total else 0
    ck = 100 * comp_kept / comp_total if comp_total else 0
    print("{:>3} {:>9} {:>6.1f}% {:>6.1f}% {:>12.1f}% {:>14} {:>11.2f}".format(
        li, mass_total, pk, pd, ck, len(dropped_routed), max_drop_rms))
    tot_mass += mass_total
    tot_kept += mass_kept
    tot_comp += comp_total
    tot_compk += comp_kept
    for eid, w, tc in dropped_routed:
        worst.append((w, li, eid, tc))

print()
print("OVERALL emit-routing mass kept: {:.2f}%  dropped: {:.2f}%".format(
    100 * tot_kept / tot_mass, 100 * (tot_mass - tot_kept) / tot_mass))
print("OVERALL competence-weighted mass kept: {:.2f}%".format(
    100 * tot_compk / tot_comp))
print()
print("Top-15 DROPPED experts by emit-RMS (highest terminator competence discarded):")
worst.sort(reverse=True)
for w, li, eid, tc in worst[:15]:
    print("  L{:>2} e{:>3}  emit-RMS={:>7.2f}  tc={}".format(li, eid, w, tc))
