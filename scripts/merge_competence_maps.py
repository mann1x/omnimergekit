"""Additively merge competence maps built from disjoint q-sets.

WHY THIS IS SOUND. expert_neuron_analysis_v5_targeted.py's serialize() keeps the
RAW accumulators per (category, layer, expert): wnsq, rnsq, wsum, tc, cc and the
per-neuron activation vector. Those are all sums over tokens, so profiling
set A then set B is arithmetically identical to profiling A+B in one pass --
PROVIDED the sets are disjoint (guaranteed upstream by tierb_split_qsets.py's
disjointness gate).

WHAT IS NOT ADDITIVE. wnorm/rnorm are the intensive per-token RMS sqrt(wnsq/tc)
written by _finalize_rms at save time. Summing them would be meaningless -- an
RMS is not a sum. This script therefore ignores the stored wnorm/rnorm entirely
and RECOMPUTES them from the merged wnsq/tc, exactly as _finalize_rms does.

THE DOUBLE-COUNT TRAP. The generic_* categories are not measured per q-set; they
are IMPORTED unchanged from the same Tier-A source file by every run. Summing
them would double tc, cc, wnsq and neuron_act. RMS happens to survive that
(sqrt(2*wnsq / 2*tc) == sqrt(wnsq/tc)), which is exactly why the bug would be
invisible -- but cc, tc and neuron_act would be silently wrong, and any scorer
reading a count rather than the RMS would inherit it. So identical categories are
CARRIED ONCE, and that decision is asserted, reported, and recorded in metadata.
"""
import argparse
import json
import math
import sys

ADDITIVE = ("wnsq", "rnsq", "wsum", "tc", "cc")
KEY = ("tc", "cc", "wnsq", "rnsq", "wsum")


def same(x, y):
    """NaN-aware equality. The Tier-A source carries NaN in `wsum` (152 cells,
    mostly layer 11), and `nan != nan`, so a naive == makes two byte-identical
    imported categories look different -- which would fire the double-count gate
    on a correct merge. NaN in the same slot on both sides is not a difference."""
    if isinstance(x, float) and isinstance(y, float) and x != x and y != y:
        return True
    return x == y


def rows_identical(a, b):
    if len(a) != len(b):
        return False
    return all(all(same(ra.get(k), rb.get(k)) for k in KEY) for ra, rb in zip(a, b))


def cat_identical(a, b):
    if set(a) != set(b):
        return False
    return all(rows_identical(a[li], b[li]) for li in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--allow-generic-sum", action="store_true",
                    help="sum generic_* even when the cells are not identical "
                         "(default: abort, because a differing generic_* means "
                         "the maps did not share a Tier-A source)")
    args = ap.parse_args()
    if len(args.maps) < 2:
        sys.exit("need at least two maps to merge")

    print("loading %d maps..." % len(args.maps), flush=True)
    maps = []
    for p in args.maps:
        with open(p) as f:
            maps.append(json.load(f))
        print("  %s  categories=%d" % (p, len(maps[-1]["categories"])), flush=True)

    base, rest = maps[0], maps[1:]
    bmeta = base.get("metadata", {})

    # ── shape gates ────────────────────────────────────────────────────────
    for p, m in zip(args.maps[1:], rest):
        mm = m.get("metadata", {})
        for k in ("model", "num_layers", "num_experts", "wnorm_mode"):
            if k in bmeta and k in mm and bmeta[k] != mm[k]:
                sys.exit("FATAL: %s disagrees on %s: %r vs %r" % (p, k, bmeta[k], mm[k]))
        if set(m["categories"]) != set(base["categories"]):
            only_a = set(base["categories"]) - set(m["categories"])
            only_b = set(m["categories"]) - set(base["categories"])
            print("  WARNING: category sets differ (%s only in first, %s only in %s) "
                  "— the union is merged, but a category present in only one input "
                  "carries that input's sample size alone." % (sorted(only_a), sorted(only_b), p))

    # ── disjointness of the SOURCE traces, not just of the files ───────────
    # Merging two maps built from the same trace would double-count it, which the
    # arithmetic cannot detect. task_ids are not stored per-map, so the best
    # available witness is the tier_b source file plus its trace count.
    srcs = [m.get("metadata", {}).get("tier_b_source") for m in maps]
    if len(set(srcs)) != len(srcs):
        sys.exit("FATAL: two inputs share tier_b_source %r — that is the same q-set "
                 "profiled twice, not two disjoint sets." % srcs)
    print("tier_b sources (must be distinct q-sets): %s" % [str(s).split("/")[-1] for s in srcs])

    # ── merge ──────────────────────────────────────────────────────────────
    out_cats = {}
    carried, summed = [], []
    nan_cells = {}
    allnames = sorted({c for m in maps for c in m["categories"]})
    for name in allnames:
        present = [m["categories"][name] for m in maps if name in m["categories"]]
        acc = present[0]
        if len(present) > 1 and all(cat_identical(acc, o) for o in present[1:]):
            # Same cells in every input => imported, not measured. Carry once.
            out_cats[name] = acc
            carried.append(name)
            continue
        if name.startswith("generic_") and len(present) > 1 and not args.allow_generic_sum:
            sys.exit("FATAL: %s differs between inputs. generic_* is imported from a "
                     "shared Tier-A file and must be byte-equal; a difference means the "
                     "inputs do NOT share a Tier-A source and summing them would mix "
                     "two different generic baselines. Pass --allow-generic-sum only if "
                     "you have verified that is what you want." % name)
        for other in present[1:]:
            for li, rows in other.items():
                brows = acc[li]
                for r, br in zip(rows, brows):
                    for k in ADDITIVE:
                        a_, b_ = br[k], r[k]
                        # nan + anything = nan. Summing a poisoned cell into a
                        # clean one would SPREAD the defect rather than preserve
                        # it, so a nan operand is skipped and counted instead.
                        an = isinstance(a_, float) and a_ != a_
                        bn = isinstance(b_, float) and b_ != b_
                        if an or bn:
                            nan_cells[k] = nan_cells.get(k, 0) + 1
                            br[k] = b_ if an and not bn else a_
                        else:
                            br[k] = a_ + b_
                    na, nb = r.get("neuron_act") or [], br.get("neuron_act") or []
                    if na and nb:
                        br["neuron_act"] = [x + y for x, y in zip(nb, na)]
                    elif na:
                        br["neuron_act"] = list(na)
        out_cats[name] = acc
        summed.append(name)

    # ── recompute the intensive fields (never sum an RMS) ──────────────────
    for cat in out_cats.values():
        for rows in cat.values():
            for r in rows:
                tc = r.get("tc", 0) or 0
                if tc > 0:
                    r["wnorm"] = math.sqrt(max(r.get("wnsq", 0.0), 0.0) / tc)
                    r["rnorm"] = math.sqrt(max(r.get("rnsq", 0.0), 0.0) / tc)
                else:
                    r["wnorm"] = 0.0
                    r["rnorm"] = 0.0

    meta = dict(bmeta)
    meta["merged_from"] = list(args.maps)
    meta["merged_categories_summed"] = summed
    meta["merged_categories_carried_once"] = carried
    meta["tier_b_source"] = "+".join(str(s) for s in srcs)
    meta["tier_b_trace_count"] = sum(m.get("metadata", {}).get("tier_b_trace_count", 0)
                                     for m in maps)
    sc = {}
    for m in maps:
        for k, v in (m.get("metadata", {}).get("tier_b_set_counts") or {}).items():
            sc[k] = sc.get(k, 0) + v
    meta["tier_b_set_counts"] = sc

    if nan_cells:
        print("\nNaN operands skipped during summation (not propagated): %s" % nan_cells)
    print("\nsummed (measured per q-set) : %s" % ", ".join(summed))
    print("carried once (imported)     : %s" % (", ".join(carried) or "(none)"))
    print("merged trace count          : %d   set_counts=%s"
          % (meta["tier_b_trace_count"], sc))
    for name in summed:
        tot = sum(r["tc"] for rows in out_cats[name].values() for r in rows)
        print("   %-28s merged tc=%s" % (name, f"{tot:,}"))

    with open(args.out, "w") as f:
        json.dump({"metadata": meta, "categories": out_cats}, f)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
