#!/usr/bin/env python
"""GATE 0 for the EOG cross-check: does the set I proposed to test even EXIST?

The draft's next step is "cross-check the experts p=12 EVICTS that p=24 RETAINS against an
EOG emit-map". make_hybrid_dropmap.py builds both doses as PREFIXES of the same two sorted
lists:

    P_p  = sorted(our_drop_set, key=-reap)[:p]      # protected
    E_p  = sorted(our_keep_set, key=+ours)[:p]      # evicted

so P_12 subset P_24 and E_12 subset E_24 BY CONSTRUCTION. If that holds in the built maps
then nothing p=12 evicts is retained by p=24 -- the proposed differential is EMPTY and the
hypothesis as worded is untestable. The real asymmetry would be the opposite direction:

    keep12 \\ keep24 = E_24 \\ E_12   (evicted ONLY at the higher dose)
    keep24 \\ keep12 = P_24 \\ P_12   (protected ONLY at the higher dose)

and p=24's LOWER empty rate could then only come from P_24\\P_12 restoring something.

Verify against the ACTUAL built maps, not the source, then name the correct target set.
"""
import json
import os

H = "/mnt/sdc/ream-work/hybrid_maps"
SHIPPED = ("/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune/results/"
           "drop_map_184e_coder_lcbmpe.json")
E_TOTAL = 256
KEEP = 184


def dropmap(p):
    return json.load(open(os.path.join(H, "drop_map_184e_hybrid_p%d.json" % p)))


def main():
    d12, d24, ship = dropmap(12), dropmap(24), json.load(open(SHIPPED))
    layers = sorted(int(k) for k in ship if k.isdigit())
    print("layers=%d  experts=%d  keep=%d" % (len(layers), E_TOTAL, KEEP))

    all_e = set(range(E_TOTAL))
    stats = {"only12_kept": [], "only24_kept": [], "E24_minus_E12": [],
             "P24_minus_P12": [], "E12_sub_E24": 0, "P12_sub_P24": 0}
    only24_prot = {}   # layer -> experts protected ONLY at p=24
    only24_evic = {}   # layer -> experts evicted ONLY at p=24

    for L in layers:
        k = str(L)
        base_drop = set(int(x) for x in ship[k])
        base_keep = all_e - base_drop
        keep12 = all_e - set(int(x) for x in d12[k])
        keep24 = all_e - set(int(x) for x in d24[k])
        assert len(keep12) == len(keep24) == KEEP, (L, len(keep12), len(keep24))

        # decompose each dose back into its P (protected) and E (evicted) sets
        P12, E12 = keep12 - base_keep, base_keep - keep12
        P24, E24 = keep24 - base_keep, base_keep - keep24
        assert len(P12) == len(E12) == 12, (L, len(P12), len(E12))
        assert len(P24) == len(E24) == 24, (L, len(P24), len(E24))

        stats["E12_sub_E24"] += E12 <= E24
        stats["P12_sub_P24"] += P12 <= P24
        stats["only12_kept"].append(len(keep12 - keep24))
        stats["only24_kept"].append(len(keep24 - keep12))
        stats["E24_minus_E12"].append(len(E24 - E12))
        stats["P24_minus_P12"].append(len(P24 - P12))
        only24_prot[L] = sorted(P24 - P12)
        only24_evic[L] = sorted(E24 - E12)

    n = len(layers)
    print("\nnesting check (per-layer, %d layers):" % n)
    print("  E_12 subset of E_24 : %d/%d layers" % (stats["E12_sub_E24"], n))
    print("  P_12 subset of P_24 : %d/%d layers" % (stats["P12_sub_P24"], n))

    def rng(key):
        v = stats[key]
        return "min=%d max=%d mean=%.1f" % (min(v), max(v), sum(v) / len(v))

    print("\nTHE SET THE DRAFT PROPOSED TO TEST:")
    print("  |kept by p12 AND dropped by p24| = |E_24 \\ E_12| = %s" % rng("only12_kept"))
    print("  ^ these are experts p=24 evicts that p=12 keeps -- the OPPOSITE direction.")
    print("  |dropped by p12 AND kept by p24| = %s" % rng("only24_kept"))
    print("     of which P_24\\P_12 (extra REAP protections): %s" % rng("P24_minus_P12"))

    empty = max(stats["only12_kept"]) == 0
    print("\n=> %s" % ("CONFIRMED EMPTY: p=12 evicts nothing that p=24 retains. The draft's "
                       "worded hypothesis is untestable."
                       if empty else
                       "NON-EMPTY: the draft's set exists; test it directly."))

    # the correct target: experts ONLY the higher dose rescues
    tgt = sorted(set(e for L in layers for e in only24_prot[L]))
    print("\nCORRECT TARGET SET = P_24 \\ P_12  (rescued only at the higher dose)")
    print("  %d (layer,expert) pairs; %d distinct expert ids across layers"
          % (sum(len(v) for v in only24_prot.values()), len(tgt)))
    out = "/mnt/sdc/ream-work/hybrid_delta_sets.json"
    json.dump({"only24_protected": {str(k): v for k, v in only24_prot.items()},
               "only24_evicted": {str(k): v for k, v in only24_evic.items()}},
              open(out, "w"))
    print("  wrote %s" % out)
    for L in layers[:3]:
        print("    L%-3d P24\\P12=%s" % (L, only24_prot[L]))
        print("         E24\\E12=%s" % only24_evic[L])


if __name__ == "__main__":
    main()
