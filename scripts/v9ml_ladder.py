"""v9-ml minimal-swap ladder over map A.

Candidates are ONLY experts map A drops. Each is scored by its per-layer rank percentile
(wnorm, 0 = lowest .. 1 = highest of 128) in the two new held-out classes:
  shared = min(r_multilingual, r_ifeval)   -- important for BOTH targets (primary ladder)
  ml     = r_multilingual                  -- attribution arm
  if     = r_ifeval                        -- attribution arm
Ties broken by the other class, then (layer, expert) for determinism.

`pins` mode prints one line per rung: "<name> <comma list of layer:expert>". The shell adds
them to map A's own force-keep list, so every pin rescues one dropped expert and the
generator evicts the lowest-scored survivor in that layer by MAP A's class scores.
`report` mode reads the built rungs and prints exact swaps vs A, the rescued and evicted
experts, and each evictee's rank in A's agentic classes (targeted_swe_* max, generic_code).
"""
import json
import sys

RUNGS = [("shared", k) for k in (2, 4, 6, 8, 12, 16)] + [("ml", 8), ("if", 8)]


def load_map(p):
    d = json.load(open(p))
    d = d.get("drop_map", d)
    return {int(k): set(v) for k, v in d.items() if str(k).isdigit()}


def pct(cat):
    out = {}
    for layer, rows in cat.items():
        order = sorted(rows, key=lambda e: (e["wnorm"], e["id"]))
        out[int(layer)] = {e["id"]: i / (len(order) - 1) for i, e in enumerate(order)}
    return out


def ranked(cats, drop, mode):
    rml, rif = pct(cats["generic_multilingual"]), pct(cats["generic_ifeval"])
    cand = []
    for layer, dropped in drop.items():
        for e in dropped:
            a, b = rml[layer][e], rif[layer][e]
            key = {"shared": (min(a, b), max(a, b)), "ml": (a, b), "if": (b, a)}[mode]
            cand.append((key, layer, e, a, b))
    cand.sort(key=lambda c: (-c[0][0], -c[0][1], c[1], c[2]))
    return cand


def main():
    mode, merged, aref = sys.argv[1:4]
    cats = json.load(open(merged))["categories"]
    A = load_map(aref)
    if mode == "pins":
        for kind, k in RUNGS:
            top = ranked(cats, A, kind)[:k]
            print(f"{kind}_k{k} " + ",".join(f"{layer}:{e}" for _, layer, e, _, _ in top))
        return
    outdir = sys.argv[4]
    rml, rif = pct(cats["generic_multilingual"]), pct(cats["generic_ifeval"])
    swe = [c for c in cats if c.startswith("targeted_swe_")]
    rswe = {c: pct(cats[c]) for c in swe}
    rcode = pct(cats["generic_code"])
    for kind, k in RUNGS:
        m = load_map(f"{outdir}/{kind}_k{k}.json")
        rescued = sorted((layer, e) for layer in A for e in A[layer] - m[layer])
        evicted = sorted((layer, e) for layer in A for e in m[layer] - A[layer])
        flag = "" if len(rescued) == len(evicted) == k else f"  !! expected {k} swaps"
        print(f"\n{kind}_k{k}: rescued {len(rescued)}, evicted {len(evicted)}{flag}")
        for layer, e in rescued:
            print(f"   + L{layer:<2} e{e:<3}  ml={rml[layer][e]:.2f} if={rif[layer][e]:.2f}")
        for layer, e in evicted:
            s = max(rswe[c][layer][e] for c in swe)
            print(f"   - L{layer:<2} e{e:<3}  swe_max={s:.2f} code={rcode[layer][e]:.2f}"
                  f"  ml={rml[layer][e]:.2f} if={rif[layer][e]:.2f}")


if __name__ == "__main__":
    main()
