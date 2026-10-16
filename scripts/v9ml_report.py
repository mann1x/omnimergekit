"""v9-ml drop-map sweep report: swaps vs map A and retention of the new classes' top experts."""
import json
import sys

O, AREF, MERGED, P12 = sys.argv[1:]
ARMS = ["ml0_if0", "ml1_if1", "ml2_if2", "ml2_if0", "ml0_if2"]


def load_map(p):
    d = json.load(open(p))
    d = d.get("drop_map", d)
    return {int(k): set(v) for k, v in d.items() if str(k).isdigit()}


A = load_map(AREF)
cats = json.load(open(MERGED))["categories"]


def top(cat, k=16):
    return {int(layer): {e["id"] for e in sorted(rows, key=lambda e: -e["wnorm"])[:k]}
            for layer, rows in cats[cat].items()}


tml, tif = top("generic_multilingual"), top("generic_ifeval")
pins = [tuple(map(int, x.split(":"))) for x in P12.split(",")]
print(f"{'arm':8} {'swaps_vs_A':>10} {'ML_top16_kept':>13} {'IF_top16_kept':>13} {'loop_pins_dropped':>17}")
for arm in ["A"] + ARMS:
    m = A if arm == "A" else load_map(f"{O}/{arm}.json")
    sw = sum(len(m[layer] - A[layer]) for layer in A)
    kml = sum(len(tml[layer] - m[layer]) for layer in tml)
    kif = sum(len(tif[layer] - m[layer]) for layer in tif)
    pd = sum(1 for layer, e in pins if e in m[layer])
    print(f"{arm:8} {sw:10d} {kml:9d}/480 {kif:9d}/480 {pd:17d}")
