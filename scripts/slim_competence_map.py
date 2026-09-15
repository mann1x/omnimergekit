"""Strip `neuron_act` from a competence map so it can be published in the repo.

WHY THIS EXISTS. The Gemma-4 maps were never published to omnimergekit, and the reason
is size, not oversight: each cell carries a 704-wide `neuron_act` array (128 experts x 30
layers x 8 categories), so a map lands at ~237 MB. The Qwen maps ARE published, at 8-10 MB
-- and they carry `neuron_act: []`. So the established publish form is the map WITHOUT the
per-neuron vector, and this produces exactly that.

SAFE BECAUSE: `make_drop_map.py` never reads `neuron_act` -- it scores on tc / wnorm /
wnorm_tc / rnorm / rnorm_tc only. The drop map built from a slimmed map is therefore
byte-identical to one built from the full map. --verify proves the precondition directly:
every field of every cell EXCEPT neuron_act is unchanged. That is stronger than running
one make_drop_map configuration and comparing, because it holds for ANY scorer that does
not read neuron_act, not just the one configuration that happened to be tested.

NOT SAFE FOR: neuron-level work (DERN-style redistribution, per-neuron pruning), which is
the whole point of the field. The full map must therefore be kept on disk and backed up;
the slim one is a publication artifact, not a replacement.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--keep-empty", action="store_true", default=True,
                help="write neuron_act as [] rather than deleting the key, matching the "
                     "published Qwen maps so downstream readers see the same schema")
ap.add_argument("--keep-cats", default="",
                help="comma-separated category names or prefixes to KEEP (e.g. 'generic_'). "
                     "Used when publishing the Tier-A baseline: its source map also holds "
                     "stale targeted_* categories from a retired bench basis, which have no "
                     "business in a public repo and are not what --load-tier-a-from imports "
                     "(the loader takes generic_* only).")
ap.add_argument("--verify", action="store_true",
                help="after writing, re-read both files and assert that every cell field "
                     "other than neuron_act is identical")
args = ap.parse_args()

d = json.load(open(args.src))
cats = d["categories"]
dropped_cats = []
if args.keep_cats:
    want = [x.strip() for x in args.keep_cats.split(",") if x.strip()]
    keep = {c for c in cats if any(c == w or c.startswith(w) for w in want)}
    if not keep:
        raise SystemExit("FATAL: --keep-cats %r matched nothing in %s" % (args.keep_cats, sorted(cats)))
    dropped_cats = sorted(set(cats) - keep)
    for c in dropped_cats:
        del cats[c]
    print("  kept %d categories, dropped %d: %s" % (len(cats), len(dropped_cats), dropped_cats))
n_cells = n_vals = 0
for cat in cats.values():
    for rows in cat.values():
        for r in rows:
            na = r.get("neuron_act")
            if na:
                n_vals += len(na)
            r["neuron_act"] = []
            n_cells += 1

meta = d.setdefault("metadata", {})
meta["neuron_act_stripped"] = True
if dropped_cats:
    meta["categories_dropped_for_publication"] = dropped_cats
    meta["categories_dropped_note"] = ("retired/stale bench categories removed for publication; "
                                       "--load-tier-a-from imports generic_* only, so nothing "
                                       "that is actually consumed was removed")
meta["neuron_act_note"] = ("per-neuron vectors removed for publication; the full map is "
                           "required for neuron-level work and is kept on disk")
with open(args.dst, "w") as fh:
    json.dump(d, fh)

a, b = os.path.getsize(args.src), os.path.getsize(args.dst)
print("  %s  %.1f MB" % (os.path.basename(args.src), a / 1048576))
print("  %s  %.1f MB   (%.1fx smaller)" % (os.path.basename(args.dst), b / 1048576, a / max(b, 1)))
print("  cells: %d   neuron_act values dropped: %s" % (n_cells, f"{n_vals:,}"))
print("  categories preserved: %s" % sorted(cats))

if args.verify:
    src = json.load(open(args.src))["categories"]
    dst = json.load(open(args.dst))["categories"]
    if dropped_cats:
        src = {k: v for k, v in src.items() if k not in set(dropped_cats)}
    if set(src) != set(dst):
        raise SystemExit("VERIFY FAILED: category sets differ (beyond the ones dropped on purpose)")
    checked = 0
    for c in src:
        if set(src[c]) != set(dst[c]):
            raise SystemExit("VERIFY FAILED: %s layer sets differ" % c)
        for li in src[c]:
            ra, rb = src[c][li], dst[c][li]
            if len(ra) != len(rb):
                raise SystemExit("VERIFY FAILED: %s/%s row count differs" % (c, li))
            for x, y in zip(ra, rb):
                ka = {k: v for k, v in x.items() if k != "neuron_act"}
                kb = {k: v for k, v in y.items() if k != "neuron_act"}
                if set(ka) != set(kb):
                    raise SystemExit("VERIFY FAILED: %s/%s/%s key set differs" % (c, li, x.get("id")))
                for k in ka:
                    va, vb = ka[k], kb[k]
                    # NaN lives in the imported generic_* cells (wsum); nan != nan, so a
                    # naive compare would report a difference the strip did not cause.
                    if isinstance(va, float) and isinstance(vb, float) and va != va and vb != vb:
                        continue
                    if va != vb:
                        raise SystemExit("VERIFY FAILED: %s/%s/%s field %s changed: %r -> %r"
                                         % (c, li, x.get("id"), k, va, vb))
                if y.get("neuron_act") not in ([], None):
                    raise SystemExit("VERIFY FAILED: neuron_act not stripped at %s/%s" % (c, li))
                checked += 1
    print("  VERIFY PASS: %d cells — every field except neuron_act is unchanged," % checked)
    print("               so any scorer that does not read neuron_act produces an")
    print("               identical drop map from the slim file.")
