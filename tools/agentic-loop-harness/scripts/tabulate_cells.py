#!/usr/bin/env python3
"""Merge per-cell result JSONs into one fail-rate table.

When templates are run as SEPARATE harness invocations (one per GPU/server, the
fast parallel layout), each writes its own result_*.json under its own out-dir.
This globs them and prints the combined table: rows = template/fixture, columns =
sampler configs, cells = loops+runaways / seeds.

  tabulate_cells.py <root-dir>      # root contains <cell>/result_*.json
"""
import glob
import json
import os
import sys


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(root, "*", "result_*.json")))
    if not files:
        sys.exit("no result_*.json under %s/*/" % root)
    rows, cfg_names = [], []
    for f in files:
        d = json.load(open(f))
        label = "%s/%s" % (d.get("template"), d.get("fixture"))
        by = {}
        for r in d.get("results", []):
            by[r["config"]] = r
            if r["config"] not in cfg_names:
                cfg_names.append(r["config"])
        rows.append((label, by))

    w = max([20] + [len(lab) for lab, _ in rows]) + 1
    header = "template/fixture".ljust(w) + "".join(c.rjust(16) for c in cfg_names)
    print("==== AGENTIC-LOOP FAIL-RATE TABLE (loops+runaways / seeds) ====")
    print(header)
    print("-" * len(header))
    for label, by in rows:
        line = label.ljust(w)
        for c in cfg_names:
            r = by.get(c)
            if r:
                cell = "%d/%d(l%d,r%d)" % (r["fails"], r["seeds"],
                                          r["loops"], r["runaways"])
            else:
                cell = "-"
            line += cell.rjust(16)
        print(line)
    print("\n(cell = fails/seeds (l=loops, r=runaways); lower is better)")


if __name__ == "__main__":
    main()
