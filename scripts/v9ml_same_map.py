"""Exit 0 iff two drop maps drop exactly the same experts in every layer."""
import json
import sys


def load_map(p):
    d = json.load(open(p))
    d = d.get("drop_map", d)
    return {int(k): sorted(v) for k, v in d.items() if str(k).isdigit()}


sys.exit(0 if load_map(sys.argv[1]) == load_map(sys.argv[2]) else 1)
