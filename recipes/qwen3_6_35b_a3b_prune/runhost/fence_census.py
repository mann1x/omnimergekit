import json, glob, os, ast, re, sys
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL)
def parses(c):
    try:
        ast.parse(c); return True
    except Exception:
        return False
hdr = ("cell", "fail", "1fence", "2+fence", "0fence", "dmg_1f", "dmg_2f")
print("%-30s %5s %7s %8s %7s %7s %7s" % hdr)
pats = ["ream_arms/lcb_v6_77q*/*/lcb_result.samples.jsonl",
        "qwen_suite/lcb_v6_77q/*/lcb_result.samples.jsonl"]
files = []
for p in pats:
    files += sorted(glob.glob(p))
tot = [0, 0, 0, 0, 0, 0]
for f in files:
    cell = os.path.basename(os.path.dirname(f))
    fail = one = two = zero = d1 = d2 = 0
    for line in open(f, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("passed"):
            continue
        fail += 1
        raw = d.get("completion") or ""
        cl = d.get("cleaned") or ""
        n = sum(1 for L in raw.split("\n") if "```" in L)
        good = any(parses(b) and "class Solution" in b for b in FENCE.findall(raw))
        dead = not parses(cl)
        if n == 1:
            one += 1
            d1 += int(dead and good)
        elif n >= 2:
            two += 1
            d2 += int(dead and good)
        else:
            zero += 1
    print("%-30s %5d %7d %8d %7d %7d %7d" % (cell, fail, one, two, zero, d1, d2))
    for i, v in enumerate((fail, one, two, zero, d1, d2)):
        tot[i] += v
print("%-30s %5d %7d %8d %7d %7d %7d" % ("TOTAL", *tot))
