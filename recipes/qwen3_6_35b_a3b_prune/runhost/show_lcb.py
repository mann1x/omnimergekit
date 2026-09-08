import json, glob, ast, re, sys
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL)
def parses(c):
    try:
        ast.parse(c); return True
    except Exception:
        return False
shown = 0
for f in glob.glob("/srv/ml/eval_results/ream_arms/lcb_v6_77q*/*/lcb_result.samples.jsonl"):
    for line in open(f, errors="ignore"):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("passed"): continue
        cleaned, raw = d.get("cleaned") or "", d.get("completion") or ""
        if parses(cleaned): continue
        good = [b for b in FENCE.findall(raw) if parses(b) and "class Solution" in b]
        if not good: continue
        shown += 1
        print("=" * 72)
        print(d.get("task_id"), "| reason:", str(d.get("reason"))[:70])
        print("--- SCORED `cleaned` (fails to parse), len", len(cleaned), "---")
        print(cleaned[:300])
        try: ast.parse(cleaned)
        except Exception as e: print("   >>> parse error:", type(e).__name__, e)
        print("--- RAW had this parseable class Solution, len", len(good[0]), "---")
        print(good[0][:300])
        print("--- raw fence-line count:", raw.count("```"))
        if shown >= 2: sys.exit(0)
