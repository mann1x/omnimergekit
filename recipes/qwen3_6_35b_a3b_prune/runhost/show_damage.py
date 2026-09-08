import json, glob, ast, re, sys
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL)
def parses(c):
    try:
        ast.parse(c); return True
    except Exception:
        return False
def flat(x):
    while isinstance(x, list) and x: x = x[0]
    return x if isinstance(x, str) else ""
pat, key = sys.argv[1], sys.argv[2]
shown = 0
for f in glob.glob(pat, recursive=True):
    for line in open(f, errors="ignore"):
        try: d = json.loads(line)
        except Exception: continue
        if d.get(key): continue
        filt, raw = flat(d.get("filtered_resps")), flat(d.get("resps"))
        if parses(filt): continue
        good = [b for b in FENCE.findall(raw) if parses(b)]
        if not good: continue
        shown += 1
        print("=" * 72); print("doc_id", d.get("doc_id"))
        print("--- SCORED ARTIFACT (fails to parse) ---")
        print(filt[:400])
        try: ast.parse(filt)
        except Exception as e: print("   >>> parse error:", type(e).__name__, e)
        print("--- RAW reply had this parseable fenced block ---")
        print(good[0][:400])
        if shown >= 2: sys.exit(0)
