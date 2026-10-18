import json, urllib.request
R = "Tesslate/OmniCoder-9B"
def get(u):
    return json.load(urllib.request.urlopen(u, timeout=30))
print("=== branches / tags ===")
try:
    j = get(f"https://huggingface.co/api/models/{R}/refs")
    for kind in ("branches", "tags", "converts"):
        for b in j.get(kind, []):
            print("  %-9s %-24s %s" % (kind[:-1], b.get("name"), str(b.get("targetCommit"))[:12]))
except Exception as e:
    print("  ERR", e)
print("\n=== commit history (main) ===")
try:
    for c in get(f"https://huggingface.co/api/models/{R}/commits/main")[:10]:
        print("  %s  %s  %s" % (str(c.get("date"))[:19], str(c.get("id"))[:10], str(c.get("title"))[:66]))
except Exception as e:
    print("  ERR", e)
