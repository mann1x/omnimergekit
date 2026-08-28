from huggingface_hub import HfApi

api = HfApi()
print("=== ManniX-ITA models matching coderx / a3b ===")
seen = []
for m in api.list_models(author="ManniX-ITA", limit=300):
    n = m.id
    low = n.lower()
    if ("coderx" in low) or ("a3b" in low):
        seen.append(n)
        print("  ", n)
if not seen:
    print("   (none matched; listing all)")
    for m in api.list_models(author="ManniX-ITA", limit=100):
        print("  ", m.id)

print()
print("=== Q6_K / imatrix inside each matched repo ===")
for r in seen:
    try:
        info = api.model_info(r, files_metadata=True)
        hits = [s for s in info.siblings
                if ("Q6_K" in s.rfilename) or ("imatrix" in s.rfilename)]
        if not hits:
            continue
        print("repo %s  private=%s" % (r, info.private))
        for s in hits:
            gb = (s.size or 0) / 1e9
            lfs = s.lfs
            sha = ""
            if isinstance(lfs, dict):
                sha = lfs.get("sha256") or ""
            elif lfs is not None:
                sha = getattr(lfs, "sha256", "") or ""
            print("   %-50s %7.2f GB  sha=%s" % (s.rfilename, gb, sha[:12]))
    except Exception as e:
        print("repo %s  ERR %s" % (r, type(e).__name__))
