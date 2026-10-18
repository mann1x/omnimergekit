import json, sys
for p in sys.argv[1:]:
    try: d = json.load(open(p))
    except Exception as e: print("ERR", e, p); continue
    ts = d.get("token_stats", {}); ct = ts.get("completion_tokens", {})
    samp = d.get("sampler") or {}
    print("--", p)
    print("   score=%s metric=%s/%s rc=%s n=%s template=%s sampler=%s" % (
        round((d.get("score") or 0)*100,2), d.get("metric"), d.get("filter"), d.get("rc"),
        ts.get("n"), d.get("template"), (samp.get("name") if isinstance(samp,dict) else samp)))
    print("   finish=%s empty=%s tok_p50=%s tok_p90=%s tok_max=%s" % (
        ts.get("finish_reasons"), ts.get("empty_completions"), ct.get("p50"), ct.get("p90"), ct.get("max")))
