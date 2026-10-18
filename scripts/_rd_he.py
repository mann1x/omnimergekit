import json, sys
for p in sys.argv[1:]:
    try:
        d = json.load(open(p))
    except Exception as e:
        print("ERR", e, p); continue
    ts = d.get("token_stats", {})
    ct = ts.get("completion_tokens", {})
    print("score=%s metric=%s/%s rc=%s n=%s empty=%s finish=%s tok_p50=%s warns=%s" % (
        round(d.get("score",0)*100,2), d.get("metric"), d.get("filter"), d.get("rc"),
        ts.get("n"), ts.get("empty_completions"), ts.get("finish_reasons"),
        ct.get("p50"), d.get("sanity_warnings")))
