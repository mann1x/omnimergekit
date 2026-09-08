"""Print '<score> <sampler_name> <cap_verdict>' from an omk summary.json.
Exists so shell runners never embed a multi-line python -c inside a quoted say() -- that
nesting is what produced the 'unexpected EOF while looking for matching quote' parse error.
"""
import json, sys
d = json.load(open(sys.argv[1]))
sm = d.get("sampler") or {}
c = d.get("generation_caps") or {}
print("%s sampler=%s verdict=%s" % (d.get("score"), sm.get("name"), c.get("verdict")))
