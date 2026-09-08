import json, glob, os
R="/srv/ml/eval_results/qwen_suite/multipl_e_100"
for arm,a in (("armJ","qwenhybridp24_q6k"),("pub","qwencodermpe_q6k")):
    f=[x for x in glob.glob("%s/%s/results/humaneval-java/HumanEval_13_*.results.json"%(R,a))][0]
    d=json.load(open(f)); r0=(d.get("results") or [{}])[0]
    print("="*70); print("%s  status=%s" % (arm, r0.get("status")))
    print("="*70)
    for i,l in enumerate(( r0.get("program") or "").split("\n"),1):
        print("%3d| %s" % (i,l))
    print()
