import glob, json
R = "/srv/ml/eval_results/qwen_suite/gpqa_diamond_full"
for n in ["qwenhybridp24_q6k", "qwena3bgepo1_q6k", "qwena3bgepo2_q6k"]:
    f = sorted(glob.glob(R + "/" + n + "/lm_eval_out/**/samples_*.jsonl", recursive=True))
    if not f:
        print("%-22s NO SAMPLES FILE" % n)
        continue
    tot = blank = 0
    for line in open(f[-1]):
        if not line.strip():
            continue
        d = json.loads(line)
        tot += 1
        r = (d.get("resps") or [[""]])[0]
        t = r[0] if r else ""
        if not (t or "").strip():
            blank += 1
    s = json.load(open(R + "/" + n + "/summary.json"))
    sc = s.get("score")
    print("%-22s n_samples=%4d  empty=%3d  score=%.4f  passes=%.1f" % (n, tot, blank, sc, sc * tot))
