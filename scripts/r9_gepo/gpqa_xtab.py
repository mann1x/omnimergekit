"""GPQA cap x pass cross-tab: does the 8192 thinking wall explain gepo1's -7?"""
import json, glob, os
from transformers import AutoTokenizer

D = "/srv/ml/eval_results/qwen_suite/gpqa_diamond_full"
ARMS = [("armJ", "qwenhybridp24_q6k", "/mnt/sdc/ream-work/armJ"),
        ("gepo1", "qwena3bgepo1_q6k", "/mnt/sdc/ml/brevity/gepo/a3b-gepo1")]
tok = AutoTokenizer.from_pretrained("/mnt/sdc/ream-work/armJ", trust_remote_code=True)

data = {}
for tag, cell, _ in ARMS:
    f = glob.glob(os.path.join(D, cell, "lm_eval_out", cell, "samples_*.jsonl"))[0]
    ncap = json.load(open(os.path.join(D, cell, "summary.json")))["generation_caps"]["capped_total"]
    st, ln, inval = {}, {}, {}
    for line in open(f):
        r = json.loads(line)
        i = r["doc_id"]
        st[i] = float(r.get("exact_match") or 0) > 0
        fr = r.get("filtered_resps") or [""]
        inval[i] = (fr and str(fr[0]).strip() == "[invalid]")
        txt = ""
        rs = r.get("resps") or []
        if rs:
            txt = rs[0][0] if isinstance(rs[0], list) else str(rs[0])
        ln[i] = len(tok(txt, add_special_tokens=False).input_ids)
    capped = {i for i, _ in sorted(ln.items(), key=lambda kv: -kv[1])[:ncap]}
    data[tag] = (st, ln, capped, inval)
    print("%-6s n=%d correct=%d capped=%d capped-correct=%d capped-[invalid]=%d"
          % (tag, len(st), sum(st.values()), len(capped),
             sum(1 for i in capped if st[i]), sum(1 for i in capped if inval[i])))

(sa, la, ca, ia), (sb, lb, cb, ib) = data["armJ"], data["gepo1"]
keys = [i for i in sa if i in sb]
both = [i for i in keys if i not in ca and i not in cb]
pa = sum(1 for i in both if sa[i]); pb = sum(1 for i in both if sb[i])
print()
print("common=%d  uncapped in BOTH=%d (dropped %d)" % (len(keys), len(both), len(keys) - len(both)))
print("PAIRED cell (neither arm hit the 8192 thinking wall):")
print("  armJ  %d/%d = %.4f" % (pa, len(both), pa / len(both)))
print("  gepo1 %d/%d = %.4f" % (pb, len(both), pb / len(both)))
print("  delta %+.2f pp (%+d q)" % ((pb - pa) / len(both) * 100, pb - pa))
b1 = sum(1 for i in both if sb[i] and not sa[i]); a1 = sum(1 for i in both if sa[i] and not sb[i])
print("  discordant: gepo1-only %d, armJ-only %d (McNemar b=%d c=%d)" % (b1, a1, b1, a1))
print()
print("  armJ score on the 10 questions gepo1 capped: %d/%d"
      % (sum(1 for i in cb if sa.get(i)), len(cb)))
print("  total [invalid] extractions: armJ %d, gepo1 %d"
      % (sum(ia.values()), sum(ib.values())))
