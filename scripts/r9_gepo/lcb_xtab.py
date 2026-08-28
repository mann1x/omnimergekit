"""LCB v6-77q cap x pass cross-tab + paired both-uncapped cell (qwen_suite cohort)."""
import json, os

R = "/srv/ml/eval_results/qwen_suite/lcb_v6_77q"
ARMS = [("armJ", "qwenhybridp24_q6k"), ("gepo1", "qwena3bgepo1_q6k")]


def rows(cell):
    p = os.path.join(R, cell, "lcb_result.samples.jsonl")
    out = {}
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[len(out)] = r
    return out


# probe schema once
_, c0 = ARMS[0]
r0 = rows(c0)
print("sample keys:", sorted(list(r0[0].keys()))[:20])
print()


def pass_of(r):
    for k in ("passed", "pass", "correct", "graded_list", "pass@1", "pass_at_1"):
        if k in r:
            v = r[k]
            if isinstance(v, list):
                return all(bool(x) for x in v) if v else None
            if isinstance(v, (bool, int, float)):
                return bool(v)
    return None


def tid_of(r, i):
    for k in ("task_id", "question_id", "id", "problem_id", "doc_id"):
        if k in r:
            return str(r[k])
    return "idx%d" % i


def toklen(r):
    for k in ("completion_tokens", "n_tokens", "gen_tokens"):
        if k in r:
            return r[k]
    for k in ("completion", "generation", "raw", "output"):
        v = r.get(k)
        if isinstance(v, str):
            return len(v)  # char proxy; only used for ranking
    return None


CAPMAX, TOL = 32768, 16
data = {}
for tag, cell in ARMS:
    rr = rows(cell)
    g = json.load(open(os.path.join(R, cell, "summary.json")))["generation_caps"]
    ncap = g["capped_total"]
    st, ln = {}, {}
    for i, r in rr.items():
        t = tid_of(r, i)
        st[t] = pass_of(r)
        ln[t] = toklen(r) or 0
    capped = {t for t, _ in sorted(ln.items(), key=lambda kv: -kv[1])[:ncap]}
    data[tag] = (st, ln, capped)
    cp = sum(1 for t in capped if st.get(t))
    n = len(st)
    tot = sum(1 for t in st if st.get(t))
    print("%-6s n=%d  pass=%d  capped=%d  capped-that-PASS=%d  uncapped-pass=%d/%d"
          % (tag, n, tot, len(capped), cp, tot - cp, n - len(capped)))

(sa, la, ca), (sb, lb, cb) = data["armJ"], data["gepo1"]
keys = [t for t in sa if t in sb]
both = [t for t in keys if t not in ca and t not in cb]
pa = sum(1 for t in both if sa[t])
pb = sum(1 for t in both if sb[t])
print()
print("common problems: %d   uncapped in BOTH arms: %d (dropped %d)"
      % (len(keys), len(both), len(keys) - len(both)))
if both:
    print("PAIRED cell (neither arm truncated):")
    print("  armJ  %d/%d = %.4f" % (pa, len(both), pa / len(both)))
    print("  gepo1 %d/%d = %.4f" % (pb, len(both), pb / len(both)))
    print("  delta %+.2f pp (%+d problems)" % ((pb - pa) / len(both) * 100, pb - pa))
    b_only = sum(1 for t in both if sb[t] and not sa[t])
    a_only = sum(1 for t in both if sa[t] and not sb[t])
    print("  discordant: gepo1-only %d, armJ-only %d (McNemar b=%d c=%d)"
          % (b_only, a_only, b_only, a_only))
