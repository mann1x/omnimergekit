"""Two-direction check: the real pass@1 is returned, and a bookkeeping-only
dict is REFUSED rather than turned into a plausible number."""
import json, pathlib, sys, tempfile
sys.path.insert(0, sys.argv[1])
import omk_eval

tpl = {"scoring": {"metric": "pass_at_1", "filter": "build_predictions"}}
root = pathlib.Path(tempfile.mkdtemp())
d = root / "lm_eval_out" / "m"
d.mkdir(parents=True)

(d / "results_2026-09-08T12-20-34.523129.json").write_text(json.dumps(
    {"results": {"humaneval_chat": {"name": "humaneval_chat", "alias": "humaneval_chat",
     "sample_len": 164, "pass@1,extract_chat": 0.8963414634146342,
     "pass@1_stderr,extract_chat": 0.024}}}))
s, _ = omk_eval.extract_canonical_score(tpl, root)
ok1 = (s == 0.8963414634146342)
print("  real pass@1 : %-22r %s" % (s, "PASS" if ok1 else "FAIL"))

for f in d.glob("results_*.json"):
    f.unlink()
(d / "results_2026-09-08T12-20-35.000000.json").write_text(json.dumps(
    {"results": {"t": {"name": "t", "sample_len": 164}}}))
s2, _ = omk_eval.extract_canonical_score(tpl, root)
ok2 = (s2 is None)
print("  floor control: %-22r %s" % (s2, "PASS (refuses)" if ok2 else "FAIL (invented)"))
print("  VERDICT:", "BOTH PASS" if (ok1 and ok2) else "*** FAILED ***")
