#!/usr/bin/env python3
"""verify_f16_tri130.py — re-score the F16 tri-130 generation WITHOUT the
closing-fence artifact. Pulls the F16 content from the run log, strips any
leading ```python / ``` fence line (the serve `until` ate the closing fence),
loads the exact HE+ doc['test'] + entry_point from the dern11-Q4 samples, and
runs check(). Reports the true PASS/FAIL plus rumination lengths."""
import json
import re

RUNLOG = "/srv/ml/agentic_loop/logs/f16_tri130_run.log"
SAMPLES = ("/srv/ml/eval_results_dern11_tradecheck/humanevalplus_full/"
           "dern11-q4/lm_eval_out/dern11-q4/"
           "samples_humaneval_plus_chat_2026-06-17T08-29-45.351520.jsonl")

# 1. test harness + entry point from the samples
doc_test = entry = None
for line in open(SAMPLES):
    r = json.loads(line)
    if r["doc"]["task_id"] == "HumanEval/130":
        doc_test = r["doc"]["test"]
        entry = r["doc"]["entry_point"]
        break
assert doc_test is not None, "tri-130 not in samples"

# 2. the F16 CONTENT block from the run log (between 'CONTENT:' and the next '===')
log = open(RUNLOG).read()
m = re.search(r"CONTENT:\n(.*?)\n=====", log, re.DOTALL)
cont = m.group(1) if m else ""

# 3. fence-robust extraction: prefer a full ```python ... ``` block; otherwise
#    strip a leading ```python / ``` line and a trailing ``` if present.
mb = re.search(r"```python\s*(.*?)```", cont, re.DOTALL)
if mb:
    code = mb.group(1)
else:
    code = re.sub(r"^\s*```[a-zA-Z]*\s*\n", "", cont)   # drop opening fence line
    code = re.sub(r"\n```\s*$", "", code)               # drop trailing fence if any
    code = re.sub(r"```\s*$", "", code)

print("extracted_code_len =", len(code))
print("first_line         =", code.strip().splitlines()[0] if code.strip() else "(empty)")

verdict = "UNKNOWN"
try:
    ns = {}
    exec(code, ns)
    exec(doc_test, ns)
    ns["check"](ns[entry])
    verdict = "PASS"
except Exception as e:
    verdict = "FAIL (%s: %s)" % (type(e).__name__, str(e)[:300])
print("TRI-130 F16 TRUE VERDICT:", verdict)
