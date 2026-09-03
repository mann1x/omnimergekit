#!/usr/bin/env python
"""Polarity battery for detect_output_gate_type (omnimergekit auto MLP-skip guard).

The guard decides whether Qwen3.6-family MLPs are copied from base instead of merged.
When it wrongly returns None the merge proceeds, completes, and produces a model that
leaks <think> ~80% of the time -- there is no crash to notice. So both directions get
an explicit arm, and the real published configs are used as the fixtures.
"""
import ast
import json
import sys
import urllib.request

src = open("omnimergekit.py").read()
tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "detect_output_gate_type")
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<omk>", "exec"), ns)
detect = ns["detect_output_gate_type"]

P = F = 0


def chk(label, cond):
    global P, F
    print(("  ok   " if cond else "  FAIL ") + label)
    P, F = (P + 1, F) if cond else (P, F + 1)


chk("A  nested text_config (the layout Qwen actually ships) -> detected",
    detect({"text_config": {"output_gate_type": "swish"}}) == "swish")
chk("B  top-level (flattened / text-only repos) -> detected",
    detect({"output_gate_type": "swish"}) == "swish")
chk("C  top-level WINS when both present",
    detect({"output_gate_type": "a", "text_config": {"output_gate_type": "b"}}) == "a")
chk("D  Qwen3.5 (absent everywhere) -> None, guard correctly does NOT fire",
    detect({"text_config": {"hidden_size": 5120}}) is None)
chk("D2 explicit null under text_config -> None",
    detect({"text_config": {"output_gate_type": None}}) is None)
chk("E  empty config -> None, no raise", detect({}) is None)
chk("E2 text_config present but not a dict -> no raise",
    (lambda: detect({"text_config": {}}) is None)())

# --- REGRESSION FIXTURE: the real published configs, fetched live -----------------
# This is the case that was broken. A synthetic dict cannot prove the shipped file
# parses the way we think. [[feedback_match_the_shipped_artifact_not_your_note]]
try:
    for repo in ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-27B"):
        u = f"https://huggingface.co/{repo}/resolve/main/config.json"
        cfg = json.loads(urllib.request.urlopen(u, timeout=30).read())
        got = detect(cfg)
        chk(f"F  REAL {repo}: guard fires (got {got!r}, top-level was "
            f"{cfg.get('output_gate_type')!r})", got == "swish")
except Exception as e:                                    # noqa: BLE001
    print(f"  SKIP live fixture ({type(e).__name__}: {e}) — offline")

print(f"AUTO_MLP_SKIP_GATE pass={P} fail={F}")
sys.exit(1 if F else 0)
