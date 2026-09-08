#!/usr/bin/env python3
"""Counterfactual extraction: OLD vs NEW chat_to_body on the SAME raw replies.

The b604 re-run both regenerated AND re-extracted, so its old-vs-new score delta
confounds the two. The paired SAME/DIFF split could not separate them either:
on byte-identical completions the extractor provably never fires (prog-diff 0/1299),
so that stratum is depleted of the population bug-604 touches, not evidence
against it.

bug-607 made the clean test possible -- the new run persists `raw_completions`.
Applying the PRE-FIX chat_to_body (git 3221739^) to those same raws builds the
counterfactual "what the old extractor would have produced from THIS generation".
Generation is then fixed by construction and the only moving part is the extractor.

CONTROL: the CURRENT chat_to_body re-applied to the same raw must reproduce the
stored `completions[0]` byte-for-byte. If it does not, this harness is not the
code that produced the run and nothing it says is a measurement.
"""
from __future__ import annotations
import json, re, shutil, sys
from pathlib import Path

sys.path.insert(0, "/shared/dev/omnimergekit/eval/multipl_e")
from multipl_e_generate import chat_to_body as new_chat_to_body, extract_code_block


def old_chat_to_body(prompt: str, code: str, stop_tokens: list) -> str:
    """Verbatim pre-fix body (omnimergekit 3221739^). The defect is the final
    `else` branch: `code[code.find("{") + 1:]` on a body-only reply cuts at the
    first brace in the BODY -- eating the opening line of the first loop/if."""
    plines = [ln for ln in prompt.splitlines() if ln.strip()]
    anchor = plines[-1] if plines else ""
    idx = code.find(anchor) if anchor else -1
    if idx < 0 and anchor:
        target = anchor.rstrip().rstrip("{").strip()
        for line in code.splitlines():
            if target and target in line:
                anchor, idx = line, code.find(line)
                break
    if idx >= 0:
        after = code[idx + len(anchor):]
        body = after if anchor.rstrip().endswith("{") else (
            after[after.find("{") + 1:] if "{" in after else after)
    else:
        body = code[code.find("{") + 1:] if "{" in code else code
    if any((t or "").strip() == "}" for t in (stop_tokens or [])):
        body = re.sub(r"\s*(?:\}\s*)+\Z", "\n", body)
    return body if body.endswith("\n") else body + "\n"


NEW = Path("/srv/ml/eval_results_b604")
OUT = Path("/srv/ml/eval_results_b604_oldext")
LANGS = ["rs", "java", "js"]


def build(bench, cells):
    print(f"\n### {bench}")
    hdr = f"{'cell':<34} {'lang':<5} {'n':>4} {'ctrl_ok':>8} {'ctrl_BAD':>9} {'ext≠':>5}"
    print(hdr); print("-" * len(hdr))
    tot = [0, 0, 0, 0]
    for cell in cells:
        src = NEW / bench / cell
        if not src.is_dir():
            continue
        for lang in LANGS:
            gd = src / "generations" / f"humaneval-{lang}"
            if not gd.is_dir():
                continue
            dd = OUT / bench / cell / "generations" / f"humaneval-{lang}"
            dd.mkdir(parents=True, exist_ok=True)
            n = ok = bad = diff = 0
            for f in sorted(gd.glob("*.json")):
                d = json.loads(f.read_text())
                raw = (d.get("raw_completions") or [""])[0] or ""
                stored = (d.get("completions") or [""])[0] or ""
                prompt, stop = d.get("prompt", ""), d.get("stop_tokens") or []
                n += 1
                repro = new_chat_to_body(prompt, extract_code_block(raw), stop, d.get("name", ""))
                if repro == stored:
                    ok += 1
                else:
                    bad += 1
                old = old_chat_to_body(prompt, extract_code_block(raw), stop)
                if old != stored:
                    diff += 1
                out = dict(d); out["completions"] = [old]
                (dd / f.name).write_text(json.dumps(out))
            print(f"{cell:<34} {lang:<5} {n:>4} {ok:>8} {bad:>9} {diff:>5}"
                  + ("   ** CONTROL FAILED" if bad else ""))
            for i, v in enumerate((n, ok, bad, diff)):
                tot[i] += v
    print("-" * len(hdr))
    print(f"{'TOTAL':<34} {'':<5} {tot[0]:>4} {tot[1]:>8} {tot[2]:>9} {tot[3]:>5}")
    return tot


if OUT.exists():
    shutil.rmtree(OUT)
t = build("qwen_suite/multipl_e_100",
          ["qwencodermpe_t10_q6k", "qwen256e_q6k", "qwencodermpe_q6k", "qwenhybridp24_q6k"])
if t[2]:
    print("\nCONTROL FAILED — this harness is not the code that produced the run. STOP.")
    sys.exit(1)
print(f"\nCONTROL OK: current extractor reproduces all {t[1]} stored bodies byte-for-byte.")
print(f"bug-604 REACH on these generations: {t[3]}/{t[0]} bodies differ under the old extractor.")
