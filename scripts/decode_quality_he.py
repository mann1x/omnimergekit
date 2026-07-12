#!/usr/bin/env python3
"""T184 decode-fix QUALITY check — HumanEval+ phase (companion to decode_quality_check.py MPE phase).

Same three decode configs on the same A2 HF-generate path; isolates the cost of
anti-repetition decoding on Python code+reasoning. HE+ is the reasoning-heavy
sister to MPE: chat completion (think + final function), so an n-gram ban also
stresses the reasoning chain (where natural repetition is common). The in-harness
`base` config is the apples-to-apples anchor; published A2 HE+ 0.909 (vLLM +
reasoning-parser + 12288 thinking budget) is backend-confounded vs HF-generate.

Generation runs in the omk env (torch); scoring shells out to the evalplus sidecar
venv (/srv/ml/envs/evalplus) so we never perturb the canonical omk env. Problem set
dumped task_id-aligned to he_plus_problems.json from get_human_eval_plus().

Run on bs2 GPU1 (concurrent with the MPE phase on GPU0). omk python.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/srv/ml/repos/omnimergekit/eval/multipl_e")
sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from multipl_e_generate import extract_code_block  # noqa: E402
from audit_full_bench import detect_loop  # noqa: E402

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
EVALPLUS_PY = "/srv/ml/envs/evalplus/bin/python"
CONFIGS = [("base", 1.0, 0), ("ng3", 1.0, 3), ("rp11ng4", 1.1, 4), ("rp105", 1.05, 0), ("rp11", 1.10, 0), ("rp115", 1.15, 0)]


def log(m):
    print("[dqhe %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def instruction(prompt):
    return ("Complete the following Python function. Reply with ONLY the complete "
            "function implementation in a single Markdown code block — include the "
            "signature exactly as given, write the full body, and do NOT add any "
            "explanation, example usage, or test code.\n\n```python\n%s\n```" % prompt)


def score_evalplus(samples_path):
    """Run evalplus.evaluate in the sidecar venv; return (base_pass1, plus_pass1)."""
    r = subprocess.run(
        [EVALPLUS_PY, "-m", "evalplus.evaluate", "--dataset", "humaneval",
         "--samples", samples_path, "--parallel", "16"],
        capture_output=True, text=True)
    out = r.stdout + "\n" + r.stderr
    # stdout prints two "pass@1: X" lines (base then plus)
    vals = re.findall(r"pass@1:\s*([0-9.]+)", out)
    base = float(vals[0]) if len(vals) >= 1 else None
    plus = float(vals[1]) if len(vals) >= 2 else None
    if base is None:
        # fallback: compute from eval_results.json
        erj = samples_path.replace(".jsonl", ".eval_results.json")
        try:
            d = json.load(open(erj))
            ev = d.get("eval", {})
            n = len(ev)
            bp = sum(1 for v in ev.values() if (v.get("base") or [[None]])[0][0] == "pass")
            pp = sum(1 for v in ev.values() if (v.get("plus") or [[None]])[0][0] == "pass")
            base, plus = (bp / n if n else None), (pp / n if n else None)
        except Exception as e:
            log("    score parse FAILED: %s\n      tail: %s" % (e, out[-500:]))
    return base, plus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=A2)
    ap.add_argument("--problems", default="/srv/ml/scripts/he_plus_problems.json")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=12288)
    ap.add_argument("--attn", default="sdpa", help="sdpa (memory-light, default) or eager")
    ap.add_argument("--limit", type=int, default=0, help="0=all 164; small for scorer smoke")
    ap.add_argument("--configs", default="")
    ap.add_argument("--out-root", default="/srv/ml/eval_results_tracks_2_3/t176_phase3/decode_quality/he")
    args = ap.parse_args()

    cfgs = CONFIGS if not args.configs else [c for c in CONFIGS if c[0] in args.configs.split(",")]
    os.makedirs(args.out_root, exist_ok=True)
    probs = json.load(open(args.problems))
    tids = list(probs)
    if args.limit:
        tids = tids[:args.limit]
    log("model=%s  problems=%d  configs=%s  max_new=%d" % (
        args.model, len(tids), [c[0] for c in cfgs], args.max_new))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation=args.attn, device_map={"": 0}).eval()

    summary = {}
    for tag, rp, ng in cfgs:
        log("===== config %s (rep_penalty=%.2f no_repeat_ngram=%d) =====" % (tag, rp, ng))
        gk = dict(max_new_tokens=args.max_new, do_sample=False, use_cache=True,
                  repetition_penalty=rp, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if ng > 0:
            gk["no_repeat_ngram_size"] = ng
        samples = []
        loop_hits = 0
        t0 = time.time()
        for i in range(0, len(tids), args.bs):
            chunk = tids[i:i + args.bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": instruction(probs[t])}],
                                             add_generation_prompt=True, tokenize=False) for t in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, **gk)
            for j, t in enumerate(chunk):
                new = o[j][enc["input_ids"].shape[1]:]
                content = tok.decode(new, skip_special_tokens=True)
                sol = extract_code_block(content)
                samples.append({"task_id": t, "solution": sol})
                if detect_loop(content):
                    loop_hits += 1
            log("    gen %d/%d (%.0fs)" % (min(i + args.bs, len(tids)), len(tids), time.time() - t0))
        sp = os.path.join(args.out_root, "samples_%s.jsonl" % tag)
        with open(sp, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        log("    scoring %s via evalplus …" % tag)
        base, plus = score_evalplus(sp)
        summary[tag] = {"repetition_penalty": rp, "no_repeat_ngram_size": ng,
                        "he_base_pass1": base, "he_plus_pass1": plus,
                        "raw_loop_rate": round(loop_hits / max(len(tids), 1), 4),
                        "n": len(tids), "wall_s": round(time.time() - t0, 0)}
        json.dump(summary, open(os.path.join(args.out_root, "summary.json"), "w"), indent=1)
        log("  DONE %s: HE base=%s HE+ plus=%s  raw_loop=%.3f  (%.0fs)" % (
            tag, base, plus, summary[tag]["raw_loop_rate"], summary[tag]["wall_s"]))

    log("=" * 60)
    log("%-10s %10s %10s %9s" % ("config", "HE_base", "HE+_plus", "loop%"))
    for tag, _, _ in cfgs:
        if tag not in summary:
            continue
        s = summary[tag]
        log("%-10s %10s %10s %8.1f%%" % (
            tag, ("%.4f" % s["he_base_pass1"]) if s["he_base_pass1"] is not None else "FAIL",
            ("%.4f" % s["he_plus_pass1"]) if s["he_plus_pass1"] is not None else "FAIL",
            100 * s["raw_loop_rate"]))
    log("DQ_HE_DONE -> %s/summary.json" % args.out_root)


if __name__ == "__main__":
    main()
