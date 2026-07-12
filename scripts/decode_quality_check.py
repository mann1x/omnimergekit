#!/usr/bin/env python3
"""T184 decode-fix QUALITY check on A2 (62e) — does anti-repetition decoding cost code quality?

T182 showed the loop floor collapses under HF-generate anti-repetition ops:
  base (rp1.0, ng0)  -> 15.5% total / 35% multilingual / 7.5% constrained
  rp1.10             ->  7.5% / 8.3% / 7.5%
  no_repeat_ngram=3  ->  0.5% / 1.7% / 0.0%
T180/T181/T183 falsified every router-math fix (survivors carry normal mass; no
orphaning), so the decoder is the only 0%-loops lever. The open question before
shipping: an n-gram ban forbids EVERY legit repeated n-gram (code indentation,
`let mut`, braces, repeated identifiers) -> does it dent code quality?

This measures HE+/MPE under three decode configs on the SAME HF-generate path that
produced the loop numbers, so the comparison isolates the decode op. CRITICAL: the
in-harness `base` config is the apples-to-apples anchor (same backend/path); the
published A2 anchors (HE+ 0.909 vLLM / MPE 0.767 llama.cpp) used different backends
+ reasoning-parser + thinking budget, so absolute-vs-anchor is secondary and
backend-confounded. We read base vs ng3 vs rp11ng4 WITHIN this harness.

SERVING CAVEAT: no_repeat_ngram_size is an HF transformers.generate feature, NOT a
llama.cpp/vLLM sampling param. ng3/ng4 ship only via a custom HF-generate endpoint
or a DRY-sampler approximation; repetition_penalty alone IS serveable on both.

MPE: reuse the T179 problem set (name/prompt/tests/stop_tokens for rs+java+js x100),
regenerate ONLY the completion via chat HF-generate + the omk runner's exact
extract_code_block/chat_to_body, then score with the same multipl_e_evaluate.sh
Docker image. HE+ phase is separate (needs evalplus) -> decode_quality_he.py.

Run on bs2 GPU0, omk python.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OMK_MPE = "/srv/ml/repos/omnimergekit/eval/multipl_e"
sys.path.insert(0, OMK_MPE)
sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from multipl_e_generate import extract_code_block, chat_to_body  # noqa: E402
from audit_full_bench import detect_loop  # noqa: E402

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
EVAL_SH = os.path.join(OMK_MPE, "multipl_e_evaluate.sh")
# T179 problem set: per-problem JSONs already carry prompt/tests/stop_tokens.
SRC_GEN = "/srv/ml/eval_results_t179/v6coder_q4km/multipl_e_100/v6coder_q4km/generations"
LANGS = ["rs", "java", "js"]
# (tag, repetition_penalty, no_repeat_ngram_size)
CONFIGS = [("base", 1.0, 0), ("ng3", 1.0, 3), ("rp11ng4", 1.1, 4), ("rp105", 1.05, 0), ("rp11", 1.10, 0), ("rp115", 1.15, 0)]


def log(m):
    print("[dq %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def load_problems(limit):
    probs = defaultdict(list)
    for lang in LANGS:
        d = os.path.join(SRC_GEN, "humaneval-%s" % lang)
        files = sorted(os.listdir(d))
        for fn in files:
            if not fn.endswith(".json"):
                continue
            doc = json.load(open(os.path.join(d, fn)))
            probs[lang].append({"name": doc["name"], "prompt": doc["prompt"],
                                "tests": doc.get("tests", ""),
                                "stop_tokens": list(doc.get("stop_tokens") or [])})
        if limit:
            probs[lang] = probs[lang][:limit]
    return probs


def chat_instruction(prompt, lang):
    # verbatim from multipl_e_generate.make_chat_request
    return ("Complete the following %s function. Reply with ONLY the complete "
            "function implementation in a single Markdown code block — include the "
            "signature exactly as given, write the full body, and do NOT add any "
            "explanation, example usage, or test code.\n\n```%s\n%s\n```" % (lang, lang, prompt))


def gen_batch(model, tok, instructions, gk, bs):
    outs = []
    for i in range(0, len(instructions), bs):
        chunk = instructions[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": c}],
                                         add_generation_prompt=True, tokenize=False) for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, **gk)
        for j in range(len(chunk)):
            new = o[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def score_lang(gen_dir, out_dir):
    """Run the omk Docker scorer; return (n_pass, n_total, pass_at_1) from _summary.json."""
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run(["bash", EVAL_SH, gen_dir, out_dir], capture_output=True, text=True)
    sp = os.path.join(out_dir, "_summary.json")
    if not os.path.exists(sp):
        log("    SCORER FAILED %s rc=%d\n      stdout: %s\n      stderr: %s" % (
            gen_dir, r.returncode, r.stdout[-400:], r.stderr[-400:]))
        return (0, 0, None)
    s = json.load(open(sp))
    return (s["n_pass"], s["n_total"], s["pass_at_1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=A2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="problems/lang (0=all 100); use 2-3 for scorer smoke")
    ap.add_argument("--out-root", default="/srv/ml/eval_results_tracks_2_3/t176_phase3/decode_quality/mpe")
    ap.add_argument("--configs", default="", help="comma tags subset, e.g. base,ng3")
    args = ap.parse_args()

    cfgs = CONFIGS if not args.configs else [c for c in CONFIGS if c[0] in args.configs.split(",")]
    os.makedirs(args.out_root, exist_ok=True)
    probs = load_problems(args.limit)
    nprob = sum(len(v) for v in probs.values())
    log("model=%s  problems=%d (%s)  configs=%s  max_new=%d" % (
        args.model, nprob, {k: len(v) for k, v in probs.items()}, [c[0] for c in cfgs], args.max_new))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0}).eval()

    summary = {}
    for tag, rp, ng in cfgs:
        log("===== config %s (rep_penalty=%.2f no_repeat_ngram=%d) =====" % (tag, rp, ng))
        gk = dict(max_new_tokens=args.max_new, do_sample=False, use_cache=True,
                  repetition_penalty=rp, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if ng > 0:
            gk["no_repeat_ngram_size"] = ng
        per_lang = {}
        loop_hits = 0
        loop_tot = 0
        t0 = time.time()
        for lang in LANGS:
            items = probs[lang]
            instrs = [chat_instruction(it["prompt"], lang) for it in items]
            raw = gen_batch(model, tok, instrs, gk, args.bs)
            gen_dir = os.path.join(args.out_root, tag, "humaneval-%s" % lang)
            os.makedirs(gen_dir, exist_ok=True)
            for it, content in zip(items, raw):
                code = extract_code_block(content)
                body = chat_to_body(it["prompt"], code, it["stop_tokens"])
                json.dump({"name": it["name"], "language": lang, "prompt": it["prompt"],
                           "completions": [body], "tests": it["tests"],
                           "stop_tokens": it["stop_tokens"]},
                          open(os.path.join(gen_dir, "%s.json" % it["name"]), "w"))
                if detect_loop(content):
                    loop_hits += 1
                loop_tot += 1
            log("    %s gen done (%d problems, %.0fs cum)" % (lang, len(items), time.time() - t0))
        # score each lang via docker
        langres = {}
        for lang in LANGS:
            gen_dir = os.path.join(args.out_root, tag, "humaneval-%s" % lang)
            out_dir = os.path.join(args.out_root, tag, "results-%s" % lang)
            np_, nt_, p1 = score_lang(gen_dir, out_dir)
            langres[lang] = {"n_pass": np_, "n_total": nt_, "pass_at_1": p1}
            log("    [%s/%s] pass@1 = %s/%s = %s" % (tag, lang, np_, nt_,
                                                     ("%.4f" % p1) if p1 is not None else "FAIL"))
        valid = [v["pass_at_1"] for v in langres.values() if v["pass_at_1"] is not None]
        macro = round(sum(valid) / len(valid), 4) if valid else None
        per_lang = langres
        summary[tag] = {"repetition_penalty": rp, "no_repeat_ngram_size": ng,
                        "per_lang": per_lang, "macro_mean_pass_at_1": macro,
                        "raw_loop_rate": round(loop_hits / max(loop_tot, 1), 4),
                        "wall_s": round(time.time() - t0, 0)}
        json.dump(summary, open(os.path.join(args.out_root, "summary.json"), "w"), indent=1)
        log("  DONE %s: macro pass@1=%s  raw_loop=%.3f  (%.0fs)" % (
            tag, macro, summary[tag]["raw_loop_rate"], summary[tag]["wall_s"]))

    log("=" * 64)
    log("%-10s %8s %8s %8s %10s %9s" % ("config", "rs", "java", "js", "macro", "loop%"))
    for tag, _, _ in cfgs:
        if tag not in summary:
            continue
        s = summary[tag]
        pl = s["per_lang"]
        def g(la):
            return "%.3f" % pl[la]["pass_at_1"] if pl[la]["pass_at_1"] is not None else "FAIL"
        log("%-10s %8s %8s %8s %10s %8.1f%%" % (
            tag, g("rs"), g("java"), g("js"),
            ("%.4f" % s["macro_mean_pass_at_1"]) if s["macro_mean_pass_at_1"] is not None else "FAIL",
            100 * s["raw_loop_rate"]))
    log("DQ_MPE_DONE -> %s/summary.json" % args.out_root)


if __name__ == "__main__":
    main()
