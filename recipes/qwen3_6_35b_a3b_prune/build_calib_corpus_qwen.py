#!/usr/bin/env python3
"""Build a Qwen-templated, domain-balanced calibration corpus for the
Qwen3.6-35B-A3B expert-competence profiler.

Pulls raw questions (+ reference solutions where available) straight from the
cached HF benchmark datasets, renders each with the *Qwen* chat template
(question -> assistant solution, or question-only for instruction/codegen
benches), balances per bench, and emits JSONL rows `{bench, text, n_tok}`.

Why a dedicated corpus: the profiler ranks experts by routing frequency (tc),
so it must see domain-representative, natively-templated inputs. The existing
`scripts/router_calib_corpus*.jsonl` are Gemma-templated (`<bos><|turn>`) and
carry pass/fail anchoring machinery irrelevant to a competence map. Domains
cover the 9-bench eval suite: science (GPQA), math (GSM8K/MATH500/AIME),
code (HumanEval/MBPP/LCB), reasoning (ARC), instruction (IFEval).

Feed to the producer WITHOUT --corpus-apply-template (text is already templated):
  expert_neuron_analysis_v5_targeted.py --model <qwen> --device cuda:0 \
      --corpus <this>.jsonl --corpus-cat-field bench --out <competence>.json

Run (CPU, offline, on a box with the HF dataset cache + the Qwen tokenizer):
  HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python build_calib_corpus_qwen.py \
      --tokenizer /path/to/Qwen3.6-35B-A3B --per-bench 80 \
      --out results/router_calib_corpus_qwen.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


# ── per-bench dataset loaders + (user, assistant|None) extractors ─────────────
def _load(name, config, split):
    return (lambda: load_dataset(name, config, split=split)) if config else \
           (lambda: load_dataset(name, split=split))


def _arc_extract(r):
    ch = r["choices"]
    body = "\n".join(f"{lab}. {txt}" for lab, txt in zip(ch["label"], ch["text"]))
    correct = dict(zip(ch["label"], ch["text"])).get(r["answerKey"], r["answerKey"])
    q = f"{r['question']}\n{body}"
    return q, f"The correct answer is {r['answerKey']}. {correct}"


def _gpqa_extract(r):
    a = (r.get("Explanation") or "").strip()
    a = (a + "\n\n" if a else "") + f"The correct answer is: {r['Correct Answer']}"
    return r["Question"], a


def _math_extract(r):
    a = (r.get("solution") or "").strip()
    if r.get("answer"):
        a = (a + "\n\n" if a else "") + f"The answer is ${r['answer']}$."
    return r["problem"], a


def _gsm8k_extract(r):
    return r["question"], (r.get("answer") or "").strip()


def _humaneval_extract(r):
    # completion = the canonical body; give the assistant the full function.
    return r["prompt"], (r["prompt"] + r["canonical_solution"])


def _mbpp_extract(r):
    tests = r.get("test_list") or []
    hint = ("\n\nYour code should satisfy:\n" + "\n".join(tests[:2])) if tests else ""
    return (r["text"] + hint), r["code"]


def _ifeval_extract(r):
    return r["prompt"], None  # instruction-following: prompt is the input, no ref answer


def _aime_extract(r):
    a = (r.get("Solution") or "").strip()
    if r.get("Answer") is not None:
        a = (a + "\n\n" if a else "") + f"The answer is {r['Answer']}."
    return r["Problem"], a


def _lcb_extract(r):
    return (r.get("question_content") or r.get("question") or ""), None


BENCHES = {
    "gpqa_diamond":  (_load("Idavidrein/gpqa", "gpqa_diamond", "train"), _gpqa_extract),
    "math500":       (_load("HuggingFaceH4/MATH-500", None, "test"), _math_extract),
    "gsm8k":         (_load("openai/gsm8k", "main", "test"), _gsm8k_extract),
    "humaneval":     (_load("openai/openai_humaneval", None, "test"), _humaneval_extract),
    "mbpp":          (_load("google-research-datasets/mbpp", "full", "test"), _mbpp_extract),
    "arc_challenge": (_load("allenai/ai2_arc", "ARC-Challenge", "test"), _arc_extract),
    "ifeval":        (_load("google/IFEval", None, "train"), _ifeval_extract),
    "aime2024":      (_load("Maxwell-Jia/aime_2024", None, "train"), _aime_extract),
    # LCB is best-effort: config name varies by release; skipped if unavailable.
    "lcb":           (None, _lcb_extract),
}
_LCB_CONFIGS = ["release_v1", "release_v2", "release_v3", "release_v4",
                "release_v5", "release_v6", "release_latest"]


def _load_lcb():
    for cfg in _LCB_CONFIGS:
        try:
            return load_dataset("livecodebench/code_generation_lite", cfg, split="test")
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Qwen model/tokenizer dir.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bench", type=int, default=80,
                    help="Rows per bench (capped by availability).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benches", default=None,
                    help="Comma list to restrict (default: all).")
    ap.add_argument("--max-assistant-chars", type=int, default=6000,
                    help="Truncate reference solutions to bound pathological rows "
                         "(the profiler chunks anyway).")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    want = [b.strip() for b in args.benches.split(",")] if args.benches else list(BENCHES)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, per_bench_tok = [], {}
    for bench in want:
        if bench not in BENCHES:
            print(f"[skip] unknown bench {bench}")
            continue
        loader, extract = BENCHES[bench]
        try:
            ds = _load_lcb() if bench == "lcb" else loader()
        except Exception as e:
            print(f"[{bench}] load ERR {type(e).__name__}: {str(e)[:100]} — skipped")
            continue
        if ds is None:
            print(f"[{bench}] unavailable — skipped")
            continue
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:args.per_bench]
        n, ntok = 0, 0
        for i in idxs:
            try:
                user, asst = extract(ds[i])
            except Exception:
                continue
            if not user or not str(user).strip():
                continue
            user = str(user).strip()
            if asst:
                asst = str(asst).strip()[:args.max_assistant_chars]
                msgs = [{"role": "user", "content": user},
                        {"role": "assistant", "content": asst}]
                text = tok.apply_chat_template(msgs, tokenize=False)
            else:
                msgs = [{"role": "user", "content": user}]
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
            n_tok = len(tok(text)["input_ids"])
            rows.append({"bench": bench, "text": text, "n_tok": n_tok})
            n += 1
            ntok += n_tok
        per_bench_tok[bench] = (n, ntok)
        print(f"[{bench}] {n} rows, {ntok} tok (avg {ntok/max(n,1):.0f})")

    rng.shuffle(rows)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total_tok = sum(t for _, t in per_bench_tok.values())
    print(f"\n[corpus] wrote {out_path}  {len(rows)} rows, {total_tok} tokens, "
          f"{len(per_bench_tok)} benches")
    print("[corpus] per-bench: "
          + ", ".join(f"{b}={n}" for b, (n, _) in per_bench_tok.items()))


if __name__ == "__main__":
    main()
