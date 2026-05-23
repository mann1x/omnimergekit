"""canary_ifeval_rumination4.py — minimal 4-prompt IFEval canary for vLLM stack debugging.

Targets 4 IFEval doc_ids that are bit-exact stack-sensitive on Gemma 4
pruned-MoE variants (98e v5-coder NVFP4A16, validated 2026-05-22):

  doc 18  — "Are hamburgers sandwiches?" + Kannada-only constraint
  doc 31  — "Write a rubric…" + Punjabi-only constraint
  doc 50  — "Write a haiku about rushing to work" + Marathi-only constraint
  doc 59  — "Short proposal on language evolution" + (no commas, <1× letter c, ≥250 words)

Baseline char counts (v5-coder × stack@1, recorded on pod 37006213 L40, all
PASS prompt_level_strict_acc):  18→28, 31→1591, 50→32, 59→1498.

Stack@2 (vLLM main 68e07d591 + Fix-E cherry-pick + lm-eval Fix-A) regressed
these to 13635 / 1795 / 10398 / 18292 — three of four are bit-exact
deterministic repetition loops on greedy decoding. The same weights, same
prompts, same greedy sampler. Hence: stack-induced.

Usage:
    python canary_ifeval_rumination4.py \
        --base-url http://localhost:8195/v1 \
        --served-name 98e_v5_coder_nvfp4a16 \
        [--out canary_result.json] \
        [--strict]

Exit codes:
    0  ALL_PASS — all 4 char counts within healthy bands
    2  ANY_FAIL — at least one rumination signature detected
    3  SETUP_ERROR — server unreachable, malformed response, etc.

Healthy bands (PASS):
    doc 18:  chars ≤ 200      (baseline 28 — short Kannada answer)
    doc 31:  chars ≤ 4000     (baseline 1591 — verbose Punjabi rubric is natural)
    doc 50:  chars ≤ 200      (baseline 32 — Marathi haiku is short by definition)
    doc 59:  chars ≤ 4000     (baseline 1498 — 250-word essay)

Additionally a script-purity check on docs 18/31/50: any answer chars must
be in the target Unicode script (Kannada / Gurmukhi / Devanagari). Non-target
script characters above a tiny tolerance (5% allowing ASCII punctuation /
digits and small English glosses) indicate multilingual contamination —
e.g. the stack@2 doc 31 Punjabi rubric had Lao + Greek + Japanese fragments
mixed in.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests required (pip install requests)", file=sys.stderr)
    sys.exit(3)


CANARY_DOCS = [
    {
        "doc_id": 18,
        "prompt": "Are hamburgers sandwiches? Please respond using only the Kannada language, no other language is allowed.",
        "instruction_id_list": ["language:response_language"],
        "language_constraint": "kn",
        "stack1_baseline_chars": 28,
        "max_chars_healthy": 200,
        "expected_script_blocks": ["KANNADA"],
    },
    {
        "doc_id": 31,
        "prompt": "Write a rubric for how to evaluate the technical skills of a software engineer only using the Punjabi language, no other language is allowed.",
        "instruction_id_list": ["language:response_language"],
        "language_constraint": "pa",
        "stack1_baseline_chars": 1591,
        "max_chars_healthy": 4000,
        "expected_script_blocks": ["GURMUKHI"],
    },
    {
        "doc_id": 50,
        "prompt": "Write a haiku about rushing to work using only the Marathi language, no other language is allowed.",
        "instruction_id_list": ["language:response_language"],
        "language_constraint": "mr",
        "stack1_baseline_chars": 32,
        "max_chars_healthy": 200,
        "expected_script_blocks": ["DEVANAGARI"],
    },
    {
        "doc_id": 59,
        "prompt": (
            "Write a short proposal for a new research project that investigates "
            "how language evolves over time. I want to make it challenging, so:\n"
            "1. Do not include any commas in your response.\n"
            "2. Do not include the letter \"c\" anywhere in your response.\n"
            "3. Your response should contain at least 250 words."
        ),
        "instruction_id_list": [
            "punctuation:no_comma",
            "keywords:letter_frequency",
            "length_constraints:number_words",
        ],
        "language_constraint": None,
        "stack1_baseline_chars": 1498,
        "max_chars_healthy": 4000,
        "expected_script_blocks": None,
    },
]


def script_block_name(ch: str) -> str:
    """Return the Unicode block name of a character, e.g. 'KANNADA' for ಕ."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"
    # First token of the canonical name is typically the script ('KANNADA LETTER KA').
    return name.split()[0] if name else "UNKNOWN"


def script_purity_check(text: str, expected_blocks: list[str]) -> dict:
    """Count characters outside expected script blocks.

    Allowed without penalty:
      - ASCII whitespace, punctuation, digits (legitimate formatting)
      - Common European punctuation (em-dash, smart quotes)
      - Newlines, control chars

    Counted as foreign:
      - Letters from any non-target script (LATIN, GREEK, CJK, LAO, etc.)
    """
    foreign = 0
    foreign_examples = []
    total_letters = 0
    for ch in text:
        if ch.isspace() or ch.isdigit():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S") or cat.startswith("C") or cat.startswith("Z"):
            continue
        if cat.startswith("L"):  # any letter
            total_letters += 1
            block = script_block_name(ch)
            if not any(exp in block for exp in expected_blocks):
                foreign += 1
                if len(foreign_examples) < 8:
                    foreign_examples.append((ch, block))
    rate = (foreign / total_letters) if total_letters > 0 else 0.0
    return {
        "total_letters": total_letters,
        "foreign_letters": foreign,
        "foreign_rate": rate,
        "foreign_examples": [{"char": c, "block": b} for c, b in foreign_examples],
    }


def send_chat(base_url: str, served_name: str, prompt: str, *,
              max_tokens: int = 16384, thinking_budget: int = 12288,
              timeout: int = 1200) -> dict:
    """Single greedy chat completion with gemma4 reasoning parser + enable_thinking."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": served_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
        "extra_body": {
            "thinking_token_budget": thinking_budget,
        },
    }
    # Some vLLM builds want thinking_token_budget at top level rather than extra_body
    payload["thinking_token_budget"] = thinking_budget
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    r.raise_for_status()
    return {"response": r.json(), "elapsed_sec": elapsed}


def classify_chars(chars: int, baseline: int, max_healthy: int) -> str:
    """CLEAN / LONG / RUMINATE classifier."""
    if chars <= max_healthy:
        return "CLEAN"
    if chars <= max_healthy * 3:
        return "LONG"
    return "RUMINATE"


def run_canary(base_url: str, served_name: str, max_tokens: int,
               thinking_budget: int, timeout: int) -> list[dict]:
    results = []
    for doc in CANARY_DOCS:
        print(f"\n=== doc {doc['doc_id']} ===", flush=True)
        print(f"  prompt: {doc['prompt'][:120]!r}", flush=True)
        try:
            out = send_chat(base_url, served_name, doc["prompt"],
                            max_tokens=max_tokens, thinking_budget=thinking_budget,
                            timeout=timeout)
        except Exception as e:
            results.append({
                "doc_id": doc["doc_id"],
                "error": f"{type(e).__name__}: {e}",
                "verdict": "ERROR",
            })
            print(f"  ERROR: {e}", flush=True)
            continue

        resp = out["response"]
        elapsed = out["elapsed_sec"]
        msg = resp["choices"][0].get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish = resp["choices"][0].get("finish_reason")
        usage = resp.get("usage", {})

        chars_content = len(content)
        chars_reasoning = len(reasoning)
        verdict_chars = classify_chars(chars_content, doc["stack1_baseline_chars"],
                                       doc["max_chars_healthy"])

        purity = None
        if doc["expected_script_blocks"]:
            purity = script_purity_check(content, doc["expected_script_blocks"])
            # 5% foreign tolerance — small English glosses allowed.
            verdict_purity = "OK" if purity["foreign_rate"] <= 0.05 else "CONTAMINATED"
        else:
            verdict_purity = "N/A"

        overall = "PASS"
        if verdict_chars != "CLEAN":
            overall = "FAIL"
        if verdict_purity == "CONTAMINATED":
            overall = "FAIL"

        result = {
            "doc_id": doc["doc_id"],
            "instruction_id_list": doc["instruction_id_list"],
            "language_constraint": doc["language_constraint"],
            "elapsed_sec": round(elapsed, 2),
            "finish_reason": finish,
            "usage": usage,
            "chars_content": chars_content,
            "chars_reasoning": chars_reasoning,
            "stack1_baseline_chars": doc["stack1_baseline_chars"],
            "max_chars_healthy": doc["max_chars_healthy"],
            "verdict_chars": verdict_chars,
            "verdict_purity": verdict_purity,
            "purity_detail": purity,
            "content_tail_150": content[-150:],
            "content_head_150": content[:150],
            "verdict": overall,
        }
        results.append(result)
        print(f"  chars_content={chars_content} (baseline {doc['stack1_baseline_chars']}, "
              f"healthy ≤{doc['max_chars_healthy']}) → {verdict_chars}", flush=True)
        if verdict_purity != "N/A":
            print(f"  script_purity: foreign={purity['foreign_letters']}/"
                  f"{purity['total_letters']} ({purity['foreign_rate']*100:.1f}%) → {verdict_purity}",
                  flush=True)
            if purity['foreign_examples']:
                print(f"    foreign examples: {purity['foreign_examples']}", flush=True)
        print(f"  finish={finish} elapsed={elapsed:.1f}s → OVERALL: {overall}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument("--base-url", default="http://localhost:8195/v1",
                    help="vLLM OpenAI base URL (default: http://localhost:8195/v1)")
    ap.add_argument("--served-name", required=True,
                    help="vLLM --served-model-name (e.g. 98e_v5_coder_nvfp4a16)")
    ap.add_argument("--out", default=None, help="optional path to write JSON results")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="max_tokens per request (default 16384, matches ifeval_100 template)")
    ap.add_argument("--thinking-budget", type=int, default=12288,
                    help="thinking_token_budget (default 12288, matches ifeval_100 template)")
    ap.add_argument("--timeout", type=int, default=1200, help="HTTP timeout per request")
    ap.add_argument("--strict", action="store_true",
                    help="exit 2 if any single doc fails (default behavior is the same)")
    args = ap.parse_args()

    print("=== IFEval rumination-trigger canary (4 docs) ===")
    print(f"server: {args.base_url}  served: {args.served_name}")
    print(f"settings: max_tokens={args.max_tokens} thinking_budget={args.thinking_budget}")

    results = run_canary(args.base_url, args.served_name, args.max_tokens,
                         args.thinking_budget, args.timeout)

    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
    n_err = sum(1 for r in results if r["verdict"] == "ERROR")

    print("\n=== SUMMARY ===")
    print(f"{'doc':>4} | {'chars':>7} | {'baseline':>8} | {'classify':>9} | {'purity':>13} | verdict")
    print("-" * 80)
    for r in results:
        if r["verdict"] == "ERROR":
            print(f"{r['doc_id']:>4} | {'ERR':>7} | {'-':>8} | {'-':>9} | {'-':>13} | ERROR ({r.get('error','?')[:30]})")
        else:
            print(f"{r['doc_id']:>4} | {r['chars_content']:>7} | "
                  f"{r['stack1_baseline_chars']:>8} | {r['verdict_chars']:>9} | "
                  f"{r['verdict_purity']:>13} | {r['verdict']}")

    print(f"\nTOTAL: pass={n_pass}/4 fail={n_fail}/4 error={n_err}/4")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "base_url": args.base_url,
            "served_name": args.served_name,
            "max_tokens": args.max_tokens,
            "thinking_budget": args.thinking_budget,
            "results": results,
            "summary": {"pass": n_pass, "fail": n_fail, "error": n_err},
        }, indent=2, ensure_ascii=False))
        print(f"wrote: {args.out}")

    if n_err:
        sys.exit(3)
    if n_fail:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
