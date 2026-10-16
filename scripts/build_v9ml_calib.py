#!/usr/bin/env python3
"""v9-ml — Tier-A calibration prompts for a multilingual + IFEval-targeted v9-agentic map.

WHY: gemma-4-20B-v9-agentic (map A) scores IFEval-100 0.90 (128e 0.97). Its misses are
low-resource-language answers in mixed script (kn, pa, mr), repetition loops and hard
format constraints. Map A has no multilingual or instruction-following class, so the
experts that carry those are never protected. This emits two Tier-A categories:

  multilingual  every IFEval "respond only in <lang>" prompt in the HELD-OUT rows, plus
                native-language aya prompts, weighted to the failing (Indic/Persian) scripts
  ifeval        held-out IFEval prompts with hard format constraints, weighted to the
                instruction types map A failed

HOLD-OUT (the whole point): omk's ifeval_100 cell runs IFEval rows 0-99 (--limit 100), and
the ifeval_v9fail10 probe lives there too. Every IFEval prompt below comes from rows
100-540; the script asserts it. A calib prompt inside rows 0-99 would let the variant be
scored on prompts it was selected on.

Deterministic: stable sorts + fixed stride, no RNG. Output JSONL {"prompt", "src", "lang"}.
"""
import argparse
import hashlib
import json
import sqlite3  # noqa: F401  must precede torch (datasets) on solidpc: feedback_torch_before_sqlite3_cxxabi

from datasets import load_dataset

HOLDOUT_ROWS = 100  # IFEval rows [0, HOLDOUT_ROWS) are the test population

# aya language -> prompts per language. Failing scripts in map A's run get more.
AYA_LANGS = {
    "Kannada": 5, "Panjabi": 5, "Marathi": 5, "Iranian Persian": 4,
    "Gujarati": 3, "Bengali": 3, "Hindi": 3, "Tamil": 3, "Telugu": 3, "Malayalam": 3,
    "Nepali": 3, "Urdu": 3, "Sinhala": 3,
    "Korean": 2, "Japanese": 2, "Simplified Chinese": 2, "Thai": 2, "Burmese": 2,
    "Standard Arabic": 2, "Swahili": 2, "Amharic": 2, "Russian": 2, "Vietnamese": 2,
    "Turkish": 2,
}
AYA_MIN_CHARS, AYA_MAX_CHARS = 40, 600

# instruction types map A failed in ifeval_100 (2026-10-15) — sampled first
FAILED_TYPES = {
    "detectable_format:number_highlighted_sections", "keywords:letter_frequency",
    "detectable_format:json_format", "punctuation:no_comma",
    "length_constraints:number_words", "length_constraints:number_paragraphs",
    "length_constraints:number_sentences", "change_case:capital_word_frequency",
    "startend:quotation", "keywords:existence",
}


def stride_pick(items, k):
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-multilingual", required=True)
    ap.add_argument("--out-ifeval", required=True)
    ap.add_argument("--n-ifeval-failed-types", type=int, default=36)
    ap.add_argument("--n-ifeval-other", type=int, default=12)
    a = ap.parse_args()

    ifv = load_dataset("google/IFEval", split="train")
    test_keys = set(ifv.select(range(HOLDOUT_ROWS))["key"])
    pool = [ifv[i] for i in range(HOLDOUT_ROWS, len(ifv))]

    ml, ife = [], []
    lang_rows = [d for d in pool if "language:response_language" in d["instruction_id_list"]]
    for d in sorted(lang_rows, key=lambda d: d["key"]):
        lang = d["kwargs"][d["instruction_id_list"].index("language:response_language")]["language"]
        ml.append({"prompt": d["prompt"], "src": f"ifeval:{d['key']}", "lang": lang})

    aya = load_dataset("CohereForAI/aya_dataset", split="train")
    for lang, k in AYA_LANGS.items():
        cands = sorted({x for x, lg in zip(aya["inputs"], aya["language"])
                        if lg == lang and AYA_MIN_CHARS <= len(x) <= AYA_MAX_CHARS})
        if len(cands) < k:
            raise SystemExit(f"aya {lang}: only {len(cands)} candidates, need {k}")
        for x in stride_pick(cands, k):
            ml.append({"prompt": x, "src": "aya", "lang": lang})

    rest = sorted((d for d in pool if "language:response_language" not in d["instruction_id_list"]),
                  key=lambda d: d["key"])
    hit = [d for d in rest if FAILED_TYPES & set(d["instruction_id_list"])]
    other = [d for d in rest if not FAILED_TYPES & set(d["instruction_id_list"])]
    for d in stride_pick(hit, a.n_ifeval_failed_types) + stride_pick(other, a.n_ifeval_other):
        ife.append({"prompt": d["prompt"], "src": f"ifeval:{d['key']}", "lang": None})

    # HOLD-OUT gate: no IFEval prompt from the test rows, by key AND by text.
    test_text = {ifv[i]["prompt"] for i in range(HOLDOUT_ROWS)}
    for r in ml + ife:
        if r["src"].startswith("ifeval:"):
            assert int(r["src"].split(":")[1]) not in test_keys, r["src"]
        assert r["prompt"] not in test_text, f"test prompt leaked: {r['prompt'][:80]!r}"
    if len({r["prompt"] for r in ml + ife}) != len(ml + ife):
        raise SystemExit("duplicate prompts across categories")

    for path, rows in ((a.out_multilingual, ml), (a.out_ifeval, ife)):
        blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        open(path, "wb").write(blob)
        print(f"{path}: {len(rows)} prompts sha256={hashlib.sha256(blob).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
