#!/usr/bin/env python3
"""T176 — build the curated MULTILINGUAL Tier-A calibration prompt set.

WHY: T175 showed the residual ~3% degenerate-loop floor of the 62e prune is a
low-resource-language + hard-format-constraint competence collapse ("respond
entirely in Korean/Swahili/Persian/French" → degenerate repetition). The
competence maps that drive the prune have NO multilingual category, so the
experts carrying those languages are never protected. This emits the prompt set
for a new `generic_multilingual` Tier-A category so the rebuilt maps measure —
and the drop map can protect — those experts.

SOURCE (targeted, not generic): the antiloop corpus is itself the failure
distribution. Its `multilingual` bucket is aya native-language inputs (Persian,
Arabic, Turkish, Hindi, Chinese, French, …) and its `constrained` bucket holds
the exact IFEval "respond only in <language>" prompts that loop (incl. Korean +
Swahili, which the corpus's aya language list omitted). We sample both, plus the
handful of verbatim loop seeds. Optional `--aya-per-lang` augments native-
language coverage for the specific failing langs straight from CohereForAI/aya.

Deterministic (sorted + fixed stride, no unseeded RNG) so the calib set is
reproducible. Output JSONL: {"prompt": str, "src": str, "lang": str|null}.
"""
import argparse
import json
import re
from collections import Counter

# Language names used to (a) tag prompts and (b) detect language-constrained
# IFEval prompts in the `constrained` bucket. Korean + Swahili FIRST — they are
# the failing langs the antiloop corpus's aya filter missed.
LANG_NAMES = [
    "Korean", "Swahili", "Persian", "Farsi", "Hindi", "Arabic", "Turkish",
    "Chinese", "Japanese", "French", "Spanish", "German", "Italian",
    "Portuguese", "Russian", "Vietnamese", "Thai", "Polish", "Dutch", "Greek",
    "Hebrew", "Bengali", "Tamil", "Urdu", "Indonesian", "Punjabi", "Telugu",
]
# Failing langs to augment natively from aya (--aya-per-lang). aya uses English
# language names in its `language` column.
AYA_LANGS = ["Korean", "Swahili", "Persian", "Hindi", "Arabic", "Turkish",
             "Chinese", "Japanese", "French"]
_LANG_RE = re.compile("|".join(re.escape(n) for n in LANG_NAMES), re.IGNORECASE)
# Phrases that mark a language/format constraint even without a named language.
_CONSTRAINT_RE = re.compile(
    r"\b(in (the )?[A-Z][a-z]+ language|respond (only|entirely)|no other language|"
    r"language only|in [A-Z][a-z]+ only|entirely in)\b")


def tag_lang(text):
    m = _LANG_RE.search(text)
    return m.group(0).title() if m else None


def deterministic_pick(items, k):
    """Evenly-spaced deterministic sample of k from a stable-sorted list."""
    items = sorted(set(items))
    if len(items) <= k:
        return items
    stride = len(items) / k
    return [items[int(i * stride)] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/mnt/sdc/ml/corpora/antiloop_sft_corpus_v2.jsonl")
    ap.add_argument("--out", default="/mnt/sdc/ml/corpora/multilingual_calib.jsonl")
    ap.add_argument("--n-multilingual", type=int, default=34,
                    help="prompts from the corpus `multilingual` bucket (aya native).")
    ap.add_argument("--n-constrained", type=int, default=22,
                    help="language-constrained prompts from the corpus `constrained` bucket.")
    ap.add_argument("--aya-per-lang", type=int, default=0,
                    help="extra native prompts per AYA_LANGS lang from CohereForAI/aya "
                         "(0 = corpus-only; set e.g. 1-2 to broaden Korean/Swahili native).")
    ap.add_argument("--min-len", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=1400)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.corpus)]
    ok = lambda p: isinstance(p, str) and args.min_len < len(p) < args.max_len  # noqa: E731

    ml_pool = [r["prompt"] for r in rows
               if r.get("bucket") == "multilingual" and ok(r.get("prompt", ""))]
    # constrained prompts that name a language or use a language/format constraint
    con_pool = [r["prompt"] for r in rows
                if r.get("bucket") == "constrained" and ok(r.get("prompt", ""))
                and (_LANG_RE.search(r["prompt"]) or _CONSTRAINT_RE.search(r["prompt"]))]
    # verbatim loop seeds (the bucket of flagged loopers), if present
    seed_pool = [r["prompt"] for r in rows
                 if r.get("bucket") == "seeds" and ok(r.get("prompt", ""))
                 and (_LANG_RE.search(r["prompt"]) or _CONSTRAINT_RE.search(r["prompt"]))]

    picks = []
    for p in deterministic_pick(ml_pool, args.n_multilingual):
        picks.append({"prompt": p, "src": "antiloop:multilingual", "lang": tag_lang(p)})
    for p in deterministic_pick(con_pool, args.n_constrained):
        picks.append({"prompt": p, "src": "antiloop:constrained", "lang": tag_lang(p)})
    for p in seed_pool:                       # take all language seeds (few)
        picks.append({"prompt": p, "src": "antiloop:seeds", "lang": tag_lang(p)})

    if args.aya_per_lang > 0:
        try:
            from datasets import load_dataset
            ds = load_dataset("CohereForAI/aya_dataset", split="train")
            by_lang = {lg: [] for lg in AYA_LANGS}
            for r in ds:
                lang = r.get("language")
                inp = r.get("inputs")
                if lang in by_lang and ok(inp):
                    by_lang[lang].append(inp.strip())
            for lang, pool in by_lang.items():
                for p in deterministic_pick(pool, args.aya_per_lang):
                    picks.append({"prompt": p, "src": "aya", "lang": lang})
        except Exception as e:
            print(f"[aya] skipped ({e})")

    # dedup on prompt text, keep first occurrence (stable)
    seen, final = set(), []
    for it in picks:
        if it["prompt"] not in seen:
            seen.add(it["prompt"])
            final.append(it)

    with open(args.out, "w") as f:
        for it in final:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    langs = Counter(it["lang"] or "unknown" for it in final)
    srcs = Counter(it["src"] for it in final)
    print(f"wrote {len(final)} multilingual calib prompts → {args.out}")
    print(f"  by src:  {dict(srcs)}")
    print(f"  by lang: {dict(langs)}")
    print(f"  pools: multilingual={len(ml_pool)} constrained(lang)={len(con_pool)} "
          f"seeds(lang)={len(seed_pool)}")


if __name__ == "__main__":
    main()
