#!/usr/bin/env python3
"""Harvest the teacher's PASSING MultiPL-E solutions into the `targeted_mpe` calib corpus.

The MultiPL-E analogue of harvest_lcb_calib_corpus.py. Input is the nuprl evaluator's
per-problem verdict files written by multipl_e_evaluate.sh:

    <results>/multipl_e_calib/<served>/results/humaneval-<lang>/<name>.results.json

Each carries the generator's payload copied through (bug-607): `prompt`,
`completions[0]` (the EXTRACTED code the compiler saw) and `raw_completions[0]`
(the model's UNTRANSFORMED reply, i.e. the full CoT), plus `results[0].status`
where "OK" is pass@1.

Keeps only status=="OK" (a wrong solution's routing is not the code-generation signal
we want to protect — the same PASS-only rule as the LCB channel) and chat-templates
each as [user=prompt, assistant=reply] into JSONL rows {bench, text, task_id, n_tok},
the format build_calib_corpus_qwen.py emits, so this file concatenates with the
balanced corpus and the competence producer reads it in Tier-C mode as category
`corpus_<bench>` (default `corpus_targeted_mpe`; pass that to make_drop_map --cat-weight).

`--text-field raw` (default) uses the full CoT reply, matching targeted_lcb. Use
`--text-field extracted` to build the code-only variant instead; rows whose raw is
absent (pre-bug-607 cache) fall back to extracted and are counted separately, so a
missing raw is never silently mistaken for an empty reply.

Usage:
  python harvest_mpe_calib_corpus.py \
    --results-root /workspace/ornith_prune/calib_results/multipl_e_calib/ornith256e_q6k/results \
    --tokenizer /workspace/models/Ornith-1.5-35B-A3B \
    --out results/router_calib_corpus_mpe_ornith.jsonl [--bench targeted_mpe] [--include-fail]
"""
import argparse
import glob
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="dir holding humaneval-<lang>/ subdirs of *.results.json")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", default="targeted_mpe",
                    help="row `bench` field -> producer category `corpus_<bench>`")
    ap.add_argument("--text-field", choices=("raw", "extracted"), default="raw",
                    help="raw = full CoT reply (default, matches targeted_lcb); "
                         "extracted = compiler-visible code only")
    ap.add_argument("--include-fail", action="store_true",
                    help="also include FAIL generations (default: PASS-only)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    root = Path(args.results_root)
    files = sorted(glob.glob(str(root / "humaneval-*" / "*.results.json")))
    if not files:
        raise SystemExit(f"[harvest] no *.results.json under {root} — nothing to harvest")

    rows = []
    n_pass = n_fail = n_empty = n_rawmiss = 0
    per_lang: dict[str, list[int]] = {}
    for fp in files:
        lang = os.path.basename(os.path.dirname(fp)).replace("humaneval-", "")
        try:
            d = json.loads(Path(fp).read_text())
        except Exception:
            continue
        res = d.get("results") or []
        passed = bool(res) and res[0].get("status") == "OK"
        n_pass += int(passed)
        n_fail += int(not passed)
        if not passed and not args.include_fail:
            continue
        user = d.get("prompt") or ""
        raw = (d.get("raw_completions") or [""])[0] or ""
        ext = (d.get("completions") or [""])[0] or ""
        if args.text_field == "raw":
            asst = raw or ext
            if not raw:
                n_rawmiss += 1
        else:
            asst = ext
        if not user or not asst.strip():
            n_empty += 1
            continue
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": asst}]
        text = tok.apply_chat_template(msgs, tokenize=False)
        n_tok = len(tok(text).input_ids)
        rows.append({"bench": args.bench, "text": text,
                     "task_id": f"{lang}::{d.get('name')}", "n_tok": n_tok})
        per_lang.setdefault(lang, []).append(n_tok)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    tot = sum(r["n_tok"] for r in rows)
    mode = "PASS+FAIL" if args.include_fail else "PASS-only"
    print(f"[harvest] verdict files: {len(files)}  PASS={n_pass} FAIL={n_fail} "
          f"(pass-rate {100*n_pass/max(n_pass+n_fail,1):.1f}%)  empty-skipped={n_empty}")
    if n_rawmiss:
        print(f"[harvest] NOTE: {n_rawmiss} rows had no raw_completions "
              f"(pre-bug-607 cache) — fell back to the extracted body")
    for lang in sorted(per_lang):
        v = per_lang[lang]
        print(f"[harvest]   {lang}: {len(v)} rows, {sum(v)} tokens (avg {sum(v)//max(len(v),1)})")
    print(f"[harvest] wrote {len(rows)} rows ({mode}, text={args.text_field}), {tot} tokens "
          f"(avg {tot//max(len(rows),1)}) -> bench='{args.bench}' -> {outp}")
    if len(rows) < 30:
        print(f"[harvest] WARNING: only {len(rows)} rows — thin targeted channel; "
              f"consider --include-fail or a larger calib set.")


if __name__ == "__main__":
    main()
