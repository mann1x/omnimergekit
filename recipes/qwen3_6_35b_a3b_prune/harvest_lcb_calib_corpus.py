#!/usr/bin/env python3
"""Harvest the teacher's PASSING LCB solutions into the `targeted_lcb` calib corpus (Qwen).

Input: the lcb_calib generation samples written by lcb_llama_server.py
(`<results>/lcb_calib/<served>/lcb_result.samples.jsonl`), one JSON record per problem:
  {task_id, passed(bool), prompt(user turn), completion(full CoT+code), cleaned, ...}

Keeps only PASS records (the v7-coder 128e-PASS channel — a wrong solution's routing is not
the code-generation signal we want to protect), and Qwen-chat-templates each as
[user=prompt, assistant=completion] into JSONL rows {bench, text, task_id, n_tok} — the exact
format build_calib_corpus_qwen.py emits, so this corpus concatenates with the balanced one and
the competence producer reads it in Tier-C mode as category `corpus_<bench>` (default
`corpus_targeted_lcb`; pass that name to make_drop_map --cat-weight).

Usage:
  python harvest_lcb_calib_corpus.py \
    --samples /srv/ml/eval_results/qwen_calib/lcb_calib/qwen256e_q6k/lcb_result.samples.jsonl \
    --tokenizer /srv/ml/models/Qwen3.6-35B-A3B \
    --out results/router_calib_corpus_lcb_qwen.jsonl [--bench targeted_lcb] [--include-fail]
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", default="targeted_lcb",
                    help="row `bench` field -> producer category `corpus_<bench>` (default targeted_lcb)")
    ap.add_argument("--include-fail", action="store_true",
                    help="also include FAIL generations (default: PASS-only, the v7-coder channel)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows, n_pass, n_fail, n_empty = [], 0, 0, 0
    with open(args.samples) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            passed = bool(rec.get("passed"))
            n_pass += int(passed)
            n_fail += int(not passed)
            if not passed and not args.include_fail:
                continue
            user = rec.get("prompt") or ""
            asst = rec.get("completion") or ""
            if not user or not asst.strip():
                n_empty += 1
                continue
            msgs = [{"role": "user", "content": user},
                    {"role": "assistant", "content": asst}]
            text = tok.apply_chat_template(msgs, tokenize=False)
            n_tok = len(tok(text).input_ids)
            rows.append({"bench": args.bench, "text": text,
                         "task_id": rec.get("task_id"), "n_tok": n_tok})

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tot = sum(r["n_tok"] for r in rows)
    mode = "PASS+FAIL" if args.include_fail else "PASS-only"
    print(f"[harvest] samples: PASS={n_pass} FAIL={n_fail} "
          f"(pass-rate {100*n_pass/max(n_pass+n_fail,1):.1f}%)  empty-skipped={n_empty}")
    print(f"[harvest] wrote {len(rows)} rows ({mode}), {tot} tokens "
          f"(avg {tot//max(len(rows),1)}) -> bench='{args.bench}' -> {outp}")
    if len(rows) < 30:
        print(f"[harvest] WARNING: only {len(rows)} rows — thin targeted channel; "
              f"consider --include-fail or a larger calib set.")


if __name__ == "__main__":
    main()
