#!/usr/bin/env python3
"""Harvest the teacher 256e PASSING IFEval responses into the `targeted_ifeval`
calib corpus (Qwen). The instruction-following analogue of harvest_mpe/lcb.

Input: an lm-eval ifeval samples_*.jsonl (from the 256e eval run). Each line has
  doc.prompt                 -> the instruction (user turn)
  filtered_resps[0]          -> the teacher response
  prompt_level_strict_acc    -> True == PASS (all verifiable constraints satisfied)

PASS == prompt_level_strict_acc is True (strict instruction-following). For each PASS
row, emit a Qwen-chat-templated [user=instruction, assistant=response] row
{bench, text, task_id, n_tok} in the exact format harvest_mpe/lcb emit, so it
concatenates with the balanced+LCB+MPE corpus and the producer reads it in Tier-C
mode as category `corpus_targeted_ifeval`.

Not disjoint from the eval set by design: this is a competence *map* (per-expert
routing frequency), not weight training — there is nothing to overfit, and IFEval
has no train/test split. Calibrating on the IFEval distribution is the most direct
signal for the instruction-following experts.

Usage:
  python harvest_ifeval_calib_corpus.py --samples <256e samples_ifeval_*.jsonl> \
    --tokenizer /srv/ml/models/Qwen3.6-35B-A3B \
    --out results/router_calib_corpus_ifeval_qwen.jsonl [--bench targeted_ifeval] \
    [--loose] [--include-fail]
"""
import argparse, json
from pathlib import Path


def resp_text(d):
    for rk in ("filtered_resps", "resps"):
        if rk in d and d[rk]:
            v = d[rk]
            s = v[0] if isinstance(v, list) else v
            s = s[0] if isinstance(s, list) and s else s
            if isinstance(s, str):
                return s
    return ""


def passed(d, loose):
    keys = ("prompt_level_loose_acc",) if loose else ("prompt_level_strict_acc",)
    for k in keys:
        if k in d:
            return d[k] in (True, 1, 1.0)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="lm-eval ifeval samples_*.jsonl (256e teacher)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", default="targeted_ifeval")
    ap.add_argument("--loose", action="store_true", help="use prompt_level_loose_acc for PASS")
    ap.add_argument("--include-fail", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows, n_pass, n_fail, n_empty = [], 0, 0, 0
    for line in open(args.samples):
        d = json.loads(line)
        doc = d.get("doc", {})
        p = passed(d, args.loose)
        n_pass += int(p); n_fail += int(not p)
        if not p and not args.include_fail:
            continue
        instr = doc.get("prompt") or ""
        resp = resp_text(d)
        if not instr.strip() or not resp.strip():
            n_empty += 1
            continue
        msgs = [{"role": "user", "content": instr},
                {"role": "assistant", "content": resp}]
        text = tok.apply_chat_template(msgs, tokenize=False)
        tid = doc.get("key", d.get("doc_id"))
        rows.append({"bench": args.bench, "text": text,
                     "task_id": f"ifeval::{tid}", "n_tok": len(tok(text).input_ids)})

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tot = sum(r["n_tok"] for r in rows)
    mode = "PASS+FAIL" if args.include_fail else ("PASS-loose" if args.loose else "PASS-strict")
    print(f"[harvest-ifeval] totals PASS={n_pass} FAIL={n_fail} "
          f"(pass-rate {100*n_pass/max(n_pass+n_fail,1):.1f}%) empty={n_empty}")
    print(f"[harvest-ifeval] wrote {len(rows)} rows ({mode}), {tot} tok "
          f"(avg {tot//max(len(rows),1)}) -> bench='{args.bench}' -> {outp}")
    if len(rows) < 30:
        print(f"[harvest-ifeval] WARNING: only {len(rows)} rows — thin channel.")


if __name__ == "__main__":
    main()
