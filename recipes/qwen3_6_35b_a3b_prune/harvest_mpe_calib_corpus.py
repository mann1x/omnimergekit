#!/usr/bin/env python3
"""Harvest the teacher 256e PASSING MultiPL-E solutions into the `targeted_mpe`
calib corpus (Qwen). The MPE analogue of harvest_lcb_calib_corpus.py.

Input: the omk multipl_e backend out_dir for the calib run, which holds
  <out>/generations/humaneval-<lang>/<name>.json  {name,prompt,completions:[body],tests}
  <out>/results/humaneval-<lang>/<name>.results.json  {results:[{status:"OK"|...}]}
PASS == results[0].status == "OK" (multipl_e_evaluate.sh pass@1 definition).

For each PASS problem, reconstruct the full function (prompt signature + generated body)
and emit a Qwen-chat-templated [user=complete-this-<lang>-fn, assistant=fenced full fn]
row {bench, text, task_id, n_tok} — the exact format build_calib_corpus_qwen.py /
harvest_lcb_calib_corpus.py emit, so it concatenates with the balanced+LCB corpus and the
producer reads it in Tier-C mode as category `corpus_targeted_mpe`.

Usage:
  python harvest_mpe_calib_corpus.py --out-dir <RES>/multipl_e_calib/<served> \
    --tokenizer /srv/ml/models/Qwen3.6-35B-A3B \
    --out results/router_calib_corpus_mpe_qwen.jsonl [--bench targeted_mpe] [--include-fail]
"""
import argparse, glob, json
from pathlib import Path


def instruction(lang, prompt):
    # byte-match multipl_e_generate.make_chat_request so the corpus user turn is
    # what the model actually saw during generation.
    return (f"Complete the following {lang} function. Reply with ONLY the complete "
            f"function implementation in a single Markdown code block — include the "
            f"signature exactly as given, write the full body, and do NOT add any "
            f"explanation, example usage, or test code.\n\n```{lang}\n{prompt}\n```")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="omk multipl_e calib out_dir (has generations/ + results/)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", default="targeted_mpe")
    ap.add_argument("--include-fail", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    base = Path(args.out_dir)

    rows = []
    per_lang = {}
    n_pass = n_fail = n_empty = n_nores = 0
    for gen_dir in sorted((base / "generations").glob("humaneval-*")):
        lang = gen_dir.name.replace("humaneval-", "")
        res_dir = base / "results" / gen_dir.name
        lp = lf = 0
        for gf in sorted(gen_dir.glob("*.json")):
            name = gf.stem
            try:
                gd = json.loads(gf.read_text())
            except Exception:
                continue
            rf = res_dir / f"{name}.results.json"
            passed = False
            if rf.exists():
                try:
                    rd = json.loads(rf.read_text())
                    res = rd.get("results") or []
                    passed = bool(res) and res[0].get("status") == "OK"
                except Exception:
                    pass
            else:
                n_nores += 1
            n_pass += int(passed); n_fail += int(not passed)
            lp += int(passed); lf += int(not passed)
            if not passed and not args.include_fail:
                continue
            prompt = gd.get("prompt") or ""
            body = (gd.get("completions") or [""])[0] or ""
            if not prompt or not body.strip():
                n_empty += 1
                continue
            full_fn = f"```{lang}\n{prompt}{body}\n```"
            msgs = [{"role": "user", "content": instruction(lang, prompt)},
                    {"role": "assistant", "content": full_fn}]
            text = tok.apply_chat_template(msgs, tokenize=False)
            rows.append({"bench": args.bench, "text": text,
                         "task_id": f"{lang}::{name}", "n_tok": len(tok(text).input_ids)})
        per_lang[lang] = (lp, lp + lf)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tot = sum(r["n_tok"] for r in rows)
    mode = "PASS+FAIL" if args.include_fail else "PASS-only"
    print(f"[harvest-mpe] per-lang PASS: " +
          ", ".join(f"{l}={p}/{n}" for l, (p, n) in per_lang.items()))
    print(f"[harvest-mpe] totals PASS={n_pass} FAIL={n_fail} "
          f"(pass-rate {100*n_pass/max(n_pass+n_fail,1):.1f}%) empty={n_empty} no-result={n_nores}")
    print(f"[harvest-mpe] wrote {len(rows)} rows ({mode}), {tot} tok "
          f"(avg {tot//max(len(rows),1)}) -> bench=\x27{args.bench}\x27 -> {outp}")
    if len(rows) < 30:
        print(f"[harvest-mpe] WARNING: only {len(rows)} rows — thin channel; consider --include-fail.")


if __name__ == "__main__":
    main()
