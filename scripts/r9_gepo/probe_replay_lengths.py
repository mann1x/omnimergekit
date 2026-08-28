#!/usr/bin/env python3
"""Measure how many completion tokens each replay tier actually NEEDS.

WHY: run4's smoke (2026-08-28, attempt 3) showed the two replay tiers diverge sharply
at max_completion=12288 --

    mbpp_exec   clipped=0.000  pass=0.250   mean_tok~3400   -> fits comfortably
    mc_letter   clipped=1.000  pass=0.000   8/8 at ceiling  -> cannot finish

A tier pinned at the ceiling tells you it needs MORE, never HOW MUCH. Raising the cap
by guesswork risks a second 25h run that clips just as hard, so this measures the
distribution directly: generate at a deliberately generous cap and record, per tier,
the token length and whether the answer marker was ever emitted.

The number that matters is not the mean -- it is the quantile at which the ANSWER
MARKER appears. A rollout that rambles to 30k without ever emitting
"The correct answer is (X)" is not a long success, it is a failure that a bigger cap
will not fix, and the two must not be averaged together.
[[project_qwen36_ifeval_is_rumination]]

Run (on the GPU host, both GPUs free):
  python scripts/r9_gepo/probe_replay_lengths.py --model /mnt/sdc/ream-work/armJ \\
      --pool eval/replay/gepo_replay_pool.jsonl --n 6 --max-new 32768
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import statistics as st
import sys
import time

MARKER = re.compile(r"correct answer is\s*\(?([ABCD])\)?", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n", type=int, default=6, help="problems per tier")
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", default="", help="comma-separated tiers; default all")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(x) for x in open(a.pool) if x.strip()]
    byk: dict[str, list] = {}
    for r in rows:
        byk.setdefault(r["meta"]["reward_kind"], []).append(r)
    rng = random.Random(a.seed)
    want = {x for x in a.kinds.split(",") if x} or set(byk)
    sel = {k: rng.sample(v, min(a.n, len(v)))
           for k, v in sorted(byk.items()) if k in want}
    if not sel:
        sys.exit(f"REFUSE: --kinds {a.kinds!r} matched nothing in {sorted(byk)}")
    print(f"pool {a.pool}: " + " ".join(f"{k}={len(v)}" for k, v in sel.items()), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()

    # Left-padding: decoder-only batched generation puts pads on the LEFT, otherwise
    # the pads sit between the prompt and the first generated token and corrupt it.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    results = []
    for kind, items in sel.items():
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                         add_generation_prompt=True, tokenize=False)
                 for r in items]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        plen = enc["input_ids"].shape[1]
        print(f"  generating {len(items)} x {kind} in ONE batch "
              f"(prompt_pad_len={plen}, max_new={a.max_new}) ...", flush=True)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        secs = round(time.time() - t0, 1)
        for i, r in enumerate(items):
            gen = out[i][plen:]
            # Strip trailing pad/eos so a short answer in a padded batch is not
            # reported as having run to the cap.
            keep = [t for t in gen.tolist() if t != tok.pad_token_id]
            text = tok.decode(keep, skip_special_tokens=True)
            n_tok = len(keep)
            hit_cap = n_tok >= a.max_new
            m = MARKER.search(text) if kind == "mc_letter" else None
            tok_to_marker = None
            if m:
                tok_to_marker = len(tok(text[:m.end()], add_special_tokens=False).input_ids)
            has_answer = bool(m) if kind == "mc_letter" else ("def " in text)
            results.append({"kind": kind, "id": r["id"], "n_tok": n_tok,
                            "hit_cap": hit_cap, "has_answer": has_answer,
                            "tok_to_marker": tok_to_marker, "secs": secs})
            print(f"  [{kind} {i+1}/{len(items)}] tok={n_tok} cap_hit={hit_cap} "
                  f"answer={has_answer} tok_to_marker={tok_to_marker}", flush=True)
        print(f"  batch took {secs}s", flush=True)

    print("\n=== per-tier summary ===")
    for kind in sel:
        rs = [x for x in results if x["kind"] == kind]
        toks = [x["n_tok"] for x in rs]
        caps = sum(x["hit_cap"] for x in rs)
        ans = sum(x["has_answer"] for x in rs)
        marks = [x["tok_to_marker"] for x in rs if x["tok_to_marker"]]
        print(f"{kind}: n={len(rs)} cap_hit={caps}/{len(rs)} answered={ans}/{len(rs)}")
        print(f"  tokens  min={min(toks)} p50={int(st.median(toks))} max={max(toks)}")
        if marks:
            print(f"  tokens-to-answer-marker  min={min(marks)} "
                  f"p50={int(st.median(marks))} max={max(marks)}  "
                  f"<- the budget this tier actually needs")
        elif kind == "mc_letter":
            print("  NO rollout ever emitted the answer marker -- a bigger cap will "
                  "NOT fix this tier; the prompt or the tier is wrong.")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
