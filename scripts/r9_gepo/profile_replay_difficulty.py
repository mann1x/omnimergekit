#!/usr/bin/env python3
"""Measure each replay problem's pass rate AT THE TRAINING GROUP SIZE, so the mixer can
skip the ones that can never produce a gradient.

WHY
---
GRPO normalises advantage by the WITHIN-GROUP spread. A replay problem the model gets
right every time (or wrong every time) yields identical rewards across the group, zero
advantage, and therefore no reward-gradient at all. run4's attempt-4 smoke showed this
happening: mbpp_exec came back pass=1.000, mean=1.000, nz=8/8 -- alive by the tier
gate, and contributing nothing through the reward.

It is not wasted entirely: with beta != 0 the KL term still anchors the policy to the
reference on those prompts (grpo_trainer.py:3121 adds beta*per_token_kl independently
of the advantage), which is real capability defence. But the reward signal on top of
that anchor is free to have, and it is what makes the tier more than a KL leash.

WHAT TO SELECT ON -- NOT p ~= 0.5
---------------------------------
The quantity that matters is P(a group of G is not unanimous) = 1 - p^G - (1-p)^G.
At G=8 even p=0.9 gives 0.57, so near-saturated problems are still useful more often
than not. The genuinely dead ones are p pinned at exactly 1.0 or 0.0. So this measures
at the SAME G the run will use and reports the unanimity rate directly, rather than
chasing a p~=0.5 band that would discard most of the pool for no benefit.

MUST MATCH THE RUN OR THE NUMBERS DO NOT TRANSFER
-------------------------------------------------
  * sampler: temperature/top_p/top_k identical to gepo_brevity.py's defaults
    (0.6 / 0.95 / 20). Pass rates measured greedy would be a different quantity.
  * rendering: no-think, exactly as --replay-no-think renders replay rows.
  * scoring: the CANONICAL verifiers imported from gepo_reward_v2 -- never a
    re-implementation, or the profile measures a different predicate than the reward.
    [[feedback_eval_methodology]]

Run:
  python scripts/r9_gepo/profile_replay_difficulty.py --model /mnt/sdc/ream-work/armJ \\
      --pool eval/replay/gepo_replay_pool.jsonl --n 96 -G 8 \\
      --out eval/replay/gepo_replay_pool.DIFFICULTY.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from gepo_reward_v2 import verify_mbpp_exec, verify_mc_letter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96, help="problems profiled PER TIER")
    ap.add_argument("-G", "--group", type=int, default=8,
                    help="samples per problem; MUST match the run's --num-generations")
    ap.add_argument("--batch-prompts", type=int, default=8)
    ap.add_argument("--kinds", default="", help="comma-separated tiers; default all")
    # max-new MUST cover what the RUN allows, or the profile measures a different
    # predicate than the reward does. Measured no-think: mbpp_exec ~205 tokens,
    # mc_letter p50 1595 but max 7612. Profiling GPQA at 4096 would score every
    # 5k-7.6k answer as a failure and mislabel exactly the rows being selected on.
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
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
    print("profiling: " + " ".join(f"{k}={len(v)}" for k, v in sel.items())
          + f"  G={a.group} sampler=t{a.temperature}/p{a.top_p}/k{a.top_k}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    torch.manual_seed(a.seed)

    out: dict[str, dict] = {}
    for kind, items in sel.items():
        t_kind = time.time()
        for b0 in range(0, len(items), a.batch_prompts):
            chunk = items[b0:b0 + a.batch_prompts]
            # Replay rows are rendered no-think, exactly as --replay-no-think does.
            texts = [tok.apply_chat_template(
                        [{"role": "user", "content": r["prompt"]}],
                        add_generation_prompt=True, tokenize=False,
                        enable_thinking=False) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            plen = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True,
                                     temperature=a.temperature, top_p=a.top_p,
                                     top_k=a.top_k, num_return_sequences=a.group,
                                     pad_token_id=tok.pad_token_id)
            for i, r in enumerate(chunk):
                npass = 0
                for g in range(a.group):
                    seq = gen[i * a.group + g][plen:]
                    keep = [t for t in seq.tolist() if t != tok.pad_token_id]
                    text = tok.decode(keep, skip_special_tokens=True)
                    ok = (verify_mc_letter(text, str(r.get("gold") or ""))
                          if kind == "mc_letter"
                          else verify_mbpp_exec(text, r["meta"]))
                    npass += int(bool(ok))
                out[r["id"]] = {"kind": kind, "G": a.group, "pass": npass,
                                "unanimous": npass in (0, a.group)}
            done = min(b0 + a.batch_prompts, len(items))
            print(f"  [{kind}] {done}/{len(items)}  "
                  f"({time.time() - t_kind:.0f}s)", flush=True)

    print("\n=== per-tier difficulty profile ===")
    for kind in sel:
        rs = [v for v in out.values() if v["kind"] == kind]
        uni = sum(1 for v in rs if v["unanimous"])
        allp = sum(1 for v in rs if v["pass"] == a.group)
        zero = sum(1 for v in rs if v["pass"] == 0)
        usable = len(rs) - uni
        hist = {}
        for v in rs:
            hist[v["pass"]] = hist.get(v["pass"], 0) + 1
        print(f"{kind}: n={len(rs)}  USABLE(mixed)={usable}  "
              f"unanimous={uni} (all-pass={allp}, all-fail={zero})")
        print("  pass-count histogram: "
              + " ".join(f"{k}/{a.group}:{hist[k]}" for k in sorted(hist)))
        if usable == 0:
            print("  WARNING: no mixed problem in this tier -- selecting on difficulty "
                  "cannot help; the tier can only ever contribute the KL anchor.")

    # MERGE, so per-tier invocations (each with its own max-new) build one profile.
    dst = pathlib.Path(a.out)
    prev = json.loads(dst.read_text()) if dst.is_file() else {}
    prev.update(out)
    dst.write_text(json.dumps(prev, indent=2) + "\n")
    print(f"\nwrote {a.out}  (+{len(out)} this run, {len(prev)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
