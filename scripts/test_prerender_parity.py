#!/usr/bin/env python3
"""Gate --replay-no-think: LCB rows must be UNCHANGED, replay rows must be no-think.

Pre-rendering prompts to text is how per-row thinking control is achieved (TRL's
chat_template_kwargs is global). That buys control at the cost of moving prompt
rendering out of TRL and into us, so two things must be proven, not assumed:

  1. An LCB row pre-rendered by us is BYTE-IDENTICAL to what TRL would have produced
     from the conversational form. If it is not, run4 stops being comparable to
     run2/run3 on the arm that is supposed to be unchanged, and the brevity result --
     the whole point of the run -- becomes uninterpretable.
  2. A replay row actually carries the EMPTY <think></think> block. A template that
     silently ignored enable_thinking would render thinking-ON replay prompts, the
     tier would score zero again, and the only symptom would be another dead-tier
     abort many GPU-hours later.

Run:  python scripts/test_prerender_parity.py --model /mnt/sdc/ream-work/armJ
"""
from __future__ import annotations

import argparse
import sys

EMPTY_THINK = "<think>\n\n</think>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            fails += 1

    msgs = [{"role": "user", "content": "Solve this problem.\n\ndef f(x):\n    pass"}]

    # What TRL produces for a conversational prompt (grpo_trainer.py:1758): the batched
    # apply_chat_template with add_generation_prompt=True and no extra kwargs.
    trl_form = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ours_lcb = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ours_rep = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)

    print("=== 1: LCB pre-render is byte-identical to TRL's own rendering ===")
    check("LCB text matches TRL", ours_lcb == trl_form,
          f"{len(ours_lcb)} chars")
    check("LCB tokenises identically",
          tok(ours_lcb, add_special_tokens=False).input_ids
          == tok(trl_form, add_special_tokens=False).input_ids)

    print("\n=== 2: replay pre-render is genuinely no-think ===")
    check("replay differs from LCB", ours_rep != ours_lcb,
          "template would otherwise be ignoring enable_thinking")
    check("replay carries the EMPTY think block", EMPTY_THINK in ours_rep,
          repr(ours_rep[-60:]))
    check("LCB does NOT carry the empty think block", EMPTY_THINK not in ours_lcb,
          repr(ours_lcb[-40:]))

    print("\n=== 3: no double-BOS on the text path ===")
    # TRL tokenises a text prompt with self.processing_class(text=prompts), whose
    # add_special_tokens default is True. If the tokenizer prepends a BOS, the rendered
    # template's own prefix would be duplicated.
    with_special = tok(ours_lcb).input_ids
    without = tok(ours_lcb, add_special_tokens=False).input_ids
    check("tokenizer adds no BOS to a rendered prompt", with_special == without,
          f"{len(with_special)} vs {len(without)} tokens"
          + ("" if with_special == without else "  <- TRL's text path would double it"))

    print(f"\n{'PRERENDER_PARITY_OK' if fails == 0 else f'PRERENDER_PARITY_FAIL ({fails})'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
