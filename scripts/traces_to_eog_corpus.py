#!/usr/bin/env python3
"""traces_to_eog_corpus.py — render Tier-B traces into the JSONL an agentic_eog pass eats.

`expert_neuron_analysis_v5_targeted.py --variant agentic_eog` reads a JSONL whose lines
carry a pre-rendered `text` field, chunks each line into --window-tokens windows, and
accumulates expert routing ONLY at positions t where token t+1 == --emit-token-id.

WHICH TERMINATOR TO ASK FOR -- they are not interchangeable
----------------------------------------------------------
    <tool_call|>      49   closes ONE tool call's syntax. A model that loses this emits
                           malformed JSON. Abundant in the SWE pool (573 model-generated
                           closes in qset_A alone). NOT the loop terminator.
    <|tool_response>  50   an OPENER, not a terminator. Predicting it means "my emission
                           ends here and a tool result follows" -- the per-step stop. This
                           is what v7-coder's shipped 46 pins were built on, and v7-coder
                           terminates but does not converge.
    <turn|>          106   the DELIVERY close: the agent stops calling tools and answers.
                           This is the terminator that bears on non-convergence. The SWE
                           pool has ZERO model-generated instances (573/573 assistant
                           messages carry tool calls); the manic omnimerge-v4 runs are the
                           only source we hold.

So: build the 49-corpus from the SWE traces, and the 106-corpus from the manic v4 runs.
Do not mix them into one category -- whether the three terminators share experts is the
open question, and merging them makes it unanswerable.

RENDERING REGIME. This used to hardcode preserve_thinking=True "to match Tier-B". That is
wrong here for the same reason it was wrong in the agreement probe (bug-371): Tier-B needs
history reasoning present so its routing is accumulated, but an emit-position map asks
which experts fire when the model is about to STOP — a question about the context the
model is actually served in, and the template gates history reasoning off by default. Its
own comment says replaying it "drives verbatim repetition (the v7-coder loop)". Default is
now the SERVED regime; --preserve-thinking is an explicit opt-in.

ROLE MATTERS FOR TURN TERMINATORS. Most `<turn|>` in a trace close the SYSTEM and USER
turns — 213 total vs 71 model-owned on the manic-v4 corpus, i.e. 66.7% prompt-side. The
map pass filters these with --emit-role model; this script reports the model-owned count
so a corpus can be judged before the GPU is spent. `<tool_call|>` (49) needs no filter:
2523 closes vs 2523 tool calls, 1:1.
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", action="append", required=True,
                    help="Trace JSON ({'traces':[{messages:...}]}). Repeatable.")
    ap.add_argument("--out", required=True, help="Output JSONL.")
    ap.add_argument("--model", default="/mnt/sdc/v7rework/base/gemma-4-26B-A4B-it")
    ap.add_argument("--template",
                    default="/mnt/sdc/v7rework/gemma4_a4b_fixed_chat_template.jinja")
    ap.add_argument("--require-token-id", type=int, default=None,
                    help="Drop rendered lines that do not contain this token id, and "
                         "report the count. Use 49 for the tool-call corpus, 106 for the "
                         "delivery corpus -- a line without the terminator contributes "
                         "zero emit positions and only costs GPU time.")
    ap.add_argument("--emit-role", choices=["model", "any"], default="model",
                    help="Which terminators to COUNT (and, with --require-token-id, to "
                         "require). Must match what the map pass will be given.")
    ap.add_argument("--preserve-thinking", action="store_true",
                    help="Render history reasoning (Tier-B regime). OFF by default: the "
                         "served template gates it off and its own comment blames it for "
                         "the v7-coder loop.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from expert_neuron_analysis_v5_targeted import emit_mask_full
    tok = AutoTokenizer.from_pretrained(args.model)
    template = open(args.template).read()

    kept = dropped = 0
    total_emit = 0
    total_raw = 0
    with open(args.out, "w") as fh:
        for path in args.traces:
            for tr in json.load(open(path))["traces"]:
                msgs = tr.get("messages")
                if not msgs:
                    continue
                ids = tok.apply_chat_template(
                    msgs, chat_template=template, add_generation_prompt=False,
                    preserve_thinking=args.preserve_thinking, enable_thinking=True,
                    tokenize=True)
                if hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                ids = list(ids)
                # Count what the MAP will actually accumulate, not raw occurrences --
                # a line whose every terminator is prompt-side contributes nothing and
                # only costs GPU time.
                n = raw = None
                if args.require_token_id:
                    raw = ids.count(args.require_token_id)
                    n = sum(emit_mask_full(ids, args.require_token_id, args.emit_role))
                    if not n:
                        dropped += 1
                        continue
                    total_emit += n
                    total_raw += raw
                fh.write(json.dumps({
                    "text": tok.decode(ids, skip_special_tokens=False),
                    "task_id": tr.get("task_id"),
                    "model": tr.get("model"),
                    "n_tokens": len(ids),
                    "n_emit": n,
                    "n_emit_raw": raw,
                    "emit_role": args.emit_role,
                    "preserve_thinking": args.preserve_thinking,
                }) + "\n")
                kept += 1

    print("wrote %s: %d lines kept, %d dropped (no %s-owned terminator)  "
          "preserve_thinking=%s"
          % (args.out, kept, dropped, args.emit_role, args.preserve_thinking))
    if args.require_token_id:
        print("  emit positions for id %d, role=%s: %d of %d raw occurrences "
              "(%.1f%% prompt-side, mean %.1f/line)"
              % (args.require_token_id, args.emit_role, total_emit, total_raw,
                 100.0 * (total_raw - total_emit) / max(total_raw, 1),
                 total_emit / max(kept, 1)))
        print("  SANITY for the map run: total tc should land near "
              "emit_positions * top_k(8) * layers(30) = %d. Millions => the emit mask did "
              "NOT restrict accumulation; REJECT the map."
              % (total_emit * 8 * 30))


if __name__ == "__main__":
    main()
