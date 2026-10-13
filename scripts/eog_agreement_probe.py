#!/usr/bin/env python3
"""eog_agreement_probe.py — does the TEACHER agree the turn ends where the trace says?

WHY THIS EXISTS
---------------
An emit-position competence map (`--variant agentic_eog`) accumulates expert routing
only at positions t where token t+1 is a terminator. That carries a silent assumption:
**the model agrees the turn ends there.** For traces the teacher generated itself the
assumption holds by construction. For traces from a DIFFERENT model (omnimerge-v4 =
Qwen-27B family) it does not: if the teacher would have continued, the experts firing
there are the ones that CONTINUE, and pinning them protects the wrong behaviour. That
inversion is how v7-coder ended up with `<|tool_response>` (an OPENER) in its
force-keep set.

WHAT IT MEASURES
----------------
    agreement = P(teacher argmax == <turn|> | the trace closes an assistant turn here)

CONTROLS -- the first is a GATE, not a nicety
---------------------------------------------
1. POSITIVE CONTROL (`--self-control`): the teacher greedily generates a short answer,
   emitting <turn|> itself; we feed its own output back and take the argmax at exactly
   that position. Same computation twice, so it MUST read ~100%. Below that, the
   harness is broken and every corpus number is void. A corpus rate quoted without
   this passing is meaningless.

2. Corpus arms, measured ONLY at assistant turn closes. A trace carries few <turn|>
   because Gemma-4 treats an entire tool loop as ONE turn -- and most of those closes
   belong to the SYSTEM and USER turns, which the model never predicts (they are
   prompt, not completion). Scoring those produced 6.8% on the teacher's OWN traces on
   2026-09-16, which is what exposed version 1 of this probe as broken.

Context handling: the full prefix is used up to --ctx, and <bos> is preserved when
truncating from the left. A window sliced mid-stream with no bos suppresses the
terminator prediction independently of who wrote the text.
"""
from __future__ import annotations

import argparse
import collections
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TURN_END = 106


def render(tok, template, messages, preserve_thinking=False):
    """Render the trace. Returns a plain list of token ids.

    preserve_thinking defaults to FALSE, which is the SERVED regime, and it is NOT what
    Tier-B uses. The two passes want opposite things and both are right:

      * Tier-B (competence map) renders with True on purpose -- "the overwhelming
        majority of a coder's expert usage lives in the thinking channel", so the
        history reasoning must be present for its routing to be accumulated.
      * THIS probe asks whether the teacher would stop here. That is a question about
        the context the model is actually served in, and the template gates history
        reasoning behind `preserve_thinking`, whose default is false. Its own comment
        records that replaying it "drives verbatim repetition (the v7-coder loop)" --
        i.e. True is the regime documented to CAUSE the pathology being mapped.

    Measured on the 71-episode v4 corpus: True renders p50 66,858 tokens, False renders
    34,575. So the wrong regime also costs a 2x context window for nothing.
    """
    ids = tok.apply_chat_template(
        messages, chat_template=template, add_generation_prompt=False,
        preserve_thinking=preserve_thinking, enable_thinking=True, tokenize=True)
    # BatchEncoding is a UserDict, NOT a dict -- isinstance(x, dict) misses it.
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


TURN_OPEN = 105       # <|turn>
ROLE_MODEL = 4368     # the 'model' role marker emitted right after <|turn>


def assistant_closes(tok, template, messages, preserve_thinking=False):
    """[(position, prefix_ids)] for every <turn|> that closes a MODEL turn.

    Single pass over the FULL render. Each turn opens `<|turn>` + a role marker
    (`system` 9731 / `user` 2364 / `model` 4368), so tracking the most recent marker
    tells us who owns each close.

    DO NOT locate these by rendering messages[:i+1] and checking the tail -- that was the
    original implementation and it is WRONG: the template closes whatever turn is open at
    the end of a prefix, so an assistant message that really ends mid-loop on a tool call
    renders as if it closed a turn. Those positions do not exist in the full render. It
    inflated a v4 trace from 1 real close to 3 and voided every 'earlier' number measured
    on 2026-09-16. It is also O(n) renders per trace, which is intractable for the Gemma
    runs (up to 538 assistant turns).
    """
    ids = render(tok, template, messages, preserve_thinking)
    out = []
    role = None
    for i, v in enumerate(ids):
        if v == TURN_OPEN and i + 1 < len(ids):
            role = ids[i + 1]
        elif v == TURN_END and role == ROLE_MODEL:
            out.append((i, ids[:i]))   # prefix ends on the token BEFORE <turn|>
    return out


def trim(ids, ctx):
    """Left-truncate to ctx tokens but KEEP <bos>."""
    if len(ids) <= ctx:
        return list(ids)
    return list(ids[:1]) + list(ids[-(ctx - 1):])


TOOL_CALL_OPEN = 48   # <|tool_call>
TOOL_CALL_CLOSE = 49  # <tool_call|>


def closes_within(model, ids, ctx, k):
    """Let the teacher CONTINUE from here: does it close the turn within k tokens,
    before starting a tool call?

    Strict next-token agreement conflates two different disagreements:
      * "the episode is not over"      -> fatal for borrowing the corpus
      * "I'd write one more sentence"  -> harmless; the terminator still fires, later
    Measured 2026-09-16, the tokens the teacher preferred over <turn|> were overwhelmingly
    `\\n\\n`, `\\n` and `.` -- i.e. more prose, NOT a tool call. So the strict test was
    answering the wrong question. Returns (closed, emitted_tool_call, n_tokens).
    """
    t = torch.tensor([trim(ids, ctx)], device=model.device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=k, do_sample=False, use_cache=True)
    tail = out[0].tolist()[t.shape[1]:]
    for n, tokid in enumerate(tail, 1):
        if tokid in (TOOL_CALL_OPEN, TOOL_CALL_CLOSE):
            return False, True, n
        if tokid == TURN_END:
            return True, False, n
    return False, False, len(tail)


def argmax_next(model, ids, ctx):
    t = torch.tensor([trim(ids, ctx)], device=model.device)
    with torch.no_grad():
        # use_cache=False is the load-bearing one: a single forward needs no KV cache,
        # and building one for 30 layers at ctx=16384 held ~34 GiB on top of the 52 GiB
        # of bf16 weights -> OOM inside SDPA. Measured: with logits_to_keep alone the
        # allocation was byte-identical (86.07 GiB), which is what proved the logits
        # tensor was NOT the consumer. logits_to_keep=1 is kept as a smaller, real win
        # (seq_len x 262k vocab, all but one row discarded).
        out = model(t, logits_to_keep=1, use_cache=False)
    return int(out.logits[0, -1].argmax())


def self_control(model, tok, ctx):
    """Teacher generates, then must agree with itself. MUST be ~100%."""
    prompts = ["Name the capital of France.",
               "What is 17 * 23?",
               "Explain in one sentence what a hash map is.",
               "What does the HTTP status code 404 mean?",
               "Sort this list in Python: [3,1,2].",
               "Write a Python function that reverses a string."]
    ok = tot = 0
    print("\n=== POSITIVE CONTROL (teacher vs itself) — must read ~100% ===", flush=True)
    for p in prompts:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
            return_dict=True)["input_ids"].to(model.device)
        gen_start = int(ids.shape[1])
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=2048, do_sample=False)
        seq = gen[0].tolist()
        # Search ONLY the generated tail. The rendered PROMPT already contains a
        # <turn|> closing the USER turn; scanning the whole sequence finds that one and
        # scores the model on predicting its own prompt, which it never does. That bug
        # made this control read 0/6 on 2026-09-16.
        tail = seq[gen_start:]
        if TURN_END not in tail:
            print("  %-46s NO <turn|> emitted — skipped" % p[:46], flush=True)
            continue
        t = gen_start + tail.index(TURN_END) - 1   # position whose NEXT token is <turn|>
        pred = argmax_next(model, seq[:t + 1], ctx)
        tot += 1
        ok += (pred == TURN_END)
        print("  %-46s pred=%-7d %s" % (
            p[:46], pred,
            "AGREE" if pred == TURN_END else "DISAGREE(%r)" % tok.convert_ids_to_tokens(pred)),
            flush=True)
    rate = 100.0 * ok / max(tot, 1)
    print("  -> %d/%d = %.1f%%  %s" % (
        ok, tot, rate,
        "PASS" if rate >= 90 else "FAIL — harness broken, corpus numbers are VOID"),
        flush=True)
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", action="append", default=[], metavar="ARM=PATH",
                    help="Arm label and trace JSON. Repeatable.")
    ap.add_argument("--model", default="/mnt/sdc/v7rework/base/gemma-4-26B-A4B-it")
    ap.add_argument("--template",
                    default="/mnt/sdc/v7rework/gemma4_a4b_fixed_chat_template.jinja")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--max-traces", type=int, default=0, help="0 = all")
    ap.add_argument("--continuation-k", type=int, default=0,
                    help="If >0, ALSO let the teacher generate up to k tokens from each "
                         "close and record whether it closes the turn before starting a "
                         "tool call. Separates 'episode is not over' from 'I would write "
                         "one more sentence first'.")
    ap.add_argument("--self-control", action="store_true",
                    help="Run the positive control. Do this before trusting any arm.")
    ap.add_argument("--require-control", action="store_true",
                    help="Abort if the positive control does not pass.")
    ap.add_argument("--preserve-thinking", action="store_true",
                    help="Render history reasoning (Tier-B's regime). OFF by default: "
                         "the served template gates it off, and it is the regime the "
                         "template's own comment blames for the v7-coder loop. Use it "
                         "only as an explicit comparison ARM, never as the baseline.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open(args.template).read()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print("terminator: id %d = %r | ctx=%d | preserve_thinking=%s (%s regime)"
          % (TURN_END, tok.convert_ids_to_tokens(TURN_END), args.ctx,
             args.preserve_thinking,
             "Tier-B" if args.preserve_thinking else "SERVED"), flush=True)

    if args.self_control:
        rate = self_control(model, tok, args.ctx)
        if args.require_control and rate < 90:
            raise SystemExit("positive control FAILED (%.1f%%) — refusing to report "
                             "corpus numbers from a broken harness" % rate)

    rows = []
    for spec in args.traces:
        arm, _, path = spec.partition("=")
        traces = json.load(open(path))["traces"]
        if args.max_traces:
            traces = traces[:args.max_traces]
        print("\n=== ARM %s — %d traces from %s ===" % (arm, len(traces), path), flush=True)
        for tr in traces:
            msgs = tr.get("messages")
            if not msgs:
                continue
            closes = assistant_closes(tok, template, msgs, args.preserve_thinking)
            if not closes:
                continue
            final_idx = closes[-1][0]
            agree_d = agree_m = n_d = n_m = 0
            for i, prefix in closes:   # i is now a TOKEN POSITION, not a message index
                pred = argmax_next(model, prefix, args.ctx)
                ok = pred == TURN_END
                is_final = (i == final_idx)
                if is_final:
                    n_d += 1
                    agree_d += ok
                else:
                    n_m += 1
                    agree_m += ok
                # A position whose prefix exceeds ctx was LEFT-TRUNCATED: the model
                # judged an episode that begins mid-stream. Those rows are not the same
                # measurement as a whole-episode row and must never be pooled silently.
                row = {"arm": arm, "task_id": tr.get("task_id"),
                       "final": is_final, "agree": ok, "pred": pred, "pos": i,
                       "prefix_len": len(prefix),
                       "whole": len(prefix) <= args.ctx,
                       "coverage": min(1.0, args.ctx / max(len(prefix), 1)),
                       "pred_tok": tok.convert_ids_to_tokens(pred)}
                if args.continuation_k:
                    closed, tooled, n = closes_within(model, prefix, args.ctx,
                                                      args.continuation_k)
                    row.update(closed=closed, tool_first=tooled, gen_tokens=n)
                rows.append(row)
            print("  %-26s asst_closes=%-3d final %d/%d  earlier %d/%d"
                  % (str(tr.get("task_id"))[:26], len(closes), agree_d, n_d, agree_m, n_m),
                  flush=True)

    if rows:
        print("\n=== AGREEMENT SUMMARY (assistant turn closes only) ===")
        print("preserve_thinking=%s  ctx=%d" % (args.preserve_thinking, args.ctx))
        print("%-12s %-14s %-9s %7s %7s  %6s  %s"
              % ("arm", "kind", "prefix", "n", "agree", "rate", "mean cov"))
        for arm in dict.fromkeys(r["arm"] for r in rows):
            for kind, want in (("final/delivery", True), ("earlier", False)):
                # Split WHOLE from TRUNCATED. Pooling them answers neither question:
                # a truncated row measures agreement given an arbitrary tail.
                for lab, pred_f in (("whole", lambda r: r["whole"]),
                                    ("truncated", lambda r: not r["whole"]),
                                    ("ALL", lambda r: True)):
                    sel = [r for r in rows
                           if r["arm"] == arm and r["final"] is want and pred_f(r)]
                    if not sel:
                        continue
                    a = sum(r["agree"] for r in sel)
                    print("%-12s %-14s %-9s %7d %7d  %5.1f%%  %6.2f"
                          % (arm, kind, lab, len(sel), a, 100.0 * a / len(sel),
                             sum(r["coverage"] for r in sel) / len(sel)))
        if args.continuation_k and any("closed" in r for r in rows):
            print("\n=== CONTINUATION TEST (teacher generates up to %d tokens) ==="
                  % args.continuation_k)
            print("%-12s %-14s %7s %8s %9s %9s" % ("arm", "kind", "n", "closes", "tool-first", "neither"))
            for arm in dict.fromkeys(r["arm"] for r in rows):
                for kind, want in (("final/delivery", True), ("earlier", False)):
                    sel = [r for r in rows if r["arm"] == arm and r["final"] is want
                           and "closed" in r]
                    if not sel:
                        continue
                    c = sum(r["closed"] for r in sel)
                    t_ = sum(r["tool_first"] for r in sel)
                    print("%-12s %-14s %7d %7d(%4.1f%%) %6d %9d"
                          % (arm, kind, len(sel), c, 100.0 * c / len(sel), t_,
                             len(sel) - c - t_))
        dis = [r for r in rows if not r["agree"]]
        if dis:
            c = collections.Counter(r["pred_tok"] for r in dis)
            print("\ntop tokens the teacher preferred where it DISAGREED (n=%d):" % len(dis))
            for tk, n in c.most_common(8):
                print("   %-24s %d" % (tk, n))

    if args.out and rows:
        json.dump(rows, open(args.out, "w"))
        print("\nwrote %d rows -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
