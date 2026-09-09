#!/usr/bin/env python3
"""GRPO efficiency run: the an-finetune method, on Jprime-p3, vLLM-colocate over 2 GPUs.

Knowledge doc (READ FIRST, keep updated): docs/METHOD_grpo_efficiency.md

METHOD PROVENANCE -- THIS IS NOT GEPO
-------------------------------------
The run that worked on an-finetune (`simpo/train_grpo_e2b.py`) is plain GRPO with TRL's
DEFAULTS, plus an absolute-budget length penalty:

    loss_type='dapo'  importance_sampling_level='token'  scale_rewards='group'
    epsilon=0.2 (clipping ON)   beta=0.04 (KL leash to the frozen reference)
    r = (1 - lambda*min(ntok/budget, 1)) if correct else 0

GEPO is what that script turns on with `--gepo`: loss_type='grpo',
importance_sampling_level='sequence', and clipping disabled (epsilon_low=1.0,
epsilon_high=1e4). We are deliberately NOT doing that -- GEPO Entropy has been tried
across run1-run4 and did not produce brevity at any dose. So every knob here is the
default/beta=0.04 path, and `--gepo` does not exist in this script on purpose.

beta=0.04 IS PART OF THE METHOD, not a leftover. TRL's default is beta=0.0 (no KL
term at all). an-finetune anchored to the frozen reference, and the run that held
capability while moving length is the one that had that leash on. Do not "modernise"
it to 0.0 without treating that as a separate arm.

PER-ROW THINKING NEEDS PRE-RENDERED PROMPTS
-------------------------------------------
`meta.think` is a per-row property (the efficiency driver reasons; the GPQA brevity
tier does not -- GPQA with thinking runs 16-18k tokens and half the rollouts never
reach an answer). But TRL's `chat_template_kwargs` is GLOBAL, and the Gemma-4 template
honours no inline `/no_think` marker: it only reads the `enable_thinking` kwarg. So a
single trainer-level setting cannot express a per-row choice.

This script therefore renders every prompt ITSELF, with that row's own
`enable_thinking`, and hands TRL plain strings. TRL treats a string `prompt` column as
non-conversational and passes it through untouched, so the template is applied exactly
once, by us, with the right per-row flag.

EOG IDS ARE LITERAL INTEGERS
----------------------------
Gemma-4's turn-end token is `<turn|>` = 106 (with 50 `<|tool_response>` and 1 `<eos>`);
`generation_config.json` carries `eos_token_id: [1, 106, 50]`. It is NOT `<end_of_turn>`
-- that is a Gemma-2/3 name, absent from Gemma-4's vocab, and
`convert_tokens_to_ids('<end_of_turn>')` returns 3 = `<unk>`. Resolving stop tokens by
NAME would therefore silently produce a stop set that never fires: every rollout would
run to the completion cap, be censored, score 0.0, and the length term would die while
the loss curve looked ordinary. So the ids are read from generation_config as integers
and asserted.  [[feedback_gemma4_double_terminator_trl_add_eos]]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import shutil
import signal
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "eval" / "lcb"))

EXPECTED_EOG = [1, 106, 50]


def load_pool(path: str) -> list[dict]:
    p = pathlib.Path(path)
    if not p.is_file():
        sys.exit(f"REFUSE: missing pool {p}")
    rows = [json.loads(x) for x in p.open() if x.strip()]
    if not rows:
        sys.exit(f"REFUSE: empty pool {p}")
    return rows


def resolve_eog(model_dir: str) -> list[int]:
    gc = pathlib.Path(model_dir) / "generation_config.json"
    if not gc.is_file():
        sys.exit(f"REFUSE: no generation_config.json in {model_dir}; EOG ids are not "
                 "guessable and must not be resolved by token NAME.")
    ids = json.loads(gc.read_text()).get("eos_token_id")
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if sorted(ids) != sorted(EXPECTED_EOG):
        sys.exit(f"REFUSE: generation_config eos_token_id={ids}, expected "
                 f"{EXPECTED_EOG}. A wrong stop set means every rollout runs to the "
                 "completion cap and scores 0.0 while the loss curve looks normal.")
    return ids


def assert_lora_scope(model_dir: str):
    """Resolve and VERIFY the LoRA scope on a meta-device model, before any GPU work.

    Mirrors gepo_brevity's gate: the count alone cannot prove the scope (a regex that
    dropped an attention projection and picked up a router would still total right),
    so the forbidden families are asserted absent BY NAME as well.
    """
    import re

    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from gepo_brevity import lora_scope_for

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    with init_empty_weights():
        meta = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    regex, expected, desc = lora_scope_for(meta)
    hit = [n for n, m in meta.named_modules()
           if isinstance(m, torch.nn.Linear) and re.fullmatch(regex, n)]
    if len(hit) != expected:
        sys.exit(f"REFUSE: LoRA regex matched {len(hit)} modules, expected {expected}. "
                 "The module naming changed; fix the regex rather than training a "
                 "different scope than the one that was reviewed.")
    forbidden = [n for n in hit
                 if "vision_tower" in n or "router" in n or n.endswith("lm_head")]
    if forbidden:
        sys.exit(f"REFUSE: the LoRA scope reached {len(forbidden)} module(s) it must "
                 f"never train, e.g. {forbidden[:3]}.")
    return regex, len(hit), desc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/sdc/v7rework/arms/Jprime-p3-bf16")
    ap.add_argument("--pool", default=str(REPO / "eval/efficiency/grpo_pool_v3.jsonl"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--budgets", default="",
                    help="JSON tier->budget. Required unless --smoke (the smoke's job "
                         "is to MEASURE the lengths a budget is derived from).")
    # --- generation / topology ---
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-len", type=int, default=8192)
    ap.add_argument("--max-prompt-len", type=int, default=2048,
                    help="Prompt headroom. TRL 1.12.0 has NO max_prompt_length: it "
                         "sizes the vLLM window instead, so this is added to "
                         "--max-completion-len to form vllm_max_model_length. A "
                         "prompt+completion over that window is rejected at "
                         "generation time, not silently truncated.")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="an-finetune used 0.9. This is the TRAINING sampler and is "
                         "unrelated to the canonical greedy EVAL sampler.")
    ap.add_argument("--vllm-tp", type=int, default=2,
                    help="Model split across both GPUs for generation (colocate).")
    ap.add_argument("--vllm-mem", type=float, default=0.30)
    ap.add_argument("--no-vllm-sleep", dest="vllm_sleep", action="store_false",
                    help="Keep vLLM resident through the training step. DEFAULT IS "
                         "SLEEP ON: at TP=2 colocate on 2x96 GB, a 39 GB DDP policy "
                         "replica plus vLLM's held 28.7 GB leaves ~5 GB, and the "
                         "reference-logprob fp32 logits upcast alone wants 8.00 GB. "
                         "Only pass this if the model is small enough that vLLM's "
                         "share is genuinely spare.")
    ap.set_defaults(vllm_sleep=True)
    # --- optimisation (an-finetune values) ---
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--save-steps", type=int, default=1,
                    help="Checkpoint every N optimiser steps. 1 = every step, so an "
                         "interruption costs at most one step of GPU time.")
    ap.add_argument("--save-total-limit", type=int, default=3,
                    help="How many rolling checkpoints to keep in --output. A full "
                         "checkpoint is adapter + optimiser + scheduler + RNG, so "
                         "keeping all 450 would be hundreds of GB. Durable retention is "
                         "--archive-every, which copies OUT of the rolling window.")
    ap.add_argument("--archive-every", type=int, default=10,
                    help="Copy every Nth checkpoint into --archive-dir, where "
                         "save_total_limit can never delete it. 0 disables archiving.")
    ap.add_argument("--archive-dir", default="",
                    help="Durable checkpoint archive (default: <output>/archive).")
    ap.add_argument("--resume", default="",
                    help="'auto' = resume from the highest-numbered checkpoint-* in "
                         "--output; or an explicit checkpoint path. Restores weights, "
                         "optimiser, scheduler, RNG and the step counter.")
    ap.add_argument("--stop-file", default="",
                    help="Touch this path to stop cleanly AT THE NEXT STEP BOUNDARY, "
                         "after a full checkpoint (default: <output>/STOP). SIGUSR1 and "
                         "SIGTERM create it rather than killing the process, so an "
                         "interrupted run is always resumable. Removed at startup.")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="Gradient-clipping ceiling, stated EXPLICITLY. HF's "
                         "TrainingArguments default is also 1.0, but inheriting it "
                         "silently is how the GEPO pilot's step-9 event (grad_norm 7.5, "
                         "clipped 7.5x) went unnoticed, and the 2026-09-09 lensignal24 "
                         "run clipped steps 20 (2.637) and 21 (1.135) without a word in "
                         "its header. Set it, log it, own it.")
    ap.add_argument("--beta", type=float, default=0.04,
                    help="KL leash to the frozen reference. TRL defaults to 0.0. "
                         "CAUTION: 0.04 is an-finetune's grpo_v8_full (v1) -- the arm "
                         "whose length went NOWHERE (OLS t=-0.36). The only AN arm that "
                         "measurably shortened, grpo_v8_v2 (t=-2.43), ran beta=0.01. "
                         "beta also caps how far a KL excursion can distort the loss: "
                         "lensignal24 step 20 hit kl=545, which at 0.04 contributes "
                         "~21.8 to the loss and at 0.01 contributes ~5.4. "
                         "[[feedback_anchor_on_the_arm_that_worked_not_the_family]]")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=1,
                    help="Completions per device per micro-step. THE LOGITS TENSOR "
                         "DOMINATES, not the weights: Gemma-4's vocab is 262,144, so "
                         "one forward over B completions of L tokens materialises "
                         "B*L*262144 logits -- at B=8, L=8192 that is ~34 GB in bf16 "
                         "and ~69 GB once TRL casts the lm_head to fp32, which OOMs a "
                         "95 GB card that already holds 39 GB of policy and ~26 GB of "
                         "vLLM. Keep B small and buy the effective batch back with "
                         "--grad-accum.")
    ap.add_argument("--measure-only", action="store_true",
                    help="Budget-derivation pass: TOLERATE rows with length_lambda>0 and "
                         "no length_budget, scoring them on correctness alone so their "
                         "passing lengths can be recorded for --measure-out. NEVER set "
                         "this for a training run -- every driver tier whose budget is "
                         "missing would train as pure correctness, silently. Was bound "
                         "to --smoke until 2026-09-09, which disabled the refusal on "
                         "every run this program has launched.")
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="16, NOT 32. This sets how fast data becomes OPTIMISER "
                         "UPDATES, which is what a short run is short of. AN's working "
                         "arm ran batch 8x4x1 = 32 completions per update; at "
                         "grad_accum 32 on 2 devices we were at 64 -- HALF AN's update "
                         "rate for the same data, so a 2-epoch run bought only 62 "
                         "steps and had no power. 16 restores AN's ratio (bsz 1 x 16 x "
                         "2 = 32) and roughly halves step time, since generation "
                         "dominates and the step generates half as many completions.")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--warmup-steps", type=int, default=6,
                    help="TRL 1.12.0's GRPOConfig exposes only warmup_steps -- the "
                         "ratio form was removed. an-finetune used a 0.1 ratio; at 248 "
                         "prompts / 4 prompts-per-step = ~62 steps, that is ~6. "
                         "Recompute if pool size or batch geometry changes.")
    ap.add_argument("--liger", action="store_true",
                    help="Use Liger's fused linear GRPO loss. It computes the SAME "
                         "quantity in chunks instead of materialising the full "
                         "[B, L, 262144] logits, which is the tensor that forces "
                         "--bsz down to 1 -- so it is the lever that buys the batch "
                         "back. It is an exact reformulation, not an approximation, "
                         "but the floating-point reduction order differs, so it is a "
                         "BASIS CHANGE: A/B it against a non-Liger baseline on the "
                         "same seed (per-tier pass rate / mean reward / mean tok) "
                         "before adopting it, rather than assuming equivalence.")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--log-every", type=int, default=64)
    ap.add_argument("--smoke", action="store_true",
                    help="Short run over a stratified slice: validates topology, "
                         "reward dispatch and EOG, and MEASURES per-tier passing "
                         "lengths so budgets can be derived rather than guessed.")
    ap.add_argument("--smoke-rows-per-tier", type=int, default=6)
    ap.add_argument("--smoke-steps", type=int, default=4)
    ap.add_argument("--length-lambda", type=float, default=0.0,
                    help="Override meta.length_lambda on EVERY row that already carries "
                         "lambda>0, leaving lambda==0 replay tiers untouched. AN's only "
                         "brevity arm that measurably shortened (grpo_v8_v2, t=-2.43) "
                         "ran a single lambda=0.8; its FLAT sibling ran 0.5. Our pool "
                         "ships 0.1/0.1/0.3. This flattens the per-tier weighting on "
                         "purpose -- it is a different design, not a tweak.")
    ap.add_argument("--budget-quantile", type=float, default=None,
                    help="Quantile of PASSING lengths used to derive budgets in the "
                         "measure path (default: grpo_reward_efficiency.BUDGET_QUANTILE "
                         "= 0.35). AN's winning arm sat at budget ~0.82x its observed "
                         "mean length; P35 puts us at 0.54x, i.e. MORE aggressive than "
                         "the only configuration known to work. ~0.6 matches AN.")
    ap.add_argument("--dump-rollouts", default="",
                    help="Path PREFIX for a per-rollout JSONL dump (one file per rank: "
                         "<prefix>.rank<N>.jsonl). Each row carries ntok, ok, r, lam, "
                         "budget, the CLAMPED penalty applied and the UNCLAMPED ratio "
                         "nt/budget, which is everything needed to re-score the same "
                         "rollouts under a different penalty shape offline -- no "
                         "regeneration, no GPU. Analyse with scripts/analyze_clamp.py.")
    ap.add_argument("--measure-out", default="",
                    help="Where the smoke writes measured lengths + derived budgets.")
    a = ap.parse_args()

    if not a.smoke and not a.budgets:
        sys.exit("REFUSE: a real run needs --budgets. Without it every lambda>0 tier "
                 "trains as pure correctness while the pool still calls it a driver.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from grpo_reward_efficiency import budget_from_lengths, make_efficiency_reward

    eog = resolve_eog(a.model)
    print(f">>> EOG ids (literal, from generation_config): {eog}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    # ---- reassert the EOG that transformers collapses at load -------------------
    # transformers "aligns" model + generation config to the TOKENIZER's scalar and
    # announces it:
    #     Updated tokens: {'eos_token_id': 1, 'bos_token_id': 2, 'pad_token_id': 0}
    # so eos collapses from [1, 106, 50] to 1. resolve_eog() cannot catch this: it
    # validates the FILE, and the override happens afterwards, in memory.
    #
    # Generation is NOT affected -- vLLM reads generation_config.json from disk
    # itself, which the 2026-09-09 smoke proved empirically (reward clip=0.000, mean
    # completion 961-2871 tok against an 8192 cap). What breaks is TRL's ACCOUNTING:
    #     grpo_trainer.py:1917   is_eos = completion_ids == self._tokenizer.eos_token_id
    # a SCALAR compare. With eos=1 nothing ever matches, every completion is called
    # truncated, mask_truncated_completions=True masks all of them, and the run
    # reports loss 0 / grad_norm nan -- i.e. it trains on NOTHING while looking alive.
    #
    # TRL's check is scalar, so it CANNOT represent {1, 106, 50}. We give it the id
    # the model actually ends turns with: <turn|> = 106. Any completion ending on 1
    # or 50 is still miscounted, which is why the assertion below is a LOUD warning
    # and not a silent fix -- if clipped_ratio stays 1.000 after this, 106 is the
    # wrong choice and the terminal-token distribution must be measured directly.
    TURN_END = 106
    prev_eos = tok.eos_token_id
    if TURN_END not in eog:
        sys.exit(f"REFUSE: turn terminator {TURN_END} is not in the resolved EOG {eog}; "
                 "do not guess a stop id.")
    if prev_eos != TURN_END:
        tok.eos_token_id = TURN_END
        print(f">>> EOS REASSERTED for TRL accounting: tokenizer.eos_token_id "
              f"{prev_eos} -> {TURN_END} (<turn|>). Full EOG {eog} stays in "
              f"generation_config for vLLM. TRL's is_eos check is a SCALAR compare, "
              f"so completions ending on {sorted(set(eog) - {TURN_END})} are still "
              f"counted truncated; if completions/clipped_ratio stays ~1.000 this "
              f"choice is wrong -- measure the terminal-token distribution.",
              flush=True)
    else:
        print(f">>> EOS already {TURN_END}; no reassertion needed", flush=True)
    # -----------------------------------------------------------------------------

    rows = load_pool(a.pool)

    # ------------------------------------------------------------ BUDGET OVERLAY
    # `--budgets` was DECLARED and GATED ON (the REFUSE above) but never actually
    # READ. A real run could pass --budgets, satisfy the guard, and then train on
    # whatever meta.length_budget the pool happened to carry -- including None, the
    # exact condition the guard exists to prevent. A gate that checks a flag instead
    # of the thing the flag promises is not a gate.
    #
    # Accepts BOTH shapes: the nested {"tiers":..., "budgets":...} that
    # --measure-out writes, and a flat {tier: int}.
    if a.budgets:
        bp = pathlib.Path(a.budgets)
        if not bp.is_file():
            sys.exit(f"REFUSE: --budgets {bp} does not exist.")
        raw = json.loads(bp.read_text())
        prov = raw.get("tiers") or {} if isinstance(raw, dict) else {}
        flat = raw["budgets"] if isinstance(raw, dict) and isinstance(
            raw.get("budgets"), dict) else raw
        table = {k: int(v) for k, v in flat.items()}
        n = 0
        for r in rows:
            m = r.setdefault("meta", {})
            if float(m.get("length_lambda") or 0) <= 0:
                continue                       # replay tier: correctness only, by design
            k = f"{m.get('reward_kind')}/{'T' if m.get('think') else 'N'}"
            if k not in table:
                sys.exit(f"REFUSE: pool tier {k} carries length_lambda>0 but "
                         f"--budgets has no entry for it. Measure it; do not guess.")
            m["length_budget"] = table[k]
            n += 1
        print(f">>> budgets overlaid onto {n} rows from {bp}", flush=True)
        for k in sorted(table):
            t = prov.get(k) or {}
            npl = t.get("n_pass_lens")
            thin = "  <-- THIN (<20 passing samples)" if isinstance(npl, int) and npl < 20 else ""
            print(f"      {k:16s} budget={table[k]:6d}  n_pass_lens={npl}{thin}", flush=True)

    # ---------------------------------------------------------- LAMBDA OVERRIDE
    # Applied AFTER the budget overlay so the provenance print above still shows the
    # budgets as measured. Rows with lambda==0 (replay) are deliberately untouched:
    # they exist to hold capability and must stay pure correctness.
    if a.length_lambda > 0:
        n_over = 0
        for r in rows:
            m = r.get("meta") or {}
            if float(m.get("length_lambda") or 0) > 0:
                m["length_lambda"] = float(a.length_lambda)
                n_over += 1
        print(f">>> LAMBDA OVERRIDE: {n_over} rows set to lambda={a.length_lambda} "
              f"(replay rows with lambda==0 untouched)", flush=True)

    # A lambda>0 row with no budget is the failure this whole path exists to stop.
    # Check the ROWS, not the flag -- for a real run it is fatal.
    missing = [r for r in rows
               if float((r.get("meta") or {}).get("length_lambda") or 0) > 0
               and not (r.get("meta") or {}).get("length_budget")]
    if missing and not a.smoke:
        sys.exit(f"REFUSE: {len(missing)}/{len(rows)} rows carry length_lambda>0 with no "
                 "length_budget. Those tiers would train as pure correctness while the "
                 "pool calls them drivers.")
    if missing:
        print(f">>> NOTE {len(missing)}/{len(rows)} lambda>0 rows have no budget "
              "(measure-only smoke: they score correctness and length_share will be 0)",
              flush=True)
    else:
        print(f">>> all {len(rows)} rows carry a length_budget where lambda>0 "
              "-- the length term IS active", flush=True)

    if a.smoke:
        bytier: dict[str, list] = {}
        for r in rows:
            m = r["meta"]
            bytier.setdefault(
                f"{m['reward_kind']}/{'T' if m.get('think') else 'N'}", []).append(r)
        rows = [x for v in bytier.values() for x in v[:a.smoke_rows_per_tier]]
        print(f">>> SMOKE: {len(rows)} rows, "
              f"{ {k: min(len(v), a.smoke_rows_per_tier) for k, v in bytier.items()} }",
              flush=True)

    # Render each prompt ONCE, here, with that row's own thinking flag.
    recs = []
    for r in rows:
        m = r["meta"]
        text = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=bool(m.get("think")))
        recs.append({"prompt": text, "gold": str(r.get("gold") or ""), "meta": m})
    ds = Dataset.from_list(recs)

    n_think = sum(1 for x in recs if x["meta"].get("think"))
    print(f">>> rendered {len(recs)} prompts  (think={n_think}, "
          f"nothink={len(recs) - n_think})", flush=True)

    lcb_verifier = None
    if any(x["meta"]["reward_kind"] == "lcb_exec" for x in recs):
        from lcb_helpers import score_lcb_problem

        def lcb_verifier(text, meta):  # noqa: F811
            # Signature is (code, tests, method_name, timeout=10.0) -- the 4th
            # positional is the TIMEOUT, not starter_code. Passing starter_code there
            # hands a string to a float comparison and the scorer never runs.
            from grpo_reward_efficiency import extract_code
            try:
                ok, why = score_lcb_problem(extract_code(text),
                                            meta.get("tests") or [],
                                            meta.get("method_name") or "")
                return bool(ok), why
            except Exception as e:
                return False, repr(e)

    # allow_unset_budget is bound to --measure-only, NOT to --smoke. A budget-derivation
    # pass necessarily runs before any budget exists and must tolerate unset budgets; a
    # REAL RUN must keep the hard refusal, because a row that reaches the reward with
    # lambda>0 and no budget trains as pure correctness and the length objective is
    # silently absent from that tier.
    #
    # These were coupled until 2026-09-09. Every arm in this program passes --smoke
    # (it is how the whole pool is selected, via --smoke-rows-per-tier), so the refusal
    # was disabled on every run that has ever been launched here. Same overload that
    # caused bug-698, where --smoke also decided save_strategy and nothing was ever
    # checkpointed. --smoke selects rows and caps steps. It decides NOTHING else.
    # [[feedback_never_skip_silently]] [[feedback_an_unchecked_case_is_silent]]
    reward = make_efficiency_reward(tok, lcb_verifier, a.max_completion_len,
                                    log_every=a.log_every,
                                    allow_unset_budget=a.measure_only,
                                    dump_rollouts=(a.dump_rollouts or None))

    cfg = GRPOConfig(
        output_dir=a.output,
        # LOAD DTYPE IS NOT `bf16=True`. `bf16` selects the training autocast; the
        # model is loaded by TRL via from_pretrained, which defaults to FLOAT32 unless
        # told otherwise. Jprime-p3 is 39 GB of bf16 on disk, so an fp32 load takes
        # ~78 GB -- which leaves 17 GB of a 95 GB card and makes vLLM refuse to
        # allocate its share. The symptom is a vLLM memory error, but the cause is the
        # dtype of the policy load.
        # transformers 5.x renamed torch_dtype -> dtype; the arm's config declares
        # `"dtype": "bfloat16"` and this makes the loader honour it.
        model_init_kwargs={"dtype": "bfloat16"},
        # --- an-finetune method: TRL defaults, explicit so a default change is loud ---
        loss_type="dapo",
        importance_sampling_level="token",
        # The line ABOVE sets the POLICY-side IS granularity. `vllm_importance_sampling_mode`
        # is a DIFFERENT knob -- the vLLM-vs-trainer distribution correction -- and TRL
        # 1.12 defaults it to `sequence_mask` with clip_max=3.0 / clip_min=None. That
        # default is wrong for this run and was never a choice; pinning it here for the
        # same reason every other TRL default in this block is pinned: so a default
        # change is LOUD.
        #
        # `sequence_mask` forms exp(SUM of signed per-token logp deltas) over the WHOLE
        # completion, then masks any sequence above clip_max to EXACTLY 0. Our
        # completions run 1600-8200 tokens, so that sum has no realistic chance of
        # landing inside (0, 3.0]: it either clears the cap and is zeroed, or underflows
        # toward 0. Measured on the 2026-09-09 eos-fix smoke, with the completion mask no
        # longer hiding it:
        #     step1  logp_diff/mean 1.128 max 37.64   is_ratio min 0 mean 0.0925 max 1.776
        #     step2  logp_diff/mean 0.923 max 35.17   is_ratio min 0 mean 0.1696 max 1.406
        # i.e. ~83-91% of the batch contributed NOTHING, and DAPO's token-count
        # normaliser can then divide by zero -- which is the shape of the `grad_norm nan`
        # seen from step 1 onward, BEFORE any weight update (step 1 ran at lr=0).
        #
        # `token_truncate` matches the granularity the method already declares above and
        # CLAMPS instead of zeroing, so no exact-zero ratios and no 0/0 denominator.
        vllm_importance_sampling_mode="token_truncate",
        scale_rewards="group",
        epsilon=0.2,
        beta=a.beta,
        num_iterations=1,
        # --- generation ---
        num_generations=a.num_generations,
        max_completion_length=a.max_completion_len,
        vllm_max_model_length=a.max_prompt_len + a.max_completion_len,
        temperature=a.temperature,
        # FALSE, matching an-finetune's grpo_v8_v2 -- the ONLY brevity arm in this
        # house with a significant length slope (OLS t=-2.43 over 446 steps). It never
        # set this flag, so it ran at TRL's default False, with clipped_ratio 0.53-0.69:
        # more than half of every batch truncated, each one scoring 0 and staying in the
        # gradient as a NEGATIVE example. That -- not the lenpen term -- is the strongest
        # brevity pressure that config had, and True cancels it: our reward computes
        # r=0.0 for a censored rollout and the mask then deletes the row.
        #
        # It also removes a dependency on a known-broken check. TRL's truncation test is
        # is_eos = completion_ids == eos_token_id, a SCALAR compare that cannot represent
        # Gemma-4's EOG set {1, 106, 50}, so it miscounts any completion ending on 1 or
        # 50. Our reward decides censoring from the TOKEN COUNT (nt >= max_completion)
        # and is correct either way; only the mask rode on the broken compare.
        # [[feedback_anchor_on_the_arm_that_worked_not_the_family]]
        # [[feedback_a_fully_masked_run_reports_constants_not_measurements]]
        mask_truncated_completions=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=a.vllm_tp,
        vllm_gpu_memory_utilization=a.vllm_mem,
        # vLLM's share is a CAP IT FILLS AND HOLDS, not a floor it grows into. Without
        # sleep mode it keeps its weight shard AND its whole KV pool resident THROUGH
        # the training step -- ~28.7 GiB per card that the optimiser can never use.
        # That is what killed the 2026-09-09 smoke: the OOM was NOT in the loss (Liger
        # already fuses that) but in _get_per_token_logps_and_entropies, where
        # accelerate's convert_to_fp32 upcasts the reference-logprob logits. 8192 tok x
        # 262,144 vocab x 4 B = exactly the 8.00 GiB it failed to allocate, against
        # ~5 GiB free. Liger does not cover that path.
        #
        # sleep(level=2) after generation discards weights AND kv_cache and gives the
        # memory back for the forward/backward; TRL re-pushes weights on wake. vLLM's
        # own kv_cache_memory_bytes would be the finer knob but TRL 1.12 does not
        # expose it -- releasing the pool entirely is strictly more headroom anyway.
        # Cost is a per-step weight re-push, which is the right trade when memory, not
        # time, is the binding constraint.
        vllm_enable_sleep_mode=a.vllm_sleep,
        # --- optimisation ---
        learning_rate=a.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=(1 if a.smoke else a.warmup_steps),
        max_grad_norm=a.max_grad_norm,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bsz,
        gradient_accumulation_steps=a.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        optim="adamw_8bit",
        max_steps=(a.smoke_steps if a.smoke else -1),
        logging_steps=1,
        # SAVING IS NEVER DISABLED. It used to read `"no" if a.smoke else "steps"`,
        # and since every arm in this program runs --smoke, NOTHING WAS EVER SAVED --
        # grpo_lensignal24/ held only measured.json after 24 steps and 2.5 h of GPU.
        # --smoke selects a row subsample and allow_unset_budget; it must not also
        # decide durability.  [[feedback_always_checkpoint_long_runs]]
        save_strategy="steps",
        save_steps=a.save_steps,
        save_total_limit=a.save_total_limit,
        # False is the default, but state it: True would write the adapter WITHOUT the
        # optimiser/scheduler/RNG, which resumes to a DIFFERENT trajectory while looking
        # like a resume. Everything needed to continue must be in the checkpoint.
        save_only_model=False,
        seed=a.seed,
        # TRL raises if Liger is combined with an lm_head adapter, prompt-learning
        # PEFT, off-policy masking, top_entropy_quantile<1, an entropy bonus, or an
        # importance_sampling_level outside {token, sequence}. This recipe satisfies
        # all six (grpo_trainer.py:794-844), and loss_type/beta/temperature are passed
        # straight through to LigerFusedLinearGRPOLoss, so dapo + beta=0.04 survive.
        use_liger_kernel=a.liger,
        report_to=[],
    )

    # LoRA scope: reuse the REVIEWED regex from gepo_brevity rather than a third
    # spelling of it. A bare suffix list like ["q_proj", ...] does not work here --
    # it also matches the vision tower, whose projections are Gemma4ClippableLinear
    # (a wrapper around an inner nn.Linear), and peft raises:
    #     ValueError: Target module Gemma4ClippableLinear(...) is not supported
    # The regex is anchored on model.language_model.layers.N so the tower, the
    # router (router.proj) and lm_head are all structurally out of scope.
    lora_regex, n_expected, scope_desc = assert_lora_scope(a.model)
    print(f">>> LORA scope OK: {n_expected} modules -- {scope_desc}", flush=True)

    peft_cfg = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=lora_regex,
    )

    print(f">>> torch {torch.__version__} | devices {torch.cuda.device_count()} | "
          f"vllm colocate TP={a.vllm_tp} mem={a.vllm_mem}", flush=True)

    # ------------------------------------------------------------- nan probe
    # `grad_norm: nan` on EVERY step of the 2026-09-09 eos-fix smoke, including step 1
    # which ran at lr=0 -- i.e. BEFORE any weight update. The obvious theory says that
    # must be fatal: clip_grad_norm_ gets total_norm=nan, clip_coef = max/(nan+eps) is
    # nan, every grad is scaled by nan, and AdamW's addcdiv_ poisons the params even at
    # lr=0 because 0*nan == nan. But the tiers did NOT collapse -- mc_letter/T held
    # pass=1.000 across all four steps -- so the theory is wrong somewhere and the
    # difference matters: a nan that never reaches the weights is a broken METRIC and a
    # lost update, not a destroyed policy.
    #
    # Scan at three points, because they answer different questions:
    #   on_substep_end        PRE-clip  -- which params the BACKWARD actually made nan.
    #   on_pre_optimizer_step POST-clip -- whether clipping SMEARED nan across all of
    #                                      them (it will, if total_norm is nan).
    #   on_step_end           params    -- whether the optimiser wrote nan into WEIGHTS.
    # Without the pre-clip scan the post-clip reading is uninformative by construction.
    # Cheap: LoRA only, a few hundred small tensors.
    from transformers import TrainerCallback

    class NanProbe(TrainerCallback):
        def __init__(self, model):
            self.m, self.sub = model, 0

        def _scan(self, grads: bool):
            bad, tot, names = 0, 0, []
            for n, prm in self.m.named_parameters():
                if not prm.requires_grad:
                    continue
                t = prm.grad if grads else prm
                if t is None:
                    continue
                tot += 1
                if not torch.isfinite(t).all():
                    bad += 1
                    if len(names) < 4:
                        names.append(n.split("base_model.model.")[-1])
            return bad, tot, names

        def on_substep_end(self, args, state, control, **kw):
            bad, tot, names = self._scan(grads=True)
            if bad or self.sub < 2:
                print(f">>> NANPROBE step={state.global_step} sub={self.sub} "
                      f"PRE-CLIP grads nonfinite={bad}/{tot} first={names}", flush=True)
            self.sub += 1

        def on_pre_optimizer_step(self, args, state, control, **kw):
            bad, tot, names = self._scan(grads=True)
            print(f">>> NANPROBE step={state.global_step} POST-CLIP grads "
                  f"nonfinite={bad}/{tot} first={names}", flush=True)

        def on_step_end(self, args, state, control, **kw):
            bad, tot, names = self._scan(grads=False)
            print(f">>> NANPROBE step={state.global_step} PARAMS nonfinite={bad}/{tot} "
                  f"first={names}", flush=True)
            self.sub = 0

    trainer = GRPOTrainer(model=a.model, args=cfg, train_dataset=ds,
                          reward_funcs=[reward], peft_config=peft_cfg,
                          processing_class=tok)
    class KLProbe(TrainerCallback):
        """Print kl/grad_norm PERCENTILES per interval, not just the step value.

        A single-step `kl` of 545 next to neighbours of 0.01 reads as a mystery number
        in the log. What matters is the DISTRIBUTION: AN's brevity arm that worked ran
        p50 3.9e-5 -> 0.0036 -> 0.069 -> 0.255 by quarter, i.e. a policy steadily
        moving, with spikes riding on top. The arm with no spikes at all is the arm
        where nothing happened. So an excursion is only readable against its own p50.
        Also flags every step whose grad_norm exceeded the clipping ceiling, since a
        clipped update is a SCALED update and that never appears in the step dict.
        """

        def __init__(self, clip_at: float, every: int = 8):
            self.kl, self.gn, self.every = [], [], every
            self.clip_at, self.n_clipped, self.seen = clip_at, 0, 0

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or "kl" not in logs:
                return
            self.seen += 1
            try:
                self.kl.append(float(logs["kl"]))
                g = float(logs.get("grad_norm", 0.0))
            except (TypeError, ValueError):
                return
            self.gn.append(g)
            if g > self.clip_at:
                self.n_clipped += 1
                print(f">>> KLPROBE step={self.seen} CLIPPED grad_norm={g:.4g} > "
                      f"max_grad_norm={self.clip_at:g} (update scaled by "
                      f"{self.clip_at / g:.3f}x)", flush=True)
            if self.seen % self.every == 0:
                k = sorted(self.kl[-self.every:])
                gg = sorted(self.gn[-self.every:])
                q = lambda v, f: v[min(int(len(v) * f), len(v) - 1)]  # noqa: E731
                print(f">>> KLPROBE last {self.every} steps: "
                      f"kl p50={q(k, .5):.4g} p90={q(k, .9):.4g} max={k[-1]:.4g} | "
                      f"grad_norm p50={q(gg, .5):.4g} max={gg[-1]:.4g} | "
                      f"clipped {self.n_clipped}/{self.seen} steps so far", flush=True)

    # ------------------------------------------------------- GRACEFUL STOP + ARCHIVE
    # This run is expected to be interrupted: it trains overnight, is stopped during the
    # day, and resumes. So an interruption must cost at most one step and must never
    # produce a checkpoint that resumes to a different trajectory.
    STOP_FILE = pathlib.Path(a.stop_file or (pathlib.Path(a.output) / "STOP"))
    ARCHIVE = pathlib.Path(a.archive_dir or (pathlib.Path(a.output) / "archive"))

    # A HF/PEFT checkpoint is only resumable if ALL of these are present. Verified, not
    # assumed -- a missing optimizer.pt resumes silently onto a fresh Adam state.
    # Verified against a real PEFT checkpoint (an-finetune e2b-an-v16-gepo/checkpoint-150):
    # adapter_config.json, adapter_model.safetensors, optimizer.pt, scheduler.pt,
    # rng_state.pth, trainer_state.json, training_args.bin.
    REQUIRED = ("adapter_model.safetensors", "adapter_config.json",
                "optimizer.pt", "scheduler.pt", "trainer_state.json")

    def _rng_ok(d) -> bool:
        """RNG file is rng_state.pth single-process but rng_state_<rank>.pth under DDP,
        so match the family, not one name. Without it a resume replays a DIFFERENT
        sampling stream while every other file says the resume was clean."""
        d = pathlib.Path(d)
        return bool(list(d.glob("rng_state*.pth")))

    def _touch_stop(signum, _frame):
        # Do NOT exit here. Killing mid-step loses the step and can leave a half-written
        # checkpoint; the callback stops us at the next boundary, after a full save.
        try:
            STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
            STOP_FILE.write_text(f"signal {signum} at {time.strftime('%FT%TZ', time.gmtime())}\n")
        except OSError:
            pass
        print(f">>> STOP REQUESTED (signal {signum}) -- will checkpoint and exit at the "
              f"next step boundary", flush=True)

    for _sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(_sig, _touch_stop)

    class Checkpointer(TrainerCallback):
        """Stop-on-request at a step boundary; archive every Nth checkpoint."""

        def __init__(self, is_main: bool):
            self.is_main, self.announced = is_main, False

        def on_step_end(self, args, state, control, **kw):
            if STOP_FILE.exists():
                if not self.announced:
                    print(f">>> STOP FILE {STOP_FILE} seen at step {state.global_step} "
                          f"-- saving a full checkpoint, then exiting cleanly. "
                          f"Resume with --resume auto.", flush=True)
                    self.announced = True
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kw):
            if not self.is_main:
                return control
            ck = pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}"
            missing = [f for f in REQUIRED if not (ck / f).exists()]
            if not _rng_ok(ck):
                missing.append("rng_state*.pth")
            if missing:
                # Loud, but do NOT kill the run: a bad checkpoint is recoverable from an
                # earlier one; a killed run is not.
                print(f">>> CHECKPOINT WARNING step {state.global_step}: missing "
                      f"{missing} in {ck} -- this checkpoint may NOT be resumable",
                      flush=True)
            if a.archive_every and state.global_step % a.archive_every == 0:
                dst = ARCHIVE / ck.name
                try:
                    ARCHIVE.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(ck, dst)
                    sz = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
                    print(f">>> ARCHIVED {ck.name} -> {dst} ({sz / 1e9:.2f} GB, "
                          f"{'complete' if not missing else 'INCOMPLETE'})", flush=True)
                except OSError as e:
                    print(f">>> ARCHIVE FAILED for {ck.name}: {e}", flush=True)
            return control

    _is_main = int(os.environ.get("RANK", "0")) == 0
    if _is_main and STOP_FILE.exists():
        # A stale STOP from the previous segment would halt this one at step 1.
        print(f">>> clearing stale stop file {STOP_FILE}", flush=True)
        STOP_FILE.unlink()

    resume = None
    if a.resume:
        if a.resume == "auto":
            cks = glob.glob(os.path.join(a.output, "checkpoint-*"))
            cks = [c for c in cks if re.search(r"checkpoint-(\d+)$", c)]
            if cks:
                resume = max(cks, key=lambda c: int(re.search(r"(\d+)$", c).group(1)))
        else:
            resume = a.resume
        if resume:
            miss = [f for f in REQUIRED if not os.path.exists(os.path.join(resume, f))]
            if not _rng_ok(resume):
                miss.append("rng_state*.pth")
            if miss:
                sys.exit(f"REFUSE: --resume {resume} is missing {miss}. Resuming from an "
                         "incomplete checkpoint silently restarts the optimiser state "
                         "and produces a DIFFERENT trajectory that still looks like a "
                         "resume. Point at a complete checkpoint or an archived one.")
            done = json.load(open(os.path.join(resume, "trainer_state.json")))
            print(f">>> RESUMING from {resume} at global_step="
                  f"{done.get('global_step')} / max {a.smoke_steps if a.smoke else '-'}",
                  flush=True)
        else:
            print(f">>> --resume {a.resume!r} found no checkpoint in {a.output}; "
                  "starting from scratch", flush=True)

    trainer.add_callback(NanProbe(trainer.model))
    trainer.add_callback(KLProbe(a.max_grad_norm))
    trainer.add_callback(Checkpointer(_is_main))
    print(f">>> CHECKPOINTING: every {a.save_steps} step(s), keep {a.save_total_limit} "
          f"rolling, archive every {a.archive_every} to {ARCHIVE}; stop file {STOP_FILE}",
          flush=True)
    trainer.train(resume_from_checkpoint=resume)

    if a.smoke:
        st = reward._state["byk"]
        # UNDER DDP EACH RANK SEES ONLY ITS OWN HALF of the completions. A budget
        # derived from one rank's slice is a per-rank view of a population statistic,
        # so gather the per-tier passing lengths across ranks before taking any
        # quantile.  [[feedback_zero_assertions_need_a_population_witness]]
        import torch.distributed as dist
        is_main = True
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            is_main = dist.get_rank() == 0
            local = {k: {"n": b["n"], "pass": b["pass"], "tok": b["tok"],
                         "clip": b["clip"], "lam": b["lam"],
                         "pass_lens": list(b["pass_lens"])} for k, b in st.items()}
            bucket = [None] * world
            dist.all_gather_object(bucket, local)
            merged: dict = {}
            for part in bucket:
                for k, b in (part or {}).items():
                    m = merged.setdefault(k, {"n": 0, "pass": 0, "tok": 0, "clip": 0,
                                              "lam": b["lam"], "pass_lens": []})
                    for f in ("n", "pass", "tok", "clip"):
                        m[f] += b[f]
                    m["pass_lens"].extend(b["pass_lens"])
            st = merged
            print(f">>> gathered smoke stats across {world} ranks", flush=True)
        out = {"tiers": {}, "budgets": {}}
        for k, b in sorted(st.items()):
            lens = b.get("pass_lens") or []
            out["tiers"][k] = {
                "n": b["n"], "pass_rate": b["pass"] / max(b["n"], 1),
                "mean_tok": b["tok"] / max(b["n"], 1),
                "clipped": b["clip"] / max(b["n"], 1),
                "lambda": b["lam"], "n_pass_lens": len(lens),
                # The lengths themselves, not just their count: a budget is a QUANTILE
                # of this list, so without it a published budget cannot be audited or
                # re-derived at a different quantile without re-running the model.
                "pass_lens": sorted(int(x) for x in lens),
            }
            if lens:
                out["budgets"][k] = (budget_from_lengths(lens, a.budget_quantile)
                                     if a.budget_quantile is not None
                                     else budget_from_lengths(lens))
        if not is_main:
            return 0
        print("\n=== SMOKE MEASUREMENT ===")
        print(json.dumps(out, indent=2))
        if a.measure_out:
            pathlib.Path(a.measure_out).write_text(json.dumps(out, indent=2))
            print(f"wrote {a.measure_out}")
    else:
        trainer.save_model(a.output)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
    # Expandable segments allocate through the CUDA VMM API (cuMemCreate/cuMemMap),
    # and memory obtained that way CANNOT be exported with cudaIpcGetMemHandle. vLLM's
    # custom all-reduce kernel is built on exactly that IPC handle, so enabling
    # expandable segments makes TP>1 die with:
    #     Failed: Cuda error custom_all_reduce.cuh:164 'invalid argument'
    # It was never the fix for the OOM either -- the binding constraint was the
    # [B, L, 262144] logits tensor, which --bsz addresses directly.
    sys.exit(main())
