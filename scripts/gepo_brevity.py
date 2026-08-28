#!/usr/bin/env python3
"""R9 (#845, the GEPO leg): shorten CoderX on LiveCodeBench WITHOUT breaking it.

    reward = (1 - lambda * min(ntok / budget, 1))  if the tests pass  else 0.0

This is the method #845 was scoped as. It is on-policy RL, not preference learning:
the model generates G rollouts per problem, each is EXECUTED against the real LCB
tests, and the group's length spread is the gradient. Nothing about it depends on a
teacher model.

WHY THE GROUP IS NOT DEGENERATE HERE
------------------------------------
The usual GRPO failure on an easy pool is that every rollout in a group gets the same
binary reward, the advantage is identically zero, and the run silently learns nothing.
Measured on the R7 generations, CoderX solves 750/760 of this pool and 181/190 problems
are solved by EVERY sample -- under a binary correctness reward that would be 95% dead
groups. It is alive here only because correctness is a GATE and the reward inside the
gate is continuous in length: eight correct rollouts of different lengths have eight
different rewards. Correctness is the floor, brevity is the signal.

THE BUDGET IS THE LOAD-BEARING HYPERPARAMETER, AND IT WAS MEASURED
-----------------------------------------------------------------
`min(ntok/budget, 1)` saturates. Every rollout past `budget` scores the same 1-lambda,
so a budget under the bulk of the length distribution silently flattens whole groups
back to zero advantage -- the same dead gradient by another route. Simulated on the
760 real R7 rollouts (fraction of groups with zero reward variance / mean within-group
reward std, which IS the gradient):

    budget   dead groups   within-group reward std
      4096   34.7%         0.061      <- lcb_brevity_reward.py's own default
      6144   16.8%         0.090
      8192    8.9%         0.105
     12288    1.6%         0.118      <- maximum signal; the default here
     16384    0.0%         0.096      <- no dead groups but the penalty barely bites

12288 is chosen because it MAXIMISES the measured signal, not because it is round.
[[feedback_grpo_completion_cap_kills_gradient]]

THE COMPLETION CAP IS A SECOND, INDEPENDENT WAY TO KILL THE RUN
---------------------------------------------------------------
A truncated rollout fails its tests and scores 0.0, which reads as "wrong" rather than
"too long". At 16384 that costs 1% of rollouts (measured p90=12503, p99=14564); at 8192
it would be 33%, poisoning a third of the signal with a length artefact. Hence
max_completion_length=16384 even though it is expensive.

TOPOLOGY (both bs2 GPUs, explicitly authorised for this run)
------------------------------------------------------------
  GPU0  vLLM rollout server (scripts/vllm_serve_qwen35.py -- vLLM 0.20.2 implements
        Qwen3_5MoeForCausalLM but never registered it)
  GPU1  this trainer; LoRA only, so the 49 GB bf16 base is resident once

The two live in different conda envs on purpose (`vllm` serves, `omnimergekit` trains --
their transformers pins diverge and must never be merged).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import random
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODEL = "/mnt/sdc/ream-work/armJ"
POOL = str(REPO / "eval/lcb/lcb_rl_pool.jsonl")

# Identical scope to dpo_brevity.py: attention (both kinds) + the always-on shared
# expert. These are RUNTIME module paths, not safetensors keys (bug-627).
#
# WHAT THIS SCOPE REACHES, MEASURED (2026-08-26 audit, run2 adapter + meta-device):
#   40/40 layers, correctly split across the hybrid -- self_attn on the 10
#   `full_attention` layers (3..39), linear_attn on the 30 `linear_attention`
#   layers (0..38), shared_expert on all 40. 620 tensors = 310 targets x 2.
#   Depth coverage is COMPLETE; there is no missing-layer bug here.
#
#   But the MoE is entirely frozen. With moe_intermediate_size=512, top_k=8 and
#   shared_expert_intermediate_size=512, the ACTIVE FFN width per token is
#   8*512 routed + 512 shared = 4608, of which this scope reaches 512 = 11.1%.
#   88.9% of the active FFN, plus the router that decides which experts fire,
#   carries no gradient.
#
# THE ROUTER EXCLUSION WAS WRONG, AND THE REASON GIVEN FOR IT WAS TWO REASONS:
#   (a) "the 184 routed experts are fused grouped-GEMM params, not nn.Linear, so
#       un-LoRA-able by construction" -- the nn.Parameter half is TRUE
#       (Qwen3_5MoeExperts stores gate_up_proj/down_proj as 3D nn.Parameter, and
#       mlp.gate.weight is nn.Parameter too), so `target_modules` cannot reach
#       either. But "by construction" is false on this stack: peft 0.20.0's
#       `target_parameters` (lora.ParamWrapper) reaches both. Measured cost at
#       r=32: experts 1,326,448,640 params = 5.18% of base (31x this scope --
#       that one is defensible on COST), router 2,856,960 = 0.011% (1/15th of
#       what this scope already trains -- essentially free).
#   (b) "every router-only lever in this repo has failed (T158, T196.SFT-3)" --
#       those are Gemma-4 98e/62e router-KD runs against loop/rumination. Other
#       architecture, other objective, other failure mode. Never tested on
#       Qwen3.6 brevity-RL.
#
# So --router-lora is OPT-IN and OFF by default: run1/run2 must stay bit-for-bit
# reproducible from this file. Turning it on ALSO forces dropout to 0, because
# lora.ParamWrapper refuses lora_dropout != 0 -- that is a real confound (scope
# AND regularisation move together), so a clean attribution needs a dropout-0
# control at the base scope. A NULL result does not need one: removing dropout
# can only increase adaptation, so it cannot manufacture a null.
LORA_REGEX = (
    r"model\.layers\.\d+\."
    r"(self_attn\.[qkvo]_proj"
    r"|linear_attn\.(in_proj_(qkv|a|b|z)|out_proj)"
    r"|mlp\.shared_expert\.(gate|up|down)_proj)"
)
N_LORA_TARGETS = 10 * 4 + 30 * 5 + 40 * 3   # = 310

# The router: one [num_experts, hidden] nn.Parameter per layer.
ROUTER_PARAM_RE = re.compile(r"model\.layers\.\d+\.mlp\.gate\.weight")
ROUTER_PARAMS = ["mlp.gate.weight"]
N_ROUTER_TARGETS = 40


def log(m):
    print("[gepo %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def load_pool(path: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(ln) for ln in Path(path).open() if ln.strip()]
    if limit:
        rows = rows[:limit]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vllm-base-url", default="http://127.0.0.1:8477")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole pool")
    # --- the two measured knobs -------------------------------------------------
    ap.add_argument("--length-budget", type=int, default=12288,
                    help="saturation point of the length penalty; 12288 maximises "
                         "measured within-group reward std on the R7 rollouts")
    ap.add_argument("--length-lambda", type=float, default=0.7)
    # --- run4: reward version + capability replay ---------------------------------
    # Default stays v1 so run1/run2/run3 reproduce from this file unchanged. v2 is
    # opt-in; see scripts/gepo_reward_v2.py for the measured reason it exists (v1's
    # length term owned 11.3% of within-group variance and DECAYED to 0.8% as lengths
    # converged -- GRPO normalises by group std, so that is 11% of the gradient
    # falling to ~1%).
    ap.add_argument("--reward", choices=["v1", "v2"], default="v1")
    ap.add_argument("--replay-pool", default="",
                    help="lambda=0 capability-replay pool (build_gepo_replay_pool.py). "
                         "Empty = no replay. Requires --reward v2: v1 has no "
                         "reward_kind dispatch and would score every replay row 0.0.")
    ap.add_argument("--replay-n", type=int, default=0,
                    help="number of replay problems to mix in (0 = all of them)")
    ap.add_argument("--replay-seed", type=int, default=0)
    ap.add_argument("--replay-balance", choices=["equal", "proportional"],
                    default="equal",
                    help="how --replay-n is split across reward_kinds. equal = "
                         "same count per tier (default; the pool ratio is the "
                         "inverse of the loss ratio it defends)")
    ap.add_argument("--replay-no-think", action="store_true",
                    help="render REPLAY rows with enable_thinking=False (LCB rows "
                         "unchanged). Requires pre-rendering every prompt to text, "
                         "because TRL's chat_template_kwargs is global. Without this, "
                         "GPQA replay rollouts run 16-18k tokens, mostly never reach "
                         "an answer, and the tier scores zero for every rollout.")
    ap.add_argument("--max-completion", type=int, default=16384)
    ap.add_argument("--max-prompt", type=int, default=4096)
    ap.add_argument("--server-max-model-len", type=int, default=24576,
                    help="must match the --max_model_len the vLLM rollout server was "
                         "launched with; prompt+completion is checked against it")
    # --- GEPO / GRPO ------------------------------------------------------------
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="with per_device=1 this is the generation batch; must be "
                         "divisible by --num-generations")
    ap.add_argument("--beta", type=float, default=0.02, help="KL to the frozen base")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--epsilon-low", type=float, default=1.0)
    ap.add_argument("--epsilon-high", type=float, default=1e6)
    ap.add_argument("--no-gepo", action="store_true",
                    help="GSPO control arm: per-sample q_i instead of the group "
                         "expectation. Same code path, one denominator changed.")
    # --- LoRA -------------------------------------------------------------------
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--router-lora", action="store_true",
                    help="ALSO LoRA the MoE router (mlp.gate.weight, 40 params, "
                         "+0.011%% of base) via peft target_parameters. Forces "
                         "--dropout 0: lora.ParamWrapper refuses nonzero dropout. "
                         "OFF by default so run1/run2 stay reproducible.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--preflight", action="store_true",
                    help="build everything, gate everything, then stop before train()")
    args = ap.parse_args()

    # The divisibility that matters is PER RANK, not global. Each DDP rank
    # generates its own batch of per_device(1) x grad_accum samples, and GEPO's
    # group reduction (logsumexp over the group) needs every member of a group
    # resident on the SAME rank -- a group split 2-and-2 across two ranks has no
    # complete group to reduce over. Measured 2026-08-23: relaxing this to the
    # global batch (grad_accum x world_size) let a G=4 / grad_accum=2 / 2-GPU
    # config through, and GEPOTrainer refused at the first step with
    # "GENERATION batch 2 divisible by num_generations 4".
    #
    # So grad_accum must be a multiple of num_generations on EVERY path; adding
    # GPUs multiplies the number of groups per step, it does not relax this.
    _world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.grad_accum % args.num_generations:
        sys.exit(f"REFUSE: --grad-accum {args.grad_accum} is not divisible by "
                 f"--num-generations {args.num_generations}. Each rank must hold "
                 f"whole generation groups; groups cannot span ranks.")
    log(f"GROUPS_OK {args.grad_accum // args.num_generations} group(s)/rank x "
        f"world={_world} = {args.grad_accum * _world // args.num_generations} "
        f"group(s)/step (G={args.num_generations})")

    # ---------------------------------------------------------------- reward FIRST
    # LcbVerifier forks a persistent worker, and forking a process that already holds
    # a CUDA context is a deadlock source (bug-622). Everything torch/CUDA below this
    # point; nothing above it.
    from scripts.lcb_brevity_reward import LcbVerifier, make_lcb_brevity_reward
    log("constructing LCB verifier (pre-CUDA fork)")
    verifier = LcbVerifier(timeout=10.0)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from scripts.gepo_trainer import GEPOTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.reward == "v2":
        from scripts.gepo_reward_v2 import make_gepo_reward_v2
        # log_every=8, not 64: the smoke runs ~8-24 rollouts total, and the launcher's
        # REPLAY_TIER_ALIVE gate READS this line. An interval longer than the smoke
        # would make a healthy run fail the gate for lack of output.
        reward = make_gepo_reward_v2(tok, verifier, max_completion=args.max_completion,
                                     log_every=8)
        log("REWARD_V2 group-relative brevity + reward_kind dispatch "
            f"(max_completion={args.max_completion})")
    else:
        reward = make_lcb_brevity_reward(tok, budget=args.length_budget,
                                         lam=args.length_lambda, verifier=verifier,
                                         log_every=64)
        log("REWARD_V1 fixed-budget length penalty "
            f"(budget={args.length_budget} lam={args.length_lambda})")

    # ---------------------------------------------------------------- dataset
    rows = load_pool(args.pool, args.limit)
    for r in rows:                      # LCB rows predate reward_kind; name it explicitly
        r.setdefault("meta", {}).setdefault("reward_kind", "lcb_exec")

    # --- capability replay ---------------------------------------------------------
    # GEPO's reward has support ONLY on LCB coding problems, and the paired census
    # measured what that costs: GPQA -7.58pp (p=0.0167) and HumanEval -3.66pp
    # (p=0.0312), monotone with dose, and run3 ended -12.99pp vs the untrained base on
    # LCB itself. Replay rows carry meta.length_lambda=0.0 -> pure correctness, so they
    # defend capability WITHOUT putting length pressure on the reasoning GEPO erodes.
    #
    # Mixed at the ROW level, which is safe: a GRPO group is G rollouts of ONE prompt,
    # so no group ever spans an LCB and a replay problem, and each tier gets its own
    # advantage normalisation.
    n_lcb = len(rows)
    if args.replay_pool:
        if args.reward != "v2":
            sys.exit("REFUSE: --replay-pool requires --reward v2. v1 has no "
                     "reward_kind dispatch and would silently score every replay row "
                     "0.0 -- an all-zero tier has no within-group spread, contributes "
                     "no gradient, and drags the policy while looking healthy.")
        rrows = load_pool(args.replay_pool, 0)
        if not rrows:
            sys.exit(f"REFUSE: replay pool {args.replay_pool} is empty")
        bad = [r for r in rrows if float(r.get("meta", {}).get("length_lambda", 1.0)) != 0.0]
        if bad:
            sys.exit(f"REFUSE: {len(bad)} replay rows carry a nonzero length_lambda -- "
                     "replay must be a pure correctness gate")
        random.Random(args.replay_seed).shuffle(rrows)
        if args.replay_n:
            # STRATIFY by reward_kind rather than taking a proportional slice. The pool
            # is 250 mc_letter / 471 mbpp_exec (35/65), but the losses these two tiers
            # defend run the OTHER way: GPQA -6.57pp vs HumanEval -3.05pp against the
            # untrained base. A proportional draw would spend most of the replay budget
            # on the smaller loss. Equal counts is the neutral choice; --replay-balance
            # proportional restores the pool's own ratio if that is ever wanted.
            if args.replay_balance == "equal":
                by: dict[str, list] = {}
                for r in rrows:
                    by.setdefault(r["meta"].get("reward_kind", "?"), []).append(r)
                per = args.replay_n // max(len(by), 1)
                short = [k for k, v in by.items() if len(v) < per]
                if short:
                    sys.exit(f"REFUSE: replay tier(s) {short} have fewer than {per} rows; "
                             "an equal split is impossible. Lower --replay-n or use "
                             "--replay-balance proportional.")
                rrows = [r for k in sorted(by) for r in by[k][:per]]
                random.Random(args.replay_seed).shuffle(rrows)
            else:
                rrows = rrows[:args.replay_n]
        kinds: dict[str, int] = {}
        for r in rrows:
            k = r["meta"].get("reward_kind", "?")
            kinds[k] = kinds.get(k, 0) + 1
        rows = rows + rrows
        random.Random(args.replay_seed).shuffle(rows)
        log(f"REPLAY_MIX lcb={n_lcb} replay={len(rrows)} {kinds} "
            f"-> {len(rows)} problems ({len(rrows)/len(rows)*100:.1f}% replay)")
    # `meta` carries lists-of-dicts with heterogeneous keys, which pyarrow will either
    # reject or silently coerce. It travels as a JSON STRING; the reward already
    # json.loads a str meta, which is exactly why that branch exists.
    # `gold` is a REQUIRED column, not an optional one. TRL forwards every dataset
    # column to the reward as a list, and the mc_letter verifier scores against gold;
    # if the column is absent the reward receives gold=None, falls back to "", and
    # EVERY replay multiple-choice row scores 0.0 -- a silently dead tier. LCB and
    # MBPP rows carry "" because their verifiers execute tests instead.
    # --replay-no-think: PRE-RENDER every prompt to a string so thinking can be turned
    # off PER ROW. Why it has to work this way:
    #   * TRL's chat_template_kwargs is GLOBAL (grpo_config.py:556, applied to the whole
    #     batch at grpo_trainer.py:1758), so it cannot express "thinking off for replay,
    #     on for LCB" -- using it would silently change the LCB arm and break
    #     comparability with run2/run3.
    #   * armJ's template honours no inline /no_think marker; only the enable_thinking
    #     kwarg (chat_template.jinja:149 emits an empty <think></think> when false).
    #   * TRL takes a text path when the prompt is a str (grpo_trainer.py:1780,
    #     `self.processing_class(text=prompts)`), applying NO template. So a
    #     fully pre-rendered dataset gives exact per-row control.
    # LCB rows are rendered with the DEFAULT kwargs, i.e. byte-identical to what TRL
    # would have produced itself, so this changes nothing for the LCB arm.
    #
    # WHY replay is rendered no-think at all: measured 2026-08-28, GPQA rollouts with
    # unbounded thinking run 16-18k tokens and 2 of 4 never answer inside 24576 -- the
    # tier scores 0 for every rollout and cannot fit memory or wall clock. With thinking
    # off it answers 12/12 at p50 1595 tokens, 9/12 correct. That is not a departure
    # from what we score: the served GPQA eval already runs at thinking_est p50=0 and
    # completion p50 ~800 on every arm. [[project_gpqa_cot_needs_16k_on_armj]]
    if args.replay_no_think:
        probe = [{"role": "user", "content": "probe"}]
        d_on = tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
        d_off = tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False)
        # A template that ignores enable_thinking would render identically and hand the
        # replay tier thinking-ON prompts anyway -- the exact failure this flag exists
        # to prevent, and it would be invisible until the tier scored zero again.
        if d_on == d_off:
            sys.exit("REFUSE: this chat template ignores enable_thinking (rendering is "
                     "identical with and without it). --replay-no-think cannot work "
                     "here; it would silently produce thinking-ON replay prompts.")
        log(f"NO_THINK_RENDER_OK template honours enable_thinking "
            f"(delta {len(d_on)} vs {len(d_off)} chars)")

        def render(r):
            kw = {} if r["meta"].get("reward_kind", "lcb_exec") == "lcb_exec" \
                else {"enable_thinking": False}
            return tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                           add_generation_prompt=True, tokenize=False,
                                           **kw)
        ds = Dataset.from_list([
            {"prompt": render(r),
             "meta": json.dumps(r["meta"]),
             "gold": str(r.get("gold") or "")}
            for r in rows
        ])
        n_nt = sum(1 for r in rows
                   if r["meta"].get("reward_kind", "lcb_exec") != "lcb_exec")
        log(f"PRERENDER text prompts: {len(ds)} rows, {n_nt} rendered no-think")
    else:
        ds = Dataset.from_list([
            {"prompt": [{"role": "user", "content": r["prompt"]}],
             "meta": json.dumps(r["meta"]),
             "gold": str(r.get("gold") or "")}
            for r in rows
        ])
    n_gold = sum(1 for r in rows if str(r.get("gold") or ""))
    n_mc = sum(1 for r in rows if r.get("meta", {}).get("reward_kind") == "mc_letter")
    if n_mc and n_gold < n_mc:
        sys.exit(f"REFUSE: {n_mc} mc_letter rows but only {n_gold} carry a gold answer. "
                 "Those rows would score 0.0 for every rollout regardless of the model.")
    log(f"pool: {len(ds)} problems from {args.pool}"
        + (f" (+replay; {n_mc} mc_letter rows, all with gold)" if n_mc else ""))

    # `return_dict=False` is load-bearing. transformers 5.5.0 returns a BatchEncoding
    # from apply_chat_template(tokenize=True), so len() counts its KEYS -- it reports 2
    # for every prompt in the pool regardless of length, and the refusal below could
    # never fire. Measured 2026-08-23: same prompt, len()=2 as a BatchEncoding, 24 as
    # a list. A gate whose input is constant is not a gate.
    plen = [len(tok(ex["prompt"], add_special_tokens=False).input_ids)
            if isinstance(ex["prompt"], str)
            else len(tok.apply_chat_template(ex["prompt"], tokenize=True,
                                             add_generation_prompt=True,
                                             return_dict=False))
            for ex in ds]
    log(f"prompt tokens p50/p90/max: {int(st.median(plen))}/"
        f"{sorted(plen)[int(len(plen) * .9)]}/{max(plen)}  (cap {args.max_prompt})")
    # TRL 1.9.0 has no `max_prompt_length` at all -- the trainer never truncates, so
    # the hazard is not silent trimming, it is the SERVER. prompt + completion must fit
    # inside vLLM's --max_model_len or the rollout comes back as an HTTP 400 and the
    # whole group dies looking like model failure. Both halves are checked here because
    # neither is visible from inside TRL.
    if max(plen) > args.max_prompt:
        sys.exit(f"REFUSE: longest prompt is {max(plen)} tokens > --max-prompt "
                 f"{args.max_prompt}.")
    budget = args.max_prompt + args.max_completion
    if budget > args.server_max_model_len:
        sys.exit(f"REFUSE: --max-prompt {args.max_prompt} + --max-completion "
                 f"{args.max_completion} = {budget} > --server-max-model-len "
                 f"{args.server_max_model_len}; vLLM rejects the request with HTTP 400 "
                 f"once a rollout runs long, and it will do so only PART-WAY through "
                 f"the run.")
    log(f"CTX_BUDGET_OK {max(plen)} prompt + {args.max_completion} completion "
        f"<= {args.server_max_model_len}")

    # ---------------------------------------------------------------- model
    log(f"loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    # bug-629, and TRL's GRPO path carries the identical trap: MoE-ness is decided by
    # ATTRIBUTE EXISTENCE (`getattr(text_config, "output_router_logits", None) is not
    # None`), so this config flag alone does NOT disable it. GRPOConfig's
    # router_aux_loss_coef defaults to 0.001 -- NON-zero -- which flips
    # aux_loss_enabled True, and TRL then force-passes output_router_logits=True on
    # every forward (grpo_trainer.py:1509). At 184 experts x 40 layers that is the
    # 6 GiB load_balancing_loss_func OOM. The coefficient below is the real lever;
    # this flag is belt.
    model.config.output_router_logits = False
    if getattr(model.config, "router_aux_loss_coef", 0):
        model.config.router_aux_loss_coef = 0.0

    hit = [n for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and re.fullmatch(LORA_REGEX, n)]
    if len(hit) != N_LORA_TARGETS:
        sys.exit(f"REFUSE: LORA_REGEX matched {len(hit)} modules, expected "
                 f"{N_LORA_TARGETS}. The module naming changed; fix the regex, do "
                 f"not train a different scope than the one that was reviewed.")
    log(f"LORA_TARGETS_OK {len(hit)} modules")

    lora_kwargs = dict(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                       bias="none", task_type="CAUSAL_LM",
                       target_modules=LORA_REGEX)
    dropout_used = args.dropout
    if args.router_lora:
        # Gate the router the same way the Linear scope is gated: count the real
        # parameters on the real model and refuse on any mismatch, so a silent
        # rename can never quietly train a different scope than the reviewed one.
        rhit = [n for n, _ in model.named_parameters()
                if ROUTER_PARAM_RE.fullmatch(n)]
        if len(rhit) != N_ROUTER_TARGETS:
            sys.exit(f"REFUSE: router param match {len(rhit)}, expected "
                     f"{N_ROUTER_TARGETS} (mlp.gate.weight, one per layer). "
                     f"The MoE module naming changed; fix ROUTER_PARAM_RE.")
        # lora.ParamWrapper raises on nonzero dropout. Force it and say so loudly
        # rather than letting peft fail 40 minutes into model load.
        if dropout_used != 0.0:
            log(f"ROUTER_LORA: forcing lora_dropout {dropout_used} -> 0.0 "
                f"(lora.ParamWrapper does not support dropout). This is a BASIS "
                f"CHANGE vs run1/run2, which trained at 0.05.")
            dropout_used = 0.0
        lora_kwargs["lora_dropout"] = dropout_used
        lora_kwargs["target_parameters"] = ROUTER_PARAMS
        log(f"ROUTER_LORA_TARGETS_OK {len(rhit)} params")

    peft_cfg = LoraConfig(**lora_kwargs)

    # ---------------------------------------------------------------- config
    cfg = GRPOConfig(
        output_dir=args.out,
        seed=args.seed,
        per_device_train_batch_size=1,   # [1, 16384, 248320] bf16 logits = 8.1 GiB
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        beta=args.beta,
        # No max_prompt_length: TRL 1.9.0 removed the field (it does not truncate
        # prompts). The prompt budget is enforced by the CTX_BUDGET_OK gate above.
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        # GEPO/GSPO are sequence-level by construction; the group-expectation
        # denominator is what GEPOTrainer swaps in.
        importance_sampling_level="sequence",
        loss_type="grpo",
        epsilon=args.epsilon_low,
        epsilon_high=args.epsilon_high,
        router_aux_loss_coef=0.0,        # THE lever -- see the note above
        # ROLLOUTS RUN IN-PROCESS THROUGH HF TRANSFORMERS, NOT vLLM.
        #
        # The vLLM server path cost a full day and produced nothing. vLLM 0.20.2
        # ships Qwen3.5's text classes but wires none of the machinery around them,
        # so serving armJ needed a hand-written architecture wrapper patching four
        # separate upstream gaps (registry entry, weight-name mapper, IsHybrid,
        # SupportsMRoPE). Three were provably right -- the weight gate reports
        # expected=573 loaded=573 missing=0 -- and the fourth, the hand-rolled
        # M-RoPE position path, still emits token soup ("#  20f 2neduARSEa ("),
        # which is why the 2026-08-23 smoke scored 0.0 on all 32 rollouts and
        # logged grad_norm=0 on every step.
        #
        # transformers 5.9.0 supports qwen3_5_moe_text natively. This trainer
        # already loads armJ through it (693/693 weights, 310 LoRA modules), so
        # generating through the same object removes an entire class of failure:
        # there is no second engine that can disagree with the one being trained.
        # It also collapses the trainer-vs-sampler logp gap (measured at ~7.7 nats,
        # importance ratio ~4e-23) to zero BY CONSTRUCTION -- same weights, same
        # kernels, same tokenizer.
        use_vllm=False,
        # num_iterations>1 is what makes old_per_token_logps exist off the vLLM
        # path (grpo_trainer.py:2563). GEPO needs lq = logp_{theta_old}; with two
        # inner iterations that is the policy as it stood before this batch's
        # updates, which is the quantity the importance weight is defined against.
        num_iterations=2,
        gradient_checkpointing=True,
        bf16=False,                      # weights already bf16; autocast only added
                                         # a fp32 logits copy (25.5 GiB) in the DPO run
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=None,           # results/adapters are never deleted
        report_to=[],
    )

    trainer = GEPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        reward_funcs=[reward], peft_config=peft_cfg,
        processing_class=tok,
    )
    if not args.no_gepo:
        trainer.gepo = True
    else:
        trainer.gepo = False
        log("GSPO CONTROL ARM -- per-sample denominator, NOT GEPO")

    # ------------------------------------------------------------------- gates
    if getattr(trainer, "aux_loss_enabled", False):
        sys.exit("REFUSE: trainer.aux_loss_enabled is True. TRL will force "
                 "output_router_logits=True and OOM in load_balancing_loss_func "
                 "(bug-629). router_aux_loss_coef must be exactly 0.0.")
    log("AUX_LOSS_OFF — trainer.aux_loss_enabled=False")

    _vllm_is = trainer.use_vllm and trainer.vllm_importance_sampling_correction
    _multi_iter = trainer.num_iterations > 1
    if not (_vllm_is or _multi_iter):
        # This is not a preference. GEPO needs lq = logp_{theta_old}, and TRL only
        # materialises old_per_token_logps under one of these two conditions
        # (grpo_trainer.py:2563). Without it GEPOTrainer._require_old_logps refuses
        # mid-run, an hour into generation.
        sys.exit("REFUSE: GEPO has no denominator to build. TRL only computes "
                 "old_per_token_logps when (use_vllm AND "
                 "vllm_importance_sampling_correction) or num_iterations>1, and "
                 f"neither holds: use_vllm={trainer.use_vllm}, "
                 f"num_iterations={trainer.num_iterations}.")
    log(f"OLD_LOGPS_WILL_EXIST — via {'vllm-IS' if _vllm_is else 'num_iterations>1'}"
        f" (use_vllm={trainer.use_vllm}, num_iterations={trainer.num_iterations})")

    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if n_train == 0:
        sys.exit("REFUSE: 0 trainable parameters — LoRA did not attach.")
    log(f"TRAINABLE {n_train / 1e6:.1f}M params")

    meta = {"task": "R9 GEPO brevity", "model": args.model, "pool": args.pool,
            "n_problems": len(ds), "gepo": bool(trainer.gepo),
            "length_budget": args.length_budget, "length_lambda": args.length_lambda,
            # The reward function IS a basis: a v1 row and a v2 row are not comparable
            # on reward, mean_length or clipped_ratio, because v2 changes what those
            # numbers mean. Recorded so the two can never be silently tabled together.
            "reward": args.reward,
            # no_think is part of the replay tier's IDENTITY, not a tuning detail: a
            # no-think replay row trains a different behaviour from a thinking one, and
            # two runs that differ only here are not comparable.
            "replay": ({"pool": args.replay_pool, "n": len(rows) - n_lcb,
                        "n_lcb": n_lcb, "seed": args.replay_seed,
                        "no_think": bool(args.replay_no_think),
                        "prerendered": bool(args.replay_no_think),
                        "frac": round((len(rows) - n_lcb) / max(len(rows), 1), 4)}
                       if args.replay_pool else None),
            "max_completion": args.max_completion, "num_generations": args.num_generations,
            "grad_accum": args.grad_accum, "beta": args.beta, "lr": args.lr,
            "epochs": args.epochs, "seed": args.seed,
            "sampler": {"temperature": args.temperature, "top_p": args.top_p,
                        "top_k": args.top_k},
            "lora": {"r": args.r, "alpha": args.alpha, "regex": LORA_REGEX,
                     "n_targets": len(hit),
                     # scope is a BASIS fact -- a run3 row must never be tabled
                     # against run1/run2 without these three being read first.
                     "router_lora": bool(args.router_lora),
                     "n_router_targets": (N_ROUTER_TARGETS if args.router_lora
                                          else 0),
                     "dropout": dropout_used},
            "trainable_params": n_train, "started": time.strftime("%F %T %Z")}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log("run_meta.json written")

    if args.preflight:
        log("PREFLIGHT ONLY — every gate green, stopping before train()")
        print("GEPO_PREFLIGHT_OK", flush=True)
        return 0

    log("=== train ===")
    trainer.train()
    trainer.save_model(args.out)
    meta["finished"] = time.strftime("%F %T %Z")
    (Path(args.out) / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("GEPO_TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
