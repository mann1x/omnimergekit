#!/usr/bin/env python3
"""R7b (#845) — LoRA DPO for brevity on Qwen3.6-27B-A3B-CoderX (armJ).

Trains on the ON-POLICY pair set from brevity_pairs.py: both sides are CoderX's own
correct solutions to the same LCB problem, one short and one long. Correctness is
constant across the pair by construction, so the only thing the preference encodes is
length. The cross (teacher) set is deliberately NOT the default -- see #845: at matched
sampling budget coder+MPE has no measurable brevity edge (ratio p50 0.96x, win-rate CI
[49.2, 64.0] straddling 50%), while CoderX's own within-problem spread is 2.13x. The
signal is in the student's variance, not in the teacher.

RENDERING IS GOLD-CHECKED, NOT ASSUMED
--------------------------------------
The generations were produced by llama-server with `--reasoning-format deepseek`, which
splits the assistant turn into `reasoning` + `content`. To train we must put it back
together EXACTLY as the model's own chat template renders an assistant turn, or the loss
is computed on a string the model never emits. armJ's chat_template.jinja renders:

    <|im_start|>assistant\\n<think>\\n{reasoning|trim}\\n</think>\\n\\n{content}<|im_end|>\\n

and `add_generation_prompt` ends the prompt at `...<|im_start|>assistant\\n<think>\\n`.
So prompt/completion split at that boundary. `assert_render_matches_template()` proves
it by re-rendering the same pair through apply_chat_template and demanding byte
equality; if the template ever changes under us the run REFUSES instead of silently
training on a malformed string.

TRUNCATION IS FORBIDDEN
-----------------------
A truncated `rejected` no longer carries the length it is being penalised for -- the
one axis this run trains on. `max_length` is therefore a FILTER with a reported dose,
never a trim, and the run asserts post-tokenisation that nothing hit the cap. Measured
on the 165-pair on-policy set: 16384 keeps 160 (97.0%), 12288 keeps only 83 (50.3%),
so 16384 is the knee.

LORA SCOPE
----------
armJ is 40 layers: 30 `linear_attention` (SSM) + 10 `full_attention`, 184 routed
experts + 1 shared expert per layer. Adapters go on the attention paths and the SHARED
expert -- the always-on route. The 7,360 routed expert tensors are NOT adapted: that is
where the model's code knowledge lives, brevity is a policy/termination behaviour, and
adapting them is both enormous and the fastest way to damage the thing CoderX is good
at. Same scoping rationale as lora_sft_antiloop.py (T174.3).

THE LENGTH-BIAS HAZARD (why --loss-type is a flag and why the gate is empirical)
-------------------------------------------------------------------------------
DPO's implicit reward is a SUM of per-token log-ratios, so it scales with length and is
known to exploit length as a spurious feature. Here `chosen` is systematically the
SHORTER side, so that exploitation points the way we want -- which is precisely the
danger: the cheapest way to win this objective is to become blanket-terse and stop
solving problems. Nothing in the loss prevents that. The ONLY honest guard is to
re-measure pass rate AND length after training on the banked cells (lcb_v6_77q,
humaneval_full_think, multipl_e_100), and to read `pass` before `TCS`.
`sigmoid_norm` (length-normalised) and `ld_alpha` (LD-DPO desensitisation) are exposed
for the ablation; neither substitutes for the measurement.

Run on bs2 GPU1 with the omnimergekit python. --preflight first, always.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

MODEL = "/mnt/sdc/ream-work/armJ"
PAIRS = "/mnt/sdc/ml/brevity/pairs_onpolicy.jsonl"

# attention (both kinds) + the ALWAYS-ON shared expert. Explicitly NOT `mlp.gate`
# (the router; every router-only lever in this repo has failed -- T158, T196.SFT-3)
# and not `shared_expert_gate` (a scalar gate, nothing to adapt).
#
# NOTE ON NAMES: these are the RUNTIME nn.Module paths, which are NOT the
# safetensors key names. The checkpoint keys carry a `language_model.` segment
# (`model.language_model.layers.N...`) that the loader strips, so a regex written
# off the index file matches ZERO modules and peft raises "Target modules ... not
# found". Always read named_modules(), never the weight map. bug-627.
#
# The 184 routed experts do not appear here at all: at runtime they are fused
# grouped-GEMM parameters, not nn.Linear, so they are un-LoRA-able by construction.
# That happens to be the scoping we want (brevity is a policy behaviour; the routed
# experts hold the code knowledge) but it is the architecture enforcing it, not us.
LORA_REGEX = (
    r"model\.layers\.\d+\."
    r"(self_attn\.[qkvo]_proj"
    r"|linear_attn\.(in_proj_(qkv|a|b|z)|out_proj)"
    r"|mlp\.shared_expert\.(gate|up|down)_proj)"
)
# 10 full-attn layers x 4 + 30 linear-attn layers x 5 + 40 layers x 3 shared-expert
N_LORA_TARGETS = 10 * 4 + 30 * 5 + 40 * 3   # = 310


def log(m):
    print("[dpo %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def mem(tag):
    log(f"MEM {tag}: alloc {torch.cuda.memory_allocated() / 2**30:6.1f} GiB | "
        f"reserved {torch.cuda.memory_reserved() / 2**30:6.1f} GiB")


class MemProbe(TrainerCallback):
    """Gate + instrument at the only moment the answer is observable.

    `gradient_checkpointing=True` in the config is a REQUEST; HF Trainer applies
    it inside train(), and PEFT models are a known place where it silently does
    not take (frozen inputs => no grad => the checkpoint is a no-op). At 13.8k
    tokens x 40 layers the difference is ~40 GiB of stored activations, i.e. the
    difference between fitting on this card and not. Asking the MODEL beats
    trusting the flag. [[feedback_a_declared_gate_is_not_a_wired_gate]]
    """

    def on_train_begin(self, args, state, control, model=None, **kw):
        on = bool(getattr(model, "is_gradient_checkpointing", False))
        mem("train_begin")
        log(f"GRAD_CKPT={on}")
        if args.gradient_checkpointing and not on:
            raise SystemExit(
                "REFUSE: gradient_checkpointing was requested but the model "
                "reports is_gradient_checkpointing=False. Every layer's "
                "activations are being kept and this run will OOM.")


def render(tok, pair):
    """-> (prompt, chosen_completion, rejected_completion) as raw strings."""
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": pair["prompt"]}],
        tokenize=False, add_generation_prompt=True)

    def comp(side):
        r = (side["reasoning"] or "").strip()
        # Ends at `<|im_end|>` with NO trailing newline, and that is load-bearing
        # twice over. (1) It is the true generation target: at serve time the model
        # emits `<|im_end|>` and decoding stops there; the newline the template puts
        # after it only exists to separate it from the NEXT turn, which never comes.
        # (2) TRL's own `add_eos` appends the eos token whenever the completion does
        # not literally END with it (dpo_trainer.py:1007). With a trailing newline
        # that test fails and TRL silently appends a SECOND `<|im_end|>`, training
        # the model to emit terminator-newline-terminator. The gold check below
        # accounts for the one-newline difference explicitly rather than being
        # relaxed. [[feedback_gemma4_double_terminator_trl_add_eos]]
        return f"{r}\n</think>\n\n{side['content']}<|im_end|>"

    return prompt, comp(pair["chosen"]), comp(pair["rejected"])


def assert_render_matches_template(tok, pair) -> None:
    """prompt+completion must be byte-identical to the template's own two-turn render.

    This is the gold: it proves the hand-rolled split sits exactly on the
    `<think>\\n` boundary that add_generation_prompt emits, for THIS tokenizer's
    template. A silent drift here would train the model on a string it never
    produces at serve time. [[feedback_match_the_shipped_artifact_not_your_note]]
    """
    prompt, chosen, _ = render(tok, pair)
    side = pair["chosen"]
    full = tok.apply_chat_template(
        [{"role": "user", "content": pair["prompt"]},
         {"role": "assistant", "content": side["content"],
          "reasoning_content": (side["reasoning"] or "").strip()}],
        tokenize=False, add_generation_prompt=False,
        preserve_thinking=True)
    # The completion deliberately stops at `<|im_end|>`; the template appends one
    # trailing newline to separate turns. Add exactly that one newline back for the
    # comparison -- anything else diverging is still fatal.
    if full != prompt + chosen + "\n":
        # show the first divergence so a template change is diagnosable, not just fatal
        a, b = prompt + chosen + "\n", full
        i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                 min(len(a), len(b)))
        sys.exit("REFUSE: render does not match the chat template.\n"
                 f"  first divergence at char {i}\n"
                 f"  ours     ...{a[max(0, i-60):i+60]!r}\n"
                 f"  template ...{b[max(0, i-60):i+60]!r}")
    log("RENDER_GOLD_OK — prompt+completion+'\\n' is byte-identical to "
        "apply_chat_template; completion ends exactly at <|im_end|>")
    # A gate on the actual failure mode, not on the render alone: prove TRL's
    # add_eos will be a NO-OP on this string. If the tokenizer's eos ever stops
    # being <|im_end|>, this fires instead of silently double-terminating.
    if not chosen.endswith(tok.eos_token):
        sys.exit(f"REFUSE: completion does not end with the tokenizer eos "
                 f"({tok.eos_token!r}); TRL add_eos would append a second "
                 f"terminator. Completion tail: {chosen[-40:]!r}")
    log(f"EOS_NOOP_OK — completion already ends with {tok.eos_token!r}")


def build(tok, path, max_length, max_prompt_length):
    rows = [json.loads(l) for l in open(path)]
    log(f"loaded {len(rows)} pairs from {path}")
    assert_render_matches_template(tok, rows[0])

    kept, drop_len, lens = [], 0, []
    for p in rows:
        pr, ch, rj = render(tok, p)
        n_p = len(tok(pr, add_special_tokens=False).input_ids)
        n_c = len(tok(ch, add_special_tokens=False).input_ids)
        n_r = len(tok(rj, add_special_tokens=False).input_ids)
        # the REJECTED side binds: it is the long one by construction
        if n_p + n_r > max_length or n_p > max_prompt_length:
            drop_len += 1
            continue
        kept.append({"prompt": pr, "chosen": ch, "rejected": rj})
        lens.append((n_p, n_c, n_r))

    if not kept:
        sys.exit(f"REFUSE: 0/{len(rows)} pairs fit max_length={max_length}")
    log(f"DOSE: kept {len(kept)}/{len(rows)} ({100*len(kept)/len(rows):.1f}%), "
        f"dropped {drop_len} over max_length={max_length} "
        "(FILTERED, never truncated — a trimmed `rejected` would not carry the "
        "length it is penalised for)")
    log("  prompt   tok p50=%d max=%d" % (st.median([x[0] for x in lens]),
                                          max(x[0] for x in lens)))
    log("  chosen   tok p50=%d max=%d" % (st.median([x[1] for x in lens]),
                                          max(x[1] for x in lens)))
    log("  rejected tok p50=%d max=%d" % (st.median([x[2] for x in lens]),
                                          max(x[2] for x in lens)))
    ratio = [x[2] / max(x[1], 1) for x in lens]
    log("  realized length ratio p50=%.2fx (this IS the training signal)"
        % st.median(ratio))
    return Dataset.from_list(kept), lens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--loss-type", default="sigmoid",
                    help="sigmoid (default) | sigmoid_norm (length-normalised) | ipo ...")
    ap.add_argument("--ld-alpha", type=float, default=None,
                    help="LD-DPO length desensitisation; None = off")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="sdpa",
                    help="sdpa (default) | flash_attention_2 | eager. NEVER eager "
                         "at these sequence lengths — see the load-site comment.")
    ap.add_argument("--amp", action="store_true", default=False,
                    help="re-enable accelerate bf16 autocast. Off by default: the "
                         "weights are already bf16 and the adapters fp32, so it "
                         "adds only a 25 GiB fp32 copy of the logits.")
    ap.add_argument("--liger", action="store_true", default=False,
                    help="use the fused chunked DPO loss instead of precomputed "
                         "reference logps. MEASURED WORSE here (see the config "
                         "comment): liger chunks over the BATCH, and one pair is "
                         "already the smallest batch, so it fuses fwd+bwd over the "
                         "whole [2, seq, 248320] tensor and peaks HIGHER. Mutually "
                         "exclusive with --precompute-ref.")
    ap.add_argument("--no-precompute-ref", dest="precompute_ref",
                    action="store_false", default=True,
                    help="compute reference logps inside every training step "
                         "instead of once up front. Costs a second full logits "
                         "tensor at peak for no change in the numbers.")
    ap.add_argument("--preflight", action="store_true",
                    help="load + wrap + ONE forward/backward on the LONGEST kept pair, "
                         "report peak VRAM, then exit without training")
    args = ap.parse_args()

    log(f"torch {torch.__version__} | visible GPUs {torch.cuda.device_count()}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds, lens = build(tok, args.pairs, args.max_length, args.max_prompt_length)

    if args.preflight:
        # hardest example first: if the longest pair fits, the rest do.
        worst = max(range(len(lens)), key=lambda i: lens[i][0] + lens[i][2])
        ds = Dataset.from_list([ds[worst]])
        log(f"PREFLIGHT: single hardest pair, prompt+rejected="
            f"{lens[worst][0] + lens[worst][2]} tok")

    # attn_implementation: SDPA, NOT eager. lora_sft_antiloop.py uses eager because
    # Gemma-4 needs it for logit softcapping; Qwen3.5-MoE does not, and eager
    # materialises the full n x n score matrix in fp32 — at 13.8k tokens that is a
    # single 22.7 GiB softmax allocation and an instant OOM on a 96 GB card even
    # with the model resident. bug-628.
    log(f"loading {args.model} (bf16, single GPU, attn={args.attn}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, attn_implementation=args.attn)
    model.config.use_cache = False
    mem("after load")
    # MoE load-balancing aux loss OFF. Two reasons, both load-bearing:
    #  1. Correctness — DPO consumes LOGITS, not the model's own `loss`, so the aux
    #     term contributes nothing to the objective. And the router is not in
    #     LORA_REGEX, so there is nothing for a balancing term to steer anyway.
    #  2. Memory — load_balancing_loss_func materialises an expert mask over
    #     (layers x seq x 184 experts x top_k). At 13.8k tokens that is a single
    #     6 GiB allocation on top of an already-92 GiB resident footprint, and it
    #     OOMs after SDPA has already fixed the attention blowup. bug-629.
    model.config.output_router_logits = False
    if getattr(model.config, "router_aux_loss_coef", 0):
        model.config.router_aux_loss_coef = 0.0

    # Count the regex hits against the LIVE module tree before handing it to peft.
    # peft's own error only fires at zero matches; a regex that drifts to a PARTIAL
    # match (e.g. an arch change renames linear_attn) would silently train a smaller
    # adapter and look fine. Assert the exact expected count instead.
    hit = [n for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and re.fullmatch(LORA_REGEX, n)]
    log(f"LoRA targets matched: {len(hit)} (expect {N_LORA_TARGETS})")
    if len(hit) != N_LORA_TARGETS:
        sys.exit(f"REFUSE: LORA_REGEX matched {len(hit)} modules, expected "
                 f"{N_LORA_TARGETS}. The module tree changed — re-derive the regex "
                 f"from named_modules(), not from the safetensors index.")

    peft_cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=LORA_REGEX)

    cfg_kw = dict(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if args.preflight else args.grad_accum,
        num_train_epochs=1.0 if args.preflight else args.epochs,
        max_steps=1 if args.preflight else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="no" if args.preflight else "epoch",
        # AMP OFF by default, and that is not "disabling bf16". The base weights are
        # ALREADY bf16 and frozen; peft keeps the LoRA adapters in fp32 and casts
        # around them itself. So autocast has nothing to convert -- but accelerate
        # pairs it with convert_outputs_to_fp32, which upcasts the forward OUTPUT,
        # and the output here is a [2, seq, 248320] logits tensor: 12.8 GiB bf16
        # becomes 25.5 GiB fp32 for no numerical benefit. Measured: that copy alone
        # is the difference between fitting and OOM. Pass --amp to restore it.
        bf16=args.amp,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=args.beta,
        loss_type=args.loss_type,
        # NOTE: TRL 1.9 DPOConfig has NO `max_prompt_length` / `max_completion_length`
        # — only `max_length`. `--max-prompt-length` stays a CLI knob because it is
        # applied here as a PRE-FILTER in build(), which is where it belongs anyway:
        # TRL's own prompt cap truncates, and truncating is forbidden in this run.
        max_length=args.max_length,
        # THE lever that actually turns the MoE aux loss off. Setting
        # `model.config.output_router_logits = False` is NOT enough: TRL decides a
        # model is MoE by whether the config ATTRIBUTE EXISTS
        # (`getattr(text_config, "output_router_logits", None) is not None`,
        # dpo_trainer.py:882), not by its value, and then force-passes
        # `output_router_logits=True` into every policy forward unless this coef is
        # exactly 0.0. Leaving it at the default costs a 6 GiB fp32 expert mask
        # (184 experts x 13.8k tokens) on top of a 92 GiB resident footprint and
        # OOMs the preflight. It also adds a term DPO's objective never wanted.
        # bug-629.
        router_aux_loss_coef=0.0,
        # THE memory lever for this model. The step-time cost is dominated by
        # [2, seq, 248320] logits tensors -- 12.8 GiB each in bf16 at seq=13794 --
        # and the naive DPO step builds TWO: one for the policy and one for the
        # frozen reference. Precomputing the reference logps in a one-off pass
        # before training removes the second one from the peak permanently, and it
        # is EXACT, not an approximation: with ref_model=None + peft the reference
        # is the base weights with the adapter disabled, which never change during
        # training, so a logp computed once is the same number computed every step.
        # Measured peak on this pair: 90.9 GiB without it, and it OOMs 6.4 GiB short
        # on a 95 GiB card.
        #
        # Liger's fused loss is the obvious alternative and is MEASURED WORSE here:
        # it chunks over the BATCH dimension, one pair is already the minimum batch,
        # so it fuses fwd+bwd across the entire tensor and peaked 1.5 GiB HIGHER
        # (92.4 GiB) than the plain path. It is also mutually exclusive with
        # precompute_ref_log_probs in TRL. bug-630.
        use_liger_kernel=args.liger,
        precompute_ref_log_probs=args.precompute_ref,
        # ref_model=None + peft => reference is the SAME weights with the adapter
        # disabled. Avoids a second 49 GB copy on the card.
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    if args.ld_alpha is not None:
        cfg_kw["ld_alpha"] = args.ld_alpha

    trainer = DPOTrainer(model=model, ref_model=None, args=DPOConfig(**cfg_kw),
                         train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.add_callback(MemProbe())
    mem("after trainer init")

    # ASK THE TRAINER, not the config. `router_aux_loss_coef=0.0` is the intended
    # lever but TRL derives `aux_loss_enabled` itself, and a TRL version bump could
    # change that derivation without changing our flag. This is the only place the
    # answer is observable before a 6 GiB allocation decides it for us.
    if getattr(trainer, "aux_loss_enabled", False):
        sys.exit("REFUSE: trainer.aux_loss_enabled is True — the MoE aux loss will "
                 "run despite router_aux_loss_coef=0.0. It OOMs at max_length and "
                 "adds a term DPO does not use.")
    log("AUX_LOSS_OFF — trainer.aux_loss_enabled=False")
    if args.liger and args.precompute_ref:
        sys.exit("REFUSE: --liger and precomputed reference logps are mutually "
                 "exclusive in TRL. Pass --no-precompute-ref with --liger.")
    if args.liger and not getattr(trainer, "use_liger_kernel", False):
        sys.exit("REFUSE: use_liger_kernel=True was requested but TRL did not take "
                 "it. Without the fused loss the full logits tensor is built and "
                 "this run OOMs at max_length.")
    log(f"LIGER_FUSED_LOSS={getattr(trainer, 'use_liger_kernel', False)}")

    n_tr = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in trainer.model.parameters())
    log(f"trainable {n_tr/1e6:.1f}M / {n_all/1e9:.2f}B ({100*n_tr/n_all:.3f}%)")
    if n_tr == 0:
        sys.exit("REFUSE: LORA_REGEX matched nothing — 0 trainable params")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / 2**30
    log(f"peak VRAM {peak:.1f} GiB | wall {(time.time()-t0)/60:.1f} min")

    if args.preflight:
        log("DPO_PREFLIGHT_OK")
        return 0

    trainer.save_model(args.out)          # adapter only — no 49 GB copytree
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "run_meta.json"), "w") as fh:
        json.dump({"pairs": args.pairs, "n_pairs": len(ds),
                   "max_length": args.max_length, "beta": args.beta,
                   "loss_type": args.loss_type, "ld_alpha": args.ld_alpha,
                   "lr": args.lr, "epochs": args.epochs, "r": args.r,
                   "alpha": args.alpha, "lora_regex": LORA_REGEX,
                   "base": args.model, "peak_vram_gib": round(peak, 2)}, fh, indent=2)
    log(f"saved adapter -> {args.out}")
    log("DPO_TRAIN_OK  — NOT a ship signal. Re-measure pass AND length on "
        "lcb_v6_77q / humaneval_full_think / multipl_e_100 before believing anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
