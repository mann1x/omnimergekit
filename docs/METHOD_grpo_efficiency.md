# METHOD — GRPO efficiency tuning (driver + brevity replay)

Companion to `scripts/train_grpo_efficiency.py`, `scripts/grpo_reward_efficiency.py`,
`scripts/build_grpo_pool_v4.py` and `scripts/run_grpo_an_anchored.sh`.

**Living document.** Every entry below was paid for with GPU hours or a wrong
conclusion. Append as the journey continues; do not silently delete a finding — mark it
SUPERSEDED with the date and the evidence that overturned it.

Last updated: **2026-09-09**.

---

## 1. What this trains, and the words for it

| term | tier(s) | source | objective |
|---|---|---|---|
| **driver** | `mc_letter/T`, `lcb_exec/T` | `manic-arm-contrast`, `lcb_v6_easy` | **efficiency** — do the task in fewer tokens |
| **replay** | `mc_letter/N` | `gpqa_main_minus_diamond` (no-think) | **brevity** — keep short answers short |

The objective is **efficiency**, not brevity. Brevity is what the replay preserves; the
driver is where multi-turn agentic-coding efficiency is taught. The split is **60/40
driver/replay** by row count. Getting this backwards produces analyses that read the
clamp/dead-group census against the wrong tier — it happened on 2026-09-09.

## 2. The reference arm — anchor on it, not on the family

`an-finetune/out/grpo_v8_v2.log` is **the only brevity arm in this house with a
statistically significant length slope**: OLS **t = -2.43 over 446 steps**. Its sibling
`grpo_v8_full` (223 steps, λ0.5, budget 1024, **β0.04**) was **FLAT, t = -0.36**.

Cite the run, never the project. `β=0.04` was copied into our config from the arm that
**failed**; the one that worked used **β=0.01**.

```
grpo_v8_v2:  --length-lambda 0.8 --beta 0.01 --length-budget 512
             --epochs 2 --max-completion-len 1024
             pool 894 rows, K=8, batch 8 x grad_accum 4 x 1 GPU = 32 completions/update
             446 steps, 2.00 epochs, ~60 s/step
             clipped_ratio 0.53-0.69      <-- see §5.2, this is not incidental
             mask_truncated_completions   <-- never set, so TRL default False
```

### 2.1 Compare at MATCHED STEP, never against another run's max

`grpo_v8_v2` peaked at `kl 11.79`, which makes any healthy run look inert by comparison.
That 11.79 is one of **28 spikes over 446 steps**, all in Q3/Q4 — not the operating
level. At steps 173-176 AN reads:

```
kl: 0.003748 / 0.0208 / 0.0115 / 0.01389
```

Our arm at the same LR reads `0.0195 / 0.0134` — **the same band**. On 2026-09-09 the
max-vs-step-5 comparison nearly produced an unnecessary LR increase.

**KL spikes accompany a moving policy; they are not by themselves a broken run.** v1
(flat) had `kl max 0.275` and zero spikes; v2 (worked) had 28 spikes > 1.

## 3. The reward

`scripts/grpo_reward_efficiency.py`, byte-equivalent to `an-finetune/simpo/train_grpo_e2b.py:100`:

```
r = (R_CORRECT - lambda * min(ntok / budget, 1.0))   if correct else 0.0
  + FORMAT_BONUS                                     if correct and answer marker present
```

- `R_CORRECT = 1.0`, `FORMAT_BONUS = 0.05`.
- **`length_budget` is PER ROW** (`meta.length_budget`), never global. A single budget
  across tiers whose medians differ 3x puts one tier on the slope and pins the others at
  a constant. The reward **REFUSES** if `lambda > 0` and no budget is set — a fallback is
  how a tier ends up measured against another tier's operating point.
- **`lambda = 0` means replay: pure correctness, no length term at all.**
- A **censored** rollout (`ntok >= max_completion`) scores exactly `0.0`. The reward
  decides this from the TOKEN COUNT, so it is correct regardless of TRL's EOS accounting
  (§5.3).

### 3.1 Budget placement is the whole game

Selftest §5 (`test_grpo_reward_efficiency.py`) proves `frac_under_budget` discriminates
the three placements:

| budget placement | `frac_under_budget` | within-group spread |
|---|---|---|
| far BELOW the distribution | **0.00** | `pstdev == 0.000000` — dead, cancels under `scale_rewards=group` |
| at the AN operating point | **0.33** | 0.1259 — real gradient |
| far ABOVE | **1.00** | 1.2e-4 — negligible |

**`frac_under_budget` is the single most diagnostic per-tier number.** Read it before
anything else. AN's ratio was `budget = 0.82 x tier mean`.

"0.66x median" is NOT a usable rule: with lengths [3000..6000] a budget at 0.66x median
falls below the *minimum*, every rollout clips to `lenpen = 1.0`, and the length term
becomes constant.

## 4. Failure modes — the group census

Every group is one prompt x K rollouts. Three classes, **counted PER TIER**:

| class | condition | meaning |
|---|---|---|
| `graded` | >=2 passers, >=1 under budget | real length gradient |
| `CLAMP-DEAD` | >=2 passers, **ALL** above budget | `min(nt/budget,1)` ties them at λ — no length contrast |
| `dead (<2 pass)` | <2 passers | pass-rate problem, **not** the clamp |

The last two must never be merged: merging blames the clamp for a pass-rate problem.

### 4.1 The census MUST be per-tier

**2026-09-09, the mistake this section exists to prevent.** A pooled census read
`graded 0.563 / CLAMP-DEAD 0.125 / dead 0.313` and was reported as "the clamp is the
minor failure mode". Per tier:

- `lcb_exec/T` at `pass=0.125` -> P(<2 of 8) = **0.74**. Nearly three quarters of coding
  groups are silent. Benign: real capability on hard code, and only ~10% of the pool.
- `mc_letter/N` (**the brevity replay**) at `frac_under_budget = 0.125` -> the clamp was
  flattening **the entire tier carrying the objective**.

A pool-wide mean cannot see a minority-tier objective. Synthetic proof in selftest §9:
three groups with opposite failure modes read a flat, uninformative 33/33/33 when pooled.

### 4.2 `length_share` — is the reward actually about length?

`length_share` = within-group spread among *passers* / total spread. Logged per tier.

```
healthy (2026-09-09): 0.745 / 0.800 / 1.000 / 0.706
v1 (the flat arm):    0.113 and FALLING
```

Below ~0.2 the reward is being driven by correctness noise, not length. Stop the run.

## 5. Config traps — each one cost real time

### 5.1 `grad_accum` sets how fast data becomes OPTIMISER UPDATES

**A short run is short of updates, not data.** This is the single most useful finding in
this document.

| | AN v8_v2 | ours @ ga32 | ours @ ga16 |
|---|---|---|---|
| completions/update | 32 | **64** | 32 |
| steps/epoch (365-row pool) | — | 46 | **91** |
| 150 steps = | — | 3.3 epochs | **1.64 epochs** |

At `grad_accum 32 x 2 devices` we converted data to updates at **half AN's rate**, so
the only choices were "2 epochs = 62 steps = no power" or "450 steps = 14.5 epochs =
overtrained". `ga 32 -> 16` dissolves that without touching the corpus, and roughly
halves step time because generation dominates.

**Growing the pool is the expensive non-fix.** Reach for `grad_accum` first.

### 5.2 `mask_truncated_completions=True` cancels the strongest brevity signal

TRL's default is **False**. AN never set it, and ran at `clipped_ratio 0.53-0.69`:
**more than half of every batch truncated, each scoring 0 and STAYING in the gradient as
a negative example.** That, not `lenpen`, is the strongest downward pressure that config
had. `True` computes the same `0.0` and then deletes the row.

Ours was `True`. Set to **False** on 2026-09-09.

### 5.3 Gemma-4 EOG breaks TRL's truncation accounting

```python
grpo_trainer.py:1917   is_eos = completion_ids == self._tokenizer.eos_token_id   # SCALAR
```

Gemma-4's EOG is `{1, 106, 50}` and transformers collapses `eos_token_id` to `1` at load.
Nothing matched -> every completion called truncated -> with `mask_truncated=True`,
**everything masked** -> the run trained on nothing while reporting textbook-healthy
metrics (see §6). Fixed by pinning `eos_token_id = 106`; completions ending on 1 or 50
are still miscounted, which is a second reason to keep `mask_truncated=False`.

### 5.4 `--liger` fabricates metrics

TRL 1.12 `--liger` fused GRPO loss produced `grad_norm nan`, `grads nonfinite 410/410`,
a `1.391e7` loss spike, and **`kl = 0.3166` where `kl` must be exactly 0** (fresh LoRA,
`B=0`, `lora_dropout=0` -> policy and reference are the same function). Controlled A/B,
one flag: without Liger, `kl = 0`, `grad_norm = 0.02768`, `nonfinite 0/410`. Cost of
dropping it: ~13% step time. **Do not re-add without re-running that A/B.**

### 5.5 `--smoke` must decide NOTHING but row selection

`--smoke` is how the whole pool is selected (`--smoke-rows-per-tier`), so **every arm
this program has ever launched passes it.** Anything else bound to that flag is
therefore silently active on every real run. This has now bitten twice:

| bound to `--smoke` | consequence | fixed |
|---|---|---|
| `save_strategy = "no" if a.smoke else "steps"` | **nothing was ever checkpointed** — 24 steps / 2.5 h of GPU left only `measured.json` | bug-698 |
| `allow_unset_budget = a.smoke` | the hard refusal in §3 was **disabled on every run** — a row with `lambda>0` and no budget would train as pure correctness, silently | bug-702, now `--measure-only` |

`allow_unset_budget` belongs to a budget-derivation pass, which genuinely runs before
any budget exists. It is now bound to an explicit **`--measure-only`** flag. The runner
does not pass it, so a real run keeps the refusal.

**Rule: `--smoke` selects rows and caps steps. If a flag changes durability, safety, or
what is measured, give it its own name.**

### 5.5b Cap placement: read `max_length`, never infer from `mean_length`

The length distribution is heavy-tailed. Measured per step at cap 4096, `mean_length`
sat at 1032-2173 while `max_length` ranged **2283-4096** — the cap is reached in some
batches and not others. `clipped_ratio` is the fraction AT the cap, so it is the direct
statement; `mean_length` cannot stand in for it.

| cap | steps reaching cap | overall clipped | step time (ga32) |
|---|---|---|---|
| 8192 | 7/24 | 0.46% | 375 s |
| 4096 | 5/11 | 1.5-7.8% | 180 s |
| AN's 1024 (vs mean ~800) | — | **53-69%** | 60 s |

**4096 is the chosen operating point** for this pool: it clips the runaway tail while
leaving normal `lcb_exec` completions (mean 2018-2608) intact. Do NOT lower toward AN's
1.2-1.4x ratio (~1700-2100 here) — with `mask_truncated_completions=False` the truncated
rows carry NEGATIVE gradient, so a cap below the coding tier's mean actively teaches the
model not to write long code. Do NOT raise back to 8192 — it doubles step time and
suppresses the truncation signal.

### 5.6 Banned

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the CUDA VMM API breaks
`cudaIpcGetMemHandle` and kills vLLM TP>1. **The OOM message itself suggests it.** The
real fix for the `selective_log_softmax` OOM was `vllm_enable_sleep_mode` level 2.

## 6. Reading the metrics — what lies, and how

**Read the mask before anything derived from it.** A fully-masked run does not produce
missing diagnostics; it produces *clean* ones, because TRL computes them over an empty
selection and falls back to literals:

| field | fully masked | actually measured |
|---|---|---|
| `sampling_logp_difference` mean/max | **0 / 0** | 1.128 / 37.64 |
| `importance_sampling_ratio` min/mean/max | **1 / 1 / 1** | 0 / 0.0925 / 1.776 |
| `kl` | **0** | 0.3166 |
| `clip_ratio` | **0** | 0.0993 |

Every masked column reads like a perfect run. Concretely:

1. `completions/clipped_ratio == 1.000` means **you are training on nothing**, not "the
   model is verbose". `mean_terminated_length == 0` beside `mean_length == 1600` is the
   same fact twice.
2. Treat an identity-valued diagnostic as **suspect**, not as good news. A real
   importance ratio is noisy.
3. `frac_reward_zero_std` is the fraction of groups with no within-group reward spread —
   they contribute exactly zero gradient. 0.375 means the effective batch is ~5 of 8 groups.

### 6.1 Cumulative counters understate

`state["byk"]` is **never cleared** — only the print trigger resets. Every tier row is a
running mean, so a real change lags for many steps. A synthetic check showed a 1604-token
interval reported as 1204 (**25% understatement**). Both the tier rows and the group
census now print `since=` / `interval=` alongside the cumulative value. **Read the
interval.**

### 6.2 A single step is not a rate

Consecutive steps on the same config, same pool:

```
step 4:  mean_length 2322  clipped 0.156  step_time 209.5
step 5:  mean_length 1014  clipped 0.016  step_time 143.4
```

2.3x apart, driven by batch tier-mix. Do not revise an estimate on one step — this
document's author did, twice, in one hour.

## 7. Pool construction

`scripts/build_grpo_pool_v4.py`. **365 rows, exactly 60.0/40.0.**

```
driver 219 (60.0%)   manic-arm-contrast 128   <-- EXHAUSTED (97 mined + 31 authored)
                     lcb_v6_easy         91
replay 146 (40.0%)   gpqa_main_minus_diamond  (250 available)
```

- **Driver supply is the binding constraint.** `manic-arm-contrast` has been mined twice
  and yields no more. Pool size is therefore capped near 414 rows at 60/40.
- v4 is a strict **SUPERSET** of v3, so shared rows stay comparable across runs.
- `lcb_v6_easy` supply is **91**, not the 79 in `gepo_mixed_with_efficiency.jsonl` — v3
  carried 12 rows the mixed pool lacks. Take the union.

### 7.1 Holdout gates — re-derive, never trust upstream

A gate that ran once in an upstream builder proves nothing about *this* file. The v4
builder re-derives both holdouts from source and REFUSES on any leak:

- **GPQA Diamond is a SUBSET of GPQA main.** Derived two independent ways (Record ID and
  question hash); if they disagree the holdout is not well defined and that is fatal.
  `main 448 - diamond 198 = 250 usable`.
- **Every `*taskids.json` under `eval/lcb/`** — globbed, not enumerated, so a new frozen
  eval list is held out the day it lands. Union is **254** ids across 5 lists.

Current status: `HOLDOUT_OK: 0/198 Diamond and 0/254 frozen-LCB ids in a 365-row pool`.

**Withhold by tier census, not by program name** — a router-calib corpus once carried 80
GPQA *Diamond* rows because it was filtered by name.

### 7.2 Do not scale a tier you cannot vouch for

`lcb_exec/T` passes at **0.125** on the *easy* split. That is low enough to suspect a
scorer artifact rather than capability: the prompt demands "ONLY the completed Solution
class in a Python markdown block" under `think=True`, which is the bug-015 shape where a
correct answer scores 0 at `exec(prompt + generation)` because of markdown fences.
**Check `--dump-rollouts` before multiplying that tier.**

## 8. Run mechanics

`scripts/run_grpo_an_anchored.sh` — segmented, stop/resume across days.

- **Saves EVERY step** (`--save-steps 1`), keeps 3 rolling, archives every 10.
  Checkpoint is **0.26 GB** (measured, not estimated); 150 steps needs ~10 G.
- `save_total_limit` rotation deletes from the live dir but **the archive survives** —
  verified: `checkpoint-2` gone from the run dir, present in `archive/`.
- **Graceful stop:** `touch $OUT/STOP`. Seen at the next step boundary; writes a full
  checkpoint and exits **rc=0**. SIGUSR1/SIGTERM write the STOP file rather than killing.
- **Resume:** `--resume auto` takes the highest checkpoint and **REFUSES** if any of
  `adapter_model.safetensors`, `adapter_config.json`, `optimizer.pt`, `scheduler.pt`,
  `trainer_state.json`, `rng_state*.pth` is missing. Note the **glob** — DDP writes
  `rng_state_<rank>.pth`, not the single-process filename.
- Validated twice on the real model: `PHASE1_RC=0 PHASE2_RC=0`, resume re-enters at
  `global_step=3` rather than restarting at 0.

### 8.1 Historic footgun

`save_strategy` used to read `"no" if a.smoke else "steps"`, and **every arm in this
program runs `--smoke`** — so nothing was ever saved (bug-698). Saving is now
unconditional.

## 9. Open questions

- `sampling/sampling_logp_difference/mean = 1.03-1.14` nats/token between the trainer
  and vLLM, byte-identical with and without Liger. Real, cross-engine, unexplained. The
  MoE-routing-flip hypothesis was **REFUTED** (`scripts/probe_forward_determinism.py`).
- Does step time actually halve at `ga 16`? Predicted, not yet measured.
- Is `lcb_exec/T` pass=0.125 capability or a fence-scoring artifact? (§7.2)
- Does `mask_truncated=False` reproduce AN's brevity pressure on a *coding* pool, where
  long completions are legitimate rather than rambling?

## 10. Changelog

| date | change |
|---|---|
| 2026-09-09 | bug-702: `allow_unset_budget` was bound to `--smoke`, disabling §3's hard refusal on every run ever launched. Now `--measure-only`. |
| 2026-09-09 | v4 smoke @ ga16: **step_time 68-98 s (mean 84.8)** -> 150 steps ~3.5 h. Per-tier census live: driver `mc_letter/T` 100% graded; replay `mc_letter/N` never fully graded (dead-by-<2-passers in one cycle, 50% CLAMP-DEAD in another). cap 4096 truncates **1.5-8% of completions** depending on batch (v4 smoke 1/6 steps at cap, 1.56% overall; resume_check 4/5 steps, 7.81%). Weak vs AN's 53-69%, but NOT inert -- an earlier note here said 'inert' from reading only 4 of 6 steps. **`mean_length` says nothing about the cap; read `max_length` and `clipped_ratio`.** |
| 2026-09-09 | Pool **v4** (365 rows, 60/40, holdout re-verified). `grad_accum 32->16`. `mask_truncated_completions True->False`. Group census made **per-tier** with interval rows. Guard added for prompt/completion length mismatch. Run target 150 steps = 1.64 epochs. |
| 2026-09-09 | bug-699: 3 of 8 reward selftest sections had **never executed** (`[True]*4` against a 6-element `lens`) — including the clamp-death assertion. |
| 2026-09-09 | `--liger` A/B: fabricated `nan` and `kl`. Removed. |
| 2026-09-09 | EOG fix (`eos_token_id = 106`) — un-masked a fully-masked run. |
| 2026-09-09 | bug-698: `save_strategy="no"` under `--smoke` meant nothing was ever checkpointed. |
