# OmniMergeKit Eval Protocol — MANDATORY checklist for any run

**Owner:** `omnimergekit/eval/` is the canonical home of every eval script.
Do not write one-off eval shell scripts in `backup_models/scripts/` or on
pods. The pod-side launcher is allowed to be a thin shell wrapper, but the
**Python evaluator must be the version checked into this repo**.

This document is the operating manual. It is mandatory reading before any
eval run, and **mandatory reading before changing settings** on any
existing eval. Every rule below was learned the hard way (linked memory
file in parens). Do not re-litigate; comply.

---

## 1. The non-negotiables

### 1.0 Canonical sampler is GREEDY — every Gemma 4 bench, every variant

ALL canonical 9-bench templates (gpqa_diamond_full, gsm8k_100, math500_100, aime_30, arc_challenge_full, ifeval_100, humaneval_full, humanevalplus_full, lcb_medium_55*) **MUST** use:

```yaml
generation:
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  do_sample: false
```

This is the recipe of the v4 published model card and the v5-coder pod cohort. Apples-to-apples cross-variant comparisons (v4 vs v5 vs 128e vs 31B vs he1 etc.) **only work when every cohort uses this exact sampler**.

**Why this is non-negotiable:**
- Sampling (temp > 0) with Gemma 4 + `thinking_token_budget=12288` produces 10–40× longer reasoning chains than greedy on hard benches (GPQA p50: 617 → 24,055 chars in our 2026-05-17 incident).
- The score itself may end up similar via flexible-extract, but token-stats columns, completion finish-reason distributions, GPQA domain breakdowns, etc. all become meaningless across cohorts.
- A cohort that crosses the sampler boundary is unpublishable until the entire cohort is re-run on one side.

**Mandatory before any canonical 9-bench launch:**
1. `grep -E "temperature|do_sample" eval/templates/{gpqa_diamond_full,gsm8k_100,math500_100,aime_30,arc_challenge_full,ifeval_100,humaneval_full,humanevalplus_full,lcb_medium_55,lcb_medium_55_v4}.yaml` — verify every line shows `temperature: 0.0` and `do_sample: false` (or absence of `do_sample`, which defaults greedy in lm-eval).
2. After launch, peek the first cached SQLite response at 5–10 questions in. If GPQA p50 > 5000 chars, **STOP** — sampler is wrong.

**Want sampling for a separate study?** Make a **new** template file (`<bench>_sample.yaml`) and a **new** orchestrator that targets it. Never overwrite the greedy templates; never let a "new canonical" silently break cross-cohort comparison.

`memory/feedback_canonical_eval_sampler_is_greedy.md`

### 1.1 NEVER skip resume / cache

| Tool                              | Resume mechanism                        | Mandate |
|-----------------------------------|------------------------------------------|---------|
| `lm_eval` (any task)              | `--use_cache <path>` (SQLite)            | ALWAYS  |
| `lcb_llama_server.py`             | per-problem JSONL cache (built-in)       | ALWAYS  |
| `multipl_e_generate.py`           | per-problem JSON files (built-in)        | ALWAYS  |
| `humaneval_smoke.py` smoke runs   | use lm_eval with `--use_cache`           | ALWAYS  |

If a run dies (PEG parser, OOM, llama-server crash, SSH blip, network blip,
SIGKILL), the next launch must pick up from where it died. We have lost
**multi-hour** runs three times to forgetting this. No exceptions.

`memory/feedback_lm_eval_sqlite.md`, `memory/feedback_eval_validate_during_run.md`

### 1.2 ALWAYS log samples + audit during the run

Every `lm_eval` invocation:
- `--log_samples` (so the post-run sanity script has data to read)
- check `samples_<task>_*.jsonl` line count (must equal n_problems)
- check p10 generation length ≥ 60 chars
- check no markdown-fence drift, no `<|channel>` malformed tokens
- if `pass@1 == 0.0` → **STOP** and inspect samples; the scorer may be
  silently failing at exec/extract, not the model

For long runs (>30 min), spot-check progress every 10 minutes.

`memory/feedback_lm_eval_sqlite.md`, bug-015

### 1.3 Token-cap awareness — measure or die

The model **must not hit a token cap during normal answering**. Fingerprint
of cap-hit:
- `finish_reason == "length"` in OpenAI API response
- `gen_secs` p50 ≈ p90 (most gens hit the same wall)
- pass@1 dragged down by `SyntaxError at line N` from truncated code
- `completion_chars` distribution clustered at the cap

**Mandatory log fields** for every per-problem record:
- `prompt_tokens`
- `completion_tokens`
- `finish_reason`

`lcb_llama_server.py` records all three since 2026-05-10. New evaluators
must do the same.

**Default caps (only raise; do not lower without justification):**

| Use case                 | server `-c` | server `--reasoning-budget` | lm_eval / runner `max_gen_toks` |
|--------------------------|------------:|----------------------------:|---------------------------------:|
| HE / MBPP chat (greedy)  | 32768       | n/a (`--reasoning off`)     | 3072                             |
| MultiPL-E (greedy)       | 32768       | n/a                         | 4096                             |
| LCB-medium chat (greedy) | 32768       | n/a                         | 8192                             |
| GPQA Diamond (reasoning) | 32768       | 16384                       | 24576                            |

Server `-c` must be ≥ `max_gen_toks` × `--parallel` (each slot gets
`-c / --parallel` context). With `--parallel 2 -c 32768` each slot has 16K
ctx — fine for HE/MBPP/LCB chat. For GPQA's 24K gen budget, use
`--parallel 1 -c 32768`.

`memory/feedback_eval_token_cap_truncation.md` (write this if not present)

### 1.4.5 ALWAYS pin the full eval stack across a cohort — vllm, lm-eval, drivers, torch, CUDA, cuDNN, templates

**A "cohort" is any set of models whose scores will be compared in the same table or card** (e.g. `{128e, v4, v5, v5-coder}` for the canonical 9-bench, or `{v4 baseline, v5 α-sweep}` for a free-knob probe). Every model in the cohort MUST be evaluated with **bit-identical software** on every layer of the stack:

| Layer | Pin to |
|---|---|
| **vllm** | EXACT git commit if source-built, EXACT version string if wheel (`vllm==0.20.2` ≠ `vllm 0.1.dev16519+g630492da3`). If you cherry-pick, every model in the cohort must be re-evaluated **after** the cherry. |
| **lm-eval-harness** | EXACT version. `0.4.11` and `0.4.12` differ in API client retry semantics and reasoning-content fallback. |
| **PyTorch + CUDA + cuDNN** | Same wheel build line (e.g. `torch==2.11.0+cu128`). cu128 vs cu130 = different kernels = different rounding. |
| **NVIDIA driver** | Doesn't need to be identical across pod and solidpc, but driver-to-CUDA-runtime compatibility must hold (driver ≥ CUDA toolkit major required). |
| **Templates** | The template YAML hash (`sha256` of the file content) must be identical across the cohort. A passing change to `thinking_token_budget` or `max_gen_toks` requires re-running the entire cohort. |
| **Pod ↔ local stack** | Pod stacks must mirror solidpc's canonical stack as closely as possible. When the user says "match solidpc", that means same vllm commit / lm-eval version / torch wheel. Diverging means the pod's scores cannot be merged into a solidpc cohort. |

**The cherry-pick trap (real, 2026-05-18 incident, cost us today):** v4 baseline IFEval-100 = 91.41% was recorded 2026-05-14 on solidpc with `vllm 0.1.dev16519+g630492da3` (pre-Fix-E). Today (2026-05-18) the same solidpc vllm-source is at HEAD `a39e23ed0` — 10+ commits ahead, including:
- `a39e23ed0` Fix E: Gemma4 parser surfaces reasoning as content on half-open thinking
- `8c79ad658` Revert #39917 routing replay (cherry #42434)
- `fd7d858c8` hidden_pad/intermediate_pad from #34301

Today's probe1 α=1.00 on bit-identical v4 weights produced **p50 response length = 24,467 chars vs v4 baseline's 754 chars (30× drift)**. The weights were verified bit-identical via `cmp`. The drift came from Fix E surfacing reasoning content that pre-Fix-E vllm dropped as empty. The "v4 baseline 91.41% IFEval" number is **post-Fix-E unreproducible** — comparisons against it from a Fix-E-aware vllm are meaningless.

Meanwhile v5-coder (98.17% HE / 92.00% MATH / 94.00% IFEval) was scored on pod 36929284 with **stock `vllm==0.20.2` wheel** (no cherries, no Fix E). v5 was scored on solidpc source-build (cherries + Fix E). v4 was scored on solidpc source-build at an earlier commit. **All three "comparisons" are apples-to-oranges in three different ways.**

**Minimum host hardware/driver floor for the canonical Gemma 4 stack (vllm 0.20.2 + torch 2.11.0+cu130):**

| Component | Minimum |
|---|---|
| NVIDIA driver | **≥ 580.x** (CUDA 13.0 support; driver 570.x maxes at CUDA 12.8 and **cannot** load vllm 0.20.2's libcudart.so.13) |
| cuda_max_good (vast.ai field) | **≥ 13.0** |
| cuDNN | 9.x (canonical: 9.19.0) |
| GPU VRAM | **≥ 24 GB** for any Gemma 4 26B-A4B NVFP4A16 eval; **≥ 48 GB** for 31B NVFP4A16 with max-model-len 65536, or BF16 surgery work |
| Disk | ≥ 200 GB (single 26B BF16 model = 52 GB) |
| Geolocation | **Exclude CN, HK** (Great Firewall blocks HF; mirror returns 405 on POST /api/repos) |

**vllm 0.20.2 wheels are CUDA-13-only.** No cu128 build exists at `wheels.vllm.ai` or PyPI. A pod with driver 570/CUDA 12.8 (e.g. TX pod 36986767 with driver 570.181) **cannot run the canonical stack** — `import vllm._C` fails with `libcudart.so.13: cannot open shared object file`. Source-building vllm 0.20.2 against cu128 is possible but violates the cherry-pick / stack-pinning rule above. **Pick a CUDA-13 pod from the start.**

**vast.ai query helper** (one-liner; emits compliant pods sorted by price):

```bash
vastai search offers 'cuda_max_good >= 13 num_gpus = 1 gpu_ram >= 24 disk_space >= 200 verified=true geolocation!=CN geolocation!=HK' -o 'dph_total' | head -30
```

For 31B / surgery work bump `gpu_ram >= 48`. For multi-GPU evals add `num_gpus >= 2`. The `cuda_max_good` field is what vast.ai exposes as the maximum CUDA toolkit the driver supports — this is the canonical compatibility gate, NOT `cuda_vers` (which is the toolkit installed in the image and is irrelevant; we control that with conda envs).

`memory/reference_vastai_eval_pod_query.md`

**Mandatory before any eval that will be tabulated:**

1. **Print + log the full stack fingerprint:**
   ```bash
   python -c "
   import vllm, lm_eval, torch, transformers, datasets
   print('vllm', vllm.__version__, vllm.__file__)
   print('lm_eval', lm_eval.__version__)
   print('torch', torch.__version__, 'cuda', torch.version.cuda)
   print('transformers', transformers.__version__)
   print('datasets', datasets.__version__)
   "
   nvidia-smi --query-gpu=driver_version --format=csv,noheader
   ```
   Save this output to `<results_dir>/STACK.txt` for every model in the cohort. **A cohort whose entries have different STACK.txt is invalid.**

2. **Template fingerprint:** `sha256sum <template>.yaml` → log to `STACK.txt`. Two cohort entries with different template hashes → invalid.

3. **vllm git-commit pin for solidpc source builds:** any new model evaluated against an existing solidpc cohort MUST first verify `cd /srv/dev-disk-by-label-opt/dev/vllm-source && git rev-parse HEAD` equals the commit logged in the prior cohort entries' STACK.txt. If it differs, either re-evaluate the entire prior cohort on the new commit or roll back vllm-source to the cohort's pinned commit (`git checkout <commit>` + `pip install -e . --no-deps`).

4. **Cohort = entire row of a comparison table.** Adding a new column (new model) does not let you change the stack. Adding a new bench column means re-running the existing models on that bench using the existing stack — not the freshest stack.

5. **When in doubt, the pod must mirror solidpc, not vice versa.** Pod stacks drift faster (image freshness, default `+cuXXX` wheels). Solidpc's stack is the canonical anchor. If you must vary the pod stack (e.g. driver doesn't support cu130), the pod is a separate cohort — note it and don't merge the scores.

**Why this is non-negotiable:** the entire point of "publishable scores" is reproducibility. A score that depends on a cherry-pick from 3 days ago that we can't reliably re-create is not publishable. We have **three** times in the last two weeks been bitten by this exact pattern.

`memory/feedback_pin_eval_stack_across_cohort.md` (write this if not present)

### 1.4 LOCKED canonical settings per benchmark

These are the methodology anchors. Do not change them in passing without
opening a feedback memory and running ablations.

#### GPQA Diamond (`eval_gpqa_v3.sh`-canonical)

```
Server:
  -c 32768 -t 12 -ngl 99 --no-warmup --parallel 1
  --temp 1.0 --top-p 0.95 --top-k 64 --seed 42 --dry-multiplier 0.5
  --jinja --reasoning-format deepseek --reasoning-budget 16384
  (KV cache: F16 default — DO NOT use q8_0 for GPQA, it noises long CoT)

Endpoint: /v1/chat/completions  (NOT /v1/completions)
lm_eval:
  --model local-chat-completions
  --apply_chat_template
  --gen_kwargs "temperature=1.0,top_p=0.95,max_gen_toks=24576"
  num_concurrent=1   (NOT 2 — costs ~5pp on Gemma 4)
  --use_cache <path>
  --log_samples
```

Family preset for sampler (Gemma 4 official): temp 1.0, top_p 0.95,
top_k 64. Other families: see `eval_gpqa_v3.sh` `MODEL_FAMILY` switch.

**Common drift bugs (real, observed):**
- Using qwen3.5 sampler defaults (0.6/0.95/20) on Gemma 4 → -5pp on GPQA.
- Halving context (16384 instead of 32768) + halving budget (8192 not
  16384) + halving max_gen (12288 not 24576) → -10pp combined effect.
- q8_0 KV cache + `--parallel 2` for "speed" → quality regression on long
  CoT (the `feedback_gpqa_parallel_slots` memory **applied to Qwen3.5 only**;
  do not generalize across families).
- Endpoint switch `/v1/completions` instead of `/v1/chat/completions` →
  client-side vs server-side template path differs; can drop ~3pp.

`memory/feedback_lm_eval_sqlite.md`, this run 2026-05-10

#### HumanEval / HumanEval-Chat

```
Server (chat-mode, greedy):
  -c 32768 --parallel 2 (or 1)  -t 12 -ngl 99
  --temp 0 --top-p 1 --top-k 0 --seed 42
  --jinja --reasoning off
  KV cache: q8_0/q8_0 OK (greedy; deterministic enough across builds)

lm_eval:
  --model local-chat-completions
  --tasks humaneval_chat  (custom omnimergekit task; supports markdown fence handling)
  --apply_chat_template
  --gen_kwargs max_gen_toks=3072
  num_concurrent=2 OK
  HF_ALLOW_CODE_EVAL=1 + --confirm_run_unsafe_code
  --use_cache <path>  --log_samples
```

**Never** HE-eval a chat-mode reasoning model via `/v1/completions` — output
gets wrapped in markdown fences and `pass@1` collapses to 0%, looking like
a model failure when it's actually a methodology mismatch (bug-015).

`memory/feedback_gemma4_chat_only_completions_breaks.md`

#### MBPP / MBPP-Chat

Same as HE-chat with `--tasks mbpp_chat` and `max_gen_toks=2048` (MBPP
solutions are short; 3072 is overkill).

#### MultiPL-E

First-class omk backend (`backend: multipl_e`). Templates: `multipl_e_100`
(rs+java+js × 100), `multipl_e_10_smoke`. Headline = macro mean pass@1 over
langs; per-lang + micro also recorded. Resume is sqlite (`eval/cache_sqlite.py`,
keyed `lang::name`). Sampler GREEDY (frozen), like every cohort bench.

##### Generation MUST be chat mode for Gemma-4 (and any reasoning/instruct model)

`generation.mode: chat` in the template. Raw `/v1/completions` (base-model
style) is FORBIDDEN for Gemma-4: the reasoning model never emits the column-0
stop terminator (`\n}` etc.), rambles in indented comment garbage to the token
cap, and the program won't compile. **Evidence (128e Q6_K, MPE-10): raw =
3.33% (rs 0.10/java 0/js 0); chat = 100% (30/30).** It is NOT a stop-token bug —
the dataset stop tokens are correct and sent; the model just doesn't produce
them. Chat mode = `/v1/chat/completions` + chat template + fenced-code
extraction + `chat_to_body()` (anchor on the prompt's signature line, strip the
trailing brace-run only for langs whose tests supply the closer: rust 1, java 2
[method+class]; js keeps its brace). This is the same path that gives HE+ ~90%.

```
Server: --jinja --reasoning off  (-ngl 99 --parallel 2 -c 16384, KV q8_0/q8_0).
        Chat code-gen needs no thinking budget; reasoning off avoids rumination.
Generator: eval/multipl_e/multipl_e_generate.py  --mode chat
  --max-tokens 1024 (chat completions are compact; bodies are short)
  --concurrency 2 ; sqlite resume via --cache-db ; greedy temp=0
  retry on 5xx: 4/8/16/32/64/128s backoff, server errs NOT silently dropped
Eval: eval/multipl_e/multipl_e_evaluate.sh  — MPE_MODE selects:
  docker  (default; solidpc) → nuprl/multipl-e-evaluation image, --network none
  native  (pods; no docker)  → nuprl harness evaluation/src/main.py directly
                               against locally-installed toolchains, UNSANDBOXED
```

##### Native (no-docker) eval on pods — REQUIREMENTS + STACK LOCK

vast.ai pods are unprivileged containers (no docker-in-docker), so MPE eval runs
**native** (`MPE_MODE=native`, `MPE_HARNESS=/workspace/MultiPL-E`). The nuprl
harness shells out to per-language toolchains; each language has hard deps the
Docker image normally provides. Provision via
`eval/multipl_e/install_mpe_toolchains_native.sh`. **SECURITY: native mode
executes model-generated code unsandboxed — throwaway pods only, never solidpc.**

Stack lock (validated pod 37268930, 2026-05-23 — MPE-10 128e = 100% rs/java/js):

| Component | Pinned value | Notes |
|-----------|--------------|-------|
| MultiPL-E harness | commit `3025a531af74` (2026-01-28) | `git clone https://github.com/nuprl/MultiPL-E /workspace/MultiPL-E` then `git checkout 3025a531af74` |
| rustc / cargo | 1.75.0 (apt) | rust has no external deps — works out of the box |
| javac (OpenJDK) | 11.0.30 (`default-jdk`) | — |
| **javatuples** | **1.2 jar at `/usr/multiple/javatuples-1.2.jar`** | sha256 `2eda5b19…`. `eval_java.py` HARDCODES this path; without it every java problem fails `package org.javatuples does not exist`. Maven Central: `org/javatuples/javatuples/1.2/javatuples-1.2.jar` |
| **node** | **v20.20.2 (NodeSource)** | Ubuntu-22.04 apt node is **12**, too old — `require('node:assert')` fails `Cannot find module 'node:assert'`. MUST `apt purge libnode-dev libnode72` (file conflict on `/usr/include/node/common.gypi`) before installing node 20. |
| python | 3.10.12 | — |
| datasets / sqlitedict / tqdm | 4.8.5 / 2.1.0 / 4.67.3 | tqdm is a harness dep; datasets+sqlitedict are omk's |

Record this table's actual values into `<results_dir>/STACK.txt` per §1.4.5; a
cohort whose MPE entries differ on harness commit / node / javatuples is invalid.
Cross-host caveat: the **docker** image (solidpc) and the **native** stack (pods)
are different evaluators — keep a cohort on ONE eval mode, or re-baseline.

`memory/feedback_lm_eval_pod_deps.md` (Docker is rare on rented pods),
`memory/feedback_gemma4_chat_only_completions_breaks.md` (the raw-completion trap)

#### LCB-Medium (LiveCodeBench)

```
Server: same chat profile as HE  (-c 32768 --parallel 2)
Runner: omnimergekit/eval/lcb/lcb_llama_server.py
  --limit 999 (full medium ~55q post 2024-10-01)
  --max-tokens 8192  (was 2048; bumped after observing pod hitting cap)
  --output OUT.json  (samples cache auto-derived: OUT.samples.jsonl)
  Resumable on crash. Records prompt_tokens/completion_tokens/finish_reason.
```

`bug-???` (filed 2026-05-10): pod p90 gen_secs=19.6s with max_tokens=2048
→ cap-hit on ~50% of problems, 20× SyntaxError-line-8 vs local 13× on
identical prompt set. Score deltas are dominated by truncation, not
capability, until the cap is set high enough that `finish_reason="length"`
fires <5% of the time.

---

## 2. Sequencing rules

### 2.1 ALWAYS check llama.cpp build version

Pod and local llama.cpp must be on the same git commit if the comparison
is going to be published. Different builds change FP rounding in CUDA
kernels enough to flip 1-3% of greedy-temp=0 problems.

```bash
# Pod
ssh -p PORT root@HOST 'cd /workspace/llama.cpp && git rev-parse HEAD'
# Local
cd /opt/llama.cpp && git rev-parse HEAD
```

If they differ, either:
- update local to pod (`git checkout <commit> && cmake --build build -j`)
- update pod to local (same)

`bug-???` (filed 2026-05-10): pod-vs-local 3pp HE delta on bit-identical
weights, attributable to commit drift between `5755a10` (local) and
`0b04728` (pod).

### 2.2 Same llama.cpp version → also same KV-cache flags

Even with same build, q8_0 KV-cache + `--parallel 2` introduces
slot-scheduling determinism quirks across runs. For greedy temp=0, runs are
*usually* deterministic, but not always. If a comparison number is in
question, rerun with same KV-cache flags on both sides.

### 2.3 Settings drift checklist — do this BEFORE launching

For every new eval shell script:
- [ ] llama-server `-c` matches the table in §1.3
- [ ] llama-server `--parallel` matches `num_concurrent` in lm_eval
- [ ] sampler matches the family preset (Gemma 4: 1.0/0.95/64; Qwen3.5: 0.6/0.95/20)
- [ ] reasoning-budget present for reasoning evals; absent for chat evals
- [ ] max_gen_toks ≥ §1.3 table
- [ ] endpoint matches: chat-completions for chat tasks, completions only
      when explicitly required (some HE recipes)
- [ ] `--use_cache <unique path>` per (model, task)
- [ ] `--log_samples`
- [ ] `HF_ALLOW_CODE_EVAL=1 --confirm_run_unsafe_code` for HE/MBPP/LCB
- [ ] Tokenizer is the **original 128e/base model dir**, not the pruned
      variant
- [ ] HF token in env (not in CLI args; check `~/.cache/huggingface/token`)
      — pod runs reading gated datasets (Idavidrein/gpqa) need this

### 2.4 Pod build vs script lifecycle

The pod's working tree should always be a **fresh clone of omnimergekit**
or a `rsync` mirror of `/shared/dev/omnimergekit`. Never edit Python
evaluator code directly on the pod. Workflow:

1. Edit + test on solidpc in `/shared/dev/omnimergekit`.
2. `rsync -a /shared/dev/omnimergekit/eval/ pod:/workspace/scripts/`
   (or scp specific file).
3. Pod's shell launcher imports from `/workspace/scripts/` — no in-pod
   edits.

Reason: pods are ephemeral; edits in `/workspace/scripts` evaporate when
the pod dies. Memory of fixes lives in the repo.

---

### 2.5 Pod eval bootstrap — ONE canonical script

`eval/pod_eval_bootstrap.sh` is **THE** on-pod setup for a fresh eval/surgery
pod. Run it ON the pod after landing the repo:

```bash
git clone https://github.com/mann1x/omnimergekit /workspace/omnimergekit \
  && HF_TOKEN=... bash /workspace/omnimergekit/eval/pod_eval_bootstrap.sh <flags>
```

It is idempotent (re-run = no-op) and owns the on-pod **system layer**: apt deps
(incl. `sqlite3` — §3.1 needs it), miniconda, repo, llama.cpp build
(`--cuda-arch`, default 86=3090), conda env (`--env` + `--deps
{eval-full,eval-augment,train,none}`), the solidpc-path symlink farm
(`--symlink-farm`), HF pulls (`--hf-pull`), and the lm-eval patches
(`--patches unbound,fix-a`). `--dry-run` prints the plan without touching the
pod; `--help` lists every flag.

The heavy version-pinned conda envs (omnimergekit-from-requirements, modelopt
0.43.0, vllm 0.20.2) live in ONE place — `eval/pod_runners/setup_conda_envs.sh`
— sourced by both this bootstrap (`--vllm-env` / `--modelopt-env` /
`--deps eval-full`) **and** the controller wrapper
`scripts/pod_setup_eval_envs.sh`. The two lm-eval patches are standalone
idempotent helpers under `eval/patches/`.

Per-variant recipes are thin wrappers in
`eval/pod_runners/bootstrap_<variant>.sh` (c6v3lcb, reeval, gpt_oss, hep_sweep,
router_recovery), each a 3–5 line call into the bootstrap with the right flags.
`backup_models/scripts/pod_*bootstrap*.sh` are symlinks into them — one tracked
source of truth. **NEVER hand-write a new per-task bootstrap under
`backup_models/scripts/`** (that is how five divergent one-offs accumulated and
how `sqlite3` went missing on pod 37588132); add a wrapper here instead.

---

## 3. Validation procedure — what "validate the evals" means

When the user asks "validate the evals" or "check the evals", these are the
checks to run. **Run all of them, in this order. Do not skip any. Do not
report a number until every check passes.** This procedure exists because
we have multiple times reported a score that was bogus, and only found out
at the end of a multi-hour run.

### 3.1 At launch — first 3 generations check (within 60 seconds of start)

Before walking away from a freshly-launched eval, sample the first 3
generated outputs. Use `tail`/`grep` on the live log or `sqlite3` on the
lm_eval cache:

```bash
# lm_eval / SQLite cache — table is `unnamed`, values are PICKLED blobs (key, value).
#   row count (liveness): sqlite3 <cache.db> "select count(*) from unnamed"
#   to read responses, unpickle `value` in python (see feedback_eval_sqlite_spot_check)
sqlite3 <cache.db> "select count(*) from unnamed"
# or for the JSONL cache used by lcb_llama_server
head -3 <out>.samples.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    txt = d.get('completion','') or d.get('cleaned','')
    fr = d.get('finish_reason','?')
    print(f'  {d[\"task_id\"]}: chars={len(txt)} finish={fr}')
    print('    head:', repr(txt[:150]))
"
```

**STOP THE RUN** if ANY of:
- Any of the 3 generations is empty (chars < 5)
- Any has `finish_reason == "length"` and was not expected to be that long
- Any contains `<|channel>` token leakage or other malformed delimiters
- Any wraps the answer in markdown fences when the task expected raw code
- p10 length already < 60 chars at n=3

If you don't STOP, you are wasting compute and will report a bogus number.

### 3.2 During the run — every 10 minutes for long runs

For runs > 30 minutes:
- Check that the request counter is moving (not stuck on a single problem)
- Spot-check the most-recent 3 generations the same way as §3.1
- Watch the `finish_reason == "length"` count: if more than 5% of completed
  problems are length-capped, the cap is too low and the answers are
  truncated; **STOP, raise the cap, restart**. Do NOT wait for the run to
  finish — the score will be artifact-driven and unreportable.
- Watch for monotonically increasing `gen_secs.p90` — this is the
  fingerprint of generations growing into the cap. Catch it early.

### 3.3 Counts vs content — counts lie

It's not enough to count "how many empty / how many fenced". A run can be
100% non-empty and 100% fenced and still produce `pass@1 = 0` because the
fence is in the wrong place. Always print 1 actual generation in addition
to counts:

```python
# bad — counts only
print(f"  empty={n_empty}  fenced={n_fenced}  ok={n - n_empty - n_fenced}")
# good — counts AND look at one
print(f"  empty={n_empty}  fenced={n_fenced}  ok={n - n_empty - n_fenced}")
print(f"  sample completion[0:200]: {repr(samples[0]['completion'][:200])}")
```

### 3.4 Pass-count cross-check

After the run, the `pass@1` reported by the runner / lm_eval must agree
with a hand-recount on the samples file:

```python
import json, glob
samples = sum(1 for _ in open(glob.glob('samples_*.jsonl')[0]))
n_pass_recount = sum(1 for line in open(glob.glob('samples_*.jsonl')[0])
                     if 'pass' in line.lower())  # task-specific; adapt
results = json.load(open(glob.glob('results_*.json')[0]))
print(f"  samples lines: {samples}")
print(f"  results pass@1: {results['results'][...]}")
```

If they disagree by more than rounding, the scorer is broken. **STOP and
investigate**.

### 3.5 Compare to a known-good baseline

For every new variant, also run the 128e baseline on the **identical
pipeline** (same protocol settings, same patched tooling). If the baseline
doesn't match the published 128e Q6_K numbers within ±3pp, the pipeline is
wrong and **no number from the new variant is publishable**.

Known-good 128e Q6_K reference numbers (locked methodology, solidpc 3090):

| Task | 128e Q6_K |
|---|---:|
| HumanEval-164 chat @ 3072 (greedy) | 97.56% |
| MBPP-500 chat @ 2048 (greedy) | 79.20% |
| MultiPL-E Rust full (greedy) | 29.49% (46/156) |
| LCB-medium full (max_tokens=8192, -c 32768) | **87.27%** |
| GPQA Diamond 198 (gemma4 sampler, RB=16384) | _running canonical; published reference is 75.25%_ |

If the run reports 67% on LCB-medium-full for 128e, that's the
**truncation artifact** signature — the run used `max_tokens=2048` and
~30% of generations are being cut off. The real number is 87.27%.

### 3.6 Memory check — has this exact failure happened before

Before publishing or restarting, do `pgvector` recall on:
- "eval validate during run"
- "<benchmark name> truncation"
- "<model name> baseline score"
- bug-NNN if you suspect a known bug

You will repeat-mistake on benchmarks the project has seen before. The
rule "validate during run" is a memory entry exactly because it's been
violated repeatedly.

---

## 4. Post-run sanity (mandatory before reporting a number)

For each eval, before publishing a score:

1. **Sample count**: `samples_<task>_*.jsonl` line count == n_problems
2. **Empty / fence drift**: count generations that are empty, contain
   markdown fences in the wrong place, or have `<|channel>` token leakage
3. **Length distribution**: print min/p10/p50/p90/max char length of
   generations; flag if p10 < 60 chars
4. **Truncation rate**: count `finish_reason == "length"`; if > 5% of
   problems, the cap is too low, **rerun with a higher cap**
5. **Pass count cross-check**: count `passed=true` entries in samples vs
   the `pass@1` × n_problems in `results_*.json`; mismatch = scorer bug
6. **Compare to a known-good baseline**: if 128e Q6_K's GPQA score on this
   pipeline is not within ±3pp of the published 75.25%, the pipeline is
   broken; **stop and investigate** before reporting numbers for new
   variants.

---

## 5. Mistakes to never make again (chronological)

| Date       | Mistake                                                | Cost            |
|------------|--------------------------------------------------------|-----------------|
| 2026-04-09 | PEG parser killed llama-server mid-eval, no cache      | hours           |
| 2026-04-11 | Wrong tokenizer dir on pruned variant                  | re-eval         |
| 2026-04-29 | Custom regex GPQA scoring instead of flexible-extract  | -45pp           |
| 2026-05-02 | HE/MBPP without `HF_ALLOW_CODE_EVAL=1`                 | empty results   |
| 2026-05-08 | nf4 + multimodal modules saturating 24GB               | shelved 31B     |
| 2026-05-10 | LCB script imported from Mythic-RDT on pod (no pkg)    | crash           |
| 2026-05-10 | LCB max_tokens=2048 vs model gen-cap                   | **+20pp** on 128e baseline alone (67.27 → 87.27); 100% truncation artifact |
| 2026-05-10 | GPQA settings drift: qwen3.5 preset on Gemma 4         | ~10pp           |
| 2026-05-10 | Pod vs local llama.cpp commit drift                    | 3pp HE noise    |
| 2026-05-10 | Did NOT validate-during-run; let LCB finish before checking p90 gen length | wasted 2 multi-quart runs |
| 2026-05-23 | Read raw lm_eval `exact_match,strict-match` (GPQA) / `exact_match,none` (math500) instead of omk `summary.json` `.score`; compounded by a SUMMARY.md roll-up globbing the wrong single-`<served>` path → `NO_RESULT`. Falsely reported GPQA 1.52% / math500 41% when the real canonical scores were 72.73% / 94%. | hours of false alarm; nearly re-ran valid evals |

---

## 6. Repository layout (canonical paths)

```
omnimergekit/
├── eval/
│   ├── EVAL_PROTOCOL.md                  # this file
│   ├── omk_eval.py                       # THE canonical eval engine (all backends)
│   ├── omk_summarize.py                  # multi-model roll-up → reads summary.json .score
│   ├── eval_suite_llama.sh               # canonical llama.cpp Q6_K suite driver (thin over omk_eval)
│   ├── eval_suite_vllm.sh                # canonical vLLM NVFP4A16 suite driver
│   ├── eval_suite_chain.sh               # multi-variant outer chain over the suite drivers
│   ├── pod_eval_bootstrap.sh             # THE canonical on-pod pod setup (§2.5)
│   ├── tasks/                            # custom lm_eval YAMLs
│   │   ├── humaneval_chat.yaml
│   │   ├── mbpp_chat.yaml
│   │   ├── gpqa_diamond_smoke10.yaml
│   │   ├── gpqa_diamond_smoke20.yaml
│   │   └── _subset_filter.py
│   ├── lcb/
│   │   ├── lcb_llama_server.py           # patched 2026-05-10: resume + token usage + finish_reason
│   │   └── lcb_helpers.py                # dependency-free shim of LCB load/clean/score
│   ├── scripts/
│   │   ├── eval_gpqa_v3.sh               # GPQA canonical (multi-arch sampler presets)
│   │   ├── multipl_e_generate.py         # MPE generator (retry on 5xx)
│   │   ├── multipl_e_evaluate.sh         # MPE Docker eval wrapper
│   │   └── README.md
│   ├── patches/                          # standalone idempotent lm-eval patch helpers
│   │   ├── lm_eval_unbound_guard.py      #   UnboundLocalError guard (api_models.py)
│   │   └── fix_a_lm_eval_patch.py        #   reasoning_content fallback (openai_completions.py)
│   └── pod_runners/                      # wrappers call ../pod_eval_bootstrap.sh
│       ├── setup_conda_envs.sh           # shared env recipes (omk/modelopt/vllm pins) — single source
│       ├── bootstrap_<variant>.sh        # thin wrappers over pod_eval_bootstrap.sh (c6v3lcb, reeval, …)
│       └── pod_<exp>.sh                  # thin shell launchers; import from eval/
└── recipes/
    └── ...
```

**Single source of truth for scores.** The canonical score for a finished
bench is `<results>/<bench>/<served>/summary.json` `.score` (omk_eval already
selects flexible-extract / math_verify / pass@1,extract_chat / pass_at_1, and
records `metric`+`filter` provenance). NEVER read raw `results_*.json`
`strict-match` / `exact_match,none` — that mismatch caused the 2026-05-23
GPQA-1.52% / math500-41% false alarm. All SUMMARY roll-ups
(`eval_suite_*.sh`, `omk_summarize.py`) read `summary.json`, not raw results.

**Suite drivers live HERE in `eval/`,** next to the engine they wrap.
`backup_models/scripts/eval_suite_{llama,vllm,chain}.sh` are symlinks into this
dir — one tracked source of truth. New eval tooling lands in
`omnimergekit/eval/`, never as an untracked one-off under
`backup_models/scripts/`. The rest of `backup_models/scripts/` is
project-specific glue and ad-hoc one-shots only.

---

## 7. The "before you launch" 60-second check

Run this mentally before every nohup/launch:

```
□ git pull omnimergekit  (no stale local script)
□ pod and local llama.cpp on same commit
□ canonical settings table §1.3 matched
□ --use_cache present, path unique per (model, task)
□ --log_samples present
□ HF_TOKEN in env if gated dataset
□ HF_ALLOW_CODE_EVAL=1 + --confirm_run_unsafe_code if code-exec eval
□ output dir exists and is on persistent disk (not /tmp)
□ disk preflight: df -h shows ≥ 50 GB free for cache + samples
□ disown background processes (nohup ... & disown) so SSH timeout doesn't kill
```

If any item is unchecked, fix before launch. This list represents
~$500 of cumulative wasted compute. It exists to keep that number from
growing.

---

# Protocol v2 — 2026-05-12 — vLLM-first, template-driven

The v1 protocol above remains binding. The v2 layer adds rules that
apply on top of v1, and *replaces* the implicit "always llama.cpp" default
that ran through every v1 prescription.

## v2.1 Default backend: vLLM

llama.cpp `llama-server` is **no longer the default** evaluation backend.
vLLM `vllm.entrypoints.openai.api_server` is. The reason:

- Continuous-batch throughput + multi-tier quant support (BF16, NVFP4A16,
  AWQ-4bit, GPTQ-4bit) under one binary.
- Same OpenAI-compatible API as before (`/v1/chat/completions`,
  `/v1/completions`), so lm-eval calls in v1 do not have to change.
- Vendor-agnostic. No surprise drift on Gemma 4 (we already burned a
  morning on `gemma4_patched.py` divergence).

llama.cpp remains the fallback for:
- Pre-quantized GGUF tiers (Q2_K through Q6_K) that we have already
  built and want to evaluate without re-quanting for vLLM.
- Devices where vLLM isn't supported (Apple silicon today — use llama.cpp
  Metal until vLLM lands a stable Metal backend).

When you choose llama.cpp, you must justify it in the run's `summary.json`
under `backend_reason`.

### v2.1.1 Quant policy on vLLM

| Model fits in VRAM at... | Use |
|---|---|
| BF16 / FP16 (preferred when it fits) | unquantized |
| 8-bit only | AWQ-8 or GPTQ-8 |
| 4-bit only | NVFP4A16 (preferred for MoE), AWQ-4bit, or GPTQ-4bit |

VRAM budget on solidPC is 24 GB (RTX 3090). The decision is therefore:
- ≤ 12 GB params → BF16 fits, use it.
- 12 – 24 GB params → may fit BF16 with reduced KV cache. Try first; fall
  back to 8-bit if KV cache is too constrained.
- > 24 GB params → 4-bit. **NVFP4A16 is the preferred 4-bit for the 98e
  Gemma-4 family** and any other architecture modelopt supports cleanly.
  AWQ-4bit is the broad fallback. GPTQ-4bit is the last resort.

Different models may pick different quants. The choice is recorded in
`summary.json -> quant` so cross-run comparisons stay honest.

## v2.2 Templates are the unit of work

A "run" is `(model, template)` — never `(model, ad-hoc CLI flags)`.

The template lives at `eval/templates/<name>.yaml` and is the only thing
that determines:
- which problems are included (deterministic indices OR criteria filter)
- generation kwargs (max_tokens, temp, top_p, top_k, sampler)
- scoring (metric, filter, scorer, validation status)
- sqlite cache prefix
- token-stats reporting toggle

Built-in templates (each canonical, do not change without a memory entry):
- `humaneval_full`, `mbpp_full`, `humanevalplus_full`
- `gsm8k_100`, `gpqa_diamond_full`, `mmlu_pro_200`, `aime_30`
- `lcb_medium_30`, `lcb_medium_55`

A custom template is just a YAML file passed as `--template <path>`.

The runner is `eval/omk_eval.py`. It owns server lifecycle, dispatch,
sanity, and token-stats reporting. Pod-side use is the same — copy the
omnimergekit clone, `omk_eval.py --model ... --template ...`.

## v2.3 Scorer validation is mandatory before adoption

For every benchmark whose scorer is *not* the canonical upstream library,
we must show point-for-point agreement on a fixture before adopting.

The pattern (established for LCB):
1. Locate the canonical upstream scorer (`lcb_runner.evaluation.testing_util.grade_call_based`
   for LiveCodeBench; `openai/human-eval` for HumanEval; etc.).
2. Run both scorers on the same held-out fixture.
3. If 100% agreement, fill in the template's `scoring.validated_against`,
   `scoring.validation_date`, `validation_n_samples`, `validation_agreement`.
4. If disagreement, fix the local scorer until 100%, *then* adopt.

The script `eval/validate_scorers.py` drives this for the lm-eval native
scorers (HE, MBPP, GSM8K, MMLU-Pro, AIME, GPQA flexible-extract). It must
pass before a template is treated as production.

Status as of 2026-05-12:
- LCB-medium       — **13/13 agreement** vs lcb_runner. Adopted ✓
- GSM8K (flex)     — **7/7 fixture agreement**. Adopted ✓
- AIME (strict)    — **6/6 fixture agreement**. Adopted ✓
- HumanEval        — fixture insufficient (stub); need openai/human-eval install
- MBPP / HE+ / MMLU-Pro / GPQA — pending

## v2.4 Token usage statistics are part of the eval report

Every run writes to `summary.json` a `token_stats` block. The block is
not optional. Required fields:

```json
{
  "token_stats": {
    "n": <int>,
    "prompt_tokens":     {"sum": _, "p10": _, "p50": _, "p90": _, "max": _},
    "completion_tokens": {"sum": _, "p10": _, "p50": _, "p90": _, "max": _},
    "completion_chars":  {"p10": _, "p50": _, "p90": _, "max": _},
    "finish_reasons":    {"stop": _, "length": _, "null": _},
    "empty_completions": <int>
  }
}
```

The post-run sanity check (omk_eval auto-runs it) fails the run when:
- `n` ≠ template's expected `n`
- `empty_completions` > 5 % of n
- `completion_chars.p10` < 60

Thinking-token tracking: if the model emits `<think>` or reasoning-channel
tokens, the runner counts them separately as `completion_tokens.thinking`
(populated when the response contains a reasoning trace). This catches the
`<|channel>` malformed-token bug from v1 §1.2 by surface area, not by a
heuristic that has to be re-discovered every time.

### v2.4.1 Token-count provenance (stack@2, 2026-05-21)

vLLM `/v1/chat/completions` returns `usage.{prompt,completion}_tokens` in
the HTTP response, but **lm-eval discards it**. The `local-chat-completions`
adapter's `parse_generations(response_dict) → List[str]` extracts only
the text — the `usage` block never reaches the sample row, and the SQLite
cache stores only the pickled completion string (`<pickled, type=str>`).
So neither `samples_*.jsonl` nor `sqlite_cache/*.db` carries the counts.

`compute_token_stats` (omk_eval.py) recovers them by **re-tokenizing the
completion text** with the same tokenizer vLLM served, passed via
`--tokenizer` (defaults to `--model`). Behavior:

- `completion_tokens.method = "tokenizer:<id>"` — counts are from local
  HF AutoTokenizer (fast=True, `add_special_tokens=False`). Exact within
  ±0 for greedy chat-completion responses (same vocab).
- `completion_tokens.method = "fallback_zero"` and a `note` field —
  tokenizer load failed (missing files, OOM, etc.); counts are 0 but
  the bench score is unaffected. Soft-fail by design.
- `completion_tokens.method = "usage_field_only"` — caller didn't pass a
  tokenizer; the per-sample `completion_tokens` keys (rarely populated)
  are read as-is. Reserved for older runs / external callers.

Prompt tokens still come from `s.get("prompt_tokens") or 0` — re-tokenizing
the prompt would need the raw doc text + chat-template rendering, which
the sample row doesn't preserve. Treat `prompt_tokens` as best-effort
until a future patch instruments the API adapter.

Retroactively works on any older `samples_*.jsonl` — re-run
`compute_token_stats(path, tokenizer_id=...)` and the new fields land
in the recomputed stats block.

## v2.5 Pre-flight inventory

Before launching any eval:

1. `vastai show instances` (if pod) — confirm the right pod is running.
2. `nvidia-smi --query-gpu=memory.free` — confirm GPU is free.
3. `df -h <output-dir>` — confirm ≥ 30 GB free for samples + cache.
4. `python eval/templates/template_loader.py <name>` — confirm template
   loads + validates.
5. `python eval/validate_scorers.py --bench <name>` — confirm scorer is
   green if it's been changed since the last validation_date.

## v2.5.1 vLLM smoke request before dispatch (workaround for ready-race)

vLLM's `/v1/models` 200 response sometimes precedes the engine actually
being warm. Symptom: the FIRST inference request hangs until the runner's
ReadTimeout (commonly 600 s), then the SECOND request goes through in
normal time. Lost task #1 on LCB-55 NVFP4A16 128e (2026-05-12) to this.

omk_eval now performs a single warmup chat-completion (`"hi"` → 4 tokens)
against the server immediately after `/v1/models` returns 200, before
handing the base_url to the eval runner. The warmup blocks until a real
200 with a non-empty response arrives. Add ~3-10 s to startup; eliminates
the first-request timeout class.

## v2.6 Migration from v1

Existing scripts under `backup_models/scripts/*` that use the v1 protocol
(llama.cpp + custom shim) continue to work. They are not retired — they
are pinned to v1 in their header and only exist for historical
reproducibility. New evals go through `omk_eval.py` only.

## v2.7 Backend × quant decision matrix

Choosing the runtime is **per-bench-per-model-family**, not a single
global default. The trade-off: vLLM NVFP4A16 on the 3090 sustains ~22
tok/s for Gemma 4 26B; llama.cpp Q6_K does 60-80 tok/s — a 3× delta.
For LCB-medium (~14k gen tokens × 55 problems) the vLLM path costs ~1.5
wall-hours per variant. We still pay it on selected benches because of
**scorer/parser trust**, not speed.

### Gemma 4 26B-A4B family — canonical per-bench backend

| Bench | Backend | Quant | Reason |
|---|---|---|---|
| LCB-medium-55 | **vLLM** | NVFP4A16 | llama.cpp + Gemma 4 MoE produces inconsistent / truncated reasoning traces; vLLM is canonical |
| MBPP-500 | **vLLM** | NVFP4A16 | prior llama.cpp results drifted across runs on Gemma 4; vLLM was stable |
| GPQA-Diamond-198 | **vLLM** | NVFP4A16 | reasoning, kept on the scorer-validated path with the other vLLM benches |
| HumanEval-164 | llama.cpp | Q6_K | already published from llama.cpp Q6_K; do NOT re-run |
| HumanEval+ -164 | llama.cpp | Q6_K | trusted + 3× faster |
| GSM8K (100, frozen) | llama.cpp | Q6_K | trusted + 3× faster |
| MMLU-Pro (200, frozen) | llama.cpp | Q6_K | trusted + 3× faster |
| AIME (30) | llama.cpp | Q6_K | trusted + 3× faster |

This split is what `recipes/gemma4/run_a4b_publication_suite.sh` runs.

### Default speed-only rule (other model families)

When scorer trust is **not** in question, default to **llama.cpp Q6_K
when a matching GGUF exists**. vLLM is preferred only when:

1. The model has no GGUF and we don't want to build one (smoke runs).
2. We're doing a backend/quant ablation (e.g. cross-validating NVFP4A16
   against the Q6_K reference, as on 2026-05-12).
3. The model fits unquantized BF16 on the GPU and the gain from vLLM's
   continuous batching outweighs the slower per-token throughput (rare
   on a 24GB 3090; relevant on A100/H100 pods).
4. We need a vLLM-specific feature: LoRA hot-swap, paged-attn telemetry,
   structured output, parallel sampling, multi-LoRA, speculative decode.

Resulting matrix (3090 + Ampere, Gemma 4 26B-A4B family):

| Model BF16 size | GGUF available? | Default backend | Quant | Note |
|---|---|---|---|---|
| ≤ 8 GB | any | vLLM | BF16 | unquantized fits, no point in GGUF detour |
| 8-24 GB | yes | llama.cpp | Q6_K | fastest with the GGUFs we already ship |
| 8-24 GB | no | vLLM | NVFP4A16 | build NVFP4 with `quantize_any.py` |
| > 24 GB | yes | llama.cpp | Q6_K (parallel=1) | only Q6_K fits, single-slot |
| > 24 GB | no | A100 pod | NVFP4A16 or BF16 | local 3090 is OOM territory |

Per-bench overlay (where the bench biases the choice):

| Bench | Gen tokens p99 | Sensitive to throughput? | Preferred backend |
|---|---|---|---|
| HumanEval (164) | ~1k | mild | either |
| MBPP (500) | ~600 | no | either |
| HumanEval+ (164) | ~1k | mild | either |
| GSM8K (100, frozen) | ~600 | no | either |
| MMLU-Pro (200, frozen) | ~400 | no | either |
| GPQA-Diamond (198) | ~8k | yes — reasoning | llama.cpp Q6_K |
| AIME (30) | ~12k | yes — long reasoning | llama.cpp Q6_K |
| **LCB-medium (55)** | **~14k** | **YES** | **llama.cpp Q6_K** |

The LCB row is the dominant one: 14k generation tokens × 55 problems ×
3× throughput delta = ~1.5 wall-hour penalty per LCB-medium-55 run if
NVFP4A16 is chosen. Worth swallowing only when we *want* the NVFP4A16
score itself — i.e. ablation/cross-validation runs.

### Gemma 4 llama-server flags (mandatory)

- **Coding benches** (HE/MBPP/LCB):
  `--jinja --reasoning off` — disables `<|channel>thought` traces so the
  scorer sees only the answer block.
- **Reasoning benches** (GPQA/AIME/MMLU-Pro):
  `--reasoning-format deepseek --reasoning-budget 8192` — the budget
  flag is **load-bearing**: without it, Gemma 4 enters "Wait, let me
  re-read..." loops and exhausts context. Cost us 1+ hour on first
  discovery; documented as a project rule.
- **Throughput**: `--parallel 2 --cache-type-k q8_0 --cache-type-v q8_0
  -c 32768 -ngl 99 --no-warmup`. `parallel=2` is the documented
  per-project speedup for batch-2 lm-eval; drop to 1 only when context
  cache OOMs.

omk_eval's `launch_llama()` accepts these via `backend_args.llama_extra`
on the template (or `LLAMA_EXTRA` env var). Bench-typed defaults are
applied automatically when the template's `task` matches.

## v2.8 Server lifecycle — no zombies

`omk_eval.py` owns the entire server lifecycle when it launches the
backend (i.e. when `--no-server` is NOT passed). Lifecycle guarantees:

- **Pre-launch port clear.** Before bind, `kill_port(port)` SIGTERMs
  then SIGKILLs anything holding the TCP port. This catches orphan
  vLLM `EngineCore` children whose parent died while still holding the
  port and GPU.
- **Process group launch.** Both `launch_vllm` and `launch_llama` set
  `preexec_fn=os.setpgrp` so the server and all its children form one
  process group. EngineCore subprocesses and llama-server worker
  threads die together.
- **Two-wave kill.** `ServerHandle.kill()` sends `SIGTERM` to the
  pgid, waits 10 s, then `SIGKILL` if needed. Final `kill_port` sweep
  catches anything that detached from the pgid.
- **Crash safety.** `atexit` + `SIGINT/SIGTERM/SIGHUP` handlers call
  `_atexit_kill_all` so the interpreter exiting for any reason
  (uncaught exception, Ctrl-C, OOM, hangup) still tears the server
  down.
- **Per-bench teardown.** In multi-bench drivers like
  `run_a4b_publication_suite.sh`, each `omk_eval.py` invocation owns
  one server: it launches, dispatches, kills. The next invocation
  starts cleanly. No state leaks between benches.

If you ever see an orphan EngineCore holding the GPU, that is a bug —
file it; the lifecycle above should prevent it. Manual recovery:
`fuser -k -KILL 8195/tcp; pkill -9 -f EngineCore; sleep 5; nvidia-smi`.

## v2.9 — Token-cap policy + per-variant headroom (2026-05-12)

### Symptom we learned from

Gemma 4 `98e-v3` LCB-55 at vLLM NVFP4A16 with `max_tokens=16384`:
3 of the first 5 problems hit `finish_reason=length`, all 3 fail with
`SyntaxError` from mid-token truncation. `128e` on identical run: 53/55
finish with `stop`, p90 = 6655 tokens, only 2 length-caps (one was a
real algo fail anyway). **The variant — not the bench — determines the
needed headroom.** Pruned MoE variants can lose convergence shortcuts
and bloat reasoning by 2-5× vs the parent.

### Rule

Before adopting a `max_tokens` for a publication run on a new variant,
**measure on the first 5 problems**:

| signal                                  | action                              |
|-----------------------------------------|-------------------------------------|
| `finish_reason=length` rate ≤ 5%        | budget is fine — proceed            |
| 5-20% length-cap                        | raise `max_tokens` by 1.5×, retest  |
| > 20% length-cap                        | variant is fundamentally bloated — pick budget based on policy below |

### Variant-specific headroom policy

- `vllm.max_model_len` sets the **KV cache reservation** at launch. Pick
  the largest sane value the VRAM allows (Gemma 4 26B-A4B NVFP4A16 on
  3090 = 32768 fits with ~2 GB free).
- `max_tokens` (per request) is **separate** from `max_model_len`. It
  can be set up to `max_model_len - prompt_tokens` per request **with
  zero additional VRAM cost** and no CUDA graph recapture.
- Default `max_tokens = 24576` is safe for any Gemma 4 26B-A4B variant
  on the 3090; gives 1.5× headroom over the original 16384 cap.

### Apples-to-apples — when to re-run reference

When raising `max_tokens` for a new variant, **the reference number on
the parent model must use the same cap, OR you must verify the parent
didn't change at the new cap**. Path of least effort: keep parent's
existing samples.jsonl, re-run **only the parent's length-capped
problems** at the new cap, patch those rows in, recompute pass@1. (For
128e LCB-55, that was 2 problems out of 55 — ~10 min on vLLM.)

### Truncation = failure for pruned variants

If a pruned variant (98e, 109e, ...) truncates a problem at the chosen
cap, it's a real failure: the variant **could not finish the reasoning
trace within the budget**. We will **not** rerun pruned variants at a
larger cap to chase a higher score — the budget is part of the recipe.
Reporting:

- `pass@1` is computed over all 55 problems, length-caps counted as
  fails.
- HF model card MUST include **`length-cap %`** alongside pass@1 — the
  reader needs to know how much of the gap is reasoning vs budget.
- Add a row in the per-bench publication summary:
  ```
  | model | bench | pass@1 | n_pass | n_length_cap | median_completion_tokens |
  ```

### CUDA graphs vs `--enforce-eager` — when

The 128e LCB-55 reference was captured under `--enforce-eager`
(~22 tok/s on 3090). Dropping eager mode on Gemma 4 NVFP4A16 captures
51 prefill-decode + 35 full-decode graphs in ~36 s and gives
**~86 tok/s sustained** — a 4× decode throughput win. This is the
correct default for 3090 + NVFP4A16 + Gemma 4 going forward.

Risks that motivated the original eager default:
- Marlin FP4 weight-only kernel under CUDA graph capture had shape
  panics in older vLLM 0.10/0.11 — **resolved** in vLLM main as of
  2026-05-12.
- Dynamic MoE routing causes shape variance — vLLM handles this via
  multiple captured graph sizes (the 51+35 graphs above).
- Eager remains the safe fallback if a future quant scheme or merge
  triggers a Marlin/cudagraph regression. Validate with a 5-problem
  smoke before committing to a full eval.

### Action items implemented 2026-05-12

- `quantize_any.py` — env canary aborts pre-launch if
  `modelopt.torch.quantization.plugins.huggingface._QuantFusedExperts`
  is missing (PyPI stable `nvidia-modelopt==0.43.0` regressed and would
  silently produce a 37 GB BF16-equivalent "quant" otherwise).
- `quantize_any.py` — modelopt configs now exclude
  `*vision_tower*`/`*embed_vision*`/`*embed_audio*` from quantization
  by default. vLLM `gemma4_mm` expects BF16 for these.
- `build_98e_v3v4_nvfp4a16.sh` — copies `processor_config.json` (and
  `preprocessor_config.json` if present) from source post-quant.
  modelopt's `tok.save_pretrained` drops it; vLLM needs it for
  multimodal cache profiling at engine init.
- `omk_eval.launch_vllm` — **default flipped to CUDA graphs ON**
  (`--enforce-eager` dropped from the default flag set). Re-enable per
  template via `backend_args.vllm_enforce_eager: true` only if a
  variant breaks graph capture. Validated on v4 NVFP4A16 LCB-55: 75%
  pass-rate / clean stops at ~86 tok/s vs ~22 with eager.
- `omk_eval.launch_vllm` — new kwargs `enforce_eager` (default False),
  `max_num_batched_tokens` (default 4096), `extra` (passthrough). All
  override-able per template via `backend_args.vllm_enforce_eager`,
  `backend_args.vllm_max_num_batched_tokens`, `backend_args.vllm_extra`.
- Templates `gsm8k_100`, `humaneval_full`, `humanevalplus_full`,
  `mbpp_full` — `max_gen_toks` bumped 512/2048 → 16384. Cap is upper
  bound, not target; reasoning models (Gemma 4) need the headroom while
  non-reasoning models still finish early on natural stop tokens.
- New templates: `math500_100`, `arc_challenge_full`, `ifeval_full`.
  All max_gen_toks=16384.


---

## v3.0 — vLLM reasoning-parser baseline (2026-05-12)

The discovery / validation arc on 2026-05-12 established that for any
**vLLM-served reasoning model**, the right baseline is to enable vLLM's
built-in reasoning-parser plus a thinking-token budget. Without these
flags, Gemma 4 silently over-thinks past the gen-budget on hard problems
and we end up shipping numbers that reflect token-cap truncation, not
capability. With them, Gemma 4 routes its CoT through the `<|channel>`
channel format, vLLM clips at the budget, and the `content` field is
clean code the LCB / HE scorer can eat directly.

### 3.1 MANDATORY vLLM flags + per-request kwargs for reasoning models

Add these to every `omk_eval.launch_vllm` call when the model is on the
supported-parser list below:

```
--reasoning-parser <name>          # CLI flag on vLLM api_server
```

Per-request payload (in `extra_body` for OpenAI client, or top-level for
raw HTTP):

```json
{
  "chat_template_kwargs": {"enable_thinking": true},
  "thinking_token_budget": 12288
}
```

### 3.2 Budget rationale — Gemma 4 26B-A4B (validated)

Empirical from `eval_results_vllm4bit/lcb_med_55q/128e_nvfp4a16/`:
the longest *passing* 128e completion under no-parser conditions was
**11,351 tokens**. Set the budget to **12,288** (12 K) — enough to
contain every solve in the observed distribution while clipping the
"`Wait, let me re-read...`" / "`Final check on...`" repeat-loops that
consume the rest of the cap on failures.

`memory/project_t14_gemma4_reasoning_parser_validated.md` records the
5-question cross-model validation:

| task     | 128e no-parser | 128e + parser+budget | v4 no-parser | v4 + parser+budget |
|----------|:---:|:---:|:---:|:---:|
| 3683     | FAIL (length 16k)         | ✅ PASS @ 12,748 | — | ✅ PASS @ 12,994 |
| 3793     | FAIL (length 16k)         | ✅ PASS @ 12,488 | — | ✅ PASS @ 12,495 |
| 3487     | FAIL (wrong algo @ 4,481) | ✅ PASS @ 12,987 | FAIL (length 24k) | ✅ PASS @ 12,931 |
| 3566     | PASS @ 233                | ✅ PASS @ 12,493 | FAIL (length 24k) | ✅ PASS @ 12,494 |
| 3594     | PASS @ 843                | ✅ PASS @ 12,846 | FAIL (length 24k) | ✅ PASS @ 12,765 |

Two surprises baked in:
1. **Capability rescue, not just truncation rescue.** 128e was getting
   3487 wrong at 4,481 tokens with plenty of room — the parser+budget
   combo unlocked the correct algorithm. The structured reasoning
   discipline matters, not only the budget cap.
2. **The budget really clips.** Both models stop at ~12.4-13.0 K
   completion tokens. ~8-9 K of that is in the parsed-out `reasoning`
   field, ~200-2,200 chars in `content`. Scorer-clean.

### 3.3 vLLM-supported reasoning-parser names (current install)

Source: `vllm/reasoning/__init__.py` registry. Pass any of these to
`--reasoning-parser`:

| Parser name             | Model family / use                                  |
|-------------------------|-----------------------------------------------------|
| `deepseek_r1`           | DeepSeek-R1 distills (Qwen / Llama bases)           |
| `deepseek_v3`           | DeepSeek-V3 / V3.1                                  |
| `deepseek_v4`           | DeepSeek-V4 (alias to v3 parser)                    |
| `glm45`                 | GLM-4.5 thinking                                    |
| `holo2`                 | Holo2 (uses DeepSeek-V3 parser)                     |
| `qwen3`                 | Qwen3 reasoning (Qwen3-Reasoning, QwQ tail)         |
| `ernie45`               | ERNIE-4.5 reasoning                                 |
| `**gemma4**`            | **Gemma 4 thinking variants — `<\|channel>` format** |
| `granite`               | IBM Granite 3.2 reasoning                           |
| `hunyuan_a13b`          | Hunyuan A13B (Tencent)                              |
| `hy_v3`                 | Hunyuan V3 reasoning                                |
| `kimi_k2`               | Kimi K2                                             |
| `mimo`                  | MiMo                                                |
| `minimax_m2`            | MiniMax M2                                          |
| `minimax_m2_append_think` | MiniMax M2 (append-style think)                   |
| `mistral`               | Mistral reasoning                                   |
| `nemotron_v3`           | NVIDIA Nemotron V3                                  |
| `olmo3`                 | OLMo 3                                              |
| `openai_gptoss`         | OpenAI gpt-oss (`<\|channel\|>analysis\|>...`)      |
| `poolside_v1`           | Poolside v1                                         |
| `seed_oss`              | Seed-OSS                                            |
| `step3`                 | Step-3                                              |
| `step3p5`               | Step-3.5                                            |
| `cohere_command3`       | Cohere Command A (gen 3)                            |
| `cohere_command4`       | Cohere Command A (gen 4)                            |

**Decision rule:** if the model checkpoint family appears in this table
**and** the chat template recognizes an `enable_thinking` (or
equivalent) kwarg, ship the run with `--reasoning-parser <name>` AND
the per-request kwargs from §3.1. Otherwise (e.g. base-completion
models, non-thinking variants), use the standard `local-chat-completions`
path without these.

### 3.4 Chat-template prerequisite

The parser only does its job if the chat template emits the start/end
tokens. Gemma 4 26B-A4B-it `chat_template.jinja` lines 157-164 inject
`<|think|>` when `enable_thinking=true`, which triggers the model to
emit `<|channel>thought...<channel|>`. Verify on any new model:

```bash
grep -niE "enable_thinking|<\|channel|<\|think" $MODEL_DIR/chat_template.jinja
```

If the template lacks the branch, the parser has nothing to parse and
budget is a no-op. Either fix the template (preferred) or skip the
parser settings.

### 3.5 Where this lands in the code

- `omk_eval.launch_vllm` — already exposes
  `backend_args.vllm_extra` for arbitrary passthrough; add
  `--reasoning-parser gemma4` (or the family's name) there.
- Per-bench templates that should enable thinking on Gemma 4:
  - `lcb_medium_55.yaml`, `lcb_medium_30.yaml` (code + reasoning)
  - `gpqa_diamond_full.yaml` (reasoning)
  - `aime_30.yaml` (math reasoning)
  - `math500_100.yaml` (math reasoning)
  - HE / HE+ / MBPP are technically still chat — enable_thinking helps
    on hard problems but adds latency on easy ones. Keep it ON for the
    Gemma 4 baseline; ablate per benchmark when curious.

### 3.6 Action items implemented 2026-05-12 (continued)

- `scripts/validate_gemma4_reasoning_parser.py` — 2-question 128e
  validation harness. PASS on both prior-length-capped questions.
- `scripts/probe_gemma4_budget.py` — generalized N-question probe
  across any served model + tag. Used to validate 128e + v4 + v3 on
  the 5-question union.
- `scripts/score_validation_responses.py` — re-scores raw vLLM
  responses through the LCB scorer.
- `eval_results_vllm4bit/lcb_med_55q/{128e_5q,v4_5q,v3_5q}_GEMMA4PARSER/`
  — validation evidence retained (raw responses + scored.json).

## v3.1 — Gemma 4 silent-empty RCA + symmetric re-eval setup (2026-05-14)

### 3.1.1 The bug cluster

While running the canonical 9-bench suite on Gemma-4-26B-A4B NVFP4A16 at
`num_concurrent=2`, ~1.5% of responses came back with **empty `content`**
even when the same prompt produced normal output in isolated single-request
calls. RCA isolated three independent failure modes that interact under
continuous batching, each with its own fix:

1. **vLLM #42250 (closure-capture)** — `Gemma4MoE.__init__` captured
   `per_expert_scale` into a closure local before defining
   `routing_function`. Under `torch.func.functional_call` (used during
   CUDA-graph capture on batch-shape changes), parameter substitution did
   not propagate to the closure-captured local, so the routing function
   used a stale scale for affected requests → wrong experts → garbage or
   empty tokens. Cherry-picked upstream commit `d93ba4d32`.

2. **vLLM #42434 (revert of #39917)** — `#39917` ("Replace routing replay
   with device cache and async D2H pipeline") rewrote routed-experts
   capture + scheduler integration; production exposure caused tail
   silent-empty events independent of #42250. Upstream reverted in
   `#42434` (`d522283d5`). Builds between #39917 and #42434 are exposed.

3. **Parser half-open thinking (Fix E)** — even with the two upstream
   fixes, the Gemma4 parser's `extract_reasoning()` returned
   `content=None` whenever the model emitted `<|channel>thought...` but
   never the closing `<channel|>` (e.g. `max_tokens` cap mid-think).
   Local commit `3e55456ed`:

   ```python
   if content is None and reasoning and self.end_token not in model_output:
       content = reasoning
   ```

   The `end_token not in model_output` guard distinguishes half-open
   from closed-with-empty-content (where `content=None` is correct
   semantics — model closed its thinking and chose to say nothing). PR
   materials in `upstream_pr/`.

### 3.1.2 Downstream guard: Fix A in lm-eval's openai_completions

Defense-in-depth: `LocalChatCompletion.parse_generations` reads
`message.content` only. Patched
`/root/anaconda3/envs/omnimergekit/lib/python3.11/site-packages/lm_eval/models/openai_completions.py`
to fall back to `reasoning_content`/`reasoning` when content is empty.
**Load-bearing** — re-apply if the omnimergekit env is rebuilt.

### 3.1.3 Symmetric re-eval pipeline

The canonical cache (gsm8k, gpqa, arc, math500, aime, he, he+, ifeval,
lcb) was built before the patches landed. Past silent-empty events
cannot be retroactively recovered (cache + samples stored only
`content`). The fix is **symmetric re-eval** of failure-union problems:

1. **Failure-union extraction** — `scripts/compute_failure_union.py`
   reads the latest canonical samples for both 128e_nvfp4a16 and
   98e_v4_nvfp4a16, unions failing doc_ids/task_ids, writes
   `scripts/reeval_failure_manifest.json` + markdown summary. PASS rule
   per bench:

   | bench         | PASS rule                          |
   |---             |---                                 |
   | gsm8k/gpqa/arc/aime | `exact_match >= 0.5`         |
   | math500        | `exact_match >= 0.5 OR math_verify >= 0.5` (canonical is `math_verify`; `exact_match` is broken string-match that under-reports ~85pp) |
   | he / he+       | `pass@1 >= 0.5`                    |
   | ifeval         | `prompt_level_strict_acc >= 0.5`   |
   | lcb            | `passed == True`                   |

2. **Override generator** — `scripts/generate_reeval_overrides.py`
   writes:
   - `lm_eval_tasks/reeval24k/utils_reeval.py` — `process_docs`
     subset filters reading the manifest.
   - `lm_eval_tasks/reeval24k/<bench>_reeval24k.yaml` — shadow tasks
     using lm-eval `include:` to inherit from the canonical or
     existing chat-shadow parent.
   - `eval/templates/<bench>_reeval24k.yaml` — omk dispatchers with
     `thinking_token_budget: 24576`, `max_gen_toks: 49152`, fresh
     `sqlite_prefix`, and `lm_eval_include_path: lm_eval_tasks`
     (recursive — picks up reeval24k/ + sibling chat shadows).

3. **Symmetric budget** — both 128e and v4 re-run at
   thinking=24576/max=49152. Wider budget gives a fair retry on
   length-capped problems (v4 LCB-24k retry showed 5/12 cured going
   from 12k→24k thinking; the symmetric run does the same for both
   models on the full failure union).

4. **Run order** — 128e first at the new budget, then v4 (vLLM bounces
   between models). Offline merge: legacy PASS rows ∪ new re-run rows
   → final samples; rescore by aggregating the merged set. Do NOT mix
   raw caches across budget settings — see 3.1.4.

### 3.1.4 Budget changes invalidate cache hits

When `max_tokens` or `thinking_token_budget` changes, the lm-eval
request hash changes → cache misses. Do NOT reuse a legacy
`sqlite_prefix` for a re-run at a different budget. The generator's
`<bench>_reeval24k` prefix convention guarantees no collision.

### 3.1.5 Math500 scoring gotcha

`exact_match` on `minerva_math500` is a string-comparison floor (~5%
on a model that genuinely solves 92% of problems). The canonical
scorer is **`math_verify`** — sympy-based symbolic checker. When
reading samples programmatically (failure detection, offline rescore,
HF-card stats), OR the two:

```python
em_ok = isinstance(row.get("exact_match"), (int, float)) and row["exact_match"] >= 0.5
mv_ok = isinstance(row.get("math_verify"), (int, float)) and row["math_verify"] >= 0.5
passed = em_ok or mv_ok
```

This is `_pass_math500_either` in `compute_failure_union.py`.

### 3.1.6 Long-running canary launching — process group hygiene

Tail-of-2026-05-14: the canary process inherited PGID from a bash
wrapper running inside a Monitor's `command`. When the Monitor stream
ended (after emitting "canary launched"), the PGID was killed and
the canary died mid-HumanEval. `nohup` + `disown` were not enough.

**Rule:** background a long-running probe with `setsid` so it gets a
new process group, never tied to the Monitor's PGID:

```bash
setsid nohup python -u my_long_probe.py >"$LOG" 2>&1 </dev/null &
PID=$!
disown
```

Then verify it really detached:

```bash
ps -p $PID -o pid,pgid,sid --no-headers
# PGID and SID should equal the new PID, NOT the launching shell's PGID.
```

Apply this to any probe/canary/eval whose runtime exceeds the parent
shell or Monitor command's lifetime.

### 3.1.7 Where this lands in the code

- `/srv/dev-disk-by-label-opt/dev/vllm-source/` — editable install
  with cherry-picked #42250 + #42434 + local Fix E commit
  `3e55456ed`. Live for all vLLM runs from this host until upstream
  catches up.
- `scripts/compute_failure_union.py` — failure-union extractor.
- `scripts/generate_reeval_overrides.py` — re-eval template + shadow
  generator (idempotent).
- `scripts/reeval_failure_manifest.json` + `reeval_failure_summary.md`
  — current failure set; regenerate after every canonical chain run.
- `upstream_pr/{PR_BODY.md, 0001-Fix-E-*.patch, probe_silentempty_canary.py}`
  — upstream PR materials, including the canary reproducer.
- Memory: `feedback_vllm_gemma4_silentempty_rca.md` (indexed in
  `MEMORY.md`).

### 3.1.8 When to re-run failure-union extraction

Before launching a re-eval, regenerate the manifest after any of:

- New canonical chain run on 128e or v4 (new `samples_*.jsonl`).
- New variant added to the comparison set (update `BENCH_LAYOUT` in
  `compute_failure_union.py`).
- New bench added to the suite (update `BENCH_LAYOUT` and
  `BENCH_CONFIG`).

Then re-run `generate_reeval_overrides.py`; it overwrites shadow
tasks + omk templates with current state.

---

## 4. Pod setup (cloud GPU rental — vast.ai et al.)

Stand-alone reference for setting up a fresh cloud pod (bare CUDA image
+ ssh) to run quants and evals end-to-end with this protocol. Every
step here is documented because we burned **7+ hours on 2026-05-14**
not following it and had to rediscover each gotcha live.

### 4.1 Pod image — MANDATORY minimum

Image: **`nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`** (or any newer
12.8+ CUDA-devel image with cuDNN 9 pre-installed).

| Requirement | Why |
|---|---|
| CUDA runtime ≥ 12.8 | torch 2.10.0 cu128 wheel demands it. |
| cuDNN 9.x | bundled with torch wheel (9.10.2). |
| `devel` variant (not `runtime`) | nvcc/headers needed for any source-built extension. |
| Ubuntu 22.04 base | matches solidPC producer environment. |
| 2× RTX 3090 24 GB minimum | 26B BF16 won't fit a single 24 GB; you want 1 GPU per parallel eval anyway. |
| ≥ 96 GB RAM | accelerate CPU-offload + calibration buffers. <72 GB forces sequential quant. |
| ≥ 250 GB NVMe | BF16 sources (≈ 51 + 39 GB) + quant outputs (≈ 16 + 13 GB) + miniconda + envs. |
| Symmetric ≥ 500 Mbps inet | HF pull/push of 30-100 GB. |

### 4.2 Phase 1 — bootstrap (`scripts/pod_bootstrap_reeval.sh`)

Source of truth: `backup_models/scripts/pod_bootstrap_reeval.sh`. Phases:

1. apt deps (cmake, git, build-essential, etc.).
2. Miniconda3 install at `/workspace/miniconda` + accept Anaconda ToS for `main` + `r` channels.
3. **vllm-source rsync + build** — patched vLLM (cherry-pick #42250 + #42434 + Fix E) lands at `/workspace/vllm-source`. **The build step is part of the vllm env install, not bootstrap** (see § 4.3).
4. omnimergekit rsync to `/workspace/omnimergekit`.
5. **HF auth + parallel BF16 pull with `HF_XET_HIGH_PERFORMANCE=1`** — mandatory; see `feedback_hf_transfer_on_pods.md`. 5-10× speedup over single-connection HTTP.
6. Generate `pod_quant_and_push.sh` runner in `/workspace/scripts/`.

### 4.3 Phase 2 — conda envs (per `docs/CONDA_ENVS.md`)

**Always create three envs, in this order.** Each must use the
documented `requirements*.txt`; do NOT improvise pip installs.

#### 4.3.1 `omnimergekit` env (eval driver — lm-eval, omk_eval)

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n omnimergekit python=3.11 -y

PY=/workspace/miniconda/envs/omnimergekit/bin
# CRITICAL: install torch cu128 FIRST from PyTorch index (pip default pulls
# cu130 wheel which crashes nvcc on CUDA 12.8 hosts during any source build).
"$PY/pip" install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0

# Then the rest. For non-Qwen3.5/3.6 models, filter out the hybrid-attention
# CUDA extensions that need exact torch/CUDA matching:
grep -vE "^(causal-conv1d|flash-linear-attention|bitsandbytes)" \
    /workspace/omnimergekit/requirements.txt > /tmp/req_filtered.txt
"$PY/pip" install -r /tmp/req_filtered.txt
"$PY/pip" install -r /workspace/omnimergekit/requirements-eval.txt
```

Verify:
```bash
$PY/python -c "import lm_eval, transformers, torch; print(f'lm_eval={lm_eval.__version__} tf={transformers.__version__} torch={torch.__version__}')"
# Expected: lm_eval=0.4.11 tf=5.5.0 torch=2.10.0+cu128
```

#### 4.3.2 `vllm` env (vLLM server)

```bash
conda create -n vllm python=3.11 -y
PY=/workspace/miniconda/envs/vllm/bin
"$PY/pip" install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
"$PY/pip" install vllm==0.20.2
```

Verify:
```bash
$PY/python -c "import vllm, torch; print(f'vllm={vllm.__version__} torch={torch.__version__}')"
```

vLLM 0.20.2 pulls its own CUDA 13.0.2 toolkit/bindings — does not
collide with the system 12.8 because vLLM uses bundled libs at
runtime. If a patched vLLM (cherry-picks, Fix E) is required, layer
it via `PYTHONPATH=/workspace/vllm-source` in the launcher and skip
`pip install vllm`.

#### 4.3.3 `modelopt` env (NVFP4A16 / INT4_AWQ quantization)

Bit-for-bit match of solidPC's producer environment. Use the canonical
script `omnimergekit/scripts/quantize_any.py`; do NOT write a one-off
recipe (see `feedback_use_omnimergekit_canonical.md`).

```bash
conda create -n modelopt python=3.11 -y
PY=/workspace/miniconda/envs/modelopt/bin
"$PY/pip" install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
"$PY/pip" install transformers==5.8.0 safetensors==0.8.0rc0 numpy==2.2.6 \
    huggingface_hub==1.14.0 accelerate==1.13.0 tokenizers==0.22.2 datasets \
    psutil hf_transfer

# Build modelopt from source at solidPC's exact commit (release wheels NaN
# during Gemma 4 calibration; commit g7a11fb240 is validated):
cd /workspace
git clone --quiet https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
cd TensorRT-Model-Optimizer && git checkout 7a11fb240
"$PY/pip" install ninja cmake pybind11
"$PY/pip" install -e .
"$PY/python" -c "import modelopt; print(f'modelopt={modelopt.__version__}')"
```

#### 4.3.4 Symlink the eval_suite_vllm.sh expected paths

`scripts/eval_suite_vllm.sh` hardcodes `/root/anaconda3/envs/...`.
On the pod, symlink:

```bash
mkdir -p /root/anaconda3
[ -e /root/anaconda3/envs ] || ln -s /workspace/miniconda/envs /root/anaconda3/envs
```

### 4.4 Phase 3 — known gotchas (each one a bug we hit)

| Symptom | Root cause | Fix |
|---|---|---|
| `ModuleNotFoundError: vllm` from `eval` env | vllm not installed in the env you're using | Use `/workspace/miniconda/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server …`, not `eval` env. |
| `Cannot copy out of meta tensor` during `model.cpu()` | accelerate's `device_map="auto"` left some params on `meta` (offload hooks); `.cpu()` chokes | Don't bulk-`.cpu()` an offloaded model. Use `gc.collect(); torch.cuda.empty_cache()` between quantize and export. |
| `0 on cuda, N on cpu` even with `max_memory={0:"22GiB","cpu":"55GiB"}` | accelerate bin-packing was over-conservative under the cap | Use **dynamic** per-GPU max_memory (compute `min(free_now, total - 3.5GiB)` per device); no `"cpu"` key (let overflow auto-spill). |
| `AssertionError: detected nan values in amax` during calibration | modelopt 0.43 release / 0.44 NaN on Gemma 4 fused-MoE bf16 calib | Build modelopt from source at commit **`7a11fb240`** (matches solidPC producer). Other versions NaN. |
| `TypeError: list indices must be integers, not str` on `quant_cfg["quant_cfg"]["*vision*"]` | modelopt 0.44+ changed config shape dict→list | omnimergekit's `quantize_any.py` already uses `.extend([…])` list-form. Always use this script — never write one-off. |
| `Failed building wheel for causal-conv1d / flash-linear-attention` | torch wheel ABI mismatch vs system CUDA (cu130 wheel + 12.8 nvcc) | Install cu128 torch FIRST from PyTorch index, then `pip install -r req` with these two packages filtered out (they're Qwen3.5/3.6 only). |
| `import vllm` works but `import vllm.entrypoints.openai.api_server` fails on `cbor2`/`uvloop`/`openai_harmony`/`xgrammar` | rsync'd vllm-source but never `pip install`d it — runtime deps missing | Either `pip install vllm==0.20.2` in a dedicated env (then run from that env), or `pip install -r /workspace/vllm-source/requirements*.txt` if cherry-picks are needed. |
| `FileNotFoundError: reeval24k/<task>` when lm-eval discovers tasks | `include:` in lm-eval YAML is **path-based**, not task-name based — discovery time error blocks ALL task registration | Use `scripts/inline_reeval24k_yamls.py` to merge parent task content into each reeval24k YAML, drop the `include:` line, copy any `!function`-referenced `utils*.py` into `reeval24k/` alongside. |
| `module 'omnimergekit' not found` etc. via `--include_path` | lm-eval doesn't follow nested `include:` chains; only resolves first-level | Same fix as above: inline. The inliner is recursive. |
| sshd intermittently dead after pod boot/reboot (vast reports `running`) | vast.ai container init race; sshd not always up when `actual_status` flips to `running` | Use `--cancel-unavail` on `vastai create` to fail-fast; if you've already rented, reboot once and probe TCP. If sshd doesn't come back in 5 min, destroy + re-rent — don't wait hours. |
| `Connection refused` on the API port from vast | vast assigns an ssh port that may differ from `--port` you specified | Always read `ssh_port` from `vastai show instance <id> --raw` JSON; do not trust your local cache. |
| `UnboundLocalError: cannot access local variable 'outputs'` at `lm_eval/models/api_models.py:545` (every "FAIL" in tmux that still wrote a results.json) | Upstream lm-eval 0.4.11 bug: `outputs` is referenced in `except BaseException` without being initialized. Fires when ALL retries raise before `outputs = await response.json()` lands (transport-level errors: `ServerDisconnectedError`, `ClientPayloadError`, persistent 5xx). On the pod the rate is **~10-30× local** because two vLLM servers + two lm-eval clients on one host contend for the same loopback TCP stack — small jitter accumulates into retry-exhaust events. Bug exists locally too, just fires too rarely to surface. | (a) Patch `api_models.py` to init `outputs = None` before the try (auto-applied by `scripts/pod_setup_eval_envs.sh`). (b) **Bump retries + timeout in `omk_eval.py`'s `model_args`** so retry exhaust is rarer: `max_retries=8` (default 3), `timeout=1800` (default 86400 — set lower so a stuck request fails fast). Both overridable per template via `backend_args.max_retries` / `backend_args.request_timeout` (or `generation.http_timeout` for LCB-style). |
| `ModuleNotFoundError: No module named 'langdetect'` at lm-eval ifeval task import | `lm-eval[api,math]` extras don't pull ifeval deps; pin needs `[ifeval]` extra or explicit `langdetect`/`immutabledict`/`nltk`. Local doesn't repeat because the modules were hand-installed during earlier ifeval work. | Pin `lm-eval[api,math,ifeval]==0.4.11` + explicit `langdetect/immutabledict/nltk` in `requirements-eval.txt` (landed 2026-05-14). On existing pods: `pip install langdetect immutabledict nltk`. |

### 4.4.1 Retry / timeout tuning (`omk_eval` model_args defaults)

Cloud pods have **higher transport-jitter** than solidpc (the dedicated
home box) — two vLLM servers on one host with parallel lm-eval clients
hit the loopback TCP stack hard. The api_models.py `UnboundLocalError`
fires when retries exhaust (default `max_retries=3`); even with the
post-install patch, every exhaust still wastes that question's
work. Tuned defaults in `omk_eval.dispatch_lm_eval`:

| Knob | Old default | New default | Source |
|---|---|---|---|
| `num_concurrent` | 2 | 2 (unchanged) | template `backend_args.num_concurrent` |
| `max_retries` | 3 (upstream) | **8** | template `backend_args.max_retries` |
| `timeout` (sec, per-request aiohttp ClientTimeout) | 86400 (upstream) | **1800** | template `backend_args.request_timeout` (also reads `generation.http_timeout` for LCB-style templates) |
| stop-after-attempt backoff | exponential (lib default) | unchanged | tenacity |

Rationale: 8 retries × exponential backoff still bounded at <~5 min
extra per request worst-case; well below the gain from not losing a
25-min thinking-budgeted generation to retry exhaust. `timeout=1800`
gives a hard ceiling — at vLLM 22 tok/s × 24k thinking + 5k answer ≈
22 min nominal, so 30 min covers tail latency without letting a stuck
request burn an hour silently. See
`memory/feedback_lm_eval_retry_tuning.md` for the data behind these
choices (pod 36755693, 2026-05-14: 57 UnboundLocal hits in one 98e_v4
run; local sees 2-3 per multi-hour run with the SAME code path).

### 4.5 Phase 4 — run the eval (`scripts/pod_reeval_failures.sh`)

After all three envs exist + the patched recipes are inlined:

```bash
# On pod:
chmod +x /workspace/scripts/pod_reeval_failures.sh
tmux new-session -d -s reeval -- bash -lc "
    source /workspace/miniconda/etc/profile.d/conda.sh
    conda activate omnimergekit                # eval driver
    export PYTHONDONTWRITEBYTECODE=1
    export HF_XET_HIGH_PERFORMANCE=1
    bash /workspace/scripts/pod_reeval_failures.sh \
        2>&1 | tee /workspace/logs/pod_reeval_failures.log
"
```

The runner launches vLLM (on the `vllm` env) per GPU + per model in
parallel, then runs the 8 inlined reeval24k templates via `omk_eval`
(omnimergekit env) against each. Failure-subset only — 206 problems
× 2 models = 412 evals; ETA 2-4 h on 2× 3090.

### 4.6 Phase 5 — pre-launch validation (60-second checklist)

Run this on the pod **before** the long eval, every time:

```bash
# 1. envs healthy
for E in omnimergekit vllm modelopt; do
    /workspace/miniconda/envs/$E/bin/python -c "import sys; print(f'$E py={sys.version.split()[0]}')"
done

# 2. critical packages present
/workspace/miniconda/envs/omnimergekit/bin/python -c "
import lm_eval, transformers, torch
print(f'omk: lm_eval={lm_eval.__version__} tf={transformers.__version__} torch={torch.__version__}')"
/workspace/miniconda/envs/vllm/bin/python -c "
import vllm, torch
print(f'vllm: {vllm.__version__} torch={torch.__version__}')"

# 3. NVFP4A16 quants on disk
du -sh /workspace/models/*NVFP4A16*

# 4. failure manifest reachable + correct counts
python3 -c "
import json
m = json.load(open('/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/scripts/reeval_failure_manifest.json'))
for k,v in m.items():
    ids = [i for i in v.get('ids',[]) if isinstance(i,int)]
    print(f'  {k}: {len(ids)} ids, range={min(ids) if ids else None}-{max(ids) if ids else None}')"

# 5. lm-eval task discovery doesn't crash on reeval24k
/workspace/miniconda/envs/omnimergekit/bin/lm-eval \
    --tasks gsm8k_reeval24k --include_path /workspace/omnimergekit/eval/lm_eval_tasks \
    --model dummy 2>&1 | tail -10
# Expect: tasks discovered, no FileNotFoundError. (Dummy model will fail at run; that's fine.)

# 6. both GPUs free
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

If ANY of the 6 fails: STOP. Do not launch the eval — fix the
broken step. This list exists because every entry corresponds to a
failure we've shipped to production.

## v3.2 — `--gpu-memory-utilization` is per-(model × GPU), NEVER a fixed default

**MANDATORY:** every vLLM-driven eval (and every quant pipeline that boots
vLLM) MUST set `--gpu-memory-utilization` tuned for the SPECIFIC pairing of
{model size, GPU VRAM, max-model-len, max-num-seqs}. There is no "safe
default" that works across GPUs. Pick a value that is **just enough** to
hold (model + activations + KV-cache pool for the expected concurrency ×
context); anything higher is wasted VRAM that another job could use,
anything lower aborts vLLM with `Available KV cache memory: -X GiB` or
`No available memory for the cache blocks`.

### v3.2.1 The pathology — why a "fixed 0.92" hides bugs

vLLM's `--gpu-memory-utilization X` is a **hard reservation** of
`total_gpu_memory × X` at startup, not an upper-cap that grows as KV-cache
fills. Once allocated, that slab is held even when the actual KV cache is
at 2%. Two failure modes hit us in production:

- **Solidpc 3090 (24 GB) regression — 2026-05-17:** patched `eval_suite_vllm.sh`
  from 0.92 → 0.55 to mirror the 48 GB pod scripts. 0.55 × 24 GB = 13.2 GB,
  which equals the Gemma 4 26B-A4B NVFP4A16 model size *with no room for KV
  cache*. vLLM died at `Available KV cache memory: -0.88 GiB`. ~25 min of
  eval lost.
- **Pod 36929284 4090 (48 GB) inverse:** v5-coder eval booted at 0.92 ×
  48 GB = 44 GB held, while measured KV-cache usage stayed at 1.8–2.9%
  across an hour. That's ~30 GB of GPU left idle when a parallel NVFP4A16
  quant could have shared the card.
- **Pod 36949547 RTX 6000 Ada (48 GB) 31B dense — 2026-05-17:** dense 31B
  NVFP4A16 (~21 GB) at 0.65 × 48 GB = 31.2 GB → 9.41 GB for KV cache, but
  max-model-len=32768 needs 9.54 GB. ValueError at engine init. Bumped to
  0.70 (33.6 GB → ~12 GB KV) and it booted.

The common failure mode: copying a value that worked elsewhere instead of
computing the budget for the current pairing.

### v3.2.2 The formula — apply BEFORE every vLLM launch

```
budget_GB           = total_gpu_VRAM_GB × gpu_memory_utilization
model_VRAM_GB       = (file size of weights on GPU; ~13 GB for Gemma 4 26B
                       NVFP4A16, ~21 GB for Gemma 4 31B NVFP4A16,
                       ~52 GB for Gemma 4 26B BF16)
activation_GB       = ~1.5–2 GB constant (CUDA graphs + buffers + tiny scratch)
kv_per_token_GB     = 2 × n_layers × hidden_dim × dtype_bytes / 1e9
                       (Gemma 4 26B-A4B at bf16 KV cache: ~0.0005 GB/token)
kv_required_GB      = max_num_seqs × max_model_len × kv_per_token_GB

# Constraint
budget_GB ≥ model_VRAM_GB + activation_GB + kv_required_GB
# Solve for the minimum util that fits with ~10% slack:
util_min            = (model_VRAM_GB + activation_GB + kv_required_GB × 1.1)
                      / total_gpu_VRAM_GB
```

Pick `util_min` rounded up to the nearest 0.05. Always reserve ~10% slack
in `kv_required_GB` for CUDA graph profiling overhead (see vLLM's
own message: "the current --gpu-memory-utilization=0.5500 is equivalent
to --gpu-memory-utilization=0.5464 without CUDA graph memory profiling").

### v3.2.3 Reference table — known-good values (2026-05-17 onwards)

These are launch points; always verify with v3.2.4 below before committing
multi-hour work.

| GPU            | VRAM  | Model                       | max-model-len | gpu_mem_util | Headroom for parallel work |
|----------------|-------|-----------------------------|---------------|--------------|----------------------------|
| RTX 3090       | 24 GB | Gemma 4 26B-A4B NVFP4A16    | 32768         | **0.90**     | ~2 GB (none — solidpc is single-job) |
| RTX 3090       | 24 GB | Gemma 4 26B-A4B Q6_K (GGUF) | 32768         | n/a (llama.cpp) | — |
| RTX 4090       | 48 GB | Gemma 4 26B-A4B NVFP4A16    | 32768         | **0.55**     | ~22 GB (room for NVFP4A16 quant) |
| RTX 6000 Ada   | 48 GB | Gemma 4 31B-dense NVFP4A16  | 32768         | **0.70**     | ~14 GB (room for one small auxiliary) |
| RTX 6000 Ada   | 48 GB | Gemma 4 31B-dense NVFP4A16  | 65536         | **0.80**     | ~10 GB (tight) |
| A100 80GB      | 80 GB | Gemma 4 26B-A4B NVFP4A16    | 32768         | **0.35**     | ~52 GB (room for full BF16 + quant) |
| A100 80GB      | 80 GB | Gemma 4 31B-dense NVFP4A16  | 32768         | **0.45**     | ~44 GB |

Rows are **examples**, not a closed list. New pairings require a fresh
calculation per v3.2.2.

### v3.2.4 Validation step — first 60 seconds after vLLM boots

Before walking away, confirm the eval actually progressed past engine init:

```bash
# 1. Engine actually started (model weights on GPU)
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Expect: ≥ model_VRAM_GB; if 0 MiB → engine died at init, read vllm_server.log

# 2. /health responds 200 (vLLM bound port)
curl -fs http://localhost:$PORT/health && echo OK

# 3. Grep the boot log for the actual KV-cache budget vLLM picked
grep -E "Available KV cache memory|GPU KV cache usage" $VLLM_LOG | head -3
# Expect: positive "Available KV cache memory: N GiB" where N ≥ 1
```

If `Available KV cache memory: -X GiB` appears, vLLM aborted at
`_check_enough_kv_cache_memory` — bump util or lower max-model-len
**before** spending compute. If the engine boots but KV usage stays under
10% for an hour, util is over-reserved — lower it next run to free GPU
for parallel work.

### v3.2.5 Where to commit the value

- `scripts/pod_*_eval.sh` — one line per pod script; the value lives next
  to `--max-model-len` and `--max-num-seqs` in the vLLM launch block.
  Scripts MUST NOT share util values across different pod hardware.
- `scripts/eval_suite_vllm.sh` (solidpc orchestrator) — value is keyed to
  the 3090 (0.90 today); never copy from a pod script.
- `omnimergekit/scripts/quantize_gguf.py` — uses its own `auto_ngl()`
  for the imatrix step; not affected by this rule.

### v3.2.6 Memory entry

Cross-reference: [`feedback_vllm_gpu_memory_util_cap_not_reserve.md`]
documents the failure modes above with the dated incidents. Update it
whenever a new (model, GPU, util) pair is validated in production.

## v3.3 — Stack lock + canary verification + update procedure (2026-05-21)

The eval stack is no longer "whatever pip happened to install"; it is a
versioned artifact described by `eval/stack.lock.yaml`. Cohorts must
pin to a stack version; HF cards must cite it; stack updates must pass
the canary regime before promotion. This section is **the procedure to
follow** every time vLLM (or any stack component) needs to be updated.

### v3.3.1 Why a locked stack exists

We hit two opposite stack failures on Gemma 4 in succession:

- **Fix-E dev build** (vLLM commit `630492da3` + cherry-picks + 8-line
  parser patch): produced clean short answers on HE / IFEval (p50 ≈
  600–700 chars) but under-counted **AIME** (~36 % real-vs-73 % on
  128e) — the artifact was traced to extraction edge cases on a non-
  released branch state.
- **Stock vLLM 0.20.2 release wheel**: reveals real AIME scores
  (~70 % for 26B-A4B class, matches 31B dense reference) but blows up
  **HE / IFEval / HE+** response lengths 10–30× on pruned MoE
  variants → IFEval drops to 22 %, HE drops 2–7 pp. Root cause: the
  release was tagged **before** `#42250` (Gemma4 MoE closure capture
  fix) was merged. Per-expert scale captured in a stale closure during
  `functional_call` substitution → routing distribution wrong on
  pruned MoE → model ruminates → reasoning parser sees half-open
  thinking → content polluted.

Neither stack is uniformly correct. **Both numbers from both stacks
were partly real and partly artifact.** This is exactly the kind of
silent contamination that motivates the locked-stack-with-canary
regime.

### v3.3.2 Stack contents (the four components)

`eval/stack.lock.yaml` describes:

1. **vLLM**: branch + base commit SHA + ordered list of cherry-picks.
   For NVFP4A16 Gemma 4 MoE eval, the locked stack is built from
   source — there is no released wheel that passes the canary today.
2. **lm-eval**: `lm-eval[api,math,ifeval]==0.4.11` + the **Fix-A
   `reasoning_content` fallback patch** (`openai_completions.py`).
3. **modelopt**: pinned at `0.43.0` (`0.44.0` has two Gemma 4
   regressions).
4. **omnimergekit eval/**: pinned git SHA so templates are reproducible.

Hardware floor is also in the lock (CUDA driver ≥ 580, cuDNN ≥ 9,
VRAM ≥ 24 GB) — pods that don't meet it are rejected before rent.

### v3.3.3 Canary regime (two layers)

**Layer 1 — Structural rules** (`eval/structural_canary.py`):
model-agnostic post-processing of any `samples_*.jsonl`. Empirically
calibrated thresholds catch parser/scorer breakage regardless of which
model is being evaluated. Five rules:

1. `empty_content_rate` ≤ 5 % on thinking-on benches
2. `marker_leak_in_content` == 0 (no `<|channel>`, `<bos>`, etc. in
   content)
3. `response_p10_chars` inside the per-bench-kind healthy band
   (short_answer 0–2000, thinking_reasoning 200–60000)
4. `response_p50_chars` inside the upper bound (short_answer ≤ 5000,
   thinking_reasoning ≤ 60000) — **this is the rule that caught the
   v6 IFEval cliff on stock 0.20.2** (observed p50 = 23 528)
5. `response_p99_chars` ≤ 4× p50 upper bound — catches `finish_reason
   = length` budget saturation

**Layer 2 — Reference anchor** (`eval/omk_canary.py` +
`eval/stack_anchors.yaml`): runs a fixed 30-question subset (10 GPQA +
10 AIME + 10 IFEval, see `templates/anchor30.yaml`) against a
**per-family anchor model** (e.g. `gemma-4-26B-A4B-it` for the
26B-A4B family) and compares scored result to the recorded
expectation within tolerance. This catches stack-level **scoring**
drift the structural rules cannot see — the AIME 36-vs-73 case had
normal response shape but wrong scores.

Both layers must pass. Either alone is insufficient.

### v3.3.4 Procedure to update vLLM to a newer dev build

When you want to refresh the stack (new vLLM dev commits, new Gemma 4
fix landed, etc.), follow these eight steps. **Do not skip.** Every
prior stack disaster came from skipping one.

**Step 1 — Survey the commit range.** Clone vLLM main, find the
commit range between current stack's base SHA and the new target HEAD.
Filter for Gemma / MoE / reasoning / parser changes:

```bash
cd /tmp && git clone --depth 500 https://github.com/vllm-project/vllm.git
cd vllm
git log <current_base>..<target_head> --format='%h %ai %s' \
    | grep -iE "gemma|moe|reasoning|parser|routing|chat.template"
```

For each commit in the filter, decide: **does it touch the response
path** (parser output, content/reasoning split, finish detection)? If
yes, treat it as canary-relevant.

**Step 2 — Verify the load-bearing fix is still present.**
`#42250` (Gemma4 MoE routing closure captures per_expert_scale) is the
rumination root-cause fix. Confirm it's in the target HEAD:

```bash
git log <target_head> | grep -E "42250|closure.captures|per_expert_scale"
```

If absent (e.g. tagged release before merge), abort or cherry-pick
explicitly. Stock 0.20.2 was the exact case where this PR was missing
and the stack failed.

**Step 3 — Verify nothing regressed.** Check that prior reverts are
still effective. For example, `#39917` (routing replay) was the
upstream change that broke MoE; it was reverted via `#42434`. Confirm
the revert is still effective and a sneakily-re-merged version hasn't
shown up:

```bash
git log <target_head> | grep -E "39917|routing.replay|device.cache"
```

If `#39917` appears without an offsetting revert, do NOT use this
HEAD as-is.

**Step 4 — Check the reasoning parser file directly.**
`vllm/reasoning/gemma4_reasoning_parser.py` is the file that owns
content/reasoning split. Compare against the current stack:

```bash
git log <current_base>..<target_head> -- vllm/reasoning/gemma4_reasoning_parser.py
```

If the file changed upstream, **read the diff** before adopting. A
change here can silently flip whether reasoning leaks into content.

**Step 5 — Cherry-pick Fix-E parser hardening.** The 8-line fix
(`a39e23ed0` original, currently `3d92852eb` on
`gemma4-moe-stack-v2`) handles the half-open thinking case (no
`<channel|>` close emitted). Apply on top of the target HEAD:

```bash
cd /srv/dev-disk-by-label-opt/dev/vllm-source
git fetch upstream
git checkout -B gemma4-moe-stack-vN+1 upstream/main   # bump N
git cherry-pick a39e23ed0                              # or current SHA
```

The branch name must match `stack.lock.yaml`'s
`components.vllm.source.branch`.

**Step 6 — Build wheels.** Run
`scripts/build_vllm_wheels.sh` (idempotent, prereqs documented in
the script header). Produces per-arch wheels under
`wheels/gemma4-moe-stack-vN+1/`:

```bash
bash scripts/build_vllm_wheels.sh                # all arches (sm86-sm120)
bash scripts/build_vllm_wheels.sh sm86           # 3090 only
bash scripts/build_vllm_wheels.sh --multi        # one fat wheel
```

**Do not build while GPU evals are in flight** — the build is CPU
bound but pulls ~12 cores and may stall the eval orchestrator.

**Step 7 — Bump `stack.lock.yaml`.** Increment `version:`. Update
`components.vllm.source.base_sha` to the new target HEAD. Update
`upstream_fixes_in_base` to reflect any new Gemma-4-relevant PRs
that landed. Commit to omk.

**Step 8 — Run the canary on the new stack.** Pick the anchor model
from `stack_anchors.yaml` for the family being worked on:

```bash
python eval/omk_canary.py \
    --stack eval/stack.lock.yaml \
    --anchor-model google/gemma-4-26B-A4B-it-NVFP4A16 \
    --served-name 128e_nvfp4a16 \
    --family gemma-4-26B-A4B \
    --out eval_results/canary/<stack_name>_<ts>/
```

- **All structural rules pass** AND
- **All anchor scores within recorded tolerance** (currently ±20pp
  for sub-bench n=10 entries):

  → Promote. Append entry to `STACK_HISTORY.md` with date, PRs picked
  up, canary table. Cohort runs can now reference the new version.

If any rule fails, **iterate on the stack, not on the canary**.
Adjusting tolerances to make a broken stack pass defeats the regime.

### v3.3.5 What to do when a canary fails

The two failure modes have different remediation paths:

- **Structural canary fails** (p50/p99 explode, marker leak, etc.):
  the parser/server is broken. Likely candidates: a vLLM commit
  between current and target changed channel handling, or a routing
  fix regressed. Bisect with `git bisect` between current base SHA
  and target HEAD using a single-question response-length probe as
  the test.

- **Anchor canary fails** (structural rules pass, scores drift > tol):
  the parser is mechanically OK but extraction is wrong. Look at
  template `process_results` and the filter pipeline. AIME 36 vs 73
  was this class — the `aime24_chat` shadow task fixed extraction
  while `aime24` (non-chat) kept the bug.

### v3.3.6 STACK.txt — runtime fingerprint (mandatory)

Already in §1.4.5; reinforced under v3 with one addition: every
`omk_eval.py` run writes `<result_dir>/STACK.txt` recording the
**stack version from `stack.lock.yaml`** that produced this number.
HF model cards cite the version in the cohort table footer. Cross-
stack comparisons are explicitly called out as not-apples-to-apples
in the card text.

### v3.3.7 Files in this regime

| File | Owns |
|---|---|
| `eval/stack.lock.yaml` | Versioned stack components (vLLM + cherry-picks, lm-eval + patches, modelopt, omk SHA, templates SHAs) |
| `eval/stack_anchors.yaml` | Per-family recorded reference scores |
| `eval/structural_canary.py` | Layer-1 rules over any `samples_*.jsonl` |
| `eval/omk_canary.py` | Layer-2 orchestrator (run anchor → diff to expectations) |
| `eval/templates/anchor30.yaml` | Fixed 30-question canary subset |
| `eval/canary_ifeval_rumination4.py` | Layer-3 rumination-trigger canary (4 IFEval doc_ids) — see §v3.3.8 |
| `scripts/stack_canary_4doc_run.sh` | Driver: vLLM up → 4-doc canary → teardown |
| `scripts/build_vllm_wheels.sh` | Per-arch wheel builder for the locked vLLM source |
| `scripts/install_stack.sh` | Idempotent installer (local + pod) |
| `eval/STACK_HISTORY.md` | Append-only log of stack promotions |

### v3.3.8 IFEval rumination-trigger canary — MANDATORY on every stack upgrade (2026-05-22)

The Layer-1 + Layer-2 canary regime as defined in §v3.3.3 is **necessary
but not sufficient**. It failed to catch a real regression introduced by
stack@2 (vLLM main `68e07d591` + Fix-E cherry-pick) on v5-coder NVFP4A16:

| | v5-coder × stack@1 | v5-coder × stack@2 |
|---|---:|---:|
| IFEval prompt_level_strict_acc | 94 % | 91 % |
| IFEval doc 18 chars | 28 | **13635** (66× repetition loop) |
| IFEval doc 31 chars | 1591 | 1795 (multilingual contamination — Lao + Greek + Japanese in a Punjabi rubric) |
| IFEval doc 50 chars | 32 | **10398** (60× repetition loop) |
| IFEval doc 59 chars | 1498 | **18292** (52× repetition loop) |

Layer-1 missed it because **doc 59 sits below the p50 short_answer
ceiling of 30000** — only one bad prompt out of 100 doesn't budge p50.
Layer-2 missed it because `anchor_ifeval_10` uses indices
`[0, 5, 10, …, 45]` — by construction it does **not** include 18, 31, 50,
or 59. The four worst-affected prompts on the canonical pruned-MoE
canary model were structurally invisible to the existing regime.

#### v3.3.8.1 Why these four

Empirically (2026-05-22, v5-coder NVFP4A16 weights):

| doc_id | constraint | greedy fragility |
|---:|---|---|
| 18 | Kannada-only | low-resource Indic-script tokens have tiny top-1/top-2 logit gaps; routing micro-numerics decide the path |
| 31 | Punjabi-only | same family, but with a "produce a rubric" generative scaffold that exposes language-mixing under perturbation |
| 50 | Marathi-only (haiku) | Devanagari short-answer task; near-tied logits at every step |
| 59 | no-comma + low-c + ≥250-word essay | non-language constraint cluster; English with three orthogonal hard filters |

All four sit at the **long tail of greedy decoding stability**: the
top-1/top-2 logit gap is small enough that any routing perturbation
shifts the argmax, and at least one downstream argmax lands in a
repetition attractor with no escape. They are stack-sensitive in
exactly the way Layer-1 thresholds are not designed to catch.

#### v3.3.8.2 Stack@1 baseline (the canary's gold)

Pinned 2026-05-22 from v5-coder NVFP4A16 × stock vLLM 0.20.2 wheel on
L40 pod 37006213 (T39):

```
doc 18 → 28 chars,   PASS
doc 31 → 1591 chars, PASS, Gurmukhi-only
doc 50 → 32 chars,   PASS
doc 59 → 1498 chars, PASS, ≥250 words, no commas, c<1
```

These are the targets every future stack must hit (within 2× tolerance
on chars; script-purity ≥95 % on docs 18/31/50).

#### v3.3.8.3 Runner

```bash
# Standalone canary (server must already be up):
python eval/canary_ifeval_rumination4.py \
    --base-url http://localhost:8195/v1 \
    --served-name 98e_v5_coder_nvfp4a16 \
    --out canary_result.json

# Full driver (boots vLLM, runs canary, tears down):
bash scripts/stack_canary_4doc_run.sh stack3_revert_39917
```

Output: `canary_results/4doc_<stack_label>_<TS>/` with `STACK.txt`
fingerprint, `canary_result.json`, full `run.log`, and `vllm_server.log`.

Exit codes:
- `0` ALL_PASS — promote-eligible (still needs Layer-1 + Layer-2)
- `2` ANY_FAIL — at least one doc ruminates or contaminates; stack is
  not promotable regardless of what the other two layers say
- `3` SETUP_ERROR — server didn't come up, malformed responses, etc.

#### v3.3.8.4 When to run it

**Every stack upgrade** must run this canary before promotion, in
addition to §v3.3.4 Steps 1–8. Add a Step 9:

> **Step 9 — Run the IFEval rumination canary.** Pick the canonical
> pruned-MoE canary model (currently `gemma-4-A4B-98e-v5-coder-NVFP4A16`)
> and run `scripts/stack_canary_4doc_run.sh <stack_name>`. **All 4 docs
> must PASS** (exit 0). If any FAIL, the stack regresses on
> stack-sensitive prompts and is not promotable — iterate the stack, not
> the thresholds.

Re-baseline only when the canary model itself changes (new pruning
recipe, new variant promoted to canary status). Re-baselining requires:

1. A stack that has already passed Layers 1 + 2 on its own anchor model.
2. A run of the canary against the new model on that stack.
3. Update of the `stack1_baseline_chars` + `max_chars_healthy` constants
   in `eval/canary_ifeval_rumination4.py` (CANARY_DOCS list) with the
   new values.
4. STACK_HISTORY.md entry documenting the re-baseline.

#### v3.3.8.5 Why not just add the 4 docs to anchor30

Because `anchor30.yaml` is wired into `omk_canary.py` via `--limit N`
(first-N from each parent template), not by per-index selection. Adding
indices 18/31/50/59 would require teaching the runner to honor
`selection.type=indices` end-to-end, which it currently does not for
the canary path. The standalone canary script bypasses that limitation
and ships today; folding it into anchor30 + omk_canary.py is a future
cleanup, not a blocker.
