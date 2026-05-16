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

```
Server: chat profile, --parallel 2 OK, KV q8_0/q8_0 OK
Generator: scripts/multipl_e_generate.py
  --max-tokens 2048 (Rust)  to 4096 (verbose languages like Java/Cpp)
  --concurrency 2
  HAS retry on 5xx (bumped 2026-05-10): 4/8/16/32/64/128s backoff,
    server errors NOT silently dropped (R-RUN: server errs MUST be repeated)
Eval: scripts/multipl_e_evaluate.sh (Docker; runs nuprl/multipl-e-evaluation image)
  Pod with no docker → run eval phase locally on solidpc
```

`memory/feedback_lm_eval_pod_deps.md` (Docker is rare on rented pods)

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
# lm_eval / SQLite cache — first 3 cached responses
sqlite3 <cache.db> "select task,key,length(response) from kv_cache limit 3" | head
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

---

## 6. Repository layout (canonical paths)

```
omnimergekit/
├── eval/
│   ├── EVAL_PROTOCOL.md                  # this file
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
│   └── pod_runners/
│       └── pod_<exp>.sh                  # thin shell launchers; import from eval/
└── recipes/
    └── ...
```

`backup_models/scripts/` is project-specific glue and ad-hoc one-shots only.
Anything reusable goes into `omnimergekit/eval/`.

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
5. **HF auth + parallel BF16 pull with `HF_HUB_ENABLE_HF_TRANSFER=1`** — mandatory; see `feedback_hf_transfer_on_pods.md`. 5-10× speedup over single-connection HTTP.
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
    export HF_HUB_ENABLE_HF_TRANSFER=1
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
