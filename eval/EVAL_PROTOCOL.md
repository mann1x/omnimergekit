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
