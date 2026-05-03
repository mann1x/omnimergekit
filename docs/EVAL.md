# Evaluation methodology

This kit relies on `lm-eval-harness` (the `local-completions` and
`local-chat-completions` model backends, served by `llama.cpp`'s
`llama-server`). Below are the rules we've baked into every recipe and
the gotchas that cost us multi-hour reruns.

## Universal rules — apply to EVERY `lm_eval` invocation

These are not suggestions. Skip them and you'll lose hours.

```bash
HF_ALLOW_CODE_EVAL=1 lm_eval \
    --model local-completions \
    --model_args "model=$NAME,base_url=http://localhost:8099/v1/completions,..." \
    --tasks $TASK \
    --gen_kwargs "temperature=0.0,top_p=1.0,max_gen_toks=$MAXTOK" \
    --batch_size 1 \
    --use_cache "$WORKDIR/${TASK}_cache/$NAME" \
    --log_samples \
    --confirm_run_unsafe_code \
    --output_path "$WORKDIR/${TASK}_$NAME"
```

| Flag | Why |
|------|-----|
| `--use_cache` | SQLite cache makes runs resumable. Without it, any death (PEG parser crash, OOM, llama-server hang, network blip) restarts from 0. We have lost 5+ hours of compute on this. **No exceptions.** |
| `--log_samples` | The post-run sanity check needs sample data. A `pass@1=0.0` may mean the model is bad OR the scorer crashed at `exec()` because of markdown fences, empty generations, or truncation. Always inspect samples. |
| `--confirm_run_unsafe_code` + `HF_ALLOW_CODE_EVAL=1` | Code-eval tasks (HE, MBPP, HumanEval-Plus) require BOTH. Missing the env var causes a silent banner-only log with zero results. |
| `--gen_kwargs "max_gen_toks=$MAXTOK"` | The CLI gen_kwargs OVERRIDE the task yaml. If you don't set this, AIME / MMLU-Pro use their tiny defaults and truncate reasoning chains. |
| `--model_args "...,max_length=32768"` | `local-completions` defaults to `max_length=2047` regardless of what the model supports. Always override; long reasoning tasks silently truncate at 2k otherwise. |

## Post-run sanity check

```bash
SAMPLES=$(ls $WORKDIR/${TASK}_$NAME/$NAME/samples_*.jsonl)
N_TOTAL=$(wc -l < "$SAMPLES")
N_EMPTY=$(jq -c 'select(.resps[0][0] == "" or .resps[0][0] == null)' "$SAMPLES" | wc -l)
N_FENCE=$(jq -c 'select(.resps[0][0] | contains("```"))' "$SAMPLES" | wc -l)
N_SHORT=$(jq -c 'select(.resps[0][0] | length < 5)' "$SAMPLES" | wc -l)
echo "$NAME $TASK total=$N_TOTAL empty=$N_EMPTY fenced=$N_FENCE short=$N_SHORT"
```

Anomaly thresholds:

- `empty > 1%` → **STOP**. Model is failing to respond. Check chat template, server health, prompt format.
- `fenced > 5%` on HE/MBPP via `/v1/completions` → **STOP**. Re-eval via `/v1/chat/completions + apply_chat_template`, or use `eval/rescore_humaneval_strip_fences.py` to recover.
- `short < 5 chars > 1%` → samples are being truncated mid-token; almost always `max_length` or `max_gen_toks` set too low.

## Validate during the run, not just after

For long evals (>30 min), spot-check progress every 10 min:

```bash
# count requests done so far
sqlite3 $WORKDIR/${TASK}_cache/$NAME -cmd ".mode csv" "SELECT COUNT(*) FROM ..."
# eyeball latest sample
ls -t $WORKDIR/${TASK}_$NAME/$NAME/samples_*.jsonl | head -1 | xargs tail -1 | jq .resps
```

If the request counter isn't moving, the server is hung or saturated. If
samples look truncated mid-token, a `max_*` flag is wrong; kill and fix
before the next 30 minutes burn.

## llama-server flags we always use

```bash
/opt/llama.cpp/build/bin/llama-server \
    -m $GGUF --port 8099 \
    -c 32768 -t 12 -ngl 99 \
    --no-warmup \
    --parallel 2 \
    --cache-type-k q8_0 --cache-type-v q8_0
```

Plus, for **Gemma 4** models specifically:

```bash
    --reasoning-format deepseek --reasoning-budget 8192
```

The `--reasoning-budget 8192` is **mandatory for Gemma 4**. Without it,
the model emits malformed channel tokens (`<|channel>thought` missing
closing) and lm_eval crashes mid-eval at the first chemistry-heavy
question. Wasted an hour learning this.

For 27B-class merges, use `-c 65536` and `--parallel 2`. For 4B,
`-c 32768 --parallel 2` is plenty. `--cache-type-k q8_0` halves KV cache
VRAM for parallel-2 evals.

## Common tasks and their expected ranges

| Task | Limit (full / quick) | Wall time on RTX 3090 (4B Q6_K) | Notes |
|------|-----|-----|-----|
| GPQA Diamond | 198 / 20 | 6-10h / 30 min | use `eval/gpqa/eval_gpqa_v3.sh`. Tokenizer must be the original 128e Gemma 4 dir, not the pruned variant. |
| HumanEval | 164 | 5-15 min | `--task humaneval`. Re-rescore if fences detected. |
| HumanEval-Plus | 164 | 10-30 min | `--task humaneval_plus`. Stricter scoring; lower numbers expected. |
| MBPP | 500 | 20-60 min | `--task mbpp`. |
| LCB-medium-30 | 30 | 30-90 min | use `eval/lcb/lcb_llama_server.py` (custom runner, lm_eval task missing). |
| GSM8K | 100 / 1319 | 5 min / 60 min | `--task gsm8k`, `max_gen_toks=512`. |
| MMLU-Pro | 200 / 12k | 10 min / 8h | `--task mmlu_pro`, `max_gen_toks=1024`. |
| AIME | 30 | 10-30 min | `--task aime`. AIME prompts short, but completions can hit 8k tokens — set `max_gen_toks=8192`. |

## When to use chat-completions vs completions

- **`/v1/chat/completions`** + `--apply_chat_template`: any reasoning model
  (Gemma 4, Qwen3.5 reasoning variants, DeepSeek-R1). The model's chat
  template handles the `<think>` channel framing. **Required** for GPQA.
- **`/v1/completions`** (raw text completion): code-completion tasks where
  you want the model to continue from a function signature without any
  chat wrapping. **Required** for HE/MBPP on chat models — chat mode
  wraps answers in markdown fences and the scorer can't `exec()` fenced
  code.

The "Gemma 4 chat-only HE breaks" learning: Gemma4-it variants degenerate
(markdown fences + token loops) on raw `/v1/completions`. HE drops to
0.61% (rescored 1.83%); MBPP 44.40% is OK. Use chat-completions + chat
template, or skip HE for chat-only models.

## lm_eval + transient TimeoutError patch

`lm_eval/api/api_models.py` line ~545 has an `UnboundLocalError: outputs`
on transient TimeoutErrors. Symptom: a single 504/timeout from
`local-completions` kills the whole eval with an unrelated stacktrace,
leaving the cache half-written. Fix: 1-line patch + purge `.pyc` +
`PYTHONDONTWRITEBYTECODE=1`. We've hit this 3 times on a single GPQA run.

The patched line wraps the request in try/except returning `[]` on
TimeoutError (lm-eval will retry the document via the cache). Apply
this patch to any fresh pod / fresh env install of lm-eval. See
`docs/PATCHES.md` for the diff.
