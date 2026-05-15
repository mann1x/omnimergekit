# Eval templates

Self-contained, file-based eval specifications. A template fully specifies
**which problems**, **how to generate**, **how to score**, and **where to
cache** for a single benchmark run. Templates are the unit the eval suite
loads (`--template <path|name>`).

Two forms of problem selection:

1. **Deterministic indices** — frozen list of dataset row indices. Bit-exactly
   reproducible across lm-eval versions and dataset re-shuffles. Use this for
   smoke subsets (HE-20, MBPP-40) and any "always the same 100 questions"
   need (GSM8K-100, MMLU-Pro-200).
2. **Criteria filter** — predicate over dataset rows (difficulty, date range,
   `testtype`, doc_id allowlist, etc.). Use this when the dataset itself
   already implies a stable subset, e.g. LCB-medium-post-2024-10-01.

Each backend reads its own subset of fields; unknown fields are ignored. The
common shape is:

```yaml
name: <stable identifier, used in cache paths>
backend: lm-eval | lcb_custom    # which scorer
task: <upstream task name>       # for lm-eval: gsm8k, humaneval, …
                                 # for lcb_custom: livecodebench
n: <expected sample count>       # validated at load time

selection:
  type: indices | filter
  indices: [<ints>]              # when type=indices
  difficulty: medium             # when type=filter (LCB-specific)
  min_date: "2024-10-01"         # when type=filter (LCB-specific)
  testtype: functional           # when type=filter (LCB-specific)
  doc_ids: ["abc", "def"]        # optional explicit allow list

generation:
  max_gen_toks: <int>
  temperature: <float>            # 0.0 = greedy
  top_p: <float>
  top_k: <int>
  do_sample: <bool>
  stop: ["..."]                  # optional stop strings

scoring:
  # lm-eval path:
  metric: exact_match | pass_at_1 | acc | acc_norm
  filter: flexible-extract | strict | none
  # custom-scorer path:
  scorer: lcb_helpers
  validated_against: <official scorer ref>
  validation_date: "YYYY-MM-DD"

backend_args:
  # lm-eval invocation flags passed-through to `lm_eval`
  apply_chat_template: true | false
  num_fewshot: <int>
  batch_size: <int|str>
  use_cache: true                # always true per project rule

cache:
  sqlite_prefix: <name>          # forms <out>/<prefix>_<model>/...

reports:
  token_stats: true              # protocol-mandatory: log token usage
```

The loader (`scripts/template_loader.py`) resolves a template to a plain
dict, validates required fields, and refuses to run if `n` disagrees with
`len(indices)` (indices path) or with the dataset's filtered cardinality
(filter path).

## Built-in templates (canonical)

| Template | Bench | n | Selection | Backend |
|---|---|---:|---|---|
| `humaneval_full.yaml` | HumanEval | 164 | all | lm-eval |
| `mbpp_full.yaml` | MBPP | 500 | all | lm-eval |
| `humanevalplus_full.yaml` | HumanEvalPlus | 164 | all | lm-eval |
| `gsm8k_100.yaml` | GSM8K | 100 | frozen indices | lm-eval |
| `gpqa_diamond_full.yaml` | GPQA Diamond | 198 | all | lm-eval |
| `mmlu_pro_200.yaml` | MMLU-Pro | 200 | frozen indices | lm-eval |
| `aime_30.yaml` | AIME 2024 | 30 | all | lm-eval |
| `lcb_medium_55.yaml` | LiveCodeBench-medium | 55 | filter (post 2024-10-01) | lcb_custom |
| `lcb_medium_30.yaml` | LiveCodeBench-medium | 30 | first 30 of LCB-55 (deterministic) | lcb_custom |

Smoke / curated subsets remain under `tasks/` for the lm-eval `process_docs`
path (HE-20, MBPP-40, GPQA-10/20). Templates can reference those task names
when needed.

## Loading

```bash
# By bundled name (resolves to /shared/dev/omnimergekit/eval/templates/<name>.yaml)
./eval_suite_vllm.sh --template gsm8k_100 --model <path> --backend vllm

# By absolute / relative path
./eval_suite_vllm.sh --template ./my_custom.yaml --model <path> ...
```
