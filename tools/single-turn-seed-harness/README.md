# single-turn-seed-harness

Reproduce the **June single-turn HumanEval+ / MultiPL-E seed sweep** that
isolated the Gemma-4 chat-template *reasoning re-injection* failure, and use it
to compare an **OLD vs NEW** chat template on the same model.

Where the sibling [`agentic-loop-harness`](../agentic-loop-harness/) replays a
frozen **multi-turn** agentic conversation, this harness fires **single-turn**
code-completion requests — one `user` message per problem, thinking enabled — and
measures how often the model degenerates (runaway / think-explosion / corruption)
across a seed sweep. Single-turn is the mode the June re-injection fix addressed
(stock ~33–37.5% verbatim-repetition failures → 0% with re-injection disabled).

## What it does

For every `(problem × seed)` it POSTs one non-streaming request to a running
`/v1/chat/completions` endpoint and grades the raw response with the **hammer
classifier** (`classify.py`, copied **verbatim** from
`opencode_capture/hammer_raw.py` so the numbers line up with the multi-turn wire
hammer):

| verdict         | meaning                                                            |
|-----------------|-------------------------------------------------------------------|
| `CLEAN`         | finished normally, no special-token leak, not oversized           |
| `RUNAWAY`       | `finish_reason == "length"` — hit the token cap, never terminated |
| `THINK_EXPLODE` | finished, but reasoning **or** content > 20,000 chars (ruminating) |
| `CORRUPT`       | HTTP error, PEG/parse failure, or a special-token leak            |
| `ABORT`         | `finish_reason` is `None` (server aborted / empty)                |

`fail-rate = (RUNAWAY + THINK_EXPLODE + CORRUPT) / total`.

The **template under test is chosen by which server you point at** — the harness
is template-agnostic. An OLD-vs-NEW comparison is two runs against two servers
(same GGUF, different `--chat-template-file`). `run_seed_sweep.sh` does exactly
that.

## Sampler = field default

To match the June `--field` condition, **no sampler is sent** (`temperature`,
`top_p`, `top_k`, `min_p`, `repeat_penalty` are all omitted) so llama-server uses
its own defaults. `cache_prompt` is sent `false` to force a fresh prefill on every
request. Requests are non-streaming so the classifier reads
`choices[].message.{content,reasoning_content}` — byte-identical to what
`hammer_raw` grades.

## Prompt set

| group | source                                   | default N        |
|-------|------------------------------------------|------------------|
| `he`  | `evalplus/humanevalplus` (→ `openai_humaneval` fallback) | all 164 |
| `cpp` | `nuprl/MultiPL-E`, config `humaneval-cpp`               | 50      |
| `js`  | `nuprl/MultiPL-E`, config `humaneval-js`                | 50      |

`--seeds 12` per problem (the June sweep count), `--seed0 2000` (matches
`hammer_raw`), `--max-tokens 16384`.

## Layout

```
single-turn-seed-harness/
├── single_turn_seed_harness/
│   ├── classify.py    # hammer verdict logic, copied verbatim
│   ├── tasks.py       # HumanEval+ / MultiPL-E prompt-set loader
│   ├── sweep.py       # (problem × seed) driver → JSONL + summary.json
│   ├── tabulate.py    # OLD-vs-NEW comparison table from two summaries
│   └── cli.py         # --server --name --tasks --seeds --out
├── run_seed_sweep.sh  # serve OLD → sweep → kill → serve NEW → sweep → table
├── results/           # JSONL + summary.json per template (gitignored)
├── pyproject.toml · requirements.txt · README.md
```

## Run

Everything on **GPU0 only** (GPU1 may hold a training job). One server per
template, killed by explicit PID between runs.

```bash
# health smoke first (2 problems, 1 seed) — guards against a bad template/serve
SMOKE=1 bash run_seed_sweep.sh

# full OLD-vs-NEW sweep
bash run_seed_sweep.sh

# custom scale / port
TASKS='he:40,cpp:20,js:20' SEEDS=12 PORT=8097 bash run_seed_sweep.sh
```

Or drive one server yourself and point the CLI at it:

```bash
python -m single_turn_seed_harness.cli \
    --server http://127.0.0.1:8097 --name old-gmain \
    --tasks all --seeds 12 --out results
```

Then tabulate:

```bash
python -m single_turn_seed_harness.tabulate \
    results/summary_old-gmain.json results/summary_new-g0709.json
```

## Outputs (per cohort `<name>`)

- `results/responses_<name>.jsonl` — one line per `(problem, seed)`: verdict,
  finish reason, content/reasoning char counts, completion tokens, latency, flags.
- `results/summary_<name>.json` — per-verdict tally, `fail_rate`, and
  completion-length p50/p90 (chars and tokens).

## Discipline (bs2)

- **GPU0 only** — `run_seed_sweep.sh` pins `CUDA_VISIBLE_DEVICES=0`.
- Uses `PORT` (default **8097**); refuses to start if the port is taken. Never
  touches the opencoti supervisor on `:8240` or `sshd`.
- Kills the server by its captured `$!` PID only — never `pkill -f`.
- Logs/results on persistent disk under `results/`, never `/tmp`.
