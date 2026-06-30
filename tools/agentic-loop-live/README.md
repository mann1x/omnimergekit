# agentic-loop-live

**Drive a real agentic client through escalating multi-turn adversarial sessions against a
live model server, capture the full wire traffic, and classify degenerate agentic loops.**

This is the *live* companion to [`agentic-loop-harness`](../agentic-loop-harness/) (the
*frozen-replay* tool). Where the replay harness replays a captured tool-call matrix × N
seeds against a server, this harness drives a **real [opencode](https://opencode.ai) agent**
end-to-end — it plans, calls tools, edits files, re-reads them — through a sequence of
escalating "still broken" follow-ups, and records exactly what the agent sent the model and
exactly what the model emitted. When the model falls into an agentic loop (ruminating
forever, repeating the same tool call, or leaking control tokens into content), the loop is
in the captured wire log verbatim, and the classifier rolls it up into a per-session verdict.

It is **host-agnostic and fully parametrized**: the agent binary and the model server are
parameters. The model server can be **ollama**, a **llamafile** binary, **llama.cpp**
`llama-server`, or any **external** OpenAI-compatible endpoint you already have running.

---

## Why agentic loops happen (what this measures)

Some chat templates re-inject every prior assistant *reasoning* turn back into the prompt on
the next step. In a long agentic session (plan → build → "still broken" → fix → …) that feeds
the model its own escalating frustration, and a loop-prone model spirals: it ruminates for
tens of thousands of tokens, or repeats one tool call dozens of times, or starts emitting raw
format/control tokens into its answer. A short single-shot benchmark never surfaces this; you
need a *deep, adversarial, multi-turn* session — which is exactly what this harness
manufactures, deterministically, so you can compare templates, quants, sampler settings, and
serving engines on a single apples-to-apples loop rate.

The bundled `snake-adversarial` fixture is the canonical driver: build a curses Snake game,
then seven escalating, **unsatisfiable** visual complaints ("still not fullscreen", "it
flickers", "colors don't show") about something the model cannot verify without a real
terminal — so it keeps trying across many turns.

---

## Architecture

```
   opencode (the agent)                 wire_proxy (this tool)            model server
   ──────────────────────   POST /v1/   ─────────────────────   POST /v1/  ──────────────
   plans, calls tools,  ───────────────► logs FULL request   ───────────► ollama /
   edits + re-reads files                + FULL response (even             llamafile /
   over many turns        ◄─────────────  streamed SSE) to a   ◄─────────  llama.cpp /
                          stream back     per-session JSONL                external
                          unchanged
                                                │
                                                ▼
                           compact.py  →  per-session meta.json + summary.md
                                          + sessions/INDEX.jsonl  →  loop table
```

The agent sends **no sampler params** (real clients don't), so the **server-side default
sampler is load-bearing** — you set it per backend (CLI flags for llamafile/llama.cpp,
Modelfile `PARAMETER` for ollama). Use the model's *own* recommended sampler; an over-damped
sampler (high `repeat_penalty`, low temperature) artificially suppresses loops and gives a
misleading "0%".

---

## Install

Core (proxy + classifier + orchestrator) is **Python stdlib only** — nothing to pip-install.
You can run straight from a checkout:

```bash
git clone https://github.com/mann1x/omnimergekit
cd omnimergekit/tools/agentic-loop-live
python -m agentic_loop_live --help
```

Optional / external:

```bash
bash install.sh            # verifies python, installs optional PyYAML, checks prerequisites
# prerequisites you install separately:
#   - opencode        https://opencode.ai          (the agentic client under test)
#   - a model server  ollama  |  a llamafile binary  |  llama.cpp llama-server
```

YAML configs need `PyYAML` (`pip install PyYAML`); JSON configs work with no dependencies.

---

## Quickstart

Copy `config.example.yaml`, point it at your agent + server, and run. Or pass everything on
the CLI. Three backend recipes:

**llama.cpp `llama-server`:**
```bash
python -m agentic_loop_live run \
  --backend-kind llamacpp \
  --bin /opt/llama.cpp/build/bin/llama-server \
  --model /models/model-Q6_K.gguf --model-name mut \
  --port 8101 --ctx 32768 --gpu 0 \
  --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 \
  --n 15 --label llamacpp --out ./runs
```

**llamafile binary:**
```bash
python -m agentic_loop_live run \
  --backend-kind llamafile --bin /path/to/model.llamafile \
  --model /models/model-Q6_K.gguf --model-name mut \
  --port 8101 --ctx 131072 --gpu 0 \
  --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 \
  --n 15 --label llamafile
```

**ollama** (sampler + `num_ctx` are pinned via a generated Modelfile):
```bash
python -m agentic_loop_live run \
  --backend-kind ollama --bin ollama \
  --model /models/model-Q6_K.gguf --model-name mut \
  --port 11434 --ctx 32768 \
  --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 \
  --n 15 --label ollama
```

**external** (a server you already started — the harness only probes + records it):
```bash
python -m agentic_loop_live run \
  --backend-kind external --host 127.0.0.1 --port 8101 --model-name mut \
  --n 15 --label external
```

Each `run` boots the backend, drives N sessions, tears the backend down, and prints the loop
table. Run several `run`s with different `--label` (and otherwise identical settings) into the
same `--out`, then `tabulate` to compare engines/quants/samplers side by side.

---

## Output

```
runs/
  backend_<label>.log
  sessions/
    INDEX.jsonl                         # one row per session (the source of the loop table)
    <ts>_<label>_<task>/
      root/                             # the agent's blank working dir (+ files it created)
      wirelog/session-*.jsonl           # FULL captured request/response per turn (ground truth)
      wirelog/raw/req-*.json            # raw request bodies (replayable as fixtures)
      server_props.json                 # the server's sampler + n_ctx at run time (provenance)
      opencode.log, proxy.log
      meta.json                         # classified turns + session verdict
      summary.md                        # human-readable
```

---

## Verdict taxonomy

`compact.py` classifies each model turn, then rolls up a session verdict.

**LOOP set = {`DEGENERATE`, `TOOL_LOOP`} only.**

| verdict | meaning |
|---|---|
| `DEGENERATE` | ≥1 turn is `RUNAWAY` (finish=`length` & `completion_tokens` > 4096), `THINK_EXPLODE` (content\|reasoning > 20000 chars), or `CORRUPT` (control-token leak: `<\|channel`, `<\|"\|>`, `<tool_call\|>`, …) |
| `TOOL_LOOP` | ≥4 consecutive turns issuing the **same `(tool, full-args)` signature** |
| `TIMEOUT` | a turn hit the per-turn wall (slow decode) — **not** a loop |
| `CONTEXT_EXHAUSTED` | accreted multi-turn history filled the window — **not** a loop |
| `SERVER_DOWN` | upstream crashed (requests sent, 0 responses) — invalid, **not** a loop |
| `COMPLETED` / `NO_TURNS` | finished cleanly / agent never reached the model |

Two correctness details baked into the detector:

- **Full-args loop signature.** The tool-loop signature hashes `name + FULL args` (sha1), not a
  truncated prefix — otherwise two distinct full-file `write`s that share a boilerplate prefix
  would collapse into a *fake* repeat-run and over-count loops.
- **Runaway vs context-bound.** A `finish=length` turn is only a `RUNAWAY` if it actually burned
  > 4096 completion tokens; a tiny length-stop means the prompt had already filled the window
  (context-bound), which is not a loop.

---

## Faithfulness audit

Before trusting a loop count, re-derive it from the **untruncated** wire log:

```bash
python -m agentic_loop_live audit --out ./runs                 # all loop verdicts
python -m agentic_loop_live audit --out ./runs --label llamacpp
python -m agentic_loop_live audit --out ./runs --session-id <sid>
```

For each `TOOL_LOOP` it re-runs the run-detector on **full** args (and flags it `SUSPECT` if
the run only existed because of prefix truncation); for each `DEGENERATE` it recomputes
finish-reason, completion tokens, zlib compression ratio and repeated-line fraction, and shows
head+tail of the offending text (flagging a `RUNAWAY` as `SUSPECT` if the text is actually a
coherent, non-repetitive long answer). Clean runs print `ALL … FAITHFUL`.

---

## Commands

| command | purpose |
|---|---|
| `run` | boot backend → N sessions → teardown → loop table |
| `session` | one session (`run` with n=1) |
| `proxy` | run the logging reverse proxy standalone |
| `compact` | (re)classify one session dir |
| `audit` | faithfulness audit of loop verdicts |
| `tabulate` | per-label loop-rate table from `sessions/INDEX.jsonl` |
| `fixtures` | list bundled fixtures |

Every `run`/`session` flag has a `config.example.yaml` equivalent; CLI flags override the file.

---

## Write your own fixture

A fixture is a JSON file `{id, description, init, followups[]}`. Point at it with
`--fixture /path/to/mytask.json` (or drop it in `fixtures/` and use its id). `init` is the
first task; each `followups` entry is an escalating complaint sent as the next turn. Make the
complaints **unverifiable without the real environment** so the agent keeps retrying.

---

## Reproducing a cross-backend study

To answer "is engine X unstable, or is the model just loop-prone?", run the **same model, same
fixture, the model's own official sampler**, varying only `--backend-kind`/`--bin`, into one
`--out`, then `tabulate`. Because the agent sends no sampler, set the server-side sampler to the
model's published values (e.g. for Gemma 4: `--temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0`).
A difference that survives that controlled setup is an engine/parser effect; a difference that
vanishes when you fix the sampler was a sampler artifact.

---

## License

Apache-2.0. Part of [omnimergekit](https://github.com/mann1x/omnimergekit).
