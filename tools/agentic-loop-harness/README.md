# agentic-loop-harness

A small, self-contained harness that measures **how often a chat-served LLM falls
into a degenerate agentic loop**, and lets you isolate the cause to a single
variable — most usefully the **chat template**, but also the sampler and the
reasoning settings.

It was built to answer a concrete question for the Gemma-4 family: *does a given
chat template cause the model to ruminate in agentic coding sessions, and does a
candidate template fix actually stop it?* But nothing here is specific to one
model — point it at any chat model with an OpenAI-compatible server.

This tool installs and runs **independently** of the rest of omnimergekit: its
own `requirements.txt`, its own `install.sh`, no shared imports.

---

## What it measures

Real agentic clients (OpenCode, Cline, …) drive a model through a multi-turn
tool-calling conversation. A degenerate model fails there in one of two ways, and
the harness detects both per generated turn:

- **Thinking-channel loop** — short 1–3 sentence cycles repeated verbatim in the
  reasoning channel until the budget is exhausted
  (*"Actually, I'll fix the meta tag. Wait, I'll also fix initialScale."* ×N).
- **Answer-channel runaway** — the answer (or a `write_file` tool argument) grows
  into a repeating cycle up to the token cap, or the stream never terminates
  cleanly (`finish_reason` ≠ `stop`/`tool_calls`).

The method: take a **frozen** agentic conversation (a *fixture* — the exact
`messages`/`tools` captured right before a model looped), replay it against a
server across a **sampler matrix × N seeds**, and report a **fail rate** per cell
(`fails = loops + runaways`, out of #seeds). Run it once per chat template and you
get an apples-to-apples table: same conversation, same sampler, same seeds — only
the template differs, so the difference in fail rate **is** the template's effect.

---

## Install (autonomous, three modes)

```bash
cd tools/agentic-loop-harness

# 1) build a pinned CUDA llama-server from source (default)
./install.sh --mode build                 # add --cuda-arch 120 for Blackwell, 86 for 3090, …

# 2) bring your own llama-server binary
./install.sh --mode byo-binary --llama-server-bin /opt/llama.cpp/build/bin/llama-server

# 3) bring your own running OpenAI-compatible endpoint (no binary built)
./install.sh --mode byo-endpoint --endpoint http://127.0.0.1:8000
```

All three modes also create a local `.venv` and `pip install -e .` (the only
third-party dependency is **PyYAML**; the harness core is pure standard library).
Use `--no-venv` to install into the active Python. After install:

```bash
source .env        # exports LLAMA_SERVER_BIN (modes 1/2) or the endpoint (mode 3)
```

### Getting a model (Gemma-4-12B example)

The harness tests a **GGUF**. To build an F16 GGUF from the official HF weights:

```bash
hf download google/gemma-4-12B-it --local-dir gemma-4-12B-it          # gated; needs HF access
python .llama.cpp/convert_hf_to_gguf.py gemma-4-12B-it \
    --outfile gemma-4-12B-it-F16.gguf --outtype f16
```

(Any quant works too — F16 just removes quantization as a confound when studying
the template.) Then set `model.gguf` in your profile to that file.

---

## The run profile — one file, everything under test

Everything you vary lives in a single YAML (or JSON) profile:
[`profiles/gemma4.example.yaml`](profiles/gemma4.example.yaml). The key blocks:

| Block | What it controls |
|---|---|
| `model.gguf` | the GGUF to test |
| `model.chat_template` | the template(s) to compare — `null`/`"embedded"`, one `.jinja`, or a **list** (one cell each) |
| `server.backend` | `llama` (launch+manage llama-server) or `endpoint` (use a running URL) |
| `server.reasoning_format` / `reasoning_budget` | Gemma-4 reasoning flags — **set these to your own settings** |
| `server.*` | gpu, port, ctx, flash-attn, KV cache type, extra llama-server args |
| `sampling.matrix` | a JSON list of named sampler configs (swept per-request) |
| `run.fixtures` | which frozen conversations to replay |
| `run.seeds` | a list, `{start,end}`, or `{count,base}` |

Because `model.chat_template` accepts a **list**, one run produces a per-template
table. The shipped example is the 4-cell comparison this tool was built for:

```yaml
chat_template:
  - { name: embedded,     path: null }                                 # the GGUF's own template
  - { name: reinject-off, path: templates/v7_reinject_disabled.jinja } # reasoning re-injection disabled
  - { name: pr35,         path: templates/google_pr35.jinja }          # Google PR#35, default mode
  - { name: pr35-ptfalse, path: templates/google_pr35_ptfalse.jinja }  # Google PR#35, pass-thinking=false
```

Google: drop in your own 12B GGUF, your own `.jinja` templates, and your own
`sampling` / `server.reasoning_*` — the harness does the rest.

---

## Run

```bash
agentic-loop-harness --profile profiles/gemma4.example.yaml
# or
python -m agentic_loop_harness --profile profiles/gemma4.example.yaml
```

Handy CLI overrides (win over the profile): `--gpu`, `--port`, `--backend`,
`--endpoint`, `--out-dir`, and `--template` (repeatable; use the literal
`embedded` for the GGUF's own template) to override the template list without
editing the profile.

### Output

Per-(template, fixture) result JSON + a `summary.json` land in `run.out_dir`, and
a table prints at the end:

```
==== FAIL-RATE TABLE (fails = loop OR runaway, per #seeds) ====
template/fixture                    minp_t0.9     minp_t0.8
-----------------------------------------------------------------
embedded/solar_build_start              31/48         28/48
reinject-off/solar_build_start           1/48          0/48
pr35/solar_build_start                   5/48          3/48
pr35-ptfalse/solar_build_start           0/48          0/48
(cell = fails/seeds; lower is better; 0/N = no loops or runaways)
```

(Numbers above are illustrative.) Each cell's per-seed detail (which channel
looped, the repeating unit, lengths, `finish_reason`) is in the result JSON.

---

## Backend-agnostic

`replay.py` speaks only OpenAI `/v1/chat/completions` (streaming), so the harness
works against **any** compatible server. `server.py` is the only llama.cpp-aware
module — it launches/tears down a `llama-server`. For vLLM or a remote gateway,
use `backend: endpoint` and point `server.endpoint` at it (template sweeps then
need one endpoint per template, since the template is fixed server-side).

---

## Layout

```
agentic-loop-harness/
├── install.sh                     # 3-mode installer
├── requirements.txt               # PyYAML (core is stdlib)
├── pyproject.toml                 # pip-installable; console_script entry point
├── README.md
├── agentic_loop_harness/
│   ├── detect.py                  # channel-aware loop detector (self-contained)
│   ├── replay.py                  # OpenAI /v1/chat/completions replay driver
│   ├── server.py                  # llama-server lifecycle (the one backend module)
│   └── cli.py                     # profile → server → replay → table orchestrator
├── profiles/gemma4.example.yaml   # the run profile (everything under test)
├── matrices/minp_2temp.json       # sampler matrix (vendor min-p @ t0.9 / t0.8)
├── fixtures/*.json                # frozen agentic conversations to replay
├── templates/*.jinja              # example chat templates to compare
└── scripts/build_llama_cpp.sh     # pinned-build helper (install.sh --mode build)
```

### Fixtures

A fixture is a captured agentic turn:
`{ "name", "messages", "tools", "base_params": {"max_tokens": …} }`. Replaying it
re-asks the model to produce the *next* assistant turn — the one that historically
looped. Bundled fixtures cover several scenarios (a Three.js build, a solar-system
build, a C# port, a deep KV-store task, plan/todo + file-write agentic flows). Add
your own by capturing the `messages`/`tools` of a turn that loops in your client.

### Sampler matrix

A JSON list of `{ "name", "params" }`. `params` is any subset of OpenAI/llama.cpp
sampler fields (`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`,
`dry_*`, …). Every config is swept against the one resident server, so adding
configs is cheap. Use the loop-prone deployment sampler here — that is the regime
where a template's effect on looping is visible.

### Detector tuning

`detect.py` is conservative by design (a hit should be almost certainly
pathological). The thresholds are module constants you can tune:
`LOOP_TAIL_RATIO` / `LOOP_SHINGLE_REPEAT` (answer channel) and
`MAX_UNIT` / `MIN_REPEAT` / `MIN_UNIT_CHARS` (thinking channel). `python -m
agentic_loop_harness.detect` runs a quick self-test.
