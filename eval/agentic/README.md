# agentic — multi-turn agentic eval (BFCL V3 via inspect_ai)

The canonical 9-bench suite is **single-turn**. A defect that only appears
across an in-request agentic loop is invisible to it *by construction*:
prompt growth, reasoning re-injection, tool-call degradation, mid-loop
truncation. This directory is the multi-turn arm of the program.

## What it runs

[`inspect_evals/bfcl`](https://github.com/UKGovernmentBEIS/inspect_evals) —
the Inspect port of the Berkeley Function-Calling Leaderboard.

| generation | category prefix | what it measures | scorer |
|---|---|---|---|
| V1/V2 | `simple_*`, `live_*`, `parallel*` | single-turn function calling | AST match |
| **V3** | **`multi_turn_*`** | **stateful multi-turn tool use** | **final backend state + execution result** |
| V4 | `memory_*`, web search | cross-session memory, multi-hop | text answer |

**`multi_turn_base` (n=200) is the program's agentic cell.** The V3 scorer is
deterministic — it compares the *final state of the backend objects* (file
system, trading bot, ...) against ground truth. There is no LLM judge, so
there is no judge drift between arms, which is what makes it usable for an
A/B where the whole effect size may be a few points.

V4's `memory_*` categories need extra backends (`html2text`,
`memory_api_metaclass`); they are not installed and not needed. Their import
warnings at startup are expected and do not affect `multi_turn_*`.

## Why not SWE-bench

Tried; rejected for this purpose. Docker-per-instance, tens of minutes per
sample, and the score is dominated by scaffold: the *same model* scores
60.47 / 59.20 / 53.73 on OpenHands / OpenCode / Codex — **6.7 pp from the
harness alone**. That variance swamps the effects we are trying to resolve.
BFCL V3 holds the scaffold fixed and runs 200 samples in well under an hour.

## Files

| file | role |
|---|---|
| `setup_agentic_env.sh` | the env, mirroring the `ensure_*_env()` contract (conda where present, venv where not) |
| `run_arm_bfcl.sh` | ONE arm: gates, serves, writes STACK.txt, evals, tears down |
| `write_stack_bfcl.sh` | STACK.txt for one arm (§1.4.5) |
| `summarize_arms.py` | per-arm table + **paired** comparison with the cap/error/basis guards |

## Usage

```bash
bash eval/agentic/setup_agentic_env.sh
export OMK_COHORT="my-cohort-2026-09-12"
export OMK_DECLARED_VARIABLE="chat_template ONLY"

bash eval/agentic/run_arm_bfcl.sh armA "$GGUF" "$TPL_A" "$OUT/armA" 0 8101
bash eval/agentic/run_arm_bfcl.sh armB "$GGUF" "$TPL_B" "$OUT/armB" 1 8102

"$OMK_AGENTIC_ENV/bin/python" eval/agentic/summarize_arms.py "$OUT/armA" "$OUT/armB"
```

Two arms on two GPUs run concurrently — the comparison costs one wall-clock
pass, not two.

## The guards, and why each exists

**PEG-degrade build is REQUIRED, and the runner refuses without it.** An
unpatched llama.cpp *throws* on a full PEG parse failure → HTTP 500 → the
harness records an empty completion → a generation the model produced
correctly is scored as a failure. On a tool-calling bench that is not a rare
event. Probe `libllama-common.so`, never the `llama-server` binary (a ~17 KB
thin wrapper — a string probe on it is a confident false negative). And
*capture* the probe rather than piping into `grep -q`: under `pipefail`,
`grep -q` exits early, SIGPIPEs `strings`, and the pipeline reports failure
on a successful match.

**Cap asymmetry turns this bench into a length meter.** If one arm inflates
generation it hits `max_tokens` more often than its sibling, and the accuracy
delta then partly measures length. `summarize_arms.py` reports cap-hits per
arm and refuses to call a delta clean when they diverge. Counting only the
*final* output misses this — a mid-loop cap is the failure mode, so the
counter walks events.

**The comparison is PAIRED.** Both arms see the same 200 sample ids, so the
test is McNemar on the discordant pairs. Two independent proportions throw the
pairing away and widen the interval for no reason.

**Chat template is part of the measurement.** A tool score grades the model
*and* the template it ships. To isolate a template, hold weights and every
serve flag byte-identical and say so in `OMK_DECLARED_VARIABLE`; the arms'
`STACK.txt` diff is the audit, and it must show nothing else.

**An errored sample is not a failed sample.** Counted separately — a delta
computed across different denominators is not a delta.
