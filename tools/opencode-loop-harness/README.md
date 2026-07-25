# opencode agentic-loop capture harness

Drives a real multi-turn **adversarial** opencode session against a model under test,
captures every request/response on the wire, and rolls up a per-session verdict. Built to
answer one question honestly: *does this model run away / loop when a frustrated user keeps
saying "still broken"?*

Ported from the bs2 reference rig (`/mnt/sdc/ml/opencode_capture`) on **2026-07-25**,
carrying the TIMEOUT-verdict fix described below.

## Files

| File | Role |
|---|---|
| `run_session_multiturn.sh` | drives ONE session: init task + escalating follow-ups, one wire proxy for the whole session |
| `wire_proxy.py` | logging reverse proxy — records every request + response to `wirelog/` |
| `compact_session.py` | per-turn classification + session verdict → `meta.json`, `summary.md`, `INDEX.jsonl` |
| `dump_transcript.py` | renders the full conversation to `conversation.md` (called automatically) |
| `rejudge_sessions.py` | re-classify every captured session with the current detector, no model re-run |
| `run_glm52_control.sh` | frontier-model control arm (harness falsification) |

## Usage

```bash
export OPENCODE_CAPTURE_ROOT=/path/to/capture-root   # default: bs2 path
export OMK_PYTHON=/path/to/python                    # default: bs2 omnimergekit env
export OPENCODE_BIN=/path/to/opencode

MODEL_PORT=8101 MODEL_NAME=my-model-q6 MODEL_LABEL=my-arm \
TASK_ID=snake-adversarial PROXY_PORT=8095 \
  bash run_session_multiturn.sh
```

`FOLLOWUPS_FILE` (one adversarial follow-up per line) retargets the same frustration engine
at any task family; combine with `INIT=` for the opening prompt. `N_FOLLOWUPS` caps turns.

Sessions are safe to run **concurrently**: `reap()` is scoped to each session's `--dir` and
the proxy teardown is scoped to its own `PROXY_PORT`. Use a distinct `PROXY_PORT` per
concurrent session.

## Verdicts

Precedence: `DEGENERATE` > `TOOL_LOOP` > `SERVER_DOWN` > `CONTEXT_EXHAUSTED` > `TIMEOUT` >
`SESSION_BUDGET` > `NO_TURNS` > `COMPLETED`.

**Only `DEGENERATE` and `TOOL_LOOP` are loop evidence.** They are derived from response
*content* (a RUNAWAY/THINK_EXPLODE/CORRUPT turn; the same tool call ≥4×), so they do not
depend on any wall budget and **are** comparable across arms that ran different budgets.
`CONTEXT_EXHAUSTED` means the accreted prompt filled the window. `SERVER_DOWN` means the
upstream returned nothing — an invalid run, not a model result.

## budget vs verdict — read before comparing arms

`PER_TURN_TIMEOUT` is a **per-turn** budget. A session is 1 init + N follow-ups, and each
turn is an agentic tool loop of many model calls. Two rules follow:

1. **Never compare a session-total wall against the per-turn budget.** `compact_session.py`
   used to do exactly that:

   ```python
   killed = (args.rc == 137) or (args.timeout and args.wall >= args.timeout)   # WRONG
   ```

   Measured on the bs2 corpus (324 sessions): of 132 `TIMEOUT` verdicts, **19** were real
   per-turn kills and **113** were this artifact — and **113/113** of those had zero
   degenerate turns, 100% `CLEAN` model calls, and had delivered **all 8** user turns
   (`rc=0` means the driver never broke early). They spent p50 **89%** of wall generating
   tokens. It flipped the dense unpruned gemma-31b control from 0% to 60–100%
   not-a-loop across three independent inference backends, which is what exposed it: three
   engines cannot share a model defect and produce an identical 100% verdict.

   Now `killed = (args.rc == 137)` only. `ran_long` records "took longer in total than one
   turn's budget" as **information**, never as a verdict. A genuine whole-session budget is
   opt-in via `--session-timeout` and gets its own `SESSION_BUDGET` verdict.

2. **A wall-kill is not loop evidence.** A large dense model legitimately exceeds a tight
   per-turn budget while doing productive work. Only the content-derived verdicts mean
   looping.

`meta.json` / `INDEX.jsonl` now record `rc`, `timeout_secs`, `session_timeout_secs`,
`ran_long`, `over_session_budget` and `n_user_turns` so comparability can be **checked**
rather than assumed. Their absence is why the bug stayed invisible for weeks.

### Comparability checklist

Before putting two arms in one table, confirm they match on:

- **per-turn budget** — the bs2 corpus mixes 150/300/600/1200s; `std16-field` alone used
  all four.
- **user turns delivered** (`n_user_turns`) — an arm that died at turn 3 saw an easier,
  less-frustrated task. In the bs2 corpus only 11 of 82 `std16-field` sessions ever
  reached turn 8.

If they differ, restrict to `n_user_turns == 8` and compare the content-derived loop rate
only. `TIMEOUT`/`COMPLETED` rates are not comparable across budgets.

The default `PER_TURN_TIMEOUT` is **1800s**, raised from 600s because 600 under-delivered
turns on large dense models — and delivered turns, not the verdict, is the axis that
actually needs to match.

## Every run saves its conversation

`compact_session.py` calls `dump_transcript.py`, so each session gets a readable
`conversation.md` next to the raw `wirelog/`. A verdict is not auditable without the
exchange that produced it. Per session on disk:

```
meta.json  summary.md  conversation.md  opencode.log  proxy.log
server_props.json  root/  wirelog/{session-*.jsonl,raw/req-*.json}
```

The final request in the wire log carries the whole accreted conversation (system + every
user turn + every assistant turn + every tool result), so the transcript is complete.
