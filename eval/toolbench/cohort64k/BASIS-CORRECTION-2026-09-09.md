# The recorded basis names a harness this cohort has never run

Found 2026-09-09 while investigating a `Traceback` in the cell logs.

## The discrepancy

`run_cohort.sh` documents the basis as:

    harness  tool-eval-bench HEAD cf54b4b (v2.6.0-45)   <- latest, per user

That is **false**. The `tool-eval-bench` on PATH is a uv tool install:

    /root/.local/bin/tool-eval-bench
      -> /root/.local/share/uv/tools/tool-eval-bench/bin/python3
      -> .../site-packages/tool_eval_bench        version 2.6.0

The git checkouts at /srv/dev-disk-by-label-opt/dev/tool-eval-bench and
/shared/dev/tool-eval-bench are both at cf54b4b, but **nothing imports them**.
The two trees are structurally different, which is how this was caught:

| | layout | TC-62 guard |
|---|---|---|
| installed v2.6.0 (WHAT RUNS)   | `evals/scenarios_planning.py` (monolithic) | **absent** |
| checkout cf54b4b (v2.6.0-45)   | `evals/scenarios/planning/tc62.py`          | present |

Verifying `git rev-parse HEAD` in the checkout does NOT establish which scorer
runs. Check `tool_eval_bench.__file__` from the CLI's own interpreter instead.

## What it costs

The README's own warning applies exactly: the ~45 commits between v2.6.0 and
v2.6.0-45 are largely scorer fixes crediting previously-penalised behaviour, so
the cohort is scored by the OLDER, more punitive scorer.

One concrete instance, and the reason this was found — TC-62 (`_tc62_eval`):

    installed:  email_after_revision = not phase_data_present or (
                    email_calls[-1].user_phase is not None and ...)     # IndexError
    cf54b4b:    email_after_revision = not phase_data_present or bool(
                    email_calls and email_calls[-1].user_phase is not None and ...)

`email_calls` is empty whenever the model sent no *valid* CFO email (empty
subject/body, or a send that did not report success). `phase_data_present` is
then True, `email_calls[-1]` raises `IndexError`, and orchestrator.py:563 catches
it and returns **status=FAIL, points=0** while KEEPING the scenario in the
denominator. That is why every cell still reads /176 and why this never showed up
as the documented `<176` defect.

Affected: **11 of 38 scored cells**, always TC-62, worth 2 pts each.

    a3b-coder     s42 s43 s44 s45 s46   (all five)
    a3b-coderx    s43 s45 s46
    ornith-1.5-35b s42
    qwen3.6-35b-a3b s43
    ornith-27b-coder s42                (new cell, 2026-09-09)

It is NOT neutral. It concentrates on the weakest models — a3b-coder is last at
123.2 and loses TC-62 to a crash at every seed. Under cf54b4b those runs would
reach the `_partial` branches instead of a hard 0, so the fix RAISES scores
unevenly. `omnimerge-v4` and `ornith-1.5-35b` are currently TIED at 146.2 and
only ornith has an affected cell, so the corrected scorer could break that tie.

## Why nothing was changed

Upgrading the harness mid-cohort would split the basis: cells scored before the
upgrade would not be comparable to cells scored after, including the 10 new
Ornith-27B cells. The cohort is currently WRONGLY DOCUMENTED but INTERNALLY
CONSISTENT — all 50 cells on v2.6.0 — and that is the safer of the two states.

`run_cohort.sh` was NOT edited: it is executing, and bash reads a script
incrementally, so an in-place edit can resume mid-statement. Fix the basis
comment after the run drains.

## The decision this needs

1. **Finish on v2.6.0** (default). Cohort stays self-consistent; correct the
   basis comment to say v2.6.0; note TC-62 as a known scorer crash affecting 11
   cells uniformly in kind, unevenly in incidence.
2. **Upgrade to cf54b4b and re-run all 50 cells** (~16 h GPU). Only this yields
   numbers that match the documented basis.

Doing neither, and leaving the comment claiming cf54b4b, is not an option.
