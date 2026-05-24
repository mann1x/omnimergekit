#!/usr/bin/env python3
"""Prefix every stdin line with a full ISO-8601 timestamp, line-buffered.

Used by eval_suite_llama.sh (and any other driver) to timestamp the raw
omk_eval / lm-eval / llama-server output stream. Chosen over `awk strftime`
because strftime is a gawk extension that older mawk builds lack — piping
through this guarantees identical, correct timestamps on every host.

Protocol (2026-05-24): no eval log line may be untimestamped; the per-bench
logs are the canonical record for recovering per-template wall time.

    ( cmd ) 2>&1 | python3 -u eval/ts_prefix.py >> bench.log
"""
import sys
import time

for line in sys.stdin:
    sys.stdout.write(time.strftime("[%Y-%m-%dT%H:%M:%S%z] ") + line)
    sys.stdout.flush()
