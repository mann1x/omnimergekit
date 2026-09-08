#!/usr/bin/env bash
set -uo pipefail
cd /srv/ml/repos/omnimergekit/tools/agentic-loop-harness
echo "=== waiting for v9 reject-gen to free GPU1 $(date +%T) ==="
until grep -q "ORPO_REJECT_v9pool_DONE" /srv/ml/an-finetune/simpo/out/run_v9pool_hard.log 2>/dev/null; do sleep 60; done
echo "=== reject-gen done; launching g0709 template re-test on GPU1 $(date +%T) ==="
.venv/bin/agentic-loop-harness --profile profiles/run_128e_g0709_retest.yaml
echo ">>> GOOGLE_G0709_RETEST_DONE $(date -u +%FT%TZ)"
