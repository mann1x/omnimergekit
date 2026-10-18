#!/usr/bin/env bash
until grep -q "T179_SUITE_DONE" /srv/ml/logs/t179/v6coder_smoke.out 2>/dev/null; do sleep 15; done
sleep 5
exec /srv/ml/scripts/t179_v6coder_q4km_suite.sh
