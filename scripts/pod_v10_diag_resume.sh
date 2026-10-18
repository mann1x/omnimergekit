#!/usr/bin/env bash
# pod_v10_diag_resume.sh — ON THE POD: resume pod_v10_diag_chain.sh after its a_core5 disk gate (97G < 100G, because
# a_ml3's 13 GB Q4_K_M was still on disk — bug-797). solidpc copies + sha-verifies a_ml3's Q4_K_M + imatrix, removes the
# Q4_K_M here, then this runs the remaining steps unchanged: quant a_core5 (same 100G floor) -> private test tags.
# Logs to diag_resume.log (diag.log keeps the earlier DIAG_FAIL line).
set -uo pipefail
W=/workspace; S=$W/omnimergekit/scripts/pod_v10_quick_build.sh; PY=$W/miniconda/envs/omnimergekit/bin/python
L=$W/v10/logs; n=a_core5
say(){ echo "[resume $(date -u +%FT%TZ)] $*"; }
f=$(df --output=avail -BG $W | tail -n1 | tr -dc 0-9)
[ "$f" -ge 100 ] || { say "RESUME_FAIL disk ${f}G < 100G before quant $n"; exit 1; }
say "disk ${f}G free (need 100G) before quant $n"
bash $S quant $n || { say "RESUME_FAIL stage quant $n"; exit 1; }
"$PY" $W/omnimergekit/scripts/pod_v10_ollama_test_tags.py \
  --gguf $W/v10/gguf/$n/gemma-4-20B-v9-agentic-$n-Q4_K_M.gguf --prefix $n > $L/push_$n.log 2>&1 \
  || { say "RESUME_FAIL push $n (see $L/push_$n.log)"; exit 1; }
grep -q "TEST_TAGS_DONE $n-Q4_K_M" $L/push_$n.log || { say "RESUME_FAIL push $n: no TEST_TAGS_DONE"; exit 1; }
say "PUSHED $n: $(grep -c PUSHED $L/push_$n.log) tags verified on the registry"
say "RESUME_DONE"
