#!/usr/bin/env bash
# pod_v10_diag_chain.sh — ON THE POD: the two diagnostic builds between map A and ml_k8 (owner 2026-10-17 21:27:
# "stop the evals ... build"): a_core5 (A + the 5 swaps shared by k8/k9/k12) and a_ml3 (A + k8's 3 multilingual-only
# swaps; byte-identical to the 10-15 ml_k3 map). Same recipe as hub_k12/hub_k9: pod_v10_quick_build.sh bf16 + quant
# (llama.cpp b11707, v5 imatrix all chunks, Q4_K_M), then private test tags <map>-Q4_K_M (+vision-/mtp-/mtp-vision-).
# Disk floors are sized from the pipeline (bug-796): quantize = F16 40 + EOG-normalised copy 40 + Q4_K_M 13 GB.
# FREE_OK is written by the solidpc side after it has copied + sha-verified the hub_k9 bf16 and the Q6_K/Q4_K_M files
# and removed them here.
set -uo pipefail
W=/workspace; S=$W/omnimergekit/scripts/pod_v10_quick_build.sh; PY=$W/miniconda/envs/omnimergekit/bin/python
L=$W/v10/logs; D=$W/v10/diag
say(){ echo "[diag $(date -u +%FT%TZ)] $*"; }
need(){ local f; f=$(df --output=avail -BG $W | tail -n1 | tr -dc 0-9); [ "$f" -ge "$1" ] || { say "DIAG_FAIL disk ${f}G < $1G before $2"; exit 1; }; say "disk ${f}G free (need $1G) before $2"; }
run(){ bash $S "$@" || { say "DIAG_FAIL stage $*"; exit 1; }; }
mkdir -p $L $D
need 56 base; run base
say "waiting for FREE_OK from solidpc (verified copies of hub_k9 bf16 + Q6_K/Q4_K_M, then removed here)"
for _ in $(seq 1 240); do [ -f $D/FREE_OK ] && break; [ -f $D/FREE_FAIL ] && { say "DIAG_FAIL solidpc copy failed: $(cat $D/FREE_FAIL)"; exit 1; }; sleep 30; done
[ -f $D/FREE_OK ] || { say "DIAG_FAIL no FREE_OK after 2 h"; exit 1; }
need 45 "bf16 a_ml3";   run bf16 a_ml3
need 45 "bf16 a_core5"; run bf16 a_core5
run rmbase
for n in a_ml3 a_core5; do
  need 100 "quant $n"; run quant $n
  "$PY" $W/omnimergekit/scripts/pod_v10_ollama_test_tags.py \
    --gguf $W/v10/gguf/$n/gemma-4-20B-v9-agentic-$n-Q4_K_M.gguf --prefix $n > $L/push_$n.log 2>&1 \
    || { say "DIAG_FAIL push $n (see $L/push_$n.log)"; exit 1; }
  grep -q "TEST_TAGS_DONE $n-Q4_K_M" $L/push_$n.log || { say "DIAG_FAIL push $n: no TEST_TAGS_DONE"; exit 1; }
  say "PUSHED $n: $(grep -c PUSHED $L/push_$n.log) tags verified on the registry"
done
say "DIAG_DONE"
