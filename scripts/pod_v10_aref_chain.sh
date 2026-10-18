#!/usr/bin/env bash
# pod_v10_aref_chain.sh — ON THE POD: the missing CONTROL for the diag test (owner 2026-10-18: a_ml3 AND a_core5 both
# unstable, same symptoms as k8/k9/k12). a_ref = map A (24c7b6cb, byte-identical to A.json) through the SAME pipeline as
# a_ml3/a_core5/hub_k*: pod_v10_quick_build.sh bf16 + quant (llama.cpp b11707, quick v5 imatrix 822 chunks, Q4_K_M) ->
# private test tags a_ref-Q4_K_M (+vision-/mtp-/mtp-vision-). The shipped map-A Q4_K_M (1daa6925, b9700 + AC imatrix
# on A) is the owner's stable reference.
#   a_ref unstable -> the v10 pipeline (quantizer / quick imatrix) breaks even A: the maps are not the cause.
#   a_ref stable   -> the swaps break A (even 3 of them).
# The quant waits for >=100G free (bug-796/797); freeing space past the base removal needs the owner's OK.
set -uo pipefail
W=/workspace; S=$W/omnimergekit/scripts/pod_v10_quick_build.sh; PY=$W/miniconda/envs/omnimergekit/bin/python
L=$W/v10/logs; n=a_ref
say(){ echo "[aref $(date -u +%FT%TZ)] $*"; }
free(){ df --output=avail -BG $W | tail -n1 | tr -dc 0-9; }
need(){ local f; f=$(free); [ "$f" -ge "$1" ] || { say "AREF_FAIL disk ${f}G < $1G before $2"; exit 1; }; say "disk ${f}G free (need $1G) before $2"; }
run(){ bash $S "$@" || { say "AREF_FAIL stage $*"; exit 1; }; }
need 56 base; run base
need 45 "bf16 $n"; run bf16 $n
run rmbase
say "waiting for >=100G free before quant $n ($(free)G now)"
for _ in $(seq 1 720); do [ "$(free)" -ge 100 ] && break; sleep 30; done
need 100 "quant $n"; run quant $n
"$PY" $W/omnimergekit/scripts/pod_v10_ollama_test_tags.py \
  --gguf $W/v10/gguf/$n/gemma-4-20B-v9-agentic-$n-Q4_K_M.gguf --prefix $n > $L/push_$n.log 2>&1 \
  || { say "AREF_FAIL push $n (see $L/push_$n.log)"; exit 1; }
grep -q "TEST_TAGS_DONE $n-Q4_K_M" $L/push_$n.log || { say "AREF_FAIL push $n: no TEST_TAGS_DONE"; exit 1; }
say "PUSHED $n: $(grep -c PUSHED $L/push_$n.log) tags verified on the registry"
say "AREF_DONE"
