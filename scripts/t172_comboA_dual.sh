#!/bin/bash
# Option A shared×PES grid — DUAL-SERVER: one model per GPU (CUDA_VISIBLE_DEVICES
# pin, NO layer-split), two pairs concurrent. bs2 omk_eval lacks gpu_planner.
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
AUDIT=$BM/scripts/audit_full_bench.py
BUILDER=$BM/scripts/build_alpha_variant.sh
PRISTINE=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pristine-it
WORK=/mnt/sdc/ml/google
RES=$BM/eval_results_tracks_2_3
BASELINE=a2-62e-fc15_25-p8-s1_0p1_20
export PATH=$BM/envs/envs/omnimergekit/bin:${PATH:-/usr/bin:/bin}
export LD_LIBRARY_PATH=$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}
exec > >(tee /srv/ml/logs/t172/comboA_dual.log) 2>&1

vname(){ local s p o; s=$(awk -v a=$1 "BEGIN{printf(\"%d\",a*100+0.5)}"); p=$(awk -v a=$2 "BEGIN{printf(\"%d\",a*100+0.5)}"); o=sf; [ "$3" = pes-first ] && o=pf; echo "gemma-4-A4B-62e-fc15_25-p8-shared${s}-pes${p}-${o}-it"; }
build_one(){ local n out q6; n=$(vname $1 $2 $3); out=$WORK/$n; q6=$out-GGUF/$n-Q6_K.gguf; [ -f "$q6" ] && { echo "[build] SKIP $n"; return 0; }; echo "[build $(date +%H:%M:%S)] $n"; bash $BUILDER --src $PRISTINE --shared-alpha $1 --pes-alpha $2 --order $3 --out $out 2>&1 | tail -6; [ -f "$q6" ] || { echo "FATAL build $n"; return 1; }; }
eval_stream(){ local gpu=$1 port=$2; shift 2; local n gdir q6 tpl sd tlim; for n in "$@"; do gdir=$WORK/$n-GGUF; q6=$WORK/$n-GGUF/$n-Q6_K.gguf; for tpl in humanevalplus_full ifeval_100 multipl_e_100; do sd=$RES/$tpl/$n; [ -f "$sd/summary.json" ] && { echo "[g$gpu] SKIP $tpl/$n"; continue; }; tlim=2400; [ "$tpl" = multipl_e_100 ] && tlim=3600; pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2; echo "[g$gpu $(date +%H:%M:%S)] eval $tpl $n"; CUDA_VISIBLE_DEVICES=$gpu timeout --kill-after=10 --signal=KILL $tlim $PY $OMK --model $q6 --tokenizer $gdir --template $tpl --backend llama --port $port --served-name $n --results-dir $RES 2>&1 | sed "s/^/[g$gpu] /" | tail -5; pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2; done; done; echo "[g$gpu] STREAM DONE"; }

N1A=$(vname 1.30 1.10 shared-first); N1B=$(vname 1.30 1.10 pes-first)
N2A=$(vname 1.20 1.20 shared-first); N2B=$(vname 1.20 1.20 pes-first)
echo "==================== comboA_dual BUILD (sequential) $(date -Iseconds) ===================="
build_one 1.30 1.10 shared-first || exit 1
build_one 1.30 1.10 pes-first    || exit 1
build_one 1.20 1.20 shared-first || exit 1
build_one 1.20 1.20 pes-first    || exit 1
echo "==================== comboA_dual EVAL  GPU0=pair(1.3,1.1)  GPU1=pair(1.2,1.2) ===================="
eval_stream 0 8195 "$N1A" "$N1B" & p0=$!
eval_stream 1 8295 "$N2A" "$N2B" & p1=$!
wait "$p0" "$p1"
echo "==================== comboA_dual AUDIT vs $BASELINE ===================="
for n in "$N1A" "$N1B" "$N2A" "$N2B"; do for tpl in humanevalplus_full ifeval_100 multipl_e_100; do $PY $AUDIT $tpl "$n" "$BASELINE" 2>/dev/null | grep "^AUDIT" || echo "AUDIT_FAIL $tpl $n"; done; done
echo "==================== comboA_dual DONE $(date +%H:%M:%S) ===================="
