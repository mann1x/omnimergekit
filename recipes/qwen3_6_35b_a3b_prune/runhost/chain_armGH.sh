#!/usr/bin/env bash
# armG / armH -- the corrected hybrid. Build -> graft MTP -> imat-Q6_K -> MPE-100 + HE+.
#
# WHY THESE TWO ARMS
# ------------------
# armC (our saliency + REAM merge) collapsed: MPE-100 0.3300 vs armD 0.7300. armB (REAP
# saliency + the SAME merge) did not: 0.7200. So the merge is not broken per se -- the
# COMBINATION is. The layer-0 grouping telemetry says why, and it is not a subtle effect:
#
#   arm   groups  size histogram              centroid weight share   centroid:member ratio
#   armB  184     {1:179, 13:1, 16:4}         0.642                   26.9x
#   armC  184     {1:179, 13:1, 16:4}         0.418                   10.8x
#
# IDENTICAL group structure -- only 5 groups per layer are merged at all, and they absorb all
# 72 dropped experts into the 5 MOST salient survivors. REAM merges a group as a
# saliency-weighted average, w = sal[members]/sum(sal[members]), so the centroid's own share
# decides what the merge does. REAP's saliency is sharply peaked, so the survivor keeps ~64%
# and the fold is gentle. Our injected scores (normalised competence + keep_offset=+2.0) have
# the same ORDERING but a nearly flat dynamic range, so the survivor keeps only ~42% and is
# genuinely averaged into 15 unrelated FFNs -- twice the dilution, landing on the experts that
# matter most. group_size=16 is what forces that concentration; with 184 centroids and only 72
# experts to absorb, the cap is doing no work except making the fold maximally lumpy.
#
#   armG  --group-size 2 : each survivor takes at most ONE neighbour. Dilution per merged
#                          expert ~58% -> ~8%, spread over 72 centroids instead of 5.
#   armH  --group-size 4 : ~25% dilution over 24 centroids. armH exists so the result is a
#                          SLOPE, not a point: if armG only recovers by doing nothing, armH
#                          says so, and if redistribution genuinely helps, armH should show
#                          more of it than armG.
#
# Everything else is armC's recipe byte-for-byte (same map, same drop map, same score/agg/
# cat-weight, same seed, same calibration default) so --group-size is the only moving part.
#
# GPU. GPU1 is the only device this work may touch. Exported, not polled -- an unset
# CUDA_VISIBLE_DEVICES lets quantize_gguf sum every physical GPU and lets llama-imatrix
# offload across both, which is how the earlier quant chain reached GPU0.
set -uo pipefail

WORK=/mnt/sdc/ream-work
GG=$WORK/gguf
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RECIPE=/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune
OMK=/srv/ml/repos/omnimergekit
QG=$OMK/scripts/quantize_gguf.py
BASE=/srv/ml/models/Qwen3.6-35B-A3B
MAP="$RECIPE/results/competence_qwen35b_coder_lcbmpe.json"
DROPMAP="$RECIPE/results/drop_map_184e_coder_lcbmpe.json"
RES=/srv/ml/eval_results/ream_arms
TMPL_FIX=/srv/ml/models/qwen36_chat_template_fixed.jinja
CLI=/opt/llama.cpp/build/bin/llama-cli
LOG=$WORK/chain_armGH.log
PORT=${PORT:-8098}
THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4

export CUDA_VISIBLE_DEVICES=1

exec >>"$LOG" 2>&1
say() { echo ">>> [$(date -u +%H:%M:%S)] $*"; }
echo "=== chain_armGH start $(date -u +%F' '%T) ==="

[ -f "$MAP" ]     || { echo "ABORT: competence map missing"; exit 2; }
[ -f "$DROPMAP" ] || { echo "ABORT: drop map missing"; exit 2; }
[ -s "$TMPL_FIX" ] || { echo "ABORT: fixed chat template missing"; exit 2; }

# ---- 1. wait for the R2 eval to release GPU1 -------------------------------------------
# Gate on the RUNNING PROCESS plus free VRAM, not on a log sentinel: chain_eval2.log already
# carries lines from an aborted earlier run, and that is exactly how the first version of the
# follow-on script fired instantly against half-built artifacts.
waited=0
while pgrep -f "eval_ream_arms.sh" >/dev/null 2>&1; do
  sleep 120; waited=$((waited+120))
  [ $((waited % 1800)) -eq 0 ] && say "R2 eval still running (${waited}s waited)"
  if [ $waited -ge 43200 ]; then say "ABORT: R2 eval still running after 12h"; exit 3; fi
done
say "R2 eval process gone after ${waited}s"
for _ in $(seq 1 60); do
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 60000 ] && break
  say "GPU1 only ${free}MiB free, waiting"; sleep 60
done
free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free" -ge 60000 ] || { say "ABORT: GPU1 never freed (${free}MiB)"; exit 3; }
say "GPU1 ready: ${free}MiB free"

avail=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
say "/mnt/sdc ${avail}G free"
[ "$avail" -ge 250 ] || { say "ABORT: need >=250G for 2 arms + quant transient"; exit 3; }

# ---- 2. build ---------------------------------------------------------------------------
build_arm() {
  local name=$1 gsize=$2
  local out=$WORK/arm$name log=$WORK/arm$name.log
  if [ -f "$out/model.safetensors.index.json" ]; then
    say "arm $name already built -- skipping"; return 0
  fi
  say "=== arm $name: armC recipe + --group-size $gsize -> $out"
  local t0=$SECONDS
  "$PY" "$RECIPE/ream/omk_ream_merge.py" \
      --model "$BASE" --merge-size 184 \
      --saliency reap --merging logits+weights \
      --data-root "$WORK" --tokenizer-name qwen36 \
      --save-path "$out" --seed 42 \
      --saliency-map "$MAP" --drop-map "$DROPMAP" \
      --score tc --agg wmax --cat-weight corpus_targeted_lcb=2.0 \
      --group-size "$gsize" >"$log" 2>&1
  local rc=$? dt=$(( SECONDS - t0 ))
  if [ $rc -ne 0 ] || ! grep -q '>>> OMK_REAM_DONE' "$log"; then
    say "arm $name BUILD FAILED (rc=$rc, ${dt}s):"; tail -15 "$log"; return 1
  fi
  say "arm $name built in ${dt}s ($((dt/60))m)"
  return 0
}

made=()
build_arm G 2 && made+=(armG)
build_arm H 4 && made+=(armH)
[ ${#made[@]} -gt 0 ] || { say "ABORT: no arm built"; exit 4; }
say "built: ${made[*]}"

# Record what the new group caps actually did to the fold -- the mechanism claim above is
# only worth anything if the telemetry confirms the cap changed the weight shares.
[ -f "$WORK/why_armc.py" ] && "$OMKPY" "$WORK/why_armc.py" "${made[@]}" 2>&1 | sed 's/^/    /'

# ---- 3. graft the published MTP block (held constant across every arm) --------------------
"$PY" "$WORK/graft_mtp.py" "${made[@]}"
say "graft rc=$?"

# ---- 4. imat Q6_K, same recipe as every other column -------------------------------------
for arm in "${made[@]}"; do
  out=$GG/${arm}_imat
  mkdir -p "$out"
  q=$(ls "$out"/*-Q6_K.gguf 2>/dev/null | head -1)
  [ -n "$q" ] && [ -s "$q" ] && { say "SKIP quant $arm (Q6_K present)"; continue; }
  avail=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  [ "$avail" -ge 85 ] || { say "ABORT quant $arm: only ${avail}G free, need ~80G transient"; break; }
  say ">>>> START quant $arm (avail ${avail}G)"
  OMK_NO_README=1 nice -n 10 "$PY" "$QG" --model "$WORK/$arm" --only Q6_K \
      --output-dir "$out" --base-precision f16 --no-upload --force-imatrix \
      --base-model-id "ManniX-ITA/Qwen3.6-27B-A3B-Coder" --threads "$THREADS"
  say "<<<< END quant $arm rc=$?"
  q=$(ls "$out"/*-Q6_K.gguf 2>/dev/null | head -1)
  if [ -z "$q" ] || [ "$(head -c4 "$q" 2>/dev/null)" != "GGUF" ]; then
    say "FAIL $arm: no valid Q6_K"; continue
  fi
  # ARTIFACT gates: the imatrix must be IN the GGUF KV (passing --force-imatrix proves intent,
  # not outcome), and blk.40 must survive so the model is servable at all.
  "$PY" - "$q" <<'PYEOF'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
im = {k: f for k, f in r.fields.items() if k.startswith("quantize.imatrix")}
n40 = sum(1 for t in r.tensors if t.name.startswith("blk.40."))
if not im:
    print("     IMATRIX MISSING from GGUF KV"); sys.exit(1)
for k, f in sorted(im.items()):
    try: print(f"     {k} = {f.contents()}")
    except Exception: print(f"     {k} = ?")
print(f"     blk.40 tensors = {n40} (want 20)")
sys.exit(0 if n40 == 20 else 1)
PYEOF
  rc=$?
  [ -s "$out/imatrix.dat" ] && say "imatrix.dat preserved ($(stat -c %s "$out/imatrix.dat" | numfmt --to=iec))" \
                            || say "WARNING: imatrix.dat NOT in $out -- archival rule violated"
  [ $rc -eq 0 ] && say "OK $arm -> $q ($(stat -c %s "$q" | numfmt --to=iec))" \
                || say "FAIL $arm: quant gate failed (imatrix and/or blk.40)"
done

# ---- 5. eval on the R2 basis --------------------------------------------------------------
# Same sampler (template-default greedy), same parallel policy, same ctx/headroom metadata,
# same results dir as every other R2 column. A new column is only comparable if nothing else
# moved: multipl_e_100 @ p4, humaneval_full_think @ p2 (per-slot ctx must stay above the
# 12288 thinking budget -- the T172.4 SAT_COLLAPSE bug).
export HF_HUB_ENABLE_HF_TRANSFER=0
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export LLAMA_EXTRA="--jinja --chat-template-file $TMPL_FIX"

smoke() {
  local g=$1 out uniq
  [ -x "$CLI" ] || return 0
  out=$("$CLI" -m "$g" -ngl 99 -n 48 --temp 0 -no-cnv -p "def fibonacci(n):" 2>/dev/null | tail -c 400)
  uniq=$(echo "$out" | tr -s ' \n' '\n\n' | sort -u | grep -c .)
  say "smoke uniq_tokens=$uniq text=[${out:0:120}]"
  [ "${uniq:-0}" -ge 5 ]
}

declare -A LABEL=([armG]=reamG_ourssal_merge_gs2 [armH]=reamH_ourssal_merge_gs4)
ran=0; failed=0
for arm in "${made[@]}"; do
  g=$(ls "$GG/${arm}_imat"/*-Q6_K.gguf 2>/dev/null | head -1)
  [ -n "$g" ] && [ -s "$g" ] || { say "SKIP eval $arm: no Q6_K"; failed=$((failed+1)); continue; }
  name=${LABEL[$arm]}
  say "==== $name -> $g"
  smoke "$g" || { say "GGUF SMOKE FAILED -- not spending bench time on a broken quant"; failed=$((failed+1)); continue; }
  for t in multipl_e_100 humaneval_full_think; do
    [ -f "$RES/$t/$name/summary.json" ] && { say "SKIP $name/$t (summary exists)"; continue; }
    case "$t" in
      multipl_e_100)        par=4 ;;
      humaneval_full_think) par=2 ;;
    esac
    say ">>>> START $name / $t (parallel=$par, template-default greedy)"
    "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$t" --quant q6_k \
      --model "$g" --tokenizer "$WORK/$arm" --served-name "$name" --port "$PORT" \
      --results-dir "$RES" --parallel "$par" \
      --metadata backend_args.llama_ctx=49152 \
      --metadata backend_args.llama_content_headroom=8192
    say "<<<< END $name / $t rc=$?"
    [ -f "$RES/$t/$name/summary.json" ] && ran=$((ran+1)) || failed=$((failed+1))
  done
done

say "=== chain_armGH done ran=$ran failed=$failed ==="
for arm in "${made[@]}"; do
  name=${LABEL[$arm]}
  for t in multipl_e_100 humaneval_full_think; do
    s=$RES/$t/$name/summary.json
    [ -f "$s" ] && echo "SCORE $name $t = $("$OMKPY" -c "
import json;d=json.load(open('$s'))
print(round(d.get('score') or 0,4), d.get('metric'), (d.get('sampler') or {}).get('name'))")"
  done
done
echo ">>> ARMGH_CHAIN_DONE ran=$ran failed=$failed"
