#!/usr/bin/env bash
# armI / armJ -- OUR ranking with a REAP STABILITY FLOOR. Build -> graft -> imat-Q6_K -> screen.
#
# THE EXPERIMENT
# --------------
# armD (ours) and armE (REAP) disagree on 38.5 experts/layer -- 53% of the drop decision --
# and score within noise of each other on every bench. What separates them is the loop axis:
# LCB-v6-77q loops 12/77 (armD) vs 5/77 (armE), and none of armE's loops pass. So REAP's
# magnitude criterion appears to retain something about generation stability that our
# capability-targeted criterion does not select for.
#
# Hybrid: keep OUR ranking, but forbid dropping the p experts REAP ranks highest among the
# ones we drop; pay for them by evicting the p we rank lowest among the ones we keep.
#
#   p=0  == armD exactly          (measured: LCB 0.6104, loop 15.6%, MPE 0.7300)
#   p=12 == armI  (this chain)
#   p=24 == armJ  (this chain)
#   p~38 ~= armE  (measured: LCB 0.6494, loop  6.5%, MPE 0.7167)
#
# Two interior points on a straight interpolation between two ALREADY-MEASURED endpoints. If
# the middle beats both ends, that is the recipe improvement. If it is monotone between them,
# the criteria are interchangeable and the answer is "pick one" -- also a result, and the
# cheapest way to get it.
#
# WHY p IS THE DOSE AND NOT A THRESHOLD. A saliency threshold would move a different number
# of experts per layer and confound dose with layer identity; a fixed p moves exactly p per
# layer, so the arms differ from armD by a known, uniform amount (gate 3 in the map builder).
#
# Everything else is armD's recipe byte-for-byte -- same base, same competence map, same
# score/agg/cat-weight, same seed, same calibration, merging=none, same MTP graft, same imat
# Q6_K, same screen. The drop map is the ONLY moving part.
#
# GPU. GPU1 only, EXPORTED not polled: an unset CUDA_VISIBLE_DEVICES lets quantize_gguf sum
# every physical GPU and lets llama-imatrix offload across both, which is how an earlier
# quant chain reached GPU0. GPU0 is not ours.
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
HMAPS=$WORK/hybrid_maps
RES=/srv/ml/eval_results/ream_arms
TMPL_FIX=/srv/ml/models/qwen36_chat_template_fixed.jinja
CLI=/opt/llama.cpp/build/bin/llama-cli
LOG=$WORK/chain_hybrid.log
PORT=${PORT:-8097}
THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4

export CUDA_VISIBLE_DEVICES=1

exec >>"$LOG" 2>&1
say() { echo ">>> [$(date -u +%H:%M:%S)] $*"; }
echo "=== chain_hybrid start $(date -u +%F' '%T) ==="

[ -f "$MAP" ]      || { echo "ABORT: competence map missing"; exit 2; }
[ -f "$DROPMAP" ]  || { echo "ABORT: shipped drop map missing"; exit 2; }
[ -s "$TMPL_FIX" ] || { echo "ABORT: fixed chat template missing"; exit 2; }

# ---- 1. wait for the REAP saliency dump ---------------------------------------------------
# Gate on the LIVE PROCESS first, then on the completion sentinel. A sentinel alone is unsafe
# (a stale log from a previous attempt fires it instantly); a process check alone is unsafe
# (a crashed process is also an absent process). Both, in that order.
waited=0
while pgrep -f "dump_reap_saliency.py" >/dev/null 2>&1; do
  sleep 60; waited=$((waited+60))
  [ $((waited % 900)) -eq 0 ] && say "REAP dump still running (${waited}s)"
  if [ $waited -ge 14400 ]; then say "ABORT: REAP dump still running after 4h"; exit 3; fi
done
say "REAP dump process gone after ${waited}s"
grep -aq ">>> OMK_REAP_DUMP_DONE" "$WORK/dump_reap.log" || {
  say "ABORT: dump_reap.log has no OMK_REAP_DUMP_DONE -- the dump died, not finished"
  tail -5 "$WORK/dump_reap.log"; exit 3; }
say "REAP dump complete: $(grep -a -o 'OMK_REAP_DUMP_DONE layers=[0-9]*' "$WORK/dump_reap.log" | tail -1)"

for _ in $(seq 1 30); do
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

# ---- 2. recover keep sets from WEIGHTS (ground truth for the dump gate) --------------------
if [ ! -s "$WORK/keepsets.json" ]; then
  say "recovering armD/armE keep sets from weights (byte-match vs base) ..."
  "$OMKPY" "$WORK/recover_keepsets.py" "$WORK/keepsets.json" \
      armD_ourssal_nomerge "$WORK/armD_ourssal_nomerge" armE "$WORK/armE" || {
    say "ABORT: keep-set recovery failed"; exit 4; }
fi
say "keepsets: $(grep -a -o 'KEEPSETS_OK[^\"]*' "$LOG" | tail -1 || echo present)"

# ---- 3. hybrid drop maps (gates 1-3 live inside the builder) -------------------------------
"$OMKPY" "$WORK/make_hybrid_dropmap.py" --protect 12 --protect 24 --out-dir "$HMAPS" || {
  say "ABORT: hybrid drop-map construction failed a gate"; exit 4; }

# ---- 4. build -- armD's recipe, ONLY --drop-map changed ------------------------------------
build_hybrid() {
  local name=$1 p=$2
  local out=$WORK/arm$name log=$WORK/arm$name.log hm=$HMAPS/drop_map_184e_hybrid_p$p.json
  [ -s "$hm" ] || { say "arm $name: no hybrid map $hm"; return 1; }
  if [ -f "$out/model.safetensors.index.json" ]; then
    say "arm $name already built -- skipping"; return 0
  fi
  say "=== arm $name: ours+REAP-floor p=$p (merging=none) -> $out"
  local t0=$SECONDS
  "$PY" "$RECIPE/ream/omk_ream_merge.py" \
      --model "$BASE" --merge-size 184 \
      --saliency reap --merging none \
      --data-root "$WORK" --tokenizer-name qwen36 \
      --save-path "$out" --seed 42 \
      --saliency-map "$MAP" --drop-map "$hm" \
      --score tc --agg wmax --cat-weight corpus_targeted_lcb=2.0 >"$log" 2>&1
  local rc=$? dt=$(( SECONDS - t0 ))
  if [ $rc -ne 0 ] || ! grep -q '>>> OMK_REAM_DONE' "$log"; then
    say "arm $name BUILD FAILED (rc=$rc, ${dt}s):"; tail -15 "$log"; return 1
  fi
  say "arm $name built in ${dt}s ($((dt/60))m)"
  # Identity IS a hard invariant here: no-merge + injected saliency means the built arm must
  # keep exactly the hybrid map's keep set, verbatim from base. Assert it, never assume it.
  "$PY" "$RECIPE/ream/verify_arm_identity.py" --base "$BASE" --built "$out" \
        --drop-map "$hm" >"$WORK/arm${name}.identity.log" 2>&1
  local irc=$?
  say "arm $name identity rc=$irc: $(tail -2 "$WORK/arm${name}.identity.log" | tr '\n' ' ')"
  [ $irc -eq 0 ] || { say "ABORT arm $name: built keep set != hybrid drop map"; return 1; }
  return 0
}

made=()
build_hybrid I 12 && made+=(armI)
build_hybrid J 24 && made+=(armJ)
[ ${#made[@]} -gt 0 ] || { say "ABORT: no hybrid arm built"; exit 4; }
say "built: ${made[*]}"

# ---- 5. graft the published MTP block (held constant across EVERY arm) ---------------------
"$PY" "$WORK/graft_mtp.py" "${made[@]}"
say "graft rc=$?"

# ---- 6. imat Q6_K -- identical recipe to every other column --------------------------------
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
  # ARTIFACT gates: imatrix must be IN the GGUF KV (--force-imatrix proves intent, not
  # outcome), and blk.40 must survive or the model is not servable.
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

# ---- 7. screen on the R2 basis -------------------------------------------------------------
# Same sampler (template-default greedy), same parallel policy, same results dir as every
# other R2 column: multipl_e_100 @ p4, humaneval_full_think @ p2 (per-slot ctx must stay above
# the 12288 thinking budget -- T172.4 SAT_COLLAPSE).
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

declare -A LABEL=([armI]=hybrid_p12_ourssal_reapfloor [armJ]=hybrid_p24_ourssal_reapfloor)
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
    say ">>>> START $name/$t (parallel=$par)"
    "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$t" --quant q6_k \
        --model "$g" --tokenizer "$WORK/$arm" --served-name "$name" --port "$PORT" \
        --results-dir "$RES" --parallel "$par"
    say "<<<< END $name/$t rc=$?"
    s=$RES/$t/$name/summary.json
    if [ -f "$s" ]; then
      ran=$((ran+1))
      say "SCORE $name/$t = $("$OMKPY" -c "
import json,sys
try: print(json.load(open(sys.argv[1])).get('score'))
except Exception: print('')" "$s")"
    else
      failed=$((failed+1)); say "FAIL $name/$t: no summary.json"
    fi
  done
done

say "=== chain_hybrid done ran=$ran failed=$failed ==="
echo ">>> HYBRID_CHAIN_DONE ran=$ran failed=$failed"
