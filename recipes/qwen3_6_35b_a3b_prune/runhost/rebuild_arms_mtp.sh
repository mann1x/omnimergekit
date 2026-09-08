#!/usr/bin/env bash
# Rebuild every REAM arm WITH the MTP block. Same recipes as before, one flag added.
#
# WHY: Qwen3.6-35B-A3B has a next-token-prediction block that is a full 41st MoE layer, and
# our published cut prunes its experts with the same drop map (published: 184 experts in
# mtp.*; base: 256). REAM's save_pretrained drops mtp.* entirely unless --mtp-safetensors is
# supplied, which we never passed. The resulting dirs still convert to GGUF -- config carries
# mtp_num_hidden_layers=1 so the converter writes block_count=41 -- but ship only blk.0..39.
# llama-quantize never builds a graph, so Q6_K "succeeded"; llama-imatrix does, and died on
# 'missing tensor blk.40.attn_norm.weight'. Every pre-MTP arm is therefore unservable and
# unevaluable, not merely imperfect.
#
# Recipes below are copied verbatim from run_ream_arms.sh / build_arm_f.sh. The ONLY change
# is --mtp-safetensors. Nothing else moves, or the arms stop being comparable to each other.
# (Noted, deliberately NOT changed: C/B/E were built with --cat-weight corpus_targeted_lcb=2.0
# while F uses the recovered 1.5/1.5. With keep_offset injection the drop map IS the keep set,
# so this only reorders within the kept/dropped groups; it is a pre-existing asymmetry between
# C and F, and silently "fixing" it mid-rebuild would break comparability with arm D.)
set -uo pipefail

PY=/srv/ml/envs/envs/omnimergekit/bin/python
RECIPE=/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune
WORK=/mnt/sdc/ream-work
BASE=/srv/ml/models/Qwen3.6-35B-A3B
MAP_SHIPPED="$RECIPE/results/competence_qwen35b_coder_lcbmpe.json"
MAP_FULL="$WORK/maps/competence_qwen35b_coder_lcbmpe_FULL.json"
DROPMAP="$RECIPE/results/drop_map_184e_coder_lcbmpe.json"
DROPMAP_RNORM="$RECIPE/results/drop_map_184e_coder_lcbmpe_rnorm.json"
MTP="$BASE/model-00025-of-00026.safetensors,$BASE/model-00026-of-00026.safetensors"
GPU=${GPU:-1}
# Replacing a superseded arm frees the 48 G its rebuild needs. Default OFF: the script
# stops rather than deleting weights on its own.
ALLOW_REPLACE=${ALLOW_REPLACE:-0}
LOG=$WORK/rebuild_arms_mtp.log

exec >>"$LOG" 2>&1
say() { echo ">>> [$(date -u +%H:%M:%S)] $*"; }
say "=== MTP rebuild start (ALLOW_REPLACE=$ALLOW_REPLACE) ==="

for f in "$MAP_SHIPPED" "$MAP_FULL" "$DROPMAP" "$DROPMAP_RNORM" \
         "${MTP%,*}" "${MTP#*,}" "$BASE/config.json"; do
  [ -s "$f" ] || { say "ABORT: missing $f"; exit 2; }
done

# name : saliency : merging : inject : map : dropmap : extra-args
ARMS=(
  "armD_ourssal_nomerge:reap:none:yes:$MAP_SHIPPED:$DROPMAP:--score tc --agg wmax --cat-weight corpus_targeted_lcb=2.0"
  "armC:reap:logits+weights:yes:$MAP_SHIPPED:$DROPMAP:--score tc --agg wmax --cat-weight corpus_targeted_lcb=2.0"
  "armF_rnorm_nomerge:reap:none:yes:$MAP_FULL:$DROPMAP_RNORM:--score rnorm --agg wmax --cat-weight corpus_targeted_lcb=1.5 --cat-weight corpus_targeted_mpe=1.5"
  "armB:reap:logits+weights:no:::"
  "armE:reap:none:no:::"
)

ok=0; bad=0
for spec in "${ARMS[@]}"; do
  IFS=: read -r name sal mrg inj map dmap extra <<<"$spec"
  out=$WORK/$name
  new=$WORK/${name}__mtp
  alog=$WORK/build_${name}_mtp.log

  # Already fixed? The index is the thing that decides, not the directory's existence.
  if [ -s "$out/model.safetensors.index.json" ] && \
     grep -q '"mtp\.' "$out/model.safetensors.index.json" 2>/dev/null; then
    say "SKIP $name -- already carries mtp.* in its index"; continue
  fi

  free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt 60 ]; then
    if [ "$ALLOW_REPLACE" = "1" ] && [ -d "$out" ]; then
      say "reclaiming superseded $out ($(du -sh "$out" | cut -f1)) to make room"
      rm -rf "$out"
      free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
    else
      say "STOP: only ${free_gb}G free and ALLOW_REPLACE=0 -- refusing to delete $out"
      break
    fi
  fi

  gpu_free=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$gpu_free" -ge 60000 ] || { say "STOP: GPU$GPU only ${gpu_free} MiB free (never preempt)"; break; }

  rm -rf "$new"
  args=(--model "$BASE" --merge-size 184 --saliency "$sal" --merging "$mrg"
        --data-root "$WORK" --tokenizer-name qwen36 --save-path "$new" --seed 42
        --mtp-safetensors "$MTP")
  [ "$inj" = "yes" ] && args+=(--saliency-map "$map" --drop-map "$dmap" $extra)

  say "=== $name (saliency=$sal merging=$mrg inject=$inj) -> $new  [${free_gb}G free]"
  t0=$SECONDS
  ( cd "$RECIPE/ream" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" omk_ream_merge.py "${args[@]}" ) >"$alog" 2>&1
  rc=$?; dt=$(( SECONDS - t0 ))

  # Gate on BOTH sentinels: OMK_REAM_DONE alone was true for every broken arm we built.
  if ! grep -q '>>> OMK_REAM_DONE' "$alog" || ! grep -q '>>> OMK_MTP_FOLDED' "$alog"; then
    say "FAIL $name (rc=$rc, ${dt}s) -- missing sentinel; tail:"; tail -12 "$alog"
    bad=$((bad+1)); continue
  fi
  n_mtp=$("$PY" -c "
import json,sys
w=json.load(open(sys.argv[1]+'/model.safetensors.index.json'))['weight_map']
print(sum(1 for k in w if k.startswith('mtp.')))" "$new")
  if [ "${n_mtp:-0}" -ne 19 ]; then
    say "FAIL $name: index carries $n_mtp mtp keys, expected 19"; bad=$((bad+1)); continue
  fi

  if [ -d "$out" ]; then
    if [ "$ALLOW_REPLACE" = "1" ]; then rm -rf "$out"
    else say "NOTE: leaving superseded $out in place; new arm is $new"; ok=$((ok+1)); continue; fi
  fi
  mv "$new" "$out"
  say "OK $name in $((dt/60))m -- 19 mtp tensors folded, at $out"
  ok=$((ok+1))
done

say "=== MTP rebuild done ok=$ok failed=$bad ==="
echo ">>> REBUILD_MTP_DONE ok=$ok failed=$bad"
