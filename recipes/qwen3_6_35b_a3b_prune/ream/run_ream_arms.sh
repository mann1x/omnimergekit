#!/usr/bin/env bash
# Build the remaining REAM arms of the 2x2 (saliency x merge) matrix, serially, on one GPU.
#
#            merge=none            merge=logits+weights
#   ours     D  (DONE, == published cut)   C  (hybrid -- the headline)
#   theirs   E                             B  (stock REAM)
#
# Arm D is already built and its identity is proven against drop_map_184e_coder_lcbmpe.json
# (verify_arm_identity.py -> ARM_IDENTITY_OK, 0/40 layers mismatched), so it anchors the
# matrix: it IS the model we published, reproduced through REAM's own pipeline.
#
# Order is C -> B -> E, by decreasing decisiveness, so a run that gets cut short still
# answers the question that matters:
#   C vs D  : same keep set, merge vs drop        -> "does REAM's merge beat our drop?"
#   C vs B  : same merge, our saliency vs REAP    -> "does our targeting add anything?"
#   E       : completes the 2x2 (their saliency, no merge)
#
# CALIBRATION. C/B/E all consume calibration (REAP saliency and the merge weights are
# computed from it), so they MUST share one basis -- REAM's default 3072x512. Arm D does not:
# with --merging none and an injected saliency the keep set is fully determined by the drop
# map, which is why D at --calib-limit 64 is still bit-comparable here. Do NOT "save time"
# by shrinking calib on one arm; that silently makes the row incomparable.
set -uo pipefail

PY=/srv/ml/envs/envs/omnimergekit/bin/python
RECIPE=/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune
WORK=/mnt/sdc/ream-work
BASE=/srv/ml/models/Qwen3.6-35B-A3B
MAP="$RECIPE/results/competence_qwen35b_coder_lcbmpe.json"
DROPMAP="$RECIPE/results/drop_map_184e_coder_lcbmpe.json"
MERGE_SIZE=184
CALIB_SFX=qwen36
MIN_FREE_GB=${MIN_FREE_GB:-150}     # each arm writes ~48 G; refuse if we cannot fit them
GPU=${GPU:-1}

say() { echo ">>> [$(date -u +%H:%M:%S)] $*"; }
die() { echo "REFUSING: $*" >&2; exit 2; }

# ---- preflight ---------------------------------------------------------------------------
[ -f "$MAP" ]     || die "competence map missing: $MAP"
[ -f "$DROPMAP" ] || die "drop map missing: $DROPMAP"

free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge "$MIN_FREE_GB" ] || die "$WORK has ${free_gb}G free, need >= ${MIN_FREE_GB}G"
say "disk ok: ${free_gb}G free on $WORK"

# Refuse to stack onto a GPU that is already loaded -- a second 35B profiling pass on top of
# a running job is how we OOM both. Checked ONCE here, and again before each arm.
gpu_free_mb=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
say "GPU$GPU free: ${gpu_free_mb} MiB"
[ "$gpu_free_mb" -ge 60000 ] || die "GPU$GPU has only ${gpu_free_mb} MiB free; wait for the \
current job to finish (this script does not preempt anything)"

run_arm() {
  local name="$1" saliency="$2" merging="$3" inject="$4"
  local out="$WORK/arm${name}"
  local log="$WORK/arm${name}.log"

  if [ -f "$out/model.safetensors.index.json" ]; then
    say "arm $name already built at $out -- skipping"
    return 0
  fi

  local free_now
  free_now=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$free_now" -lt 60000 ]; then
    say "SKIP arm $name: GPU$GPU only ${free_now} MiB free"
    return 1
  fi

  local -a args=(
    --model "$BASE" --merge-size "$MERGE_SIZE"
    --saliency "$saliency" --merging "$merging"
    --data-root "$WORK" --tokenizer-name "$CALIB_SFX"
    --save-path "$out" --seed 42
  )
  # The hybrid/control arms drive REAM's centroid choice from OUR competence map instead of
  # REAP. Injection turns on by supplying --saliency-map WITH --drop-map (the script refuses
  # one without the other), and is derived from the SHIPPED drop map, so the centroids are
  # exactly the experts we published -- that is what makes merge-vs-drop the only moving part.
  # --saliency stays 'reap' even when injecting: it also selects final_reduce in
  # run_all_experts, so flipping it would change the expert outputs the MERGE consumes and
  # quietly move a second variable.
  if [ "$inject" = "yes" ]; then
    args+=(--saliency-map "$MAP" --drop-map "$DROPMAP"
           --score tc --agg wmax --cat-weight corpus_targeted_lcb=2.0)
  fi

  say "=== arm $name: saliency=$saliency merging=$merging inject=$inject -> $out"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$RECIPE/ream/omk_ream_merge.py" "${args[@]}" >"$log" 2>&1
  local rc=$?
  local dt=$(( SECONDS - t0 ))

  if [ $rc -ne 0 ] || ! grep -q '>>> OMK_REAM_DONE' "$log"; then
    say "arm $name FAILED (rc=$rc, ${dt}s) -- last lines:"
    tail -15 "$log"
    return 1
  fi
  say "arm $name OK in ${dt}s ($((dt/60))m)"

  # Identity is only a hard invariant for the injected no-merge control (arm D). For a MERGED
  # arm the router rows are rebuilt, so we record what it kept rather than asserting it.
  "$PY" "$RECIPE/ream/verify_arm_identity.py" --base "$BASE" --built "$out" \
        --drop-map "$DROPMAP" >"$WORK/arm${name}.identity.log" 2>&1
  say "arm $name identity: $(tail -2 "$WORK/arm${name}.identity.log" | tr '\n' ' ')"
  return 0
}

built=(); failed=()
for spec in "C:reap:logits+weights:yes" "B:reap:logits+weights:no" "E:reap:none:no"; do
  IFS=: read -r n s m i <<<"$spec"
  if run_arm "$n" "$s" "$m" "$i"; then built+=("$n"); else failed+=("$n"); fi
done

say "built: ${built[*]:-none}   failed/skipped: ${failed[*]:-none}"
du -sh "$WORK"/arm* 2>/dev/null
echo ">>> REAM_ARMS_DONE built=${#built[@]} failed=${#failed[@]}"
