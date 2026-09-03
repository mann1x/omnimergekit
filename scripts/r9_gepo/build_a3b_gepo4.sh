#!/usr/bin/env bash
# Build a3b-gepo4 = armJ + the R9 run4 GEPO adapter, quantized to Q6_K **and** Q4_K_M,
# both WITH imatrix.
#
# VERBATIM CLONE of build_a3b_gepo3.sh, which clones build_a3b_gepo1.sh, which clones
# ream-work/quant_arms_imat.sh. Same corpus (calibration_datav5.txt, the bundled
# default), same --force-imatrix, same --base-precision f16, same --base-model-id.
# The ONLY intended difference from gepo1/gepo2/gepo3 is the WEIGHTS. Any recipe
# difference would sit underneath every gepo4-vs-armJ delta, which is exactly the
# comparison this file exists to enable. Q6_K is the EVAL tier -- every banked armJ and
# gepo1/2/3 cell is Q6_K, so a Q4_K_M eval would have no comparison arm. Q4_K_M is the
# deployment tier. --only Q6_K,Q4_K_M runs ONE F16 conversion and ONE compute_imatrix(),
# so both tiers share a single calibration draw; two invocations would not.
#
# TWO DIFFERENCES FROM gepo3, both deliberate:
#
# 1. THE ROUTER GATE RUNS WITH THE OPPOSITE POLARITY. run3 put LoRA on the 40 MoE
#    routers; run4 reverted to run2 scope (ROUTER_LORA=0) because run3 LOST. So the
#    routers MUST NOT have moved, and --expect-router unchanged asserts it. This is not
#    a formality: it proves run4 is genuinely run2-scope rather than having quietly
#    inherited run3's, and nothing else can see the difference -- the router merges into
#    an EXISTING tensor, so tensor count, mtp count and config are byte-identical either
#    way. The POSITIVE control (run2 scope moved) still has to pass, otherwise "routers
#    unchanged" would be satisfied trivially by a merge that did nothing at all.
#
# 2. ADAPTER IS A PARAMETER, and the output names are DERIVED FROM IT. run4 is a 424-step
#    epoch checkpointed every 2 steps, so an intermediate checkpoint is a likely subject
#    -- step 219 is dose parity with run2's entire run (sum of decayed LR: run2 1.63e-4,
#    run4-at-219 1.63e-4, run4 full epoch 2.13e-4). A fixed MERGED/OUT would let a
#    checkpoint build and a final build collide on one filename, and neither would then
#    be identifiable from the artifact.
#    [[feedback_tensor_types_maps_collide_on_a_fixed_filename]]
#
#   ADAPTER=$WORK/run4                  -> a3b-gepo4      / gguf_gepo4
#   ADAPTER=$WORK/run4/checkpoint-218   -> a3b-gepo4ck218 / gguf_gepo4ck218
#
# Adapters and checkpoints are NEVER deleted; this script only reads them.
set -uo pipefail

WORK=/mnt/sdc/ml/brevity/gepo
BASE=/mnt/sdc/ream-work/armJ
ADAPTER=${ADAPTER:-$WORK/run4}
PYM=$WORK/venv/bin/python
PYQ=/srv/ml/envs/envs/omnimergekit/bin/python
QG=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4

# TAG from the adapter path, so the artifact names itself.
case "$(basename "$ADAPTER")" in
  checkpoint-*) TAG="gepo4ck$(basename "$ADAPTER" | sed 's/checkpoint-//')" ;;
  *)            TAG="gepo4" ;;
esac
MERGED=$WORK/a3b-$TAG
OUT=$WORK/gguf_$TAG

# GPU1 by DEFAULT. GPU0 is not mine outside an explicit authorization, and the merge is
# CPU-resident anyway -- only llama-imatrix needs a device.
# BUILD_GPU overrides it, so a second arm can be built on GPU0 alongside a live eval on
# GPU1 WITHOUT contending for it: those eval cells are not resumable under sampling
# (do_sample=true makes --use_cache a no-op), so an OOM there costs a whole bench.
# User authorized GPU0 for this on 2026-08-30. Default stays 1 -- an unset BUILD_GPU must
# never silently reach for a device that is not ours.
export CUDA_VISIBLE_DEVICES=${BUILD_GPU:-1}

ts(){ date -u '+%F %T UTC'; }
say(){ echo "[$(ts)] $*"; }

say "=== a3b-$TAG build start (threads=$THREADS, adapter=$ADAPTER) ==="

# ---- gate 0: preflight -------------------------------------------------------
# REFUSE to run while run4 is still training: the trainer is writing checkpoints, both
# GPUs are committed, and llama-imatrix needs a device. A sleep is not a readiness
# predicate -- this is the observed condition. [[feedback_a_sleep_is_not_a_readiness_predicate]]
if pgrep -f "[g]epo_brevity.py" >/dev/null 2>&1; then
  say "FATAL: gepo_brevity.py is still running -- refusing to merge/quantize under a live trainer"
  exit 8
fi
for p in "$BASE/config.json" "$ADAPTER/adapter_model.safetensors" "$WORK/merge_gepo1.py" \
         "$WORK/gate_router_merged.py" "$QG" "$PYM" "$PYQ"; do
  [ -e "$p" ] || { say "FATAL: missing $p"; exit 2; }
done
AVAIL=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
# 49G merged bf16 + ~52G F16 intermediate + ~21G Q6_K + ~17G Q4_K_M = ~139G peak.
if [ "$AVAIL" -lt 160 ]; then
  say "FATAL: /mnt/sdc has ${AVAIL}G free, need ~160G peak for two tiers"; exit 2
fi
say "preflight OK (/mnt/sdc ${AVAIL}G free, no live trainer)"

# ---- step 1: merge + MTP graft ----------------------------------------------
if [ -f "$MERGED/model.safetensors.index.json" ] && [ -f "$MERGED/mtp.safetensors" ]; then
  say "SKIP merge (already present at $MERGED)"
else
  say "--- merge ---"
  "$PYM" "$WORK/merge_gepo1.py" "$BASE" "$ADAPTER" "$MERGED" || {
    say "FATAL: merge failed"; exit 3; }
fi
"$PYM" - "$MERGED" <<'PY' || exit 3
import json, sys
wm = json.load(open(sys.argv[1] + "/model.safetensors.index.json"))["weight_map"]
n_mtp = sum(1 for k in wm if k.startswith("mtp."))
print(f"[gate] merged index: {len(wm)} tensors, {n_mtp} mtp")
raise SystemExit(0 if (len(wm) == 22712 and n_mtp == 19) else 1)
PY
say "MERGE_OK"

# ---- gate 1b: run2 scope moved AND the routers did NOT -----------------------
"$PYM" "$WORK/gate_router_merged.py" "$BASE" "$MERGED" --expect-router unchanged || {
  say "FATAL: scope gate failed -- either the merge did nothing (POSITIVE control) or the"
  say "       routers MOVED, which would make gepo4 a rerun of run3's refuted arm"
  exit 7; }
say "SCOPE_RUN2_OK (routers unchanged, run2 scope moved)"

# ---- step 2: Q6_K + Q4_K_M with ONE imatrix ---------------------------------
mkdir -p "$OUT"
Q6=$(ls "$OUT"/*-Q6_K.gguf 2>/dev/null | head -1)
Q4=$(ls "$OUT"/*-Q4_K_M.gguf 2>/dev/null | head -1)
if [ -n "$Q6" ] && [ -s "$Q6" ] && [ -n "$Q4" ] && [ -s "$Q4" ]; then
  say "SKIP quantize (both tiers present)"
else
  say "--- quantize Q6_K,Q4_K_M (single imatrix, calibration_datav5) ---"
  OMK_NO_README=1 nice -n 10 "$PYQ" "$QG" --model "$MERGED" --only Q6_K,Q4_K_M \
      --output-dir "$OUT" --base-precision f16 --no-upload --force-imatrix \
      --base-model-id "ManniX-ITA/Qwen3.6-27B-A3B-Coder" --threads "$THREADS"
  say "quantize rc=$?"
fi

Q6=$(ls "$OUT"/*-Q6_K.gguf 2>/dev/null | head -1)
Q4=$(ls "$OUT"/*-Q4_K_M.gguf 2>/dev/null | head -1)
for f in "$Q6" "$Q4"; do
  if [ -z "$f" ] || [ ! -s "$f" ] || [ "$(head -c4 "$f")" != "GGUF" ]; then
    say "FATAL: a requested tier is missing or not a GGUF in $OUT (Q6_K='$Q6' Q4_K_M='$Q4')"
    exit 4
  fi
done

# ---- gate: the imatrix must be RECORDED, not merely requested -- BOTH tiers ---
for f in "$Q6" "$Q4"; do
  "$PYQ" - "$f" <<'PY' || { say "FATAL: $f built WITHOUT an imatrix -- wrong basis"; exit 5; }
import os, sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
tag = os.path.basename(sys.argv[1])
kv = {k: v for k, v in r.fields.items() if k.startswith("quantize.imatrix")}
blk = r.fields.get("qwen3_5_moe_text.block_count") or r.fields.get("block_count")
if blk is not None:
    try: print(f"[gate] {tag}: block_count = {blk.contents()}")
    except Exception: pass
if not kv:
    print(f"[gate] {tag}: IMATRIX MISSING from GGUF KV"); raise SystemExit(1)
for k, v in sorted(kv.items()):
    try: print(f"[gate] {tag}: {k} = {v.contents()}")
    except Exception: print(f"[gate] {tag}: {k} = ?")
PY
done

IM=$OUT/imatrix.dat
[ -s "$IM" ] || { say "FATAL: imatrix.dat absent from $OUT -- archival rule violated"; exit 6; }
say "imatrix.dat preserved ($(stat -c %s "$IM" | numfmt --to=iec))"

# ---- step 3: real load test on BOTH quantized files --------------------------
LS=/opt/llama.cpp/build/bin/llama-server
smoke(){  # smoke <gguf> <tag>
  local f=$1 tag=$2 log="$WORK/${TAG}_smoke_$2.log"
  if [ ! -x "$LS" ]; then
    say "WARNING: $LS not executable -- skipping load smoke for $tag"; return 0
  fi
  say "--- load smoke $tag (llama-server, 20 tok greedy) ---"
  "$LS" -m "$f" --port 8397 -c 4096 -ngl 99 --no-warmup >"$log" 2>&1 &
  local SPID=$! ok=0 i
  for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8397/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 4
  done
  if [ "$ok" = 1 ]; then
    local R
    R=$(curl -sf http://127.0.0.1:8397/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":20,"temperature":0}')
    say "smoke $tag reply: ${R:0:300}"
    say "SMOKE_LOADED_OK_$tag"
  else
    say "WARNING: $tag server never became healthy -- see $log"
    tail -20 "$log"
  fi
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
}
smoke "$Q6" Q6_K
smoke "$Q4" Q4_K_M

sha256sum "$Q6" | tee "$Q6.sha256"
sha256sum "$Q4" | tee "$Q4.sha256"
say "Q6_K:   $Q6 ($(stat -c %s "$Q6" | numfmt --to=iec))"
say "Q4_K_M: $Q4 ($(stat -c %s "$Q4" | numfmt --to=iec))"
say "imatrix: $IM ($(stat -c %s "$IM" | numfmt --to=iec))"
echo "###### A3B_${TAG^^}_BUILD_DONE $(ts) ######"
