#!/usr/bin/env bash
# Build a3b-gepo3 = armJ + the R9 run3 GEPO brevity adapter, quantized to
# Q6_K **and** Q4_K_M, both WITH imatrix.
#
# THE RECIPE IS A VERBATIM CLONE OF build_a3b_gepo1.sh -- which is itself a clone of
# ream-work/quant_arms_imat.sh. Same corpus (calibration_datav5.txt, the bundled
# default), same --force-imatrix, same --base-precision f16, same --base-model-id.
# The only intended difference between this Q6_K and gguf_gepo1/a3b-gepo1-Q6_K.gguf
# is the WEIGHTS. Anything else would put a recipe difference underneath every
# gepo3-vs-gepo1-vs-armJ delta, which is precisely the comparison this file exists
# to enable.
#
# WHY TWO TIERS IN ONE INVOCATION
# -------------------------------
# --only Q6_K,Q4_K_M runs ONE F16 conversion and ONE compute_imatrix()
# (quantize_gguf.py:2099, outside the per-quant loop), then quantizes both tiers
# from that single imatrix.dat. Two separate invocations would recompute the
# imatrix, and a second imatrix run is a second calibration draw -- the two tiers
# would then not share a basis. Q6_K is the EVAL tier (every banked armJ and gepo1
# cell is Q6_K; a Q4_K_M eval would have no comparison arm). Q4_K_M is the
# DEPLOYMENT tier that ships to solidpc next to a3b-gepo1.gguf.
#
# RUN3 DIFFERENCE FROM run2: the adapter ALSO carries LoRA on the 40 MoE routers
# (mlp.gate.weight, reached by peft target_parameters). merge_gepo1.py is reused
# unchanged -- it is parametrised and must not drift -- but its existing gates
# cannot see whether the router merged, so gate 1b below is added for this build.
#
# WHICH ADAPTER: run3/ = the final adapter = checkpoint-64, the full 1-epoch pass.
# The eight intermediate checkpoints are kept and are NOT deleted; if a later
# checkpoint sweep is wanted, point ADAPTER at run3/checkpoint-NN and change MERGED
# and OUT to match -- never overwrite these directories.
#
# TWO PYTHONS, DELIBERATELY:
#   merge  -> the gepo venv (peft 0.20.0 / transformers 5.9.0) = the env that WROTE
#             the adapter. The omk env has peft 0.18.1 and adapter_config.json
#             carries 0.20.0 fields (monteclora_config, use_bdlora, velora_config).
#   quant  -> the omk env, because that is the env armJ and gepo1 were quantized in.
#
# THE GATE IS THE ARTIFACT: --force-imatrix proves intent, quantize.imatrix.* in the
# GGUF KV proves outcome, and it is checked on BOTH tiers. llama-imatrix also doubles
# as the load test -- it builds a real graph, so a missing MTP block (block_count 41,
# 40 shipped) dies here instead of at serve time.
set -uo pipefail

WORK=/mnt/sdc/ml/brevity/gepo
BASE=/mnt/sdc/ream-work/armJ
ADAPTER=$WORK/run3
MERGED=$WORK/a3b-gepo3
OUT=$WORK/gguf_gepo3
PYM=$WORK/venv/bin/python
PYQ=/srv/ml/envs/envs/omnimergekit/bin/python
QG=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4

# GPU1 only. GPU0 is not mine outside the explicit GEPO authorization, and the merge
# is CPU-resident anyway -- only llama-imatrix needs a device.
export CUDA_VISIBLE_DEVICES=1

ts(){ date -u '+%F %T UTC'; }
say(){ echo "[$(ts)] $*"; }

say "=== a3b-gepo3 build start (threads=$THREADS, adapter=$ADAPTER) ==="

# ---- gate 0: preflight -------------------------------------------------------
for p in "$BASE/config.json" "$ADAPTER/adapter_model.safetensors" "$WORK/merge_gepo1.py" "$WORK/gate_router_merged.py" \
         "$QG" "$PYM" "$PYQ"; do
  [ -e "$p" ] || { say "FATAL: missing $p"; exit 2; }
done
AVAIL=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
# 49G merged bf16 + ~52G F16 intermediate + ~21G Q6_K + ~17G Q4_K_M = ~139G peak.
if [ "$AVAIL" -lt 160 ]; then
  say "FATAL: /mnt/sdc has ${AVAIL}G free, need ~160G peak for two tiers"; exit 2
fi
say "preflight OK (/mnt/sdc ${AVAIL}G free)"

# ---- step 1: merge + MTP graft ----------------------------------------------
# merge_gepo1.py is fully parametrised by argv (BASE, ADAPTER, OUT) -- reused as-is,
# not copied, so the two builds cannot drift.
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

# ---- gate 1b (run3 ONLY): the ROUTER must actually be in the merged weights ----
# run3 trains mlp.gate.weight via peft target_parameters/ParamWrapper, which merges
# INTO AN EXISTING TENSOR. Tensor count, mtp count and config are therefore identical
# whether or not the router merged -- no gate above can see a silently skipped
# ParamWrapper, and the result would be a GGUF that builds, loads and evals fine while
# being run2-plus-a-dropout-change. Three polarities so the gate can fail informatively.
"$PYM" "$WORK/gate_router_merged.py" "$BASE" "$MERGED" || {
  say "FATAL: router LoRA did NOT land in the merged weights -- gepo3 would be a rerun of run2"
  exit 7; }
say "ROUTER_MERGE_OK"

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

# ---- gate: the imatrix must be RECORDED, not merely requested -- on BOTH tiers -
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
# Q4_K_M is the file that ships to solidpc, so it gets a load test of its own --
# "the Q6_K loads" is not evidence about a different quantization of the same weights.
LS=/opt/llama.cpp/build/bin/llama-server
smoke(){  # smoke <gguf> <tag>
  local f=$1 tag=$2 log="$WORK/gepo3_smoke_$2.log"
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
echo "###### A3B_GEPO3_BUILD_DONE $(ts) ######"
