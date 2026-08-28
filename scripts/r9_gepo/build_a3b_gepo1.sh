#!/usr/bin/env bash
# Build a3b-gepo1 = armJ + the R9 GEPO brevity adapter, quantized to Q6_K WITH imatrix.
#
# THE RECIPE IS COPIED FROM ream-work/quant_arms_imat.sh ON PURPOSE.
# Same corpus (calibration_datav5.txt, the bundled default), same --force-imatrix, same
# --base-precision f16, same tier. The only thing that may differ between this GGUF and
# /mnt/sdc/ream-work/gguf/armJ_imat/armJ-Q6_K.gguf is the WEIGHTS. Putting the GEPO model
# on a different quant recipe than its own base would put a recipe difference underneath
# every gepo-vs-armJ delta, which is precisely the comparison this file exists to enable.
#
# TWO PYTHONS, DELIBERATELY:
#   merge  -> the gepo venv (peft 0.20.0 / transformers 5.9.0) = the env that WROTE the
#             adapter. The omk env has peft 0.18.1 and adapter_config.json carries 0.20.0
#             fields (monteclora_config, use_bdlora, velora_config, ...).
#   quant  -> the omk env, because that is the env the armJ anchor was quantized in.
#
# THE GATE IS THE ARTIFACT: --force-imatrix proves intent, quantize.imatrix.* in the GGUF
# KV proves outcome. llama-imatrix also doubles as the load test -- it builds a real graph,
# so a missing MTP block (block_count 41, 40 shipped) dies here instead of at serve time.
set -uo pipefail

WORK=/mnt/sdc/ml/brevity/gepo
BASE=/mnt/sdc/ream-work/armJ
ADAPTER=$WORK/run1
MERGED=$WORK/a3b-gepo1
OUT=$WORK/gguf_gepo1
PYM=$WORK/venv/bin/python
PYQ=/srv/ml/envs/envs/omnimergekit/bin/python
QG=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4

# GPU1 only. GPU0 is not mine outside the explicit GEPO authorization, and the merge is
# CPU-resident anyway -- only llama-imatrix needs a device.
export CUDA_VISIBLE_DEVICES=1

ts(){ date -u '+%F %T UTC'; }
say(){ echo "[$(ts)] $*"; }

say "=== a3b-gepo1 build start (threads=$THREADS) ==="

# ---- gate 0: preflight -------------------------------------------------------
for p in "$BASE/config.json" "$ADAPTER/adapter_model.safetensors" "$QG" "$PYM" "$PYQ"; do
  [ -e "$p" ] || { say "FATAL: missing $p"; exit 2; }
done
AVAIL=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
# 49G merged bf16 + ~52G F16 intermediate + ~21G Q6_K = ~122G peak.
if [ "$AVAIL" -lt 135 ]; then
  say "FATAL: /mnt/sdc has ${AVAIL}G free, need ~135G peak"; exit 2
fi
say "preflight OK (/mnt/sdc ${AVAIL}G free)"

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

# ---- step 2: Q6_K with imatrix ----------------------------------------------
mkdir -p "$OUT"
Q=$(ls "$OUT"/*-Q6_K.gguf 2>/dev/null | head -1)
if [ -n "$Q" ] && [ -s "$Q" ]; then
  say "SKIP quantize (Q6_K present: $Q)"
else
  say "--- quantize Q6_K (imatrix, calibration_datav5) ---"
  OMK_NO_README=1 nice -n 10 "$PYQ" "$QG" --model "$MERGED" --only Q6_K \
      --output-dir "$OUT" --base-precision f16 --no-upload --force-imatrix \
      --base-model-id "ManniX-ITA/Qwen3.6-27B-A3B-Coder" --threads "$THREADS"
  say "quantize rc=$?"
fi

Q=$(ls "$OUT"/*-Q6_K.gguf 2>/dev/null | head -1)
if [ -z "$Q" ] || [ ! -s "$Q" ] || [ "$(head -c4 "$Q")" != "GGUF" ]; then
  say "FATAL: no valid Q6_K produced in $OUT"; exit 4
fi

# ---- gate: the imatrix must be RECORDED, not merely requested ----------------
"$PYQ" - "$Q" <<'PY' || { say "FATAL: Q6_K built WITHOUT an imatrix -- wrong basis"; exit 5; }
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
kv = {k: f for k, f in r.fields.items() if k.startswith("quantize.imatrix")}
blk = r.fields.get("qwen3_5_moe_text.block_count") or r.fields.get("block_count")
if blk is not None:
    try: print(f"[gate] block_count = {blk.contents()}")
    except Exception: pass
if not kv:
    print("[gate] IMATRIX MISSING from GGUF KV"); raise SystemExit(1)
for k, f in sorted(kv.items()):
    try: print(f"[gate] {k} = {f.contents()}")
    except Exception: print(f"[gate] {k} = ?")
PY

IM=$OUT/imatrix.dat
[ -s "$IM" ] || { say "FATAL: imatrix.dat absent from $OUT -- archival rule violated"; exit 6; }
say "imatrix.dat preserved ($(stat -c %s "$IM" | numfmt --to=iec))"

# ---- step 3: real load test on the quantized file ---------------------------
say "--- load smoke (llama-server, 20 tok greedy) ---"
LS=/opt/llama.cpp/build/bin/llama-server
if [ -x "$LS" ]; then
  "$LS" -m "$Q" --port 8397 -c 4096 -ngl 99 --no-warmup >"$WORK/gepo1_smoke.log" 2>&1 &
  SPID=$!
  ok=0
  for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8397/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 4
  done
  if [ "$ok" = 1 ]; then
    R=$(curl -sf http://127.0.0.1:8397/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":20,"temperature":0}')
    say "smoke reply: ${R:0:300}"
    say "SMOKE_LOADED_OK"
  else
    say "WARNING: server never became healthy -- see $WORK/gepo1_smoke.log"
    tail -20 "$WORK/gepo1_smoke.log"
  fi
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
else
  say "WARNING: $LS not executable -- skipping load smoke (imatrix already loaded the F16)"
fi

sha256sum "$Q" | tee "$Q.sha256"
say "Q6_K: $Q ($(stat -c %s "$Q" | numfmt --to=iec))"
say "imatrix: $IM ($(stat -c %s "$IM" | numfmt --to=iec))"
echo "###### A3B_GEPO1_BUILD_DONE $(ts) ######"
