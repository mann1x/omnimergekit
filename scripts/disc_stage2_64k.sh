#!/usr/bin/env bash
# T87 discriminator — Stage-2 @ 64k continued-pretrain, to settle the 32k-wall RCA.
#
# QUESTION: is the extended ckpt's VT cliff a TRAINING-LENGTH wall (fixable by training
# at 64k) or an adapter-capacity limit? Continue the SAME narrow adapter (r16/α32, q/k/o
# on the 5 global-attn layers — the trainer default TARGET_MODULE_SUFFIXES) from the
# MERGED stage-1 model, at 64k packs, ~50M tokens, model-parallel across both Blackwells.
# Then merge → F16 → serve+YaRN → VT@64k.
#   PASS (length-wall):  VT@64k → ~base (≥0.90, Δ≤~0.08). Narrow adapter suffices at trained len.
#   FAIL (capacity):     VT@64k stays ~stage-1 0.74. Adapter locus/rank is the limiter.
#
# Reference points (llama.cpp F16, single-slot, verified from summary.json):
#   base VT@64k = 1.00 ; stage-1 ext VT@64k = 0.74 (Δ0.26) ; stage-1 ext VT@32k = 0.92 (Δ0.03)
#
# Design notes:
#  - --yarn-cfg-dir = the MERGED stage-1 model (weights have 250M-tok/32k baked in; config is
#    the same proportional_yarn rope, hardlinked). New LoRA learns the 64k delta on top.
#  - data_98e is a TOKEN STREAM (token_stream() packs to --pack-len at load), so the existing
#    500M-token shards are reused verbatim at 64k — no re-pack.
#  - --resume auto on a FRESH ckpt-dir: starts fresh; if the chain dies it resumes the SAME
#    stage correctly (same pack-len/tokens → same schedule). This is NOT the cross-stage
#    resume bug (that was re-using stage-1's ckpt-dir with a different pack-len).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # OOM msg's own fragmentation hint
TRAIN_PY=/srv/ml/envs/envs/omk-yarn/bin/python
OMK_PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK_BIN=$(dirname "$OMK_PY")
REPO=/srv/ml/repos/omnimergekit
TRAINER="$REPO/scripts/phase1_train_yarn_lora.py"
MERGE="$REPO/recipes/gemma4/longctx_512k/merge_512k_lora.py"
CONVERT=/srv/ml/tools/llama.cpp/convert_hf_to_gguf.py
PATCH=/srv/ml/scripts/t87_patch_proportional_config.py
LS=/opt/llama.cpp/build/bin/llama-server
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TOK=/srv/ml/google/gemma-4-26B-A4B-it

BASE=/srv/ml/longctx/gemma-4-26B-A4B-it-512k          # merged stage-1 (proportional_yarn cfg)
DATA=/srv/ml/longctx/data_98e                         # 500M-token stream (reused at 64k)
CKPT=/srv/ml/longctx/ckpt_disc_s2_64k                 # FRESH dir
MERGED=/srv/ml/longctx/disc_s2_merged
CDIR=/srv/ml/longctx/disc_s2_convert
GGUF=/srv/ml/longctx/t87_llama/disc-s2-f16.gguf
RES=/srv/ml/longctx/ruler_llama/disc_s2_64k
LOGD=/srv/ml/longctx; mkdir -p "$RES" "$(dirname "$GGUF")"

TOKENS=50000000          # ~191 steps @ tokens/step 262144 ; ~8h @ ~1686 tok/s
PACK=65536
PORT=8301
SCTX=71680               # 65536 prompt + headroom
YARN=(--rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale 2.0 --yarn-beta-fast 32 --yarn-beta-slow 1)
KV=(--cache-type-k q8_0 --cache-type-v q8_0)

say(){ echo "[disc $(date '+%F %T %Z')] $*"; }
fatal(){ echo "FATAL: $*" >&2; exit 1; }

# ---- PHASE 0: preflight (fail loud) -----------------------------------------
say "PHASE 0 preflight"
for p in "$TRAIN_PY" "$OMK_PY" "$TRAINER" "$MERGE" "$CONVERT" "$PATCH" "$LS" "$OMK"; do
  [ -e "$p" ] || fatal "missing tool: $p"; done
for d in "$BASE" "$DATA" "$TOK"; do [ -d "$d" ] || fatal "missing dir: $d"; done
ls "$DATA"/*.jsonl >/dev/null 2>&1 || fatal "no jsonl shards in $DATA"
"$OMK_PY" -c "import json;c=json.load(open('$BASE/config.json'));s=json.dumps(c);assert 'proportional_yarn' in s,'rope not proportional_yarn';print('[preflight] rope OK: proportional_yarn present')" || fatal "merged base rope config wrong"
u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
say "GPU mem used: 0=${u0}MiB 1=${u1}MiB"
{ [ "${u0:-9999}" -lt 2000 ] && [ "${u1:-9999}" -lt 2000 ]; } || fatal "a GPU is busy (need both free for model-parallel 64k)"

# ---- PHASE 1: train (model-parallel, 64k, narrow adapter) --------------------
say "PHASE 1 train: 64k packs, ${TOKENS} tok, MP gpus 0,1, narrow q/k/o r16/a32, resume auto"
"$TRAIN_PY" "$TRAINER" \
  --yarn-cfg-dir "$BASE" --data-dir "$DATA" --ckpt-dir "$CKPT" \
  --tokens "$TOKENS" --pack-len "$PACK" --lr 1e-4 --rank 16 --alpha 32 \
  --grad-accum 4 --warmup-frac 0.05 --gpus 0,1 --attn memeff \
  --grad-ckpt --max-mem-gib 40 \
  --ckpt-every-steps 20 --resume auto \
  >> "$LOGD/disc_s2_64k.train.log" 2>&1 || fatal "training exited non-zero (see disc_s2_64k.train.log)"
FINAL=$(ls -d "$CKPT"/ckpt-* 2>/dev/null | sort | tail -1)
[ -n "$FINAL" ] && [ -f "$FINAL/adapter_model.safetensors" ] || fatal "no final adapter ckpt in $CKPT"
say "PHASE 1 done: $FINAL"

# ---- PHASE 2: surgical merge (stage-2 adapter onto merged stage-1) -----------
say "PHASE 2 merge $FINAL onto $BASE"
rm -rf "$MERGED"
"$OMK_PY" "$MERGE" --base "$BASE" --adapter "$FINAL" --out "$MERGED" \
  >> "$LOGD/disc_s2_64k.merge.log" 2>&1 || fatal "merge failed (see disc_s2_64k.merge.log)"
[ -f "$MERGED/model.safetensors.index.json" ] || fatal "merge produced no index"

# ---- PHASE 3: F16 GGUF (proportional base; YaRN applied at serve runtime) -----
say "PHASE 3 convert F16 (proportional config patch)"
rm -rf "$CDIR"; mkdir -p "$CDIR"
for f in "$MERGED"/*; do ln -s "$f" "$CDIR/$(basename "$f")"; done
rm -f "$CDIR/config.json"
"$OMK_PY" "$PATCH" "$MERGED/config.json" "$CDIR/config.json" || fatal "config patch failed"
rm -f "$GGUF"
"$OMK_PY" "$CONVERT" "$CDIR" --outfile "$GGUF" --outtype f16 \
  >> "$LOGD/disc_s2_64k.convert.log" 2>&1 || fatal "convert failed (see disc_s2_64k.convert.log)"
[ -s "$GGUF" ] || fatal "F16 GGUF empty"
say "PHASE 3 done: $(ls -la "$GGUF" | awk '{print $5}') bytes"

# ---- PHASE 4: serve + YaRN, eval VT@64k -------------------------------------
say "PHASE 4 serve+YaRN @ ctx=$SCTX GPU0:$PORT, eval VT@65536"
CUDA_VISIBLE_DEVICES=0 setsid nohup "$LS" -m "$GGUF" --port "$PORT" --host 127.0.0.1 \
  -ngl 99 -fa on --parallel 1 -c "$SCTX" "${KV[@]}" "${YARN[@]}" --alias disc_s2 --no-warmup \
  > "$RES/serve.log" 2>&1 < /dev/null &
ok=0; for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 5; done
[ "$ok" = 1 ] || { tail -25 "$RES/serve.log"; pkill -f "[l]lama-server.*--port $PORT"; fatal "serve unhealthy"; }
PATH="$OMK_BIN:$PATH" "$OMK_PY" "$OMK" --backend llama --no-server --port "$PORT" --parallel 1 \
  --model "$GGUF" --served-name disc_s2 --tokenizer "$TOK" \
  --template ruler_native_vt_256k --metadata "ctx_tokens=65536" --results-dir "$RES" \
  >> "$LOGD/disc_s2_64k.eval.log" 2>&1
pkill -f "[l]lama-server.*--port $PORT"; sleep 2

# ---- PHASE 5: verdict -------------------------------------------------------
S=$("$OMK_PY" -c "import json;print(json.load(open('$RES/ruler_native_vt_256k/disc_s2/summary.json')).get('score'))" 2>/dev/null)
say "PHASE 5 verdict"
echo "=== T87 DISCRIMINATOR — VT@64k (llama.cpp F16, single-slot) ==="
printf "  %-22s %s\n" "base (no YaRN):" "1.00"
printf "  %-22s %s\n" "stage-1 ext (32k-trn):" "0.74"
printf "  %-22s %s\n" "stage-2 ext (64k-trn):" "${S:-NA}"
"$OMK_PY" - "$S" <<'PY'
import sys
s=sys.argv[1]
try:
    s=float(s); d=abs(1.0-s)
    if s>=0.90 and d<=0.08:
        print(f"  VERDICT: PASS — 32k-WALL CONFIRMED (Δvs base={d:.3f}). Narrow adapter suffices at trained length; curriculum length is the lever.")
    elif s>0.80:
        print(f"  VERDICT: PARTIAL — recovered but short of base (Δ={d:.3f}). Length helps; adapter/tokens may need more.")
    else:
        print(f"  VERDICT: FAIL — VT@64k still low ({s:.3f}, Δ={d:.3f}). Length training did NOT fix it → adapter capacity/locus implicated.")
except Exception as e:
    print(f"  VERDICT: INDETERMINATE — no score ({e}). Inspect disc_s2_64k.eval.log + serve.log.")
PY
: > "$LOGD/disc_s2_64k.DONE"
say "ALL DONE"
