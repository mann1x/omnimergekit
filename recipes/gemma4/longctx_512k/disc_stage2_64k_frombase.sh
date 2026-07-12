#!/usr/bin/env bash
# T87 discriminator v2 (CORRECTED) — the first attempt continued from the already-CE-fit
# merged stage-1 model → near-floor loss → 0.26% weight change → inconclusive (training too
# weak to test anything). This version mirrors stage-1's SUCCESSFUL recipe exactly, changing
# ONLY the pack length: fresh narrow LoRA (r16/α32, q/k/o on the 5 global layers) from the
# UN-TRAINED YaRN base, strong gradients, 64k packs. That is the proper controlled length-wall
# test. Positive control: also eval VT@32k — it must reproduce stage-1's ~0.92 (proving the
# training took); VT@64k is the actual question.
#
#   stage-1 (32k-trn, from yarn base, 250M tok): VT@32k 0.92  (worked)
#   refs:  base VT@32k 0.948 / @64k 1.00 ; stage-1-ext VT@64k 0.74
#   READ:  VT@32k≈0.92 + VT@64k≈base  -> LENGTH-WALL CONFIRMED (narrow adapter suffices, length is the lever)
#          VT@32k≈0.92 + VT@64k≈0.74  -> length-specific: capacity/locus binds at longer range (widen adapter)
#          VT@32k low                 -> training/positive-control failed: not the length question, debug first
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

BASE=/srv/ml/longctx/yarn_cfg_98e                     # UN-TRAINED YaRN base (strong gradients) — the fix
DATA=/srv/ml/longctx/data_98e                         # 500M-token stream, packed to 64k at load
CKPT=/srv/ml/longctx/ckpt_disc2_64k_frombase
MERGED=/srv/ml/longctx/disc2_64k_merged
CDIR=/srv/ml/longctx/disc2_64k_convert
GGUF=/srv/ml/longctx/t87_llama/disc2-64k-f16.gguf
RES=/srv/ml/longctx/ruler_llama/disc2_64k
LOGD=/srv/ml/longctx; mkdir -p "$RES" "$(dirname "$GGUF")"

TOKENS=150000000         # ~572 steps @ tokens/step 262144 ; ~25h @ ~1675 tok/s (strong-gradient regime)
PACK=65536
PORT=8302
SCTX=71680
YARN=(--rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale 2.0 --yarn-beta-fast 32 --yarn-beta-slow 1)
KV=(--cache-type-k q8_0 --cache-type-v q8_0)

say(){ echo "[disc2 $(date '+%F %T %Z')] $*"; }
fatal(){ echo "FATAL: $*" >&2; exit 1; }

say "PHASE 0 preflight"
for p in "$TRAIN_PY" "$OMK_PY" "$TRAINER" "$MERGE" "$CONVERT" "$PATCH" "$LS" "$OMK"; do [ -e "$p" ] || fatal "missing tool: $p"; done
for d in "$BASE" "$DATA" "$TOK"; do [ -d "$d" ] || fatal "missing dir: $d"; done
ls "$DATA"/*.jsonl >/dev/null 2>&1 || fatal "no jsonl shards in $DATA"
"$OMK_PY" -c "import json;s=json.dumps(json.load(open('$BASE/config.json')));assert 'proportional_yarn' in s;print('[preflight] rope OK')" || fatal "base rope wrong"
u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
say "GPU used: 0=${u0} 1=${u1}"; { [ "${u0:-9999}" -lt 2000 ] && [ "${u1:-9999}" -lt 2000 ]; } || fatal "a GPU is busy"

# wandb is MANDATORY on every launch — a silently-disabled run cost us the dashboard
# on the 2026-06-09 disc2 run (flag omitted; trainer defaulted wandb:False). Fail LOUD
# if the key is missing rather than training blind. Key comes from the env (bs2 ~/.bashrc),
# never from a file/literal.
[ -n "${WANDB_API_KEY:-}" ] || fatal "WANDB_API_KEY not in env — refuse to train without wandb (export it on bs2)"
say "PHASE 1 train: 64k packs, ${TOKENS} tok, FRESH from yarn base, MP 0,1, narrow q/k/o r16/a32"
"$TRAIN_PY" "$TRAINER" \
  --yarn-cfg-dir "$BASE" --data-dir "$DATA" --ckpt-dir "$CKPT" \
  --tokens "$TOKENS" --pack-len "$PACK" --lr 1e-4 --rank 16 --alpha 32 \
  --grad-accum 4 --warmup-frac 0.05 --gpus 0,1 --attn memeff \
  --grad-ckpt --max-mem-gib 40 --ckpt-every-steps 20 --resume auto \
  --wandb --wandb-project gemma4-longctx-512k --wandb-run-name disc2_64k_frombase \
  >> "$LOGD/disc2_64k.train.log" 2>&1 || fatal "training non-zero (see disc2_64k.train.log)"
FINAL=$(ls -d "$CKPT"/ckpt-* 2>/dev/null | sort | tail -1)
[ -n "$FINAL" ] && [ -f "$FINAL/adapter_model.safetensors" ] || fatal "no final ckpt"
say "PHASE 1 done: $FINAL"

say "PHASE 2 merge $FINAL onto $BASE"
rm -rf "$MERGED"
"$OMK_PY" "$MERGE" --base "$BASE" --adapter "$FINAL" --out "$MERGED" >> "$LOGD/disc2_64k.merge.log" 2>&1 || fatal "merge failed"
[ -f "$MERGED/model.safetensors.index.json" ] || fatal "merge no index"

say "PHASE 3 convert F16"
rm -rf "$CDIR"; mkdir -p "$CDIR"
for f in "$MERGED"/*; do ln -s "$f" "$CDIR/$(basename "$f")"; done
rm -f "$CDIR/config.json"
"$OMK_PY" "$PATCH" "$MERGED/config.json" "$CDIR/config.json" || fatal "patch failed"
rm -f "$GGUF"
"$OMK_PY" "$CONVERT" "$CDIR" --outfile "$GGUF" --outtype f16 >> "$LOGD/disc2_64k.convert.log" 2>&1 || fatal "convert failed"
[ -s "$GGUF" ] || fatal "F16 empty"
say "PHASE 3 done"

say "PHASE 4 serve+YaRN @ ctx=$SCTX GPU0:$PORT, eval VT@32k (control) + VT@64k (question)"
CUDA_VISIBLE_DEVICES=0 setsid nohup "$LS" -m "$GGUF" --port "$PORT" --host 127.0.0.1 \
  -ngl 99 -fa on --parallel 1 -c "$SCTX" "${KV[@]}" "${YARN[@]}" --alias disc2 --no-warmup \
  > "$RES/serve.log" 2>&1 < /dev/null &
ok=0; for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 5; done
[ "$ok" = 1 ] || { tail -25 "$RES/serve.log"; pkill -f "[l]lama-server.*--port $PORT"; fatal "serve unhealthy"; }
for C in 32768 65536; do
  PATH="$OMK_BIN:$PATH" "$OMK_PY" "$OMK" --backend llama --no-server --port "$PORT" --parallel 1 \
    --model "$GGUF" --served-name "disc2_${C}" --tokenizer "$TOK" \
    --template ruler_native_vt_256k --metadata "ctx_tokens=$C" --results-dir "$RES/c$C" \
    >> "$LOGD/disc2_64k.eval.log" 2>&1
done
pkill -f "[l]lama-server.*--port $PORT"; sleep 2

say "PHASE 5 verdict"
S32=$("$OMK_PY" -c "import json;print(json.load(open('$RES/c32768/ruler_native_vt_256k/disc2_32768/summary.json')).get('score'))" 2>/dev/null)
S64=$("$OMK_PY" -c "import json;print(json.load(open('$RES/c65536/ruler_native_vt_256k/disc2_65536/summary.json')).get('score'))" 2>/dev/null)
echo "=== T87 DISCRIMINATOR v2 — fresh-from-base 64k-trained, VT (llama.cpp F16) ==="
printf "  %-26s 32k=%s   64k=%s\n" "this model (64k-trn):" "${S32:-NA}" "${S64:-NA}"
printf "  %-26s 32k=0.948  64k=1.00\n" "base (no YaRN):"
printf "  %-26s 32k=0.92   64k=0.74\n" "stage-1 (32k-trn) ref:"
"$OMK_PY" - "$S32" "$S64" <<'PY'
import sys
try:
    a=float(sys.argv[1]); b=float(sys.argv[2])
    if a < 0.85:
        print(f"  VERDICT: INVALID — positive control VT@32k={a:.3f} below stage-1's 0.92; training did not take. Debug before concluding.")
    elif b >= 0.90:
        print(f"  VERDICT: LENGTH-WALL CONFIRMED — VT@32k={a:.3f} (control OK) AND VT@64k={b:.3f}≈base. Training at length restores VT; narrow adapter suffices. Build the curriculum.")
    elif b <= 0.80:
        print(f"  VERDICT: CAPACITY/LOCUS — VT@32k={a:.3f} recovered but VT@64k={b:.3f} stuck at baseline. Length-specific failure → widen adapter (MLP, r32) before any 512k spend.")
    else:
        print(f"  VERDICT: PARTIAL — VT@32k={a:.3f}, VT@64k={b:.3f} (above 0.74 baseline, below base). Directional length recovery; extend tokens to settle.")
except Exception as e:
    print(f"  VERDICT: INDETERMINATE ({e}) — inspect disc2_64k.eval.log + serve.log")
PY
: > "$LOGD/disc2_64k.DONE"
say "ALL DONE"
