#!/usr/bin/env bash
# T87 serve-control — complete the 2x2 {model} x {serve-rope} to disambiguate the disc2 VT@64k gap.
# PARALLEL: ARM A on GPU0, ARM B on GPU1 concurrently (each a single-GPU TP=1 F16 serve, ~50GB +
# KV << 97GB), so wall-clock ~= one arm instead of two.
#
# Council csl-2026-06-10-1632-3cfd verdict: the disc2 trained model was served WITH a YaRN 2.0 ramp
# on a natively-262144 GGUF (gguf metadata: context_length=262144, freq_base=1e6, no yarn keys).
# 64k sits far INSIDE the native window, so the --rope-scale 2.0 ramp is gratuitous and is the most
# likely cause of the 1.000 -> 0.760 drop. This re-eval (NO rebuild — reuses the two existing 50GB
# F16 GGUFs) fills the two missing cells of the factorial:
#
#   already on disk:  base + native = 0.948 / 1.000   ;   disc2 + YaRN = 0.904 / 0.760
#   ARM A (this run): disc2 + native  -> @64k ~1.0  => training is FINE, the YaRN serve flag caused it
#   ARM B (this run): base  + YaRN    -> @64k ~0.76 => YaRN@64k is itself the penalty (confound proven)
#
# VT (variable-tracking) n=50, VT@32k positive control + VT@64k question, exactly as disc_stage2.
# Usage:  bash t87_serve_control.sh     (~1-2h; both bs2 Blackwell GPUs)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OMK_PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK_BIN=$(dirname "$OMK_PY")
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
LS=/opt/llama.cpp/build/bin/llama-server
TOK=/srv/ml/google/gemma-4-26B-A4B-it
DISC2=/srv/ml/longctx/t87_llama/disc2-64k-f16.gguf
BASEG=/srv/ml/longctx/t87_llama/base-f16.gguf
RES=/srv/ml/longctx/ruler_llama/serve_control
LOGD=/srv/ml/longctx
SCTX=71680
KV=(--cache-type-k q8_0 --cache-type-v q8_0)
YARN=(--rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale 2.0 --yarn-beta-fast 32 --yarn-beta-slow 1)

mkdir -p "$RES"
say(){ echo "[svc $(date '+%F %T %Z')] $*"; }
fatal(){ echo "FATAL: $*" >&2; exit 1; }

# --- preflight ---
for p in "$OMK_PY" "$OMK" "$LS" "$DISC2" "$BASEG"; do [ -e "$p" ] || fatal "missing tool/file: $p"; done
[ -d "$TOK" ] || fatal "missing tokenizer dir: $TOK"
for g in 0 1; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 9999)
  [ "${u:-9999}" -lt 2000 ] || fatal "GPU $g busy (${u} MiB used) — free it or edit the arm GPU assignment"
done
say "preflight OK — GPU0 + GPU1 both free; running ARM A∥ARM B in parallel"

# arm: $1=label  $2=gguf  $3="native"|"yarn"  $4=gpu  $5=port
run_arm(){
  local label="$1" gguf="$2" mode="$3" gpu="$4" port="$5" rope=()
  [ "$mode" = "yarn" ] && rope=("${YARN[@]}")
  say "ARM $label : serve '$mode' @ctx=$SCTX GPU$gpu:$port  ($(basename "$gguf"))"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "$LS" -m "$gguf" --port "$port" --host 127.0.0.1 \
    -ngl 99 -fa on --parallel 1 -c "$SCTX" "${KV[@]}" "${rope[@]}" --alias "svc_$label" --no-warmup \
    > "$RES/serve_${label}.log" 2>&1 < /dev/null &
  local ok=0; for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 5; done
  [ "$ok" = 1 ] || { tail -25 "$RES/serve_${label}.log"; pkill -f "[l]lama-server.*--port $port"; say "ARM $label serve UNHEALTHY — see serve_${label}.log"; return 1; }
  for C in 32768 65536; do
    say "ARM $label eval VT@$C"
    PATH="$OMK_BIN:$PATH" "$OMK_PY" "$OMK" --backend llama --no-server --port "$port" --parallel 1 \
      --model "$gguf" --served-name "svc_${label}_${C}" --tokenizer "$TOK" \
      --template ruler_native_vt_256k --metadata "ctx_tokens=$C" --results-dir "$RES/${label}_c$C" \
      >> "$LOGD/serve_control_${label}.eval.log" 2>&1 || say "ARM $label VT@$C eval non-zero (continuing)"
  done
  pkill -f "[l]lama-server.*--port $port"; sleep 3
  say "ARM $label DONE"
}

# --- run both arms concurrently, one per GPU ---
run_arm A_disc2_native "$DISC2" native 0 8315 &
PA=$!
sleep 8   # stagger boot so the two llama-server loads don't collide on disk/PCIe
run_arm B_base_yarn    "$BASEG" yarn   1 8316 &
PB=$!
wait "$PA"; rA=$?
wait "$PB"; rB=$?
say "arms finished (rA=$rA rB=$rB)"

say "=== T87 SERVE-CONTROL — 2x2 complete ==="
"$OMK_PY" - <<'PY'
import json, glob
RES = "/srv/ml/longctx/ruler_llama/serve_control"
def sc(label, C):
    g = glob.glob(f"{RES}/{label}_c{C}/ruler_native_vt_256k/*/summary.json")
    return json.load(open(g[0])).get("score") if g else None
def f(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "  n/a"
print(f"{'cell':30s} {'VT@32k':>8s} {'VT@64k':>8s}")
print(f"{'disc2 + YaRN  (recorded)':30s} {0.904:8.3f} {0.760:8.3f}")
print(f"{'base  + native (recorded)':30s} {0.948:8.3f} {1.000:8.3f}")
for name, lab in [("disc2 + native (ARM A)", "A_disc2_native"), ("base  + YaRN   (ARM B)", "B_base_yarn")]:
    print(f"{name:30s} {f(sc(lab,32768)):>8s} {f(sc(lab,65536)):>8s}")
print()
print("READ: ARM A @64k ~1.0  => training FINE; the YaRN serve flag caused the 'wall'.")
print("      ARM B @64k ~0.76 => YaRN@64k is a serve penalty independent of training (confound proven).")
print("      => per council: drop YaRN at <=64k; re-weight toward DCA Stage-1 hd=512 LSE-merge parity gate.")
PY
