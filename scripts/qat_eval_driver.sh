#!/usr/bin/env bash
# qat_eval_driver.sh — HE+ (humanevalplus_full) + MPE-100 (multipl_e_100) on the
# QAT-Q4_0 GGUFs built by build_v7_qat.sh, greedy/canonical, llama backend.
# Resolves the shared-α-at-Q4_0 question and produces the comparison table
# against the plain (non-QAT) Q4_0 baseline from the running per-tier sweep.
#
# Usage:  qat_eval_driver.sh <GPU> <PORT>
#   Pin to a FREE gpu (the sweep owns GPU1; GPU0 frees after F16 suite+LCB-retry).
# Idempotent: a cell with an existing summary.json is skipped.
set -uo pipefail

GPU="${1:?usage: qat_eval_driver.sh <GPU> <PORT>}"
PORT="${2:?usage: qat_eval_driver.sh <GPU> <PORT>}"

BM=/srv/ml
PY="$BM/envs/envs/omnimergekit/bin/python"
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
OMK="$BM/repos/omnimergekit/eval/omk_eval.py"
TPL="$BM/repos/omnimergekit/eval/templates"
OUTGG=/mnt/sdc/ml/eval_gguf/qat
RES=/srv/ml/eval_results_v7_qat
SWEEP=/srv/ml/eval_results_v7_quant_sweep      # plain-Q4_0 baseline lives here
TOK_CODER=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
TOK_CODERX=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it
BENCHES=(humanevalplus_full multipl_e_100)
mkdir -p "$RES"
LOG="$OUTGG/qat_eval_g${GPU}_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[qateval $(date -u +%H:%M:%S)] $*"; }
sc(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['score'])" "$1" 2>/dev/null; }

L "=== qat_eval_driver gpu=$GPU port=$PORT ==="
shopt -s nullglob
GGUFS=("$OUTGG"/*-qat-Q4_0.gguf "$OUTGG"/*-qat-noshared-Q4_0.gguf)
[ "${#GGUFS[@]}" -gt 0 ] || { L "FATAL no QAT GGUFs in $OUTGG yet"; exit 1; }

for gg in "${GGUFS[@]}"; do
  [ -f "$gg" ] || continue
  served=$(basename "$gg" .gguf)                       # e.g. gemma-4-A4B-98e-v7-coder-it-qat-Q4_0
  case "$served" in *coderx*) tok="$TOK_CODERX";; *) tok="$TOK_CODER";; esac
  L ">>> $served  (tok=$(basename "$tok"))"
  for b in "${BENCHES[@]}"; do
    if [ -f "$RES/$b/$served/summary.json" ]; then
      L "  [skip] $b already scored ($(sc "$RES/$b/$served/summary.json"))"; continue
    fi
    L "  [eval] $b"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$OMK" \
        --model "$gg" --tokenizer "$tok" \
        --template "$TPL/$b.yaml" --backend llama \
        --served-name "$served" --results-dir "$RES" --port "$PORT"
    rc=$?
    [ $rc -eq 0 ] && [ -f "$RES/$b/$served/summary.json" ] \
      && L "  [ok] $b $served = $(sc "$RES/$b/$served/summary.json")" \
      || L "  [ERR] $b $served rc=$rc"
  done
done

# ---- comparison table (plain from sweep, qat shared/noshared from here) ----
L "=== QAT-Q4_0 comparison (HE+ / MPE-100) ==="
"$PY" - "$RES" "$SWEEP" <<'PYEOF'
import json, glob, os, sys
RES, SWEEP = sys.argv[1], sys.argv[2]
def sc(p):
    try: return json.load(open(p)).get("score")
    except Exception: return None
def f(x): return "  -  " if x is None else f"{x*100:5.2f}"
def cell(root, served, b): return sc(f"{root}/{b}/{served}/summary.json")
rows = [
  ("v7-coder",  "v7coder-Q4_0",  "gemma-4-A4B-98e-v7-coder-it-qat-Q4_0",  "gemma-4-A4B-98e-v7-coder-it-qat-noshared-Q4_0"),
  ("v7-coderx", "v7coderx-Q4_0", "gemma-4-A4B-98e-v7-coderx-it-qat-Q4_0", "gemma-4-A4B-98e-v7-coderx-it-qat-noshared-Q4_0"),
]
print(f"{'model':10} | {'bench':4} | {'plain Q4_0':>10} | {'qat shared':>10} | {'qat noshared':>12}")
print("-"*62)
for name, plain, qshared, qnoshared in rows:
    for b,blab in (("humanevalplus_full","HE+"),("multipl_e_100","MPE")):
        p = cell(SWEEP, plain, b); s = cell(RES, qshared, b); n = cell(RES, qnoshared, b)
        if p is None and s is None and n is None: continue
        print(f"{name:10} | {blab:4} | {f(p):>10} | {f(s):>10} | {f(n):>12}")
PYEOF
L "###### QAT_EVAL_DONE ######"
