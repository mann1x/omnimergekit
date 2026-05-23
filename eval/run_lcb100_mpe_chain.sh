#!/usr/bin/env bash
# run_lcb100_mpe_chain.sh — LCB-100 + MultiPL-E-100 on 128e + v5-coder.
#
# Designed to run ON the eval host (pod or solidpc). Drives omk_eval for the new
# comparison benches. MultiPL-E mode is env-selected:
#   MPE_MODE=native  → no docker; nuprl harness + local rustc/javac/node (pods)
#   MPE_MODE=docker  → nuprl Docker image, sandboxed (solidpc)
#
# Flow: preflight → MPE-10 smoke on 128e (GATE) → on PASS, full LCB-100 +
# MPE-100 for both models. Everything resumes through sqlite, so a death
# restarts cheaply. Results are NOT auto-uploaded.
#
# Required env (override per host):
#   OMK            repo root (default /workspace/omnimergekit)
#   OMK_PYTHON     python with omk deps (default /usr/bin/python3 on pods)
#   RESULTS        results dir (default $OMK/eval_results_lcb100_mpe)
#   GGUF_128E      128e Q6_K gguf
#   GGUF_V5C       v5-coder Q6_K gguf
#   TOKENIZER      HF tokenizer dir (original 128e model dir)
#   MPE_MODE       native|docker   MPE_HARNESS (native): MultiPL-E repo root
set -uo pipefail

OMK="${OMK:-/workspace/omnimergekit}"
OMK_PYTHON="${OMK_PYTHON:-/usr/bin/python3}"
RESULTS="${RESULTS:-$OMK/eval_results_lcb100_mpe}"
PORT="${PORT:-8099}"
export OMK_PYTHON MPE_MODE="${MPE_MODE:-native}"
EVAL="$OMK/eval/omk_eval.py"
mkdir -p "$RESULTS"

run() {  # run <template> <served> <gguf> [extra omk args...]
    local tmpl="$1" served="$2" gguf="$3"; shift 3
    echo "[chain $(date -Iseconds)] omk_eval $tmpl on $served"
    "$OMK_PYTHON" "$EVAL" --backend llama --template "$tmpl" --quant q6_k \
        --model "$gguf" --tokenizer "$TOKENIZER" --served-name "$served" \
        --port "$PORT" --results-dir "$RESULTS" "$@"
}

score_of() {  # echo numeric score from a finished bench dir, else empty
    local tmpl="$1" served="$2"
    "$OMK_PYTHON" - "$RESULTS/$tmpl/$served" <<'PY'
import json,sys,glob,os
d=sys.argv[1]
for f in (os.path.join(d,"summary.json"), os.path.join(d,"mpe_result.json"), os.path.join(d,"lcb_result.json")):
    if os.path.exists(f):
        j=json.load(open(f)); print(j.get("score", j.get("pass_at_1",""))); break
PY
}

echo "[chain $(date -Iseconds)] === preflight ==="
[ -f "$EVAL" ] || { echo "FATAL: $EVAL missing — sync the eval/ tree"; exit 2; }
[ -n "${GGUF_128E:-}" ] && [ -f "$GGUF_128E" ] || { echo "FATAL: GGUF_128E not set/found"; exit 2; }
[ -n "${GGUF_V5C:-}" ]  && [ -f "$GGUF_V5C" ]  || { echo "FATAL: GGUF_V5C not set/found"; exit 2; }
[ -n "${TOKENIZER:-}" ] && [ -d "$TOKENIZER" ] || { echo "FATAL: TOKENIZER dir not set/found"; exit 2; }
"$OMK_PYTHON" -c "import datasets, sqlitedict" || { echo "FATAL: pip install datasets sqlitedict"; exit 2; }
if [ "$MPE_MODE" = native ]; then
    for t in rustc cargo javac node; do command -v "$t" >/dev/null || { echo "FATAL: native MPE needs $t on PATH"; exit 2; }; done
    [ -f "${MPE_HARNESS:-/workspace/MultiPL-E}/evaluation/src/main.py" ] || { echo "FATAL: MultiPL-E harness missing (git clone https://github.com/nuprl/MultiPL-E)"; exit 2; }
fi
echo "[chain] preflight OK (MPE_MODE=$MPE_MODE)"

echo "[chain $(date -Iseconds)] === MPE-10 smoke on 128e (GATE) ==="
run multipl_e_10_smoke 128e_q6k "$GGUF_128E" || true
S=$(score_of multipl_e_10_smoke 128e_q6k)
echo "[chain] smoke score=$S"
if [ -z "$S" ] || awk "BEGIN{exit !($S>0)}"; then
    echo "[chain] SMOKE GATE PASS (score=$S > 0) — proceeding to full"
else
    echo "[chain] SMOKE GATE FAIL (score=$S) — HALTING. Inspect $RESULTS/multipl_e_10_smoke/128e_q6k/"
    exit 1
fi

echo "[chain $(date -Iseconds)] === FULL: LCB-100 + MPE-100 × {128e, v5-coder} ==="
# 128e uses the no-parser LCB recipe; v5-coder (pruned) uses the parser+budget one.
run lcb_medium_100      128e_q6k    "$GGUF_128E"
run multipl_e_100       128e_q6k    "$GGUF_128E"
run lcb_medium_100_v4   v5coder_q6k "$GGUF_V5C"
run multipl_e_100       v5coder_q6k "$GGUF_V5C"

echo "[chain $(date -Iseconds)] === SUMMARY ==="
for pair in "lcb_medium_100 128e_q6k" "multipl_e_100 128e_q6k" "lcb_medium_100_v4 v5coder_q6k" "multipl_e_100 v5coder_q6k"; do
    set -- $pair; echo "  $1 / $2 : $(score_of "$1" "$2")"
done
echo "[chain $(date -Iseconds)] === chain done ==="
