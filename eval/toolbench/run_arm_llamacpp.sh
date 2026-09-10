#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_arm_llamacpp.sh — ONE arm of tool-eval-bench, served by plain llama-server.
#
# Companion to run_cohort.sh, which is bound to the opencoti-llamafile binary
# and to the 10-model 64k cohort. This runner exists for cohorts that serve
# Q6_K GGUFs from stock llama.cpp and want tool-calling on the SAME greedy
# basis as the rest of the canonical suite.
#
# BASIS — every field here must be held constant across the arms of a cohort,
# and NONE of it is comparable to the run_cohort.sh cells:
#   harness   whatever `tool-eval-bench` the uv tool resolves. VERIFY IT WITH
#             THE CLI'S OWN INTERPRETER, never `git rev-parse` — the CLI
#             imports the uv install, not any checkout:
#               PY=$(head -1 "$(command -v tool-eval-bench)" | sed 's|^#!||')
#               "$PY" -c 'import importlib.metadata as m
#                         print(m.version("tool-eval-bench"))'
#             run_cohort.sh's cells are v2.6.0. A dev build off main is a
#             DIFFERENT SCORER — never pool the two.
#   sampler   GREEDY (temp 0.0 / top-p 1.0 / top-k 0). run_cohort.sh serves
#             temp 0.6 / top-p 0.95 / top-k 20 — also not poolable.
#   flags     --hardmode --weight-by-difficulty --backend llamacpp
#   server    llama-server --jinja (MANDATORY: without it the tool grammar
#             never comes from the model's chat template and every scenario
#             degrades to prose)
#
# CHAT TEMPLATE IS PART OF THE MEASUREMENT. A tool score compares the model
# AND the template it ships. Two arms with different templates do not isolate
# weights — census the templates before attributing a delta:
#   python -c 'from gguf import GGUFReader; r=GGUFReader("x.gguf");
#              print({f.name:f for f in r.fields.values()}
#                    ["tokenizer.chat_template"].contents()[:200])'
#
# Usage:
#   run_arm_llamacpp.sh <ARM> <GGUF> <OUTDIR> [SEEDS...]
# ---------------------------------------------------------------------------
set -uo pipefail

ARM="${1:?ARM name required}"
GGUF="${2:?GGUF path required}"
OUT="${3:?output dir required}"
shift 3
SEEDS=("${@:-42}")

LLAMA_BIN="${LLAMA_BIN:-/opt/llama.cpp/build/bin}"
PORT="${OMK_TB_PORT:-8265}"
CTX="${OMK_TB_CTX:-65536}"
PRESSURE="${OMK_TB_PRESSURE:-0.25}"
# 120s default drops whole scenarios to a CLIENT-side timeout, and
# tool-eval-bench removes those from the DENOMINATOR rather than scoring them
# 0 — so a timing-out cell is graded on fewer points and is not comparable.
TIMEOUT="${OMK_TB_TIMEOUT:-600}"

command -v tool-eval-bench >/dev/null || { echo "FAIL: tool-eval-bench not on PATH"; exit 1; }
[ -f "$GGUF" ] || { echo "FAIL: no such GGUF: $GGUF"; exit 1; }
mkdir -p "$OUT"

PY=$(head -1 "$(command -v tool-eval-bench)" | sed 's|^#!||')
VER=$("$PY" -c 'import importlib.metadata as m; print(m.version("tool-eval-bench"))' 2>/dev/null)
echo ">>> harness tool-eval-bench $VER   (resolved via the CLI interpreter)"
echo ">>> arm=$ARM gguf=$(basename "$GGUF") ctx=$CTX pressure=$PRESSURE timeout=${TIMEOUT}s seeds=${SEEDS[*]}"

echo ">>> starting llama-server on :$PORT"
"$LLAMA_BIN/llama-server" -m "$GGUF" --port "$PORT" -c "$CTX" -ngl 99 \
    --jinja --parallel 1 --no-warmup \
    > "$OUT/${ARM}_server.log" 2>&1 &
SRV=$!
# A sleep is not a readiness predicate — poll the endpoint, and give up loudly.
for i in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SRV" 2>/dev/null || { echo "FAIL: llama-server died during startup"; tail -20 "$OUT/${ARM}_server.log"; exit 1; }
    sleep 2
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 || {
    echo "FAIL: llama-server never became ready"; tail -20 "$OUT/${ARM}_server.log"; kill "$SRV" 2>/dev/null; exit 1; }
echo ">>> server ready"

rc_all=0
for S in "${SEEDS[@]}"; do
    J="$OUT/${ARM}_seed${S}.json"
    echo ">>> $(date -u +%H:%M:%S) $ARM seed=$S"
    tool-eval-bench run \
        --model "$ARM" \
        --backend llamacpp \
        --base-url "http://localhost:$PORT/v1" \
        --hardmode --weight-by-difficulty \
        --context-size "$CTX" --context-pressure "$PRESSURE" \
        --temperature 0.0 --top-p 1.0 --top-k 0 \
        --variant-seed "$S" --seed "$S" \
        --timeout "$TIMEOUT" \
        --json-file "$J" --no-live
    rc=$?; [ $rc -ne 0 ] && rc_all=$rc
    echo ">>> $(date -u +%H:%M:%S) $ARM seed=$S rc=$rc -> $J"
done

kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo ">>> server stopped"
[ $rc_all -eq 0 ] && echo "TOOLBENCH_OK $ARM" || echo "TOOLBENCH_FAIL $ARM rc=$rc_all"
exit $rc_all
