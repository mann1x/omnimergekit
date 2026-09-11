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
#   seeds     42 43 44 45 46 -- PAIRED across the arms of the cohort, n=5,
#             t=2.776. THE SEED SET IS PART OF THE BASIS, exactly like the
#             sampler and the harness version. The reportable tool-calling
#             cell for an arm is the MEAN OVER ALL FIVE; a 1- or 3-seed run
#             is a different denominator with a different CI and must never
#             be pooled into an arm table or compared against a 5-seed cell.
#             Passing a subset on the command line is for FILLING missing
#             seeds of an arm that will reach 42..46 -- not for producing a
#             cell. This runner prints TOOLBENCH_BASIS_COMPLETE only when
#             all five seed files exist on disk.
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
# Default is the FULL canonical basis, not seed 42 alone. A bare invocation
# must produce a reportable cell; under-powering has to be an explicit choice.
if [ "$#" -gt 0 ]; then SEEDS=("$@"); else SEEDS=(42 43 44 45 46); fi

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

if [ "${#SEEDS[@]}" -lt 5 ]; then
    echo ">>> PARTIAL BASIS: ${#SEEDS[@]}/5 seeds (${SEEDS[*]}) -- canonical cell is the mean over 42..46."
    echo ">>> This invocation alone does NOT yield a reportable cell for $ARM."
    printf '%s partial invocation seeds=%s\n' "$(date -u +%FT%TZ)" "${SEEDS[*]}" >> "$OUT/PARTIAL_BASIS.txt"
fi

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

# Basis census. The reportable unit is the ARM, not this invocation: earlier
# runs may have filled other seeds, so count the files on disk. An arm is only
# tabulatable once all five paired seeds exist.
have=(); missing=()
for S in 42 43 44 45 46; do
    if [ -f "$OUT/${ARM}_seed${S}.json" ]; then have+=("$S"); else missing+=("$S"); fi
done
if [ "${#missing[@]}" -eq 0 ]; then
    echo "TOOLBENCH_BASIS_COMPLETE $ARM seeds=42..46 (n=5, t=2.776)"
else
    echo "TOOLBENCH_BASIS_INCOMPLETE $ARM have=${have[*]:-none} missing=${missing[*]}"
    echo "TOOLBENCH_BASIS_INCOMPLETE -> do NOT report a $ARM tool-calling score yet."
fi

exit $rc_all
