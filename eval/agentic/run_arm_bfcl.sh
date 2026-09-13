#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_arm_bfcl.sh — ONE arm of the multi-turn AGENTIC bench (BFCL V3), served
# by plain llama-server. Canonical driver; see EVAL_PROTOCOL.md §8.
#
# WHY THIS HARNESS. The 9-bench canonical suite is single-turn. A defect that
# only manifests across an in-request agentic loop -- prompt growth, reasoning
# re-injection, tool-call degradation -- is INVISIBLE to it by construction.
# BFCL `multi_turn_*` runs the model against stateful backends (file system,
# trading bot, ...) and scores FINAL BACKEND STATE + execution results. The
# scorer is deterministic: no LLM judge, so no judge drift between arms.
#
# BASIS -- constant across every arm of a cohort. Any of these differing makes
# the cells non-poolable, exactly as in toolbench/run_arm_llamacpp.sh:
#   harness   inspect_ai + inspect_evals, pinned in requirements-agentic.txt.
#             inspect_evals is pinned by COMMIT -- a version string is not
#             enough, the project re-cuts 0.x tags over a moving main.
#             VERIFY WITH THE ENV'S OWN INTERPRETER, never a git checkout.
#   sampler   GREEDY (temperature 0.0) -- EVAL_PROTOCOL §1.0 anchor.
#   category  the BFCL category set. multi_turn_base = 200 samples.
#   max_tokens / max_connections / per-slot ctx  (see the CAP GATE below)
#   engine    llama.cpp build, INCLUDING the PEG-degrade patch. A PEG throw
#             turns a produced generation into an HTTP 500 and scores a
#             correct answer as a failure -- on a tool-calling bench that is
#             not a rare event. This runner REFUSES an unpatched build.
#   server    --jinja is MANDATORY. Without it the tool grammar does not come
#             from the model's chat template and every scenario degrades to
#             prose.
#
# CHAT TEMPLATE IS PART OF THE MEASUREMENT. Two arms with different templates
# do NOT isolate weights -- and conversely, to isolate a TEMPLATE the weights
# and every flag above must be byte-identical. State which one is the declared
# variable in STACK.txt and hold the rest.
#
# CAP GATE -- the reason this runner refuses rather than warns. If one arm
# inflates generation (e.g. by re-injecting prior reasoning) it hits max_tokens
# more often than its sibling, and the accuracy delta then measures LENGTH, not
# quality. summarize_arms.py reports cap-hits per arm and REFUSES to call a
# delta when the cap rates differ materially. Size the cap so neither arm is
# near it; do not equalise after the fact.
#
# Usage:
#   run_arm_bfcl.sh <ARM> <GGUF> <TEMPLATE> <OUTDIR> [GPU] [PORT] [CATEGORIES] [LIMIT]
# ---------------------------------------------------------------------------
set -uo pipefail

ARM="${1:?ARM name required}"
GGUF="${2:?GGUF path required}"
TPL="${3:?chat template path required}"
OUT="${4:?output dir required}"
GPU="${5:-0}"
PORT="${6:-8101}"
CATS="${7:-multi_turn_base}"
LIMIT="${8:-200}"

AGENTIC_ENV="${OMK_AGENTIC_ENV:-/mnt/sdc/harness/venv}"
LLAMA_BIN="${LLAMA_BIN:-/srv/ml/tools/llama.cpp/build-pegfix/bin}"
CTX="${OMK_BFCL_CTX:-131072}"
NP="${OMK_BFCL_PARALLEL:-4}"
MAXTOK="${OMK_BFCL_MAXTOK:-8192}"
THINK="${OMK_BFCL_THINKING:-8192}"

PY="$AGENTIC_ENV/bin/python"
INSPECT="$AGENTIC_ENV/bin/inspect"

[ -x "$PY" ]      || { echo "FAIL: no agentic env python at $PY (run setup_agentic_env.sh)"; exit 1; }
[ -f "$GGUF" ]    || { echo "FAIL: no such GGUF: $GGUF"; exit 1; }
[ -f "$TPL" ]     || { echo "FAIL: no such template: $TPL"; exit 1; }
mkdir -p "$OUT"

# --- GATE 1: PEG-degrade build -------------------------------------------
# Probe the LIBRARY, not the thin binary (llama-server is a ~17 KB wrapper; a
# string probe on it is a confident FALSE NEGATIVE). Capture, do not pipe into
# `grep -q`: under `set -o pipefail` grep -q exits early, SIGPIPEs strings, and
# the pipeline reports failure on a SUCCESSFUL match.
PEG_LIB="$LLAMA_BIN/libllama-common.so.0.0.1"
[ -f "$PEG_LIB" ] || { echo "FAIL: no $PEG_LIB"; exit 1; }
PEGN=$(strings "$PEG_LIB" | grep -c LLAMA_CHAT_PEG_STRICT || true)
[ "${PEGN:-0}" -ge 1 ] || {
    echo "FAIL: $LLAMA_BIN is NOT the PEG-degrade build (sentinel=$PEGN)."
    echo "      An unpatched build THROWS on a full PEG parse failure -> HTTP 500"
    echo "      -> a generated answer is scored as a failure. Refusing to run."
    exit 1; }

# --- GATE 2: per-slot ctx (EVAL_PROTOCOL §1.3a) ---------------------------
PER_SLOT=$(( CTX / NP ))
REQ=$(( MAXTOK > THINK + 4096 ? MAXTOK : THINK + 4096 ))
[ "$PER_SLOT" -ge "$REQ" ] || {
    echo "FAIL: per_slot_ctx=$PER_SLOT < required=$REQ (ctx=$CTX / np=$NP)."
    echo "      That is SILENT TRUNCATION, not an error at runtime. Refusing."
    exit 1; }

HVER=$("$PY" -c 'import importlib.metadata as m; print(m.version("inspect_evals"))' 2>/dev/null)
echo ">>> arm=$ARM gpu=$GPU port=$PORT cats=$CATS limit=$LIMIT"
echo ">>> harness inspect_evals $HVER (resolved via the env interpreter)"
echo ">>> PEG-degrade gate OK (sentinel=$PEGN); per_slot_ctx=$PER_SLOT >= $REQ OK"
echo ">>> template $(basename "$TPL") sha=$(sha256sum "$TPL" | cut -c1-16)"

# --- serve ----------------------------------------------------------------
SRVLOG="$OUT/server_${ARM}.log"
CUDA_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH="$LLAMA_BIN:${LD_LIBRARY_PATH:-}" \
nohup "$LLAMA_BIN/llama-server" \
  -m "$GGUF" --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -np "$NP" -ngl 99 --no-warmup -t 16 \
  --jinja --chat-template-file "$TPL" \
  --reasoning-format deepseek --reasoning-budget "$THINK" \
  > "$SRVLOG" 2>&1 &
SRVPID=$!
disown
echo ">>> llama-server pid=$SRVPID log=$SRVLOG"

# readiness is a PREDICATE, never a sleep
for i in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && break
    kill -0 "$SRVPID" 2>/dev/null || { echo "FAIL: server died during load; tail:"; tail -20 "$SRVLOG"; exit 1; }
    sleep 5
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] \
    || { echo "FAIL: server not ready after 600s"; tail -20 "$SRVLOG"; exit 1; }
echo ">>> server READY; per-slot ctx as reported by the server:"
grep -m2 "new slot" "$SRVLOG" || true

bash "$(dirname "${BASH_SOURCE[0]}")/write_stack_bfcl.sh" \
     "$ARM" "$OUT" "$TPL" "$GGUF" "$LLAMA_BIN" "$AGENTIC_ENV" \
     "$CTX" "$NP" "$MAXTOK" "$THINK" "$CATS" "$LIMIT" || echo "WARN: STACK.txt not written"

# --- eval -----------------------------------------------------------------
export LLAMACPP_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LLAMACPP_API_KEY="none"
"$INSPECT" eval inspect_evals/bfcl \
  -T categories="$CATS" \
  --model "openai-api/llamacpp/${ARM}" \
  --limit "$LIMIT" \
  --max-connections "$NP" \
  --max-tokens "$MAXTOK" \
  --temperature 0.0 \
  --log-dir "$OUT" \
  --no-fail-on-error
RC=$?

kill "$SRVPID" 2>/dev/null || true
echo "BFCL_ARM_DONE arm=$ARM rc=$RC $(date -u +%FT%TZ)"
exit $RC
