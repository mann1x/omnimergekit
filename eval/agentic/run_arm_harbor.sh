#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_arm_harbor.sh — ONE arm of a DEEP-CONTEXT agentic loop test (Harbor +
# terminus-2), served by llama-server. Companion to run_arm_bfcl.sh.
#
# WHY THIS EXISTS. BFCL `multi_turn` cannot exhibit reasoning-loop degeneration:
# measured reasoning p50 is 252-480 chars and the loop rate is 1.5-4.5% with no
# template effect. The pathology needs DEEP, FRUSTRATING agentic context — the
# fixture that first showed it (33% -> 0%) ran 81k-token histories with
# think_len reaching 18,419 chars. Terminal-Bench / CompileBench / SWE-bench
# produce that shape; BFCL does not. See
# memory/feedback_measure_the_endpoint_not_the_proxy.md.
#
# THE ENDPOINT IS LOOPING, NOT TASK SCORE. Task resolution is a downstream proxy
# that only moves when a loop is severe enough to wreck the task. Score the
# trajectories with loop_census-style detectors; the pass rate is secondary.
#
# BASIS — hold constant across arms:
#   agent      terminus-2 (harbor built-in)
#   summarize  DISABLED. `enable_summarize` defaults TRUE with an 8000-token
#              proactive threshold — it COMPRESSES HISTORY, which is exactly the
#              accumulation under test. Leaving it on masks the mechanism.
#   sampler    GREEDY (temperature 0.0) — EVAL_PROTOCOL §1.0 anchor.
#   server     one per GPU, bound to the DOCKER BRIDGE (172.17.0.1), never
#              0.0.0.0: bs2 has a public interface and llama-server has no auth.
#   engine     PEG-degrade llama.cpp build (a PEG throw discards a generation).
#
# Usage:
#   run_arm_harbor.sh <ARM> <GGUF> <TEMPLATE> <DATASET> <OUTDIR> [GPU] [PORT] [NTASKS]
# ---------------------------------------------------------------------------
set -uo pipefail

ARM="${1:?ARM name}"; GGUF="${2:?GGUF path}"; TPL="${3:?template path}"
DATASET="${4:?dataset dir or name}"; OUT="${5:?output dir}"
GPU="${6:-0}"; PORT="${7:-8401}"; NTASKS="${8:-0}"

HARBOR_ENV="${OMK_HARBOR_ENV:-/mnt/sdc/agentbench/venv-harbor}"
LLAMA_BIN="${LLAMA_BIN:-/srv/ml/tools/llama.cpp/build-pegfix/bin}"
BRIDGE="${OMK_DOCKER_BRIDGE:-172.17.0.1}"
NP="${OMK_HARBOR_PARALLEL:-4}"
CTX="${OMK_HARBOR_CTX:-$(( 262144 * NP ))}"
THINK="${OMK_HARBOR_THINKING:--1}"     # default: UNRESTRICTED (llama.cpp default)
NCONC="${OMK_HARBOR_NCONC:-$NP}"

[ -f "$GGUF" ] || { echo "FAIL: no GGUF $GGUF"; exit 1; }
[ -f "$TPL" ]  || { echo "FAIL: no template $TPL"; exit 1; }
mkdir -p "$OUT"

# GATE 1 — PEG-degrade build. Capture; never pipe into `grep -q` under pipefail
# (grep -q short-circuits, SIGPIPEs strings, and the pipeline reports failure
# on a SUCCESSFUL match).
PEGN=$(strings "$LLAMA_BIN/libllama-common.so.0.0.1" | grep -c LLAMA_CHAT_PEG_STRICT || true)
[ "${PEGN:-0}" -ge 1 ] || { echo "FAIL: not the PEG-degrade build (sentinel=$PEGN)"; exit 1; }

# GATE 2 — never bind a public interface.
case "$BRIDGE" in
  127.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*|10.*|192.168.*) : ;;
  *) echo "FAIL: refusing to bind non-private address '$BRIDGE'"; exit 1 ;;
esac
ip -4 addr show 2>/dev/null | grep -q "inet $BRIDGE" || { echo "FAIL: $BRIDGE not a local address"; exit 1; }

# GATE 3 — docker image pull vs the root-fs floor.
# harbor pulls one pre-built image PER TASK into the docker data root, which on
# bs2 is /var/lib/docker on `/`. The root fs must keep >=200G free AT ALL TIMES.
# Measured expansion vs registry-compressed size: 2.41x (ML-heavy) .. 4.62x
# (text-heavy); budget the pessimistic 4.62x and ~0.55 GB compressed per task.
# Refuse loudly rather than silently eating the floor.
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
FLOOR_GB="${ROOTFS_FLOOR_GB:-200}"
AVAIL_GB=$(df -BG --output=avail "$DOCKER_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
: "${AVAIL_GB:=0}"
HEADROOM_GB=$(( AVAIL_GB - FLOOR_GB ))
NEED_GB=$(( ${NTASKS:-89} * 55 * 462 / 10000 + 1 ))
echo ">>> disk: root=$DOCKER_ROOT avail=${AVAIL_GB}G floor=${FLOOR_GB}G headroom=${HEADROOM_GB}G need~${NEED_GB}G (${NTASKS:-89} tasks @4.62x)"
if [ "$HEADROOM_GB" -lt "$NEED_GB" ]; then
  echo "FAIL: docker image pull needs ~${NEED_GB}G but only ${HEADROOM_GB}G sits above the ${FLOOR_GB}G root-fs floor."
  echo "      Reduce --n-tasks, prune images (docker image prune -a), or raise ROOTFS_FLOOR_GB deliberately."
  exit 1
fi

# harbor 0.23.0: --dataset takes a REGISTRY reference (name@version, validated as
# org/name by pydantic). A local directory MUST be passed with --path, or the run
# dies with "Package name must be in 'org/name' format". Pick by inspection.
if [ -d "$DATASET" ]; then
  DATASET_FLAG=(--path "$DATASET")
  echo ">>> dataset is a LOCAL DIR -> --path"
else
  DATASET_FLAG=(--dataset "$DATASET")
  echo ">>> dataset is a REGISTRY ref -> --dataset"
fi

echo ">>> arm=$ARM gpu=$GPU port=$PORT dataset=$DATASET tasks=${NTASKS:-all}"
echo ">>> PEG gate OK (sentinel=$PEGN); binding $BRIDGE (docker bridge, host-local)"
echo ">>> template $(basename "$TPL") sha=$(sha256sum "$TPL" | cut -c1-16)"

# GATE 4 — free VRAM, not a closed port.
# 2026-09-12: the smoke server started as soon as the previous ports closed and
# died with `cudaMalloc failed: out of memory` (35.5G free of 97G) because the
# prior server had not released its buffers. Wait on the number that matters.
NEED_MIB="${NEED_MIB:-45000}"
vram_ok=0
for i in $(seq 1 120); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -dc '0-9')
  : "${FREE:=0}"
  [ "$FREE" -ge "$NEED_MIB" ] && { vram_ok=1; break; }
  echo "    waiting for VRAM on gpu$GPU: ${FREE}MiB free, need ${NEED_MIB}MiB"
  sleep 15
done
[ "$vram_ok" -eq 1 ] || { echo "FAIL: gpu$GPU never freed ${NEED_MIB}MiB — refusing to start a server that will OOM"; exit 1; }
echo ">>> VRAM gate OK (gpu$GPU free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")MiB)"

SRVLOG="$OUT/server_${ARM}.log"
CUDA_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH="$LLAMA_BIN:${LD_LIBRARY_PATH:-}" \
nohup "$LLAMA_BIN/llama-server" -m "$GGUF" --host "$BRIDGE" --port "$PORT" \
  -c "$CTX" -np "$NP" -ngl 99 --no-warmup -t 16 \
  --jinja --chat-template-file "$TPL" \
  --reasoning-format auto --reasoning-budget "$THINK" \
  > "$SRVLOG" 2>&1 &
SRVPID=$!
disown
echo ">>> llama-server pid=$SRVPID log=$SRVLOG"

for i in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://$BRIDGE:$PORT/health" 2>/dev/null)" = "200" ] && break
  kill -0 "$SRVPID" 2>/dev/null || { echo "FAIL: server died"; tail -20 "$SRVLOG"; exit 1; }
  sleep 5
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://$BRIDGE:$PORT/health" 2>/dev/null)" = "200" ] \
  || { echo "FAIL: server not ready"; tail -20 "$SRVLOG"; exit 1; }
grep -m1 "new slot" "$SRVLOG" || true

TASK_ARGS=()
[ "$NTASKS" != "0" ] && TASK_ARGS+=(--n-tasks "$NTASKS")

OPENAI_API_KEY=none OPENAI_BASE_URL="http://$BRIDGE:$PORT/v1" \
"$HARBOR_ENV/bin/harbor" run \
  "${DATASET_FLAG[@]}" \
  --agent terminus-2 \
  --model "openai/$ARM" \
  --ak "api_base=http://$BRIDGE:$PORT/v1" \
  --ak "temperature=0.0" \
  --ak "enable_summarize=false" \
  --ak "proactive_summarization_threshold=0" \
  --allow-agent-host "$BRIDGE" \
  --jobs-dir "$OUT" --job-name "$ARM" \
  -n "$NCONC" -y "${TASK_ARGS[@]}" 2>&1 | tail -40
RC=$?

kill "$SRVPID" 2>/dev/null || true
echo "HARBOR_ARM_DONE arm=$ARM rc=$RC $(date -u +%FT%TZ)"
exit $RC
