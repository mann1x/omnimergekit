#!/usr/bin/env bash
# R7 (#845) — build the CoderX brevity preference pairs.
#
# GOAL: pairs of CORRECT LCB solutions to the SAME problem, one short and one
# long, so a preference trainer sees length as the only intended difference.
#   chosen   <- coder+MPE  (the teacher; short)
#   rejected <- CoderX     (the student; long)
# coder+MPE is used ONLY as a teacher here. It is unpublished and stays that
# way; nothing in this pipeline ships it.
#
# The pool is eval/lcb/lcb_rl_pool.jsonl — 190 LCB problems that
# build_lcb_rl_pool.py verified carry ZERO overlap with any frozen eval
# task-id list. Composition {easy 135, medium 30, hard 25} is deliberate and
# well-matched to the failure being fixed: the agentic harness fails on
# ordinary subtasks that the model over-thinks, not on LCB-hard.
#
# SERVING GEOMETRY — both sides get the SAME one, or the pairs measure the
# geometry instead of the models. Per-slot ctx is pinned to 45056, which is
# what the banked lcb_v6_77q cells actually ran at (read back from their
# server.log, not from the template's llama_ctx which the planner overrode).
# Per-slot ctx must stay far above thinking_budget + max_gen or slots collapse
# mid-generation. [[feedback_llama_ctx_is_total_pin_and_read_back]]
#
# ASYMMETRIC k IS INTENTIONAL. The teacher side only has to supply "the
# shortest correct solution that exists", so k=2 is enough. The student side
# has to supply a REPRESENTATIVE correct sample (the median) and enough spread
# for the on-policy control pair (shortest vs longest of its own correct
# samples), which needs k=4.
#
# GPU1 ONLY. GPU0 is not ours to take without asking.
set -uo pipefail

REPO=/srv/ml/repos/omnimergekit
PY=/root/anaconda3/envs/omnimergekit/bin/python
LLAMA=/opt/llama.cpp/build/bin/llama-server
WORK=/mnt/sdc/ml/brevity
POOL="$REPO/eval/lcb/lcb_rl_pool.jsonl"
PORT=8473
PARALLEL=4
CTX_PER_SLOT=45056
CTX=$((CTX_PER_SLOT * PARALLEL))
THINK=12288
MAXGEN=32768

TEACHER_GGUF=/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf
STUDENT_GGUF=/mnt/sdc/ream-work/gguf/armJ_imat/armJ-Q6_K.gguf
TEACHER_K=${TEACHER_K:-2}
STUDENT_K=${STUDENT_K:-4}
LIMIT=${LIMIT:-0}          # 0 = whole pool; set small for a smoke

mkdir -p "$WORK"
ts(){ date '+%F %T %Z'; }
say(){ echo "[$(ts)] $*"; }

# ---------------------------------------------------------------- preflight
say "=== R7 preflight ==="
for f in "$POOL" "$TEACHER_GGUF" "$STUDENT_GGUF" "$LLAMA"; do
  [ -e "$f" ] || { say "FATAL: missing $f"; exit 2; }
done
say "pool: $(wc -l < "$POOL") problems"

FREE=$(df -BG --output=avail /mnt/sdc | tail -1 | tr -dc '0-9')
[ "$FREE" -ge 10 ] || { say "FATAL: /mnt/sdc only ${FREE}G free"; exit 2; }

# Wait for GPU1 rather than contending with whatever is on it (R8 gpqa today).
# Bounded: if it is still busy after 6h something else is wrong and a silent
# infinite wait would hide it.
WAITED=0
while :; do
  USED1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
  [ "$USED1" -lt 2000 ] && break
  [ "$WAITED" -ge 21600 ] && { say "FATAL: GPU1 still busy (${USED1} MiB) after 6h"; exit 2; }
  [ $((WAITED % 900)) -eq 0 ] && say "GPU1 busy (${USED1} MiB) — waiting..."
  sleep 60; WAITED=$((WAITED + 60))
done
say "GPU1 free (${USED1} MiB) after ${WAITED}s"

# ---------------------------------------------------------------- one arm
run_arm() {
  local role="$1" gguf="$2" name="$3" k="$4"
  local out="$WORK/gen_${role}.jsonl"
  local slog="$WORK/server_${role}.log"

  say "--- $role ($name) k=$k ---"
  fuser -k "${PORT}/tcp" 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES=1 "$LLAMA" -m "$gguf" --port "$PORT" -c "$CTX" \
      -ngl 99 --parallel "$PARALLEL" --no-warmup \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --reasoning-format deepseek --reasoning-budget "$THINK" \
      > "$slog" 2>&1 &
  local spid=$!
  disown

  # A sleep is not a readiness predicate — poll /v1/models until it answers.
  local ok=0
  for _ in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ok=1; break; fi
    kill -0 "$spid" 2>/dev/null || { say "FATAL: server died, see $slog"; return 3; }
    sleep 5
  done
  [ "$ok" -eq 1 ] || { say "FATAL: server not ready after 600s"; kill "$spid" 2>/dev/null; return 3; }
  # Read the geometry BACK — the server clamps silently if the KV will not fit.
  say "server up: $(grep -m1 -oE 'n_ctx_seq \([0-9]+\)' "$slog" || echo 'n_ctx_seq ?')"
  grep -c 'new slot' "$slog" | xargs -I{} echo "[$(ts)] slots: {}"

  "$PY" "$REPO/scripts/brevity_gen.py" \
      --pool "$POOL" --base-url "http://localhost:$PORT" \
      --model "$name" --role "$role" --out "$out" \
      --k "$k" --limit "$LIMIT" \
      --max-tokens "$MAXGEN" --thinking-budget "$THINK" \
      --temperature 0.6 --top-p 0.95 --top-k 20 \
      --sampler-name recommended --workers "$PARALLEL" \
      2>&1 | tee "$WORK/gen_${role}.log"
  local rc=${PIPESTATUS[0]}

  kill "$spid" 2>/dev/null; sleep 5
  fuser -k "${PORT}/tcp" 2>/dev/null
  say "$role: gen exit=$rc"
  return $rc
}

run_arm teacher "$TEACHER_GGUF" qwencodermpe_q6k "$TEACHER_K" || exit 3
run_arm student "$STUDENT_GGUF" qwenhybridp24_q6k "$STUDENT_K" || exit 3

# ---------------------------------------------------------------- pairs
say "=== building pairs ==="
"$PY" "$REPO/scripts/brevity_pairs.py" \
    --pool "$POOL" \
    --teacher "$WORK/gen_teacher.jsonl" \
    --student "$WORK/gen_student.jsonl" \
    --out-cross "$WORK/pairs_cross.jsonl" \
    --out-onpolicy "$WORK/pairs_onpolicy.jsonl" \
    2>&1 | tee "$WORK/pairs.log"

echo "###### R7_DONE $(ts) ######"
