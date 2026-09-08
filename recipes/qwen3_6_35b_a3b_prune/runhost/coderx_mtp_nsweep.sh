#!/usr/bin/env bash
# Measure the MTP draft-n curve ON CODERX ITSELF, to pick the `draft_num_predict` value
# baked into ~50 ollama tags.
#
# WHY NOT INHERIT: the 2026-08-21 solidpc reading (n=4 -> +63%, n=8 -> +73%, mean accept
# 3.82) came from omnimerge-v4-mtp -- a DIFFERENT artifact. Draft acceptance is a property
# of the weights, so a constant lifted from a proxy is not a measurement of this model.
#
# Flags mirror EXACTLY what ollama 0.32.7 passes (read off its own llama-server cmdline):
#     --spec-type draft-mtp --spec-draft-n-max N --spec-draft-backend-sampling
# so the curve measured here is the curve the shipped tag will get.
#
# bs2 discipline: GPU1 ONLY (GPU0 is not ours), NEVER pkill (bug-614 self-match), kill by
# literal PID, readiness is a PREDICATE not a sleep (bug-617 / a-sleep-is-not-a-predicate).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
BIN=/opt/llama.cpp/build/bin
GGUF=/mnt/sdc/ream-work/gguf/armJ_imat/armJ-Q6_K.gguf
WORK=/mnt/sdc/ream-work
LOG=$WORK/coderx_mtp_nsweep.log
PORT=8123
NPRED=500
say(){ echo "[nsweep $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

[ -f "$GGUF" ] || { say "REFUSE: no GGUF at $GGUF"; exit 1; }

# A live server on our port would silently answer for the wrong config.
if ss -ltnp 2>/dev/null | grep -q ":$PORT "; then say "REFUSE: port $PORT already bound"; exit 1; fi

PROMPT='<|im_start|>user\nWrite a complete Python implementation of a red-black tree with insert, delete and in-order traversal. Include docstrings and a short complexity analysis.<|im_end|>\n<|im_start|>assistant\n'

leg(){ # $1 = label   $2... = extra server flags
  local label="$1"; shift
  local slog="$WORK/nsweep_${label}.log"
  "$BIN/llama-server" -m "$GGUF" --port "$PORT" -c 8192 -ngl 99 \
      --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --no-warmup \
      --jinja --reasoning off "$@" > "$slog" 2>&1 < /dev/null &
  local pid=$!
  local ok=0 i
  for i in $(seq 1 180); do
    curl -s -m 2 "localhost:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || { say "  $label: SERVER DIED"; tail -12 "$slog" | sed 's/^/     /'; return 1; }
    sleep 2
  done
  [ "$ok" = 1 ] || { say "  $label: no health in 360s"; kill "$pid" 2>/dev/null; return 1; }

  # engagement evidence -- a tok/s number without this cannot be attributed to MTP
  local eng
  eng=$(grep -aiE "adding speculative implementation|creating MTP draft context|no implementations specified" "$slog" | head -2 | tr '\n' ';')
  local tot=0 n=0 r tps
  for r in 1 2 3; do
    tps=$(curl -s -m 600 "localhost:$PORT/completion" -H "Content-Type: application/json" \
      -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$NPRED,\"temperature\":0,\"cache_prompt\":false}" \
      | python3 -c "import json,sys;t=json.load(sys.stdin)['timings'];print('%.2f'%t['predicted_per_second'])" 2>/dev/null)
    [ -z "$tps" ] && tps=0
    tot=$(python3 -c "print($tot+$tps)"); n=$((n+1))
  done
  local mean; mean=$(python3 -c "print('%.2f'%($tot/$n))")
  local acc; acc=$(grep -aoE "draft acceptance = [0-9.]+ \([^)]*\), mean len = *[0-9.]+" "$slog" | tail -1)
  say "  $label: mean tok/s = $mean | ${acc:-no-acceptance-line} | engaged: ${eng:-NONE}"
  kill "$pid" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  # bug-617: teardown util drains asynchronously; wait for the card, don't assume
  for i in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    [ "${u:-99999}" -lt 2000 ] && break
    sleep 5
  done
}

say "===== CoderX (armJ) Q6_K MTP draft-n sweep, GPU1, greedy, ${NPRED} tok x3"
leg "baseline"
for N in 2 3 4 6 8; do
  leg "n$N" --spec-type draft-mtp --spec-draft-n-max "$N" --spec-draft-backend-sampling
done
say "NSWEEP_DONE"
