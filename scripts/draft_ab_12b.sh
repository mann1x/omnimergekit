#!/usr/bin/env bash
# draft_ab_12b.sh — DECISIVE A/B: does the 12B assistant drafter increase throughput at the
# harness's real --parallel 5, on STOCK llama.cpp b9700? (user: "use b9700"; "if the settings
# prevent it to increase speed we don't use it".) Two legs, same GPU/port/prompts:
#   BASELINE: 12B, --parallel 5, NO drafter.
#   DRAFTER : 12B, --parallel 5, -md assistant --spec-type draft-simple.
# Metric = aggregate completion tokens / wall-clock over PAR concurrent requests. Decision:
# use the drafter iff DRAFTER aggregate_tps > BASELINE aggregate_tps. GPU1, transient.
set -uo pipefail
BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server
M=/mnt/sdc/ml/google/gemma-4-12B-it-F16.gguf
D=/mnt/sdc/ml/google/gemma-4-12B-it-assistant-Q8_0.gguf
PORT=8199; GPU=1; PAR=5
WORK=/mnt/sdc/ml/draft_ab; mkdir -p "$WORK"
ts(){ date '+%T %Z'; }
for f in "$BIN" "$M" "$D"; do [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }; done

# mixed prompt: high-acceptance repetitive span + low-acceptance free text (representative)
REQ='{"model":"g","messages":[{"role":"user","content":"Write out the numbers 1 through 80, one per line as \"N: <number-in-words>\". Then in 3 sentences explain why counting is useful."}],"max_tokens":600,"temperature":0.0,"stream":false}'

leg() { # $1=label ; $2... = extra server args
  local label="$1"; shift
  local log="$WORK/${label}.server.log"
  echo "[ab $(ts)] ===== $label ===== launch (par=$PAR) extra: $*"
  CUDA_VISIBLE_DEVICES=$GPU nohup "$BIN" -m "$M" -ngl 99 -c 20480 --parallel $PAR \
    -fa on --port "$PORT" --no-warmup --reasoning-format deepseek "$@" > "$log" 2>&1 &
  local spid=$!; disown
  if ! curl -sf --retry 90 --retry-delay 2 --retry-connrefused --retry-all-errors \
            --max-time 240 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[ab $(ts)] $label NOT READY / boot failed — tail:"; tail -25 "$log"
    kill -0 "$spid" 2>/dev/null && kill "$spid"; echo "${label}_BOOT_FAIL"; return 1
  fi
  if ! kill -0 "$spid" 2>/dev/null; then echo "[ab $(ts)] $label died post-ready"; tail -25 "$log"; echo "${label}_BOOT_FAIL"; return 1; fi
  echo "[ab $(ts)] $label ready; firing $PAR concurrent requests"
  local t0 t1; t0=$(date +%s.%N)
  for i in $(seq 1 $PAR); do
    curl -sf --max-time 400 -H "Content-Type: application/json" -d "$REQ" \
      "http://127.0.0.1:$PORT/v1/chat/completions" -o "$WORK/${label}_r${i}.json" &
  done
  wait
  t1=$(date +%s.%N)
  local TT=0 ok=0
  for i in $(seq 1 $PAR); do
    local ct; ct=$(grep -oE '"completion_tokens":[0-9]+' "$WORK/${label}_r${i}.json" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    [ -n "$ct" ] && { TT=$((TT + ct)); ok=$((ok+1)); }
  done
  local wall tps; wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
  tps=$(awk "BEGIN{ if($wall>0) printf \"%.1f\", $TT/$wall; else print 0 }")
  echo "[ab $(ts)] $label RESULT: ok_resp=$ok/$PAR  total_completion_tokens=$TT  wall=${wall}s  AGGREGATE_TPS=$tps"
  echo "[ab $(ts)] $label draft/accept signal:"; grep -iE "draft|accept|n_drafted|n_accept|speculat" "$log" | tail -8 || echo "  (none)"
  echo "RESULT|$label|$tps|$TT|$wall|$ok"
  kill "$spid" 2>/dev/null
  for i in $(seq 1 12); do kill -0 "$spid" 2>/dev/null || break; done
  kill -0 "$spid" 2>/dev/null && kill -9 "$spid" 2>/dev/null
  return 0
}

echo "[ab $(ts)] BEGIN A/B (b9700, GPU$GPU, par=$PAR)"
leg BASELINE_par5 || true
leg DRAFTER_par5 -md "$D" --spec-type draft-simple --spec-draft-n-max 8 --spec-draft-p-min 0.4 -ngld 99 || true

echo ""; echo "[ab $(ts)] ===== VERDICT ====="
b=$(grep "^RESULT|BASELINE_par5|" "$WORK"/../draft_ab.stdout 2>/dev/null | tail -1 | cut -d'|' -f3)
echo "  (compare AGGREGATE_TPS lines above: DRAFTER_par5 > BASELINE_par5  => use drafter; else drop)"
echo "DRAFT_AB_DONE"
