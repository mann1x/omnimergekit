#!/usr/bin/env bash
# MTP drafter-QUANT sweep, built on mtp-bench.py (9-prompt bench). Answers "does the
# drafter's quant level move acceptance?" — i.e. is it worth keeping the assistant/drafter
# at F16/Q8 vs Q4. Fixed target + n_max; only the drafter GGUF quant varies. Bounds the
# "should we imatrix the drafter" question. One server at a time, reap by PORT.
#
# Nothing host-hardcoded — provide paths via env (see docs/MTP_BENCH.md for a bs2 example).
# DDIR must hold drafter files named <DRAFTER_STEM>.<Q>.gguf for each Q in QUANTS.
set -u
BIN="${BIN:?set BIN to the opencoti llamafile binary (has --spec-type draft-assistant)}"
TGT="${TGT:?set TGT to the target model GGUF}"
DDIR="${DDIR:?set DDIR to the directory holding the drafter GGUFs}"
DRAFTER_STEM="${DRAFTER_STEM:?set DRAFTER_STEM, e.g. gemma-4-E2B-it-assistant (files: <stem>.<Q>.gguf)}"
BENCH="${BENCH:?set BENCH to the path of mtp-bench.py}"
PY="${PY:-python3}"
PORT="${PORT:-8232}"; GPU="${GPU:-0}"; CTX="${CTX:-8192}"; NMAX="${NMAX:-3}"
QUANTS="${QUANTS:-F16 Q8_0 Q5_K_M Q4_K_M Q4_K_S}"
OUT="${OUT:-./mtp_quant_out}"; mkdir -p "$OUT"
exec > "$OUT/run.log" 2>&1
reap(){ fuser -k "$PORT/tcp" 2>/dev/null; sleep 3; }
ts(){ date +%H:%M:%S; }

echo "target=$(basename "$TGT")  drafter_stem=$DRAFTER_STEM  n_max=$NMAX  bench=9-prompt  $(ts)"
printf '%-10s %14s %12s %12s\n' "DRAFTER_Q" "accept_rate" "tok/s_avg" "draft/acc"
for Q in $QUANTS; do
  DFT="$DDIR/$DRAFTER_STEM.$Q.gguf"
  [ -r "$DFT" ] || { printf '%-10s %14s\n' "$Q" "MISSING"; continue; }
  log="$OUT/$Q.server.log"
  reap
  CUDA_VISIBLE_DEVICES=$GPU setsid bash -c "exec '$BIN' --server -m '$TGT' -c $CTX -ngl 99 --flash-attn on \
      --mtp-head '$DFT' --spec-type draft-assistant -ngld 99 --parallel 1 --spec-draft-n-max $NMAX \
      --host 127.0.0.1 --port $PORT --temp 0 --no-warmup > '$log' 2>&1" &
  ready=0; for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "couldn.t bind|error loading|GGML_ASSERT|failed to load" "$log" 2>/dev/null && break
    sleep 2; done
  [ "$ready" != 1 ] && { printf '%-10s %14s\n' "$Q" "BOOT-FAIL"; reap; continue; }
  "$PY" "$BENCH" --url "http://127.0.0.1:$PORT" --out "$OUT/$Q.json" > "$OUT/$Q.bench.txt" 2>&1
  ar=$("$PY" -c "import json;print(json.load(open('$OUT/$Q.json'))['aggregate']['aggregate_accept_rate'])" 2>/dev/null)
  tk=$("$PY" -c "import json;d=json.load(open('$OUT/$Q.json'))['results'];print(round(sum(x['predicted_per_second'] for x in d)/len(d),1))" 2>/dev/null)
  da=$("$PY" -c "import json;d=json.load(open('$OUT/$Q.json'))['aggregate'];print(f\"{d['total_draft']}/{d['total_draft_accepted']}\")" 2>/dev/null)
  printf '%-10s %14s %12s %12s\n' "$Q" "${ar:-ERR}" "${tk:-?}" "${da:-?}"
  reap
done
echo "MTP_QUANT_SWEEP_DONE $(ts)"
