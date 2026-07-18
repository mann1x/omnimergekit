#!/usr/bin/env bash
# MTP speculative-decoding n_max sweep, built on mtp-bench.py (the 9-prompt bench:
# 8 coding/topic categories + aggregate). Answers "what --spec-draft-n-max is best
# for this target+drafter?" — draft more tokens per round and you accept slightly
# more but pay an extra drafter forward each round; at modest acceptance (small
# models, high-entropy workloads) the optimum is often 2, not the default 3.
#
# Only --spec-draft-n-max varies across runs; target, drafter, and workload are fixed,
# so decode-tps and acceptance are directly comparable. One server at a time, reap by PORT.
#
# Nothing is host-hardcoded. Provide the paths via env (see docs/MTP_BENCH.md for a
# ready-to-run bs2 example). Requires an opencoti llamafile build with the Gemma-4
# MTP `draft-assistant` spec-type (upstream Mozilla llamafile does NOT have it).
set -u
BIN="${BIN:?set BIN to the opencoti llamafile binary (has --spec-type draft-assistant)}"
TGT="${TGT:?set TGT to the target model GGUF (the model under test)}"
DFT="${DFT:?set DFT to the drafter/assistant GGUF (gemma4_assistant, e.g. *-assistant.Q8_0.gguf)}"
BENCH="${BENCH:?set BENCH to the path of mtp-bench.py}"
PY="${PY:-python3}"
PORT="${PORT:-8263}"; GPU="${GPU:-0}"; CTX="${CTX:-8192}"
NMAXES="${NMAXES:-1 2 3 4 5}"
OUT="${OUT:-./mtp_nmax_out}"; mkdir -p "$OUT"
exec > "$OUT/run.log" 2>&1
reap(){ fuser -k "$PORT/tcp" 2>/dev/null; sleep 3; }
ts(){ date +%H:%M:%S; }

echo "target=$(basename "$TGT")  drafter=$(basename "$DFT")  bench=$(basename "$BENCH") (9-prompt)  $(ts)"
printf '%-8s %14s %12s %12s %12s\n' "N_MAX" "accept_rate" "tok/s_avg" "draft/acc" "wall_s"
for N in $NMAXES; do
  log="$OUT/nmax$N.server.log"
  reap
  CUDA_VISIBLE_DEVICES=$GPU setsid bash -c "exec '$BIN' --server -m '$TGT' -c $CTX -ngl 99 --flash-attn on \
      --mtp-head '$DFT' --spec-type draft-assistant -ngld 99 --parallel 1 --spec-draft-n-max $N \
      --host 127.0.0.1 --port $PORT --temp 0 --no-warmup > '$log' 2>&1" &
  ready=0; for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "couldn.t bind|error loading|GGML_ASSERT|failed to load" "$log" 2>/dev/null && break
    sleep 2; done
  [ "$ready" != 1 ] && { printf '%-8s %14s\n' "$N" "BOOT-FAIL"; tail -5 "$log"; reap; continue; }
  "$PY" "$BENCH" --url "http://127.0.0.1:$PORT" --out "$OUT/nmax$N.json" > "$OUT/nmax$N.bench.txt" 2>&1
  ar=$("$PY" -c "import json;print(json.load(open('$OUT/nmax$N.json'))['aggregate']['aggregate_accept_rate'])" 2>/dev/null)
  tk=$("$PY" -c "import json;d=json.load(open('$OUT/nmax$N.json'))['results'];print(round(sum(x['predicted_per_second'] for x in d)/len(d),1))" 2>/dev/null)
  da=$("$PY" -c "import json;d=json.load(open('$OUT/nmax$N.json'))['aggregate'];print(f\"{d['total_draft']}/{d['total_draft_accepted']}\")" 2>/dev/null)
  ws=$("$PY" -c "import json;print(json.load(open('$OUT/nmax$N.json'))['aggregate']['wall_s_total'])" 2>/dev/null)
  printf '%-8s %14s %12s %12s %12s\n' "$N" "${ar:-ERR}" "${tk:-?}" "${da:-?}" "${ws:-?}"
  reap
done
echo "MTP_NMAX_SWEEP_DONE $(ts)"
