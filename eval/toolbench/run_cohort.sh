#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# tool-eval-bench 64k cohort — 7 models x 3 seeds, HARDMODE (88 scen / 176 pts)
#
# BASIS (constant across every cell):
#   harness  tool-eval-bench HEAD cf54b4b (v2.6.0-45)   <- latest, per user
#   flags    --hardmode --weight-by-difficulty --backend llamacpp
#            --context-size 65536 --context-pressure 0.25   (~14k fill)
#   server   opencoti-llamafile 0.10.5-c7-x86_64 (git 4142df1)
#            temp 0.6 / top-p 0.95 / top-k 20 / min-p 0 / presence 0 / repeat 1
#            -ngl 99 --ubatch-size 2048 --fit-target 256 -ctk q8_0 -ctv q8_0
#            -c 65536 --parallel 1
#   seeds    42 43 44 45 46  (PAIRED across models; n=5 -> t=2.776)
#   quants   27B = Q4_K_M ; 35B = IQ4_XS (matched to each other)
#
# NOT comparable to the r/LocalLLaMA thread (64k vs 256k, HEAD vs 2.6.0 scorer,
# MTP on, different quants). Ornith-1.5 / Qwen3.8-27B / Qwen3.6-35B-A3B are
# carried as IN-HOUSE ANCHORS against their published 144.2 / 152.6 / 131.5.
#
# MTP is AUTO-DETECTED per file (nextn tensors), never assumed.
# ---------------------------------------------------------------------------
set -uo pipefail

BIN="${OMK_TB_BIN:?OMK_TB_BIN must point at the opencoti-llamafile binary}"
MDIR="${OMK_TB_MODELS:?OMK_TB_MODELS must point at the GGUF directory}"
W="${OMK_TB_OUT:?OMK_TB_OUT must point at the results directory}"
PORT="${OMK_TB_PORT:-8265}"
SEEDS="${OMK_TB_SEEDS:-42 43 44 45 46}"
CTX="${OMK_TB_CTX:-65536}"
PRESSURE="${OMK_TB_PRESSURE:-0.25}"
# tool-eval-bench --timeout default is 120s. MEASURED 2026-09-06: qwen3.6-27b
# (dense 27B, NO MTP head, ~22.8 t/s decode) needs ~131 s for a 3k-token answer;
# its turn durations reach 335.5 s with 8 turns over 120 s. At the default it lost
# whole scenarios to a CLIENT-side timeout -- and tool-eval-bench DROPS those from
# the denominator rather than scoring 0, so that cell was graded on 174/172 instead
# of 176 and was not comparable to anything. No other model exceeded 120 s more than
# once (max 127.9 s), so a higher value is a NO-OP for them and does not rebase them.
TIMEOUT="${OMK_TB_TIMEOUT:-600}"
mkdir -p "$W"
export PATH="$HOME/.local/bin:$PATH"

MODELS=(
  "a3b-coder|Qwen3.6-27B-A3B-Coder-Q4_K_M.gguf"
  "a3b-coderx|Qwen3.6-27B-A3B-CoderX-Q4_K_M.gguf"
  "omnimerge-v4|Qwen3.6-27B-Omnimerge-v4-Q4_K_M.gguf"
  "omnimerge-v6|Qwen3.8-27B-Omnimerge-v6-Q4_K_M.gguf"
  "qwen3.6-27b|Qwen3.6-27B-Q4_K_M.gguf"
  "qwen3.8-27b|Qwen3.8-27B-UD-Q4_K_M.gguf"
  "ornith-1.5-35b|Ornith-1.5-35B-A3B-IQ4_XS.gguf"
  "qwen3.6-35b-a3b|Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
)

log(){ echo "[$(date +%H:%M:%S)] $*"; }
server_pid(){ local l; l=$(ss -ltnp 2>/dev/null || true); grep ":$PORT" <<<"$l" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true; }
kill_server(){
  local p; p=$(server_pid)
  if [ -n "${p:-}" ]; then
    log "  stopping server pid $p"
    kill "$p" 2>/dev/null                      # literal PID only, never pkill
    for _ in $(seq 1 30); do sleep 2; [ -z "$(server_pid)" ] && break; done
    [ -n "$(server_pid)" ] && { kill -9 "$p" 2>/dev/null; sleep 5; }
  fi
  sleep 10
}
have_report(){ [ -d "$1" ] && grep -rlq "Total Points" "$1" 2>/dev/null; }

# auto-detect a NextN/MTP head in a GGUF (never assume from the filename)
has_nextn(){ python3 - "$1" <<'PY'
import struct,sys
def u64(f): return struct.unpack('<Q',f.read(8))[0]
def u32(f): return struct.unpack('<I',f.read(4))[0]
def s(f):  return f.read(u64(f)).decode('utf-8','replace')
def sk(f,t):
    if t in (0,1,7): f.read(1)
    elif t in (2,3): f.read(2)
    elif t in (4,5,6): f.read(4)
    elif t in (10,11,12): f.read(8)
    elif t==8: s(f)
    elif t==9:
        et=u32(f); n=u64(f)
        for _ in range(n): sk(f,et)
try:
    with open(sys.argv[1],'rb') as f:
        f.read(4); u32(f); nt=u64(f); nkv=u64(f)
        for _ in range(nkv): s(f); sk(f,u32(f))
        for _ in range(nt):
            nm=s(f); nd=u32(f)
            for _ in range(nd): u64(f)
            u32(f); u64(f)
            if 'nextn' in nm or nm.startswith('mtp'): print("YES"); sys.exit(0)
    print("NO")
except Exception as e:
    print("NO")
PY
}

for SEED in $SEEDS; do
log "######## SEED $SEED — balanced pass over all ${#MODELS[@]} models ########"
for entry in "${MODELS[@]}"; do
  NAME="${entry%%|*}"; GGUF="${entry##*|}"; MODEL="$MDIR/$GGUF"

  # wait up to 40 min for a still-downloading file
  for _ in $(seq 1 40); do [ -f "$MODEL" ] && break; log "waiting for $GGUF ..."; sleep 30; done
  if [ ! -f "$MODEL" ]; then
    log "!! $NAME — $GGUF ABSENT, recorded and skipped"
    echo "$NAME MISSING $GGUF (seed $SEED) — will retry next seed pass" >> "$W/SKIPPED.txt"; continue
  fi

  have_report "$W/$NAME-s$SEED" && { log "== $NAME s$SEED already done, skipping"; continue; }
  todo="$SEED"

  MTP=$(has_nextn "$MODEL")
  SPEC=""; [ "$MTP" = "YES" ] && SPEC="--spec-type draft-mtp --spec-draft-n-max 3"
  log "== $NAME ($GGUF) nextn=$MTP seed=$SEED"

  nohup "$BIN" --server -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
      --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
      --presence-penalty 0.0 --repeat-penalty 1.0 \
      -ngl 99 --ubatch-size 2048 --fit-target 256 \
      -ctk q8_0 -ctv q8_0 -c "$CTX" --seed 42 --parallel 1 \
      $SPEC --jinja --metrics --alias "$NAME" > "$W/$NAME.server.log" 2>&1 &
  disown
  sleep 10

  # NOTE: `cmd | grep -q` under `set -o pipefail` INVERTS the result -- grep -q
  # exits on first match, SIGPIPEs cmd (141), pipefail propagates the 141 and the
  # test fails even though it matched. Capture first, then grep a herestring.
  ready=0
  for _ in $(seq 1 90); do
    lports=$(ss -ltnp 2>/dev/null || true)
    if grep -q ":$PORT" <<<"$lports"; then
      hz=$(curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
      if grep -q '"status"' <<<"$hz"; then ready=1; break; fi
    fi
    sleep 5
  done
  [ "$ready" != "1" ] && { log "!! $NAME never ready"; echo "$NAME SERVER_NOT_READY" >> "$W/SKIPPED.txt"; kill_server; continue; }
  log "  ready — VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

  if [ "$MTP" = "YES" ]; then
    acc=$(curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"'"$NAME"'","messages":[{"role":"user","content":"Count from 1 to 40."}],"max_tokens":140,"temperature":0.6}' \
      | python3 -c "import sys,json;t=json.load(sys.stdin).get('timings',{});print(t.get('draft_n',0),t.get('draft_n_accepted',0))" 2>/dev/null)
    dn=$(echo "$acc" | awk '{print $1}')
    if [ -z "${dn:-}" ] || [ "$dn" -eq 0 ] 2>/dev/null; then
      log "!! $NAME MTP_NOT_ENGAGED — recorded, running anyway WITHOUT spec (score unaffected)"
      echo "$NAME MTP_NOT_ENGAGED" >> "$W/SKIPPED.txt"
    else
      log "  MTP engaged: drafted=$dn accepted=$(echo "$acc"|awk '{print $2}')"
    fi
  fi

  for s in $todo; do
    OUT="$W/$NAME-s$s"; log "  -> $NAME seed $s"
    t0=$(date +%s)
    tool-eval-bench run --hardmode --weight-by-difficulty --backend llamacpp \
        --base-url "http://127.0.0.1:$PORT" \
        --context-size "$CTX" --context-pressure "$PRESSURE" \
        --seed "$s" --model "$NAME" --timeout "$TIMEOUT" --output-dir "$OUT" > "$W/$NAME-s$s.log" 2>&1
    rc=$?; el=$(( $(date +%s) - t0 ))
    pts=$(grep -rhoE "\*\*Total Points\*\*:[[:space:]]*[0-9]+[[:space:]]*/[[:space:]]*[0-9]+" "$OUT" 2>/dev/null | head -1)
    log "     rc=$rc  $((el/60))m$((el%60))s  ${pts:-NO_POINTS}"
    sleep 10
  done
  kill_server
done
log "######## SEED $SEED PASS COMPLETE — balanced cohort at n=$(ls -d "$W"/*-s$SEED 2>/dev/null | wc -l) ########"
done

log "COHORT COMPLETE"
[ -f "$W/SKIPPED.txt" ] && { echo "--- ANOMALIES ---"; cat "$W/SKIPPED.txt"; }
