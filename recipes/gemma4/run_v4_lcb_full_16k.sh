#!/usr/bin/env bash
# Re-run v4 LCB-medium full with --max-tokens 16384 to decontaminate the
# 8192-token cap-truncation from the apparent -9.09pp regression vs 128e.
#
# 8192-run baseline: 78.18% (43/55), 12 cap-hits (21.8%).
# Inspection (gemma4-A4B-98e-v4-it-Q6K_lcb_full.samples.jsonl):
#   ~5 of 8 regressions are degenerate loops (will still fail at 16384);
#   ~3 borderline cases (3699, 3716, 3776) may recover with more budget.
#
# Waits for the in-flight v4_full_local_canonical.sh chain to finish before
# claiming port 8099 / the 3090.
set -euo pipefail
WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
LOGS=$WS/logs
GGUF=$WS/google/gemma-4-A4B-98e-cd-Q6_K.gguf
NAME=gemma4-A4B-98e-v4-it-Q6K
PORT=8099
RESULTS=$WS/eval_results_v4/lcb_16k
mkdir -p "$RESULTS"

eval "$(/root/anaconda3/bin/conda shell.bash hook)" && conda activate omnimergekit
ts() { date +%Y%m%d_%H%M%S; }
log() { echo "[$(date -Iseconds)] $*"; }

# ── 1. Wait for the in-flight canonical chain to exit ──────────
CHAIN_PID=2737556
log "waiting for in-flight canonical chain pid=$CHAIN_PID ..."
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
log "  chain exited; settling 30 s for any straggler kill ..."
sleep 30
# Belt-and-braces: kill anything still on the port
pkill -KILL -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 5

# ── 2. Start chat-profile server, c=32768 (LCB protocol §1.3) ───
SLOG=$LOGS/server_v4_lcb_16k_$(ts).log
log "starting llama-server (LCB chat profile) ..."
/opt/llama.cpp/build/bin/llama-server -m "$GGUF" --port $PORT \
    -c 32768 -t 12 -ngl 99 --no-warmup --parallel 2 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --temp 0 --top-p 1 --top-k 0 --seed 42 \
    --jinja --reasoning off > "$SLOG" 2>&1 &
SPID=$!
disown
log "  pid=$SPID log=$SLOG"
for i in $(seq 1 60); do
    curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { log "  server ready"; break; }
    sleep 2
done

# ── 3. Run LCB with bumped max-tokens ──────────────────────────
T0=$(date +%s)
log "running LCB-medium full @ max_tokens=16384 ..."
python3 /shared/dev/omnimergekit/eval/lcb/lcb_llama_server.py \
    --name "$NAME" --base-url "http://localhost:$PORT" \
    --limit 999 --max-tokens 16384 \
    --output "$RESULTS/${NAME}_lcb_full_16k.json" \
    > "$LOGS/v4_lcb_16k_$(ts).log" 2>&1 || true
log "  LCB wall=$(($(date +%s)-T0))s"

# ── 4. Teardown ────────────────────────────────────────────────
kill -TERM $SPID 2>/dev/null || true
sleep 2; kill -KILL $SPID 2>/dev/null || true
pkill -KILL -f "llama-server.*--port $PORT" 2>/dev/null || true

# ── 5. Summary: pass@1 + cap-hits + diff vs 8192 run ───────────
python3 - <<EOF
import json
new = json.load(open("$RESULTS/${NAME}_lcb_full_16k.json"))
print(f"\n=== v4 LCB-medium full @ 16384 ===")
print(f"  pass@1 = {new['pass_at_1']*100:.2f}%  ({new['n_pass']}/{new['n']})")
caps = sum(1 for r in new.get('samples', []) if r.get('finish_reason')=='length')
print(f"  finish_reason=length: {caps}/{new['n']} = {caps/new['n']*100:.1f}%")

# Cross-diff vs 8192 run
old_p = "$WS/eval_results_v4/lcb/${NAME}_lcb_full.samples.jsonl"
try:
    old = {json.loads(l)['task_id']: json.loads(l) for l in open(old_p)}
    samples = {r['task_id']: r for r in new.get('samples', [])} if isinstance(new.get('samples'), list) else {}
    # If the new file doesn't carry samples inline, fall back to JSONL sidecar
    if not samples:
        new_p = "$RESULTS/${NAME}_lcb_full_16k.samples.jsonl"
        try:
            samples = {json.loads(l)['task_id']: json.loads(l) for l in open(new_p)}
        except FileNotFoundError:
            print("  (no samples file to diff — check runner output)")
            raise SystemExit(0)
    recovered = []   # old fail → new pass
    new_regr  = []   # old pass → new fail (sanity — should be 0)
    for tid in set(samples) | set(old):
        op = old.get(tid, {}).get('passed')
        np_ = samples.get(tid, {}).get('passed')
        if op is False and np_ is True:  recovered.append(tid)
        if op is True  and np_ is False: new_regr.append(tid)
    print(f"  recovered (8192 fail → 16384 pass): {len(recovered)} -> {recovered}")
    print(f"  new regressions (sanity, expect 0): {len(new_regr)} -> {new_regr}")
except FileNotFoundError:
    print("  (no 8192 samples to diff against)")
EOF

log "DONE — results in $RESULTS"
