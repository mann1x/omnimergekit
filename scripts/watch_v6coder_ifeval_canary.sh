#!/usr/bin/env bash
# Watches for v6-coder stack@2 IFEval cache to populate, then extracts the
# 4 known-trigger doc_ids (18, 31, 50, 59) and reports rumination status.
#
# Diagnostic: tells us whether stack@2 rumination on those docs is
#   (a) drop-map-specific to v5-coder, or
#   (b) stack-wide on all pruned Gemma 4 MoE variants.
#
# Runs as a polling background watcher; fires the diagnostic the moment all
# 4 doc_ids appear in the sqlite cache. Exits after report.

set -euo pipefail
ROOT="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"
LOGTS=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/logs/v6coder_ifeval_canary_${LOGTS}.log"
mkdir -p "$ROOT/logs"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

V6_CACHE_DIR="$ROOT/eval_results_vllm_suite/v6coder/ifeval_100/98e_v6_coder_nvfp4a16/sqlite_cache"
V6_DB="$V6_CACHE_DIR/ifeval_98e_v6_coder_nvfp4a16_rank0.db"

log "Waiting for v6-coder IFEval sqlite to appear at $V6_DB ..."
while [ ! -f "$V6_DB" ]; do sleep 60; done
log "v6-coder IFEval cache materialized."

log "Polling for all 4 critical doc_ids (cache rows accumulate live)..."
# lm-eval stores cached responses keyed by prompt hash; we can't easily
# map doc_id → hash from outside, so wait for v6-coder IFEval to fully
# complete (samples_*.jsonl gets written at end of bench).
SAMP_DIR="$ROOT/eval_results_vllm_suite/v6coder/ifeval_100/98e_v6_coder_nvfp4a16/lm_eval_out"
while true; do
    SAMP=$(find "$SAMP_DIR" -name "samples_ifeval*.jsonl" 2>/dev/null | sort | tail -1)
    if [ -n "$SAMP" ] && [ "$(wc -l < "$SAMP")" -ge 100 ]; then
        log "v6-coder IFEval samples complete (100/100). Extracting canary docs."
        break
    fi
    sleep 90
done

log "=== v6-coder stack@2 IFEval CANARY: docs 18, 31, 50, 59 ==="
python3 - <<PY | tee -a "$LOG"
import json
samp = "$SAMP"
target = {18, 31, 50, 59}
with open(samp) as f:
    for line in f:
        r = json.loads(line)
        did = r['doc_id']
        if did not in target: continue
        resp = (r.get('resps') or [['']])[0][0]
        clen = len(resp)
        verdict = "CLEAN" if clen < 2000 else ("LONG" if clen < 5000 else "RUMINATE")
        tail = resp[-150:].replace('\n', ' ')
        iids = r.get('doc', {}).get('instruction_id_list', [])
        strict = r.get('prompt_level_strict_acc')
        print(f"doc {did:>2}: chars={clen:>6}  strict={strict}  verdict={verdict}")
        print(f"  inst_ids: {iids}")
        print(f"  tail(150): {tail!r}")
        print()
PY

log "Canary diagnostic complete. Reading the table:"
log "  If all 4 docs CLEAN  → stack@2 + v6-coder drop map is FINE; the v5-coder rumination is C6-drop-map-specific (drop map interacts badly with stack@2 routing micro-numerics)."
log "  If any doc RUMINATES → stack@2 has structural issue on these prompts independent of drop map; broader fix needed."
log "  Compare per-doc against v5-coder stack@2 (ruminating) and v5-coder stack@1 (clean) for the full 4-cell matrix."
