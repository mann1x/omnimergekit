#!/usr/bin/env bash
# lcb_retry_driver.sh — surgical LCB length-cap retry across the published-table
# cells, gated behind the F16 suite. For each cell, re-generate ONLY the
# finish_reason==length problems at max_gen_toks=24576 (the canonical retry),
# IN PLACE (same dir / same sqlite, resumed), then rewrite lcb_result.json +
# summary.json over the full set.
#
# Self-describing per cell: gguf from server.log, tokenizer from summary.json,
# template from the parent dir name. Eval results are SACRED → every cell dir is
# copied to <cell>.preretry_celldir.bak before any mutation; nothing is deleted.
#
# Usage: lcb_retry_driver.sh [GPU] [PORT] [--no-wait]
#   GPU   GPU id to pin (default 0)
#   PORT  llama-server port (default 8230)
#   --no-wait  skip the F16_SUITE_ALL_DONE gate (run immediately)
set -uo pipefail

PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TPL_DIR=/srv/ml/repos/omnimergekit/eval/templates
EVICT=/srv/ml/scripts/lcb_evict_lengthcaps.py
SUITE_LOG=/srv/ml/logs/eval_f16_suite.log
RETRY_CAP=24576
WORK=/srv/ml/scripts/lcb_retry_work; mkdir -p "$WORK"
LOG="$WORK/lcb_retry_$(date -u +%Y%m%d_%H%M%S).log"

GPU=0; PORT=8230; NOWAIT=0
for a in "$@"; do
  case "$a" in
    --no-wait) NOWAIT=1 ;;
    *[!0-9]*) : ;;                       # ignore non-numeric
    *) if [ "$GPU" = 0 ] && [ "$a" -lt 8 ]; then GPU="$a"; else PORT="$a"; fi ;;
  esac
done

exec > >(tee -a "$LOG") 2>&1
echo "[lcbretry] START $(date -u)  gpu=$GPU port=$PORT cap=$RETRY_CAP nowait=$NOWAIT"

if [ "$NOWAIT" -eq 0 ]; then
  echo "[lcbretry] gating on F16_SUITE_ALL_DONE in $SUITE_LOG"
  while ! grep -q "F16_SUITE_ALL_DONE" "$SUITE_LOG" 2>/dev/null; do sleep 60; done
  echo "[lcbretry] suite done — waiting 30s for servers to exit"; sleep 30
fi

# Published-table cells only:  results_root|served_name
ROOTS=(
  "/srv/ml/eval_results_f16|128e-f16"
  "/srv/ml/eval_results_f16|v7coder-f16"
  "/srv/ml/eval_results_f16|v7coderx-f16"
  "/srv/ml/eval_results_128e_bs2|128e-q6k"
  "/srv/ml/eval_results_v6coder_bs2built|v6coder-bs2built-q6k"
  "/srv/ml/eval_results_v7coder_g15f2440|v7coder-g15f2440-q6k"
  "/srv/ml/eval_results_v7coder_fs2440|v7coder-fs2440-q6k"
)
TEMPLATES=(lcb_medium_55 lcb_medium_100 lcb_medium_55_v4 lcb_medium_100_v4)

mk_retry_tpl() {  # $1 orig template name ; writes $WORK/<name>.yaml (same basename!)
  local tname="$1"; local dst="$WORK/$tname.yaml"
  "$PY" - "$TPL_DIR/$tname.yaml" "$dst" "$RETRY_CAP" >&2 <<'PYEOF'
import sys, yaml
src, dst, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = yaml.safe_load(open(src))
d.setdefault("generation", {})["max_gen_toks"] = cap
d.setdefault("backend_args", {})["llama_ctx"] = 53248  # per_slot>=26624 (>=cap+prompt) @ parallel=2; works v4(bump) & non-v4(reasoning-off)
# name + cache.sqlite_prefix MUST stay identical so omk writes to the same dir
# and resumes the same sqlite db.
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
print(f"  tpl name={d.get('name')} prefix={d.get('cache',{}).get('sqlite_prefix')} "
      f"max_gen_toks={d['generation']['max_gen_toks']} llama_ctx={d['backend_args']['llama_ctx']}")
PYEOF
  echo "$dst"
}

retry_cell() {
  local cell="$1"
  [ -d "$cell" ] || return 0
  local tname; tname=$(basename "$(dirname "$cell")")
  local served; served=$(basename "$cell")
  [ -f "$cell/summary.json" ] || { echo "[skip] $tname/$served (no summary.json)"; return 0; }

  local caps; caps=$("$PY" "$EVICT" "$cell" --inspect 2>/dev/null | grep -c '^  - ')
  if [ "$caps" -eq 0 ]; then echo "[skip] $tname/$served caps=0"; return 0; fi

  local oldscore gguf tok
  oldscore=$("$PY" -c "import json;print(json.load(open('$cell/summary.json'))['score'])")
  gguf=$(grep -oE "load_model: loading model '[^']+'" "$cell/server.log" 2>/dev/null | head -1 | sed "s/.*'\(.*\)'/\1/")
  # >>> gguf-fallback override (relocated bins) <<<
  if [ -z "$gguf" ] || [ ! -f "$gguf" ]; then
    case "$served" in
      v7coder-g15f2440-q6k) gguf=/mnt/sdc/ml/eval_gguf/v7coder-g15f2440-Q6_K.gguf ;;
      v7coder-fs2440-q6k)   gguf=/mnt/sdc/ml/eval_gguf/v7coderx-Q6_K.gguf ;;
    esac
    [ -f "$gguf" ] && echo "        gguf(fallback)=$gguf"
  fi
  tok=$("$PY" -c "import json;print(json.load(open('$cell/summary.json'))['token_stats']['completion_tokens']['method'].split('tokenizer:')[-1])" 2>/dev/null)
  echo "[retry] $tname/$served caps=$caps old=$oldscore"
  echo "        gguf=$gguf"
  echo "        tok=$tok"
  if [ -z "$gguf" ] || [ ! -f "$gguf" ]; then echo "[ERR] gguf missing for $served: '$gguf'"; return 1; fi
  if [ -z "$tok" ]; then echo "[ERR] tokenizer unresolved for $served"; return 1; fi

  cp -a "$cell" "$cell.preretry_celldir.bak" || { echo "[ERR] backup failed"; return 1; }
  "$PY" "$EVICT" "$cell" --force || { echo "[ERR] evict failed"; return 1; }
  local rtpl; rtpl=$(mk_retry_tpl "$tname")
  local root; root=$(dirname "$(dirname "$cell")")

  echo "[run ] omk_eval resume CUDA_VISIBLE_DEVICES=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$OMK" --model "$gguf" --tokenizer "$tok" --template "$rtpl" \
        --backend llama --served-name "$served" --results-dir "$root" \
        --port "$PORT"
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[ERR] omk_eval rc=$rc for $served"; return 1; fi

  local nrows newscore newcaps
  nrows=$(wc -l < "$cell/lcb_result.samples.jsonl")
  newscore=$("$PY" -c "import json;print(json.load(open('$cell/summary.json'))['score'])")
  newcaps=$("$PY" "$EVICT" "$cell" --inspect 2>/dev/null | grep -c '^  - ')
  echo "[DONE] $tname/$served rows=$nrows  ${oldscore} -> ${newscore}  remaining_caps=${newcaps}"
  "$PY" -c "
o=float('$oldscore'); n=float('$newscore')
print('[WARN] score DROPPED after retry — investigate' if n < o-1e-9 else '[ok] non-decreasing')
"
}

for spec in "${ROOTS[@]}"; do
  root="${spec%%|*}"; served="${spec##*|}"
  for t in "${TEMPLATES[@]}"; do
    retry_cell "$root/$t/$served"
  done
done
echo "###### LCB_RETRY_ALL_DONE $(date -u) ######"
