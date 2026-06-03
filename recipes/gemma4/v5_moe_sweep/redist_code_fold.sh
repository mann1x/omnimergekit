#!/usr/bin/env bash
# T193 code-driver redistribution: capture(code corpus) -> closed-form fold of dropped
# code-experts into A2 fixed survivors -> loop_screen(regression) -> KEEP model for HE+/MPE eval.
# Code fails FLUENTLY, so loop_screen is a no-regression gate; HE+/MPE pass@1 is the primary
# metric (run separately via redist_code_eval.sh). Closed-form (REAM) = no label leakage.
#
# Usage: redist_code_fold.sh [--run] <method> [gpu] [max_seqs] [max_tokens]
#   Dry-run by default (prints the plan, exits 0). Pass --run to execute.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$SCRIPT_DIR/redist_config.sh" ] && . "$SCRIPT_DIR/redist_config.sh"
DO_RUN=0; ARGS=()
for a in "$@"; do [ "$a" = "--run" ] && DO_RUN=1 || ARGS+=("$a"); done
set -- "${ARGS[@]+"${ARGS[@]}"}"

METHOD="${1:-ream}"; GPU="${2:-0}"; MAXSEQ="${3:-200}"; MAXTOK="${4:-512}"
REDIST_PY="${REDIST_PY:-python}"
REDIST_SCRIPTS_DIR="${REDIST_SCRIPTS_DIR:-$REPO_ROOT/scripts}"
WORK="${REDIST_WORK:-$PWD/redist_work}"
RES="${REDIST_RESULTS:-$PWD/eval_results_redist}"
OUT_BASE="${REDIST_OUT_BASE:-$PWD/redist_models}"
CORPUS="${REDIST_CALIB_CODE:-}"
CAP="$WORK/capture_code_${METHOD}.pt"
OUT="$OUT_BASE/redist_code_${METHOD}_62e"

cat <<PLAN
=== redist_code_fold plan (method=$METHOD) ===
  python      : $REDIST_PY
  scripts-dir : $REDIST_SCRIPTS_DIR
  teacher     : ${REDIST_TEACHER:-<unset REDIST_TEACHER>}
  student     : ${REDIST_STUDENT:-<unset REDIST_STUDENT>}
  keep-meta   : ${REDIST_KEEP_META:-<unset REDIST_KEEP_META>}
  sample      : ${REDIST_SAMPLE:-<unset REDIST_SAMPLE>}
  code corpus : ${CORPUS:-<unset REDIST_CALIB_CODE>}
  emit ->     : $OUT   (KEPT for HE+/MPE eval)
  gpu/seqs/tok: $GPU / $MAXSEQ / $MAXTOK
PLAN
[ "$DO_RUN" = 1 ] || { echo "(dry-run: pass --run to execute)"; exit 0; }

: "${REDIST_TEACHER:?set REDIST_TEACHER (128e teacher dir)}"
: "${REDIST_STUDENT:?set REDIST_STUDENT (pruned 62e dir)}"
: "${REDIST_KEEP_META:?set REDIST_KEEP_META (a2 keep metadata json)}"
: "${REDIST_SAMPLE:?set REDIST_SAMPLE (loop_screen jsonl)}"
: "${REDIST_CALIB_CODE:?set REDIST_CALIB_CODE (code capture corpus jsonl)}"
mkdir -p "$RES" "$WORK" "$WORK/logs" "$OUT_BASE"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a "$WORK/logs/redist_code_fold_${METHOD}.log") 2>&1
L(){ echo "[code_$METHOD $(date -u +%H:%M:%S)] $*"; }

L "=== [1/3] capture (driver=code corpus=$(basename "$CORPUS") seqs=$MAXSEQ tok=$MAXTOK) ==="
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/redist.py" capture --driver code --method "$METHOD" \
  --teacher "$REDIST_TEACHER" --corpus "$CORPUS" --keep-meta "$REDIST_KEEP_META" \
  --max-seqs "$MAXSEQ" --max-tokens "$MAXTOK" --device cuda:0 --workdir "$WORK" \
  --scripts-dir "$REDIST_SCRIPTS_DIR" || { L "CAPTURE FAIL"; exit 1; }

L "=== [2/3] $METHOD fit + emit (dropped code-experts -> A2 survivors, KEEP) ==="
rm -rf "$OUT"
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/redist.py" redistribute --method "$METHOD" --driver code \
  --capture "$CAP" --teacher "$REDIST_TEACHER" --student "$REDIST_STUDENT" \
  --keep-meta "$REDIST_KEEP_META" --emit "$OUT" --device cuda:0 \
  --scripts-dir "$REDIST_SCRIPTS_DIR" || { L "FIT/EMIT FAIL"; exit 1; }

L "=== [3/3] loop_screen REGRESSION check (200-prompt, greedy, max-new 2048) ==="
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/loop_screen.py" --model "$OUT" --out "$RES/loop_code_${METHOD}.json" \
  --name "code_${METHOD}" --sample "$REDIST_SAMPLE" --bs 16 --max-new 2048 || { L "SCREEN FAIL"; exit 1; }
"$REDIST_PY" - "$RES/loop_code_${METHOD}.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); bb = d.get("by_bucket", {})
print("[%s] loop_pct=%s loops=%s/%s" % (d["name"], d.get("loop_pct"), d.get("loops"), d.get("n")))
print("  by_bucket: " + "  ".join("%s=%s/%s" % (b, v.get("loops"), v.get("n")) for b, v in sorted(bb.items())))
print("  A2 ANCHOR 15.5%% (constr 6 ML 21 oe 2)  -> regression gate: must not WORSEN")
PYEOF
L "FOLD_DONE method=$METHOD out=$OUT  (model KEPT for HE+/MPE eval)"
