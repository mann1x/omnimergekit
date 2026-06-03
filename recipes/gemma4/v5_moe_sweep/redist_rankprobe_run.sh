#!/usr/bin/env bash
# T191 E-RankProbe orchestrator (the LEAD diffuse-multilingual capacity test):
#   capture(expert_kd, router_in tap; DISJOINT multilingual calib) -> single-layer
#   trainable rank probe -> held-out block-output divergence verdict.
# The capture corpus MUST be disjoint from any test the verdict is judged against;
# this is a CAPACITY probe (held-out divergence), not a loop_screen.
#
# Usage: redist_rankprobe_run.sh [--run] [gpu] [layers] [max_seqs]
#   Dry-run by default (prints the plan, exits 0). Pass --run to execute.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$SCRIPT_DIR/redist_config.sh" ] && . "$SCRIPT_DIR/redist_config.sh"
DO_RUN=0; ARGS=()
for a in "$@"; do [ "$a" = "--run" ] && DO_RUN=1 || ARGS+=("$a"); done
set -- "${ARGS[@]+"${ARGS[@]}"}"

GPU="${1:-0}"; LAYERS="${2:-5,12,18,25}"; MAXSEQ="${3:-120}"
REDIST_PY="${REDIST_PY:-python}"
REDIST_SCRIPTS_DIR="${REDIST_SCRIPTS_DIR:-$REPO_ROOT/scripts}"
WORK="${REDIST_WORK:-$PWD/redist_work}"
CORPUS="${REDIST_CALIB_MULTILINGUAL:-}"
CAP="$WORK/capture_multilingual_expert_kd.pt"

cat <<PLAN
=== redist_rankprobe_run plan ===
  python      : $REDIST_PY
  scripts-dir : $REDIST_SCRIPTS_DIR
  teacher     : ${REDIST_TEACHER:-<unset REDIST_TEACHER>}
  student     : ${REDIST_STUDENT:-<unset REDIST_STUDENT>}
  keep-meta   : ${REDIST_KEEP_META:-<unset REDIST_KEEP_META>}
  ml corpus   : ${CORPUS:-<unset REDIST_CALIB_MULTILINGUAL>}
  workdir     : $WORK
  gpu/layers/seqs: $GPU / $LAYERS / $MAXSEQ
PLAN
[ "$DO_RUN" = 1 ] || { echo "(dry-run: pass --run to execute)"; exit 0; }

: "${REDIST_TEACHER:?set REDIST_TEACHER (128e teacher dir)}"
: "${REDIST_STUDENT:?set REDIST_STUDENT (pruned 62e dir)}"
: "${REDIST_KEEP_META:?set REDIST_KEEP_META (a2 keep metadata json)}"
: "${REDIST_CALIB_MULTILINGUAL:?set REDIST_CALIB_MULTILINGUAL (rankprobe corpus jsonl)}"
mkdir -p "$WORK" "$WORK/logs"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a "$WORK/logs/redist_rankprobe.log") 2>&1
L(){ echo "[rankprobe-run $(date -u +%H:%M:%S)] $*"; }

L "=== [1/2] capture (expert_kd: router_in+swiglu_in+block_out; corpus=$(basename "$CORPUS") seqs=$MAXSEQ) ==="
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/redist.py" capture --driver multilingual --method expert_kd \
  --teacher "$REDIST_TEACHER" --corpus "$CORPUS" --keep-meta "$REDIST_KEEP_META" \
  --max-seqs "$MAXSEQ" --max-tokens 512 --device cuda:0 --workdir "$WORK" \
  --scripts-dir "$REDIST_SCRIPTS_DIR" || { L "CAPTURE FAIL"; exit 1; }

L "=== [2/2] rank probe (layers=$LAYERS, 400 steps, held-out tail 20%) ==="
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/redist_rank_probe.py" --student "$REDIST_STUDENT" --capture "$CAP" \
  --layers "$LAYERS" --steps 400 --lr 1e-3 --heldout 0.2 \
  --device cuda:0 --out "$WORK/rankprobe_multilingual.json" || { L "PROBE FAIL"; exit 1; }
L "RANKPROBE_RUN_DONE"
