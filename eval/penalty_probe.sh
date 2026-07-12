#!/usr/bin/env bash
# T201-code generalized penalty probe: run the {none, min_p 0.05, repeat_penalty
# 1.1, both} penalty matrix for ANY code bench at the gemma vendor sampler, on
# v7-coder Q4_K_M. Reused for MPE (multipl_e_sampler_probe) and LCB
# (lcb_sampler_probe). The sampler lives in the shadow TEMPLATE's generation
# block; this driver varies ONLY the penalty axis via --metadata
# backend_args.llama_extra (min_p/repeat_penalty are llama-server launch flags —
# lm-eval / the MPE generator never send them; the deployment topology).
#
# Required env: TEMPLATE (shadow template name), TAGPFX (cell-tag prefix).
# Optional:    PROBE (results root), MODEL, QUANT, LIMIT, LANE.
#   LANE=all (default) GPU1:8167 both cells of each lane + preflight
#   LANE=A   GPU0:8157 {<pfx>_none <pfx>_rep}
#   LANE=B   GPU1:8167 {<pfx>_minp <pfx>_both}
#   LANE=summary       print the score table
# Idempotent: a cell whose summary.json already has a numeric score is SKIPPED.
set -uo pipefail

cd /shared/dev/omnimergekit
PY=/root/anaconda3/envs/omnimergekit/bin/python
export PATH="/root/anaconda3/envs/omnimergekit/bin:$PATH"
export HF_ALLOW_CODE_EVAL=1
TOK=/srv/ml/google/gemma-4-26B-A4B-it
MODEL=${MODEL:-/mnt/sdc/ml/quant_sweep_gguf/gemma-4-A4B-98e-v7-coder-it-Q4_K_M.gguf}
QUANT=${QUANT:-q4_k_m}
TEMPLATE=${TEMPLATE:?TEMPLATE must be set (shadow template name)}
TAGPFX=${TAGPFX:?TAGPFX must be set (cell-tag prefix)}
PROBE=${PROBE:-/mnt/sdc/ml/penalty_probe_t201/$TAGPFX}
LIMIT=${LIMIT:-0}
LANE=${LANE:-all}
mkdir -p "$PROBE"

NONE='["--min-p","0.0"]'
MINP='["--min-p","0.05"]'
REP='["--min-p","0.0","--repeat-penalty","1.1"]'
BOTH='["--min-p","0.05","--repeat-penalty","1.1"]'

# tag | extra
CELLS=(
  "${TAGPFX}_none|$NONE"
  "${TAGPFX}_minp|$MINP"
  "${TAGPFX}_rep|$REP"
  "${TAGPFX}_both|$BOTH"
)
LANE_A="${TAGPFX}_none ${TAGPFX}_rep"
LANE_B="${TAGPFX}_minp ${TAGPFX}_both"

print_summary() {
  $PY - "$PROBE" "$TAGPFX" <<'PY'
import json, sys, glob, os
probe, pfx = sys.argv[1], sys.argv[2]
order = [f"{pfx}_none", f"{pfx}_minp", f"{pfx}_rep", f"{pfx}_both"]
print(f"\n===== T201-code PENALTY PROBE [{pfx}] — gemma sampler, v7-coder Q4_K_M =====")
print(f"{'cell':14s} {'score':>7s} {'cp50':>6s} {'cp90':>6s} {'cmax':>7s} {'empty':>5s}")
for c in order:
    sj = glob.glob(os.path.join(probe, c, "**", "summary.json"), recursive=True)
    if not sj:
        print(f"{c:14s} {'PENDING':>7s}"); continue
    j = json.load(open(sj[0])); ts = j.get("token_stats") or {}; ct = ts.get("completion_tokens") or {}
    sc = j.get("score"); sc = f"{sc:.4f}" if isinstance(sc, float) else str(sc)
    print(f"{c:14s} {sc:>7s} {str(ct.get('p50')):>6s} {str(ct.get('p90')):>6s} {str(ct.get('max')):>7s} {str(ts.get('empty_completions')):>5s}")
PY
}

if [ "$LANE" = "summary" ]; then print_summary; exit 0; fi

: "${HF_TOKEN:?HF_TOKEN must be exported}"

case "$LANE" in
  A)   GPU=0; PORT=8157; WANT="$LANE_A";;
  B)   GPU=1; PORT=8167; WANT="$LANE_B";;
  all) GPU=1; PORT=8167; WANT="";;
  *)   echo "bad LANE=$LANE"; exit 2;;
esac

cell_done() {
  $PY - "$PROBE/$1" <<'PY' >/dev/null 2>&1
import json,sys,glob,os
sj=glob.glob(os.path.join(sys.argv[1],"**","summary.json"),recursive=True)
sys.exit(0 if (sj and isinstance(json.load(open(sj[0])).get("score"),(int,float))) else 1)
PY
}

run_cell() {  # tag extra [limit]
  local tag="$1" extra="$2" lim="${3:-$LIMIT}"
  if cell_done "$tag"; then echo "[cell $tag] SKIP (already scored)"; return 0; fi
  local rd="$PROBE/$tag"
  echo "===== [cell $tag] GPU=$GPU port=$PORT extra=$extra limit=$lim ($(date +%T)) ====="
  $PY eval/omk_eval.py --template "$TEMPLATE" --backend llama \
    --model "$MODEL" --quant "$QUANT" --tokenizer "$TOK" \
    --served-name "$tag" --port "$PORT" --limit "$lim" \
    --gpus "$GPU" --parallel 2 --results-dir "$rd" \
    --metadata "backend_args.llama_extra=$extra" \
    > "$PROBE/${tag}.log" 2>&1
  echo "[cell $tag] omk exit=$? ($(date +%T))"
}

if [ "$LANE" = "all" ]; then
  echo "[penalty-probe $TAGPFX] PREFLIGHT: ${TAGPFX}_minp @ ${PRE_LIMIT:-6}q ($(date +%T))"
  run_cell _preflight "$MINP" "${PRE_LIMIT:-6}"
  if ! cell_done _preflight; then
    echo "[penalty-probe $TAGPFX] PREFLIGHT FAILED — see $PROBE/_preflight.log"; exit 1
  fi
  echo "[penalty-probe $TAGPFX] PREFLIGHT PASSED ($(date +%T))"
fi

echo "[penalty-probe $TAGPFX LANE=$LANE] GPU=$GPU port=$PORT cells=[${WANT:-ALL}] limit=$LIMIT ($(date +%T))"
for row in "${CELLS[@]}"; do
  IFS='|' read -r tag extra <<<"$row"
  if [ -n "$WANT" ]; then case " $WANT " in *" $tag "*) : ;; *) continue;; esac; fi
  run_cell "$tag" "$extra"
done
echo "[penalty-probe $TAGPFX LANE=$LANE] LANE DONE ($(date +%T))"

if [ "$LANE" = "all" ]; then print_summary; fi
