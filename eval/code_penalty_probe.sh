#!/usr/bin/env bash
# T201-code — CODE-side confirmation of the T201 GPQA finding before any public
# card edit: does min_p 0.05 HELP and repeat_penalty 1.1 HARM at the deployment
# gemma vendor sampler, on the canonical Python code bench (HE+ pass@1, 164q)?
#
# v7-coder Q4_K_M, gemma sampler (temp 1.0 / top_p 0.95 / top_k 64), serving
# canonical HE+ (--jinja --reasoning off). Four penalty cells:
#   hep_none  : --min-p 0.0
#   hep_minp  : --min-p 0.05
#   hep_rep   : --min-p 0.0  --repeat-penalty 1.1
#   hep_both  : --min-p 0.05 --repeat-penalty 1.1
# Shadow template humanevalplus_sampler_probe (NEVER the frozen humanevalplus_full).
# Penalties are llama-server launch flags (lm-eval never sends them — deployment
# topology). Scored from summary.json (.score). Idempotent: a cell whose
# summary.json already has a numeric score is SKIPPED.
#
# LANE env splits across both idle bs2 GPUs:
#   LANE=all (default) GPU1:8167 all 4 cells, with preflight
#   LANE=A   GPU0:8157 {hep_none hep_rep}
#   LANE=B   GPU1:8167 {hep_minp hep_both}
#   LANE=summary       print the score table
set -uo pipefail

cd /shared/dev/omnimergekit
PY=/root/anaconda3/envs/omnimergekit/bin/python
export PATH="/root/anaconda3/envs/omnimergekit/bin:$PATH"   # bare `lm-eval` resolves
export HF_ALLOW_CODE_EVAL=1                                  # HE+ exec scorer
TOK=/srv/ml/google/gemma-4-26B-A4B-it
V7Q4=/mnt/sdc/ml/quant_sweep_gguf/gemma-4-A4B-98e-v7-coder-it-Q4_K_M.gguf
PROBE=${PROBE:-/mnt/sdc/ml/code_penalty_probe_t201}
LIMIT=${LIMIT:-0}     # 0 = full 164; >0 = smoke
LANE=${LANE:-all}
mkdir -p "$PROBE"

NONE='["--min-p","0.0"]'
MINP='["--min-p","0.05"]'
REP='["--min-p","0.0","--repeat-penalty","1.1"]'
BOTH='["--min-p","0.05","--repeat-penalty","1.1"]'

# tag | extra
CELLS=(
  "hep_none|$NONE"
  "hep_minp|$MINP"
  "hep_rep|$REP"
  "hep_both|$BOTH"
)
LANE_A="hep_none hep_rep"
LANE_B="hep_minp hep_both"

print_summary() {
  $PY - "$PROBE" <<'PY'
import json, sys, glob, os
probe = sys.argv[1]
order = ["hep_none", "hep_minp", "hep_rep", "hep_both"]
print("\n===== T201-code HE+ PENALTY PROBE — gemma sampler, v7-coder Q4_K_M =====")
print(f"{'cell':12s} {'score':>7s} {'cp50':>6s} {'cp90':>6s} {'cmax':>7s} {'empty':>5s}")
for c in order:
    sj = glob.glob(os.path.join(probe, c, "**", "summary.json"), recursive=True)
    if not sj:
        print(f"{c:12s} {'PENDING':>7s}"); continue
    j = json.load(open(sj[0])); ts = j.get("token_stats") or {}; ct = ts.get("completion_tokens") or {}
    sc = j.get("score"); sc = f"{sc:.4f}" if isinstance(sc, float) else str(sc)
    print(f"{c:12s} {sc:>7s} {str(ct.get('p50')):>6s} {str(ct.get('p90')):>6s} {str(ct.get('max')):>7s} {str(ts.get('empty_completions')):>5s}")
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
  $PY eval/omk_eval.py --template humanevalplus_sampler_probe --backend llama \
    --model "$V7Q4" --quant q4_k_m --tokenizer "$TOK" \
    --served-name "$tag" --port "$PORT" --limit "$lim" \
    --gpus "$GPU" --parallel 2 --results-dir "$rd" \
    --metadata "backend_args.llama_extra=$extra" \
    > "$PROBE/${tag}.log" 2>&1
  echo "[cell $tag] omk exit=$? ($(date +%T))"
}

if [ "$LANE" = "all" ]; then
  echo "[code-probe] PREFLIGHT: hep_minp @ 6q ($(date +%T))"
  run_cell _preflight "$MINP" 6
  if ! cell_done _preflight; then
    echo "[code-probe] PREFLIGHT FAILED — see $PROBE/_preflight.log"; exit 1
  fi
  echo "[code-probe] PREFLIGHT PASSED ($(date +%T))"
fi

echo "[code-probe LANE=$LANE] GPU=$GPU port=$PORT cells=[${WANT:-ALL}] limit=$LIMIT ($(date +%T))"
for row in "${CELLS[@]}"; do
  IFS='|' read -r tag extra <<<"$row"
  if [ -n "$WANT" ]; then case " $WANT " in *" $tag "*) : ;; *) continue;; esac; fi
  run_cell "$tag" "$extra"
done
echo "[code-probe LANE=$LANE] LANE DONE ($(date +%T))"

if [ "$LANE" = "all" ]; then print_summary; fi
