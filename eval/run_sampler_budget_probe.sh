#!/usr/bin/env bash
# T201 — sampler × budget × penalty × thinking disentangling probe on GPQA-diamond.
#
# Answers: (1) was --reasoning-budget's necessity a GREEDY artifact (greedy vs
# gemma t=1.0 × budget 8192/off, v7-Q4KM + 128e-Q6K); (2) do min_p0.05 /
# repeat_penalty1.1 help or HARM at the vendor sampler (v7 gemma+bud none/minp/
# rep/both); (3) thinking-off none/rep premature-stop check. Scored from
# summary.json. Shadow template gpqa_sampler_probe (NEVER the frozen
# gpqa_diamond_full). Sampler via --metadata; min_p/repeat_penalty as llama
# server flags (lm-eval never sends them — the deployment topology).
#
# LANE env selects GPU + port + cell subset so the run can be split across both
# bs2 GPUs (GPU0 freed 2026-06-15):
#   LANE=all (default) — GPU1:8137, all 13 cells, with preflight gate
#   LANE=A             — GPU1:8137, lane-A subset (no preflight)
#   LANE=B             — GPU0:8147, lane-B subset (no preflight)
#   LANE=summary       — just print the score table over all cells
# Every cell is idempotent: a cell whose summary.json already has a numeric
# score is SKIPPED, and a half-finished cell resumes from its sqlite cache.
set -uo pipefail

cd /shared/dev/omnimergekit
PY=/root/anaconda3/envs/omnimergekit/bin/python
# omk_eval shells out to a bare `lm-eval`; put the omk env bin on PATH so it resolves.
export PATH="/root/anaconda3/envs/omnimergekit/bin:$PATH"
TOK=/srv/ml/google/gemma-4-26B-A4B-it            # original 128e tokenizer (both models)
V7Q4=/mnt/sdc/ml/quant_sweep_gguf/gemma-4-A4B-98e-v7-coder-it-Q4_K_M.gguf
M128=/mnt/sdc/ml/eval_gguf/128e-Q6_K.gguf
PROBE=/mnt/sdc/ml/sampler_probe_t201
LIMIT=${LIMIT:-40}
LANE=${LANE:-all}
mkdir -p "$PROBE"

CLEAN='["--min-p","0.0"]'
MINP='["--min-p","0.05"]'
REP='["--min-p","0.0","--repeat-penalty","1.1"]'
BOTH='["--min-p","0.05","--repeat-penalty","1.1"]'

# tag | model | quant | temp | top_p | top_k | budget | extra
CELLS=(
  "128e_greedy_bud|$M128|q6_k|0.0|1.0|0|8192|$CLEAN"
  "128e_greedy_nobud|$M128|q6_k|0.0|1.0|0|-1|$CLEAN"
  "128e_gemma_bud|$M128|q6_k|1.0|0.95|64|8192|$CLEAN"
  "128e_gemma_nobud|$M128|q6_k|1.0|0.95|64|-1|$CLEAN"
  "v7_greedy_bud|$V7Q4|q4_k_m|0.0|1.0|0|8192|$CLEAN"
  "v7_greedy_nobud|$V7Q4|q4_k_m|0.0|1.0|0|-1|$CLEAN"
  "v7_gemma_bud|$V7Q4|q4_k_m|1.0|0.95|64|8192|$CLEAN"
  "v7_gemma_nobud|$V7Q4|q4_k_m|1.0|0.95|64|-1|$CLEAN"
  "v7_gemma_bud_minp|$V7Q4|q4_k_m|1.0|0.95|64|8192|$MINP"
  "v7_gemma_bud_rep|$V7Q4|q4_k_m|1.0|0.95|64|8192|$REP"
  "v7_gemma_bud_both|$V7Q4|q4_k_m|1.0|0.95|64|8192|$BOTH"
  "v7_gemma_nothink_none|$V7Q4|q4_k_m|1.0|0.95|64|0|$CLEAN"
  "v7_gemma_nothink_rep|$V7Q4|q4_k_m|1.0|0.95|64|0|$REP"
)
# 128e_greedy_bud already complete; lanes split the remaining 12, balancing the
# slow budget-off greedy/gemma cells one per lane.
LANE_A="128e_greedy_nobud 128e_gemma_bud 128e_gemma_nobud v7_gemma_bud v7_gemma_bud_rep v7_gemma_nothink_none"
LANE_B="v7_greedy_bud v7_greedy_nobud v7_gemma_nobud v7_gemma_bud_minp v7_gemma_bud_both v7_gemma_nothink_rep"

print_summary() {
  $PY - "$PROBE" <<'PY'
import json, sys, glob, os
probe = sys.argv[1]
rows = []
for cell in sorted(os.listdir(probe)):
    d = os.path.join(probe, cell)
    if not os.path.isdir(d) or cell == "_preflight":
        continue
    sj = glob.glob(os.path.join(d, "**", "summary.json"), recursive=True)
    if not sj:
        rows.append((cell, None, None, None)); continue
    try:
        j = json.load(open(sj[0]))
    except Exception as e:
        rows.append((cell, f"ERR:{e}", None, None)); continue
    ts = j.get("token_stats") or {}
    p50 = ts.get("completion_p50") or ts.get("p50") or ts.get("think_p50")
    rows.append((cell, j.get("score"), f"{j.get('metric')}/{j.get('filter')}", p50))
print("\n===== T201 SAMPLER×BUDGET PROBE — GPQA-diamond =====")
print(f"{'cell':26s} {'score':>7s} {'metric/filter':>22s} {'tok_p50':>8s}")
for cell, score, mf, p50 in rows:
    ss = f"{score:.4f}" if isinstance(score, float) else str(score)
    print(f"{cell:26s} {ss:>7s} {str(mf):>22s} {str(p50) if p50 is not None else '-':>8s}")
PY
}

if [ "$LANE" = "summary" ]; then print_summary; exit 0; fi

: "${HF_TOKEN:?HF_TOKEN must be exported (GPQA is gated)}"

case "$LANE" in
  A)   GPU=1; PORT=8137; WANT="$LANE_A";;
  B)   GPU=0; PORT=8147; WANT="$LANE_B";;
  all) GPU=1; PORT=8137; WANT="";;
  *)   echo "bad LANE=$LANE"; exit 2;;
esac

cell_done() {  # tag -> 0 if summary.json has a numeric score
  $PY - "$PROBE/$1" <<'PY' >/dev/null 2>&1
import json,sys,glob,os
sj=glob.glob(os.path.join(sys.argv[1],"**","summary.json"),recursive=True)
sys.exit(0 if (sj and isinstance(json.load(open(sj[0])).get("score"),(int,float))) else 1)
PY
}

run_cell() {  # tag model quant temp top_p top_k budget extra [limit]
  local tag="$1" model="$2" quant="$3" t="$4" p="$5" k="$6" bud="$7" extra="$8" lim="${9:-$LIMIT}"
  if cell_done "$tag"; then echo "[cell $tag] SKIP (already scored)"; return 0; fi
  local rd="$PROBE/$tag"
  echo "===== [cell $tag] GPU=$GPU port=$PORT t=$t p=$p k=$k budget=$bud extra=$extra limit=$lim ($(date +%T)) ====="
  $PY eval/omk_eval.py --template gpqa_sampler_probe --backend llama \
    --model "$model" --quant "$quant" --tokenizer "$TOK" \
    --served-name "$tag" --port "$PORT" --limit "$lim" \
    --gpus "$GPU" --parallel 2 --results-dir "$rd" \
    --metadata generation.temperature=$t \
    --metadata generation.top_p=$p \
    --metadata generation.top_k=$k \
    --metadata generation.thinking_token_budget=$bud \
    --metadata generation.max_gen_toks=24576 \
    --metadata "backend_args.llama_extra=$extra" \
    > "$PROBE/${tag}.log" 2>&1
  echo "[cell $tag] omk exit=$? ($(date +%T))"
}

# Preflight only in monolithic mode (lanes are post-validation).
if [ "$LANE" = "all" ]; then
  echo "[probe] PREFLIGHT: v7 gemma+bud @ 2q ($(date +%T))"
  run_cell _preflight "$V7Q4" q4_k_m 1.0 0.95 64 8192 "$CLEAN" 2
  if ! cell_done _preflight; then
    echo "[probe] PREFLIGHT FAILED — aborting. See $PROBE/_preflight.log"; exit 1
  fi
  echo "[probe] PREFLIGHT PASSED ($(date +%T))"
fi

echo "[probe LANE=$LANE] GPU=$GPU port=$PORT cells=[${WANT:-ALL}] limit=$LIMIT ($(date +%T))"
for row in "${CELLS[@]}"; do
  IFS='|' read -r tag model quant t p k bud extra <<<"$row"
  if [ -n "$WANT" ]; then case " $WANT " in *" $tag "*) : ;; *) continue;; esac; fi
  run_cell "$tag" "$model" "$quant" "$t" "$p" "$k" "$bud" "$extra"
done
echo "[probe LANE=$LANE] LANE DONE ($(date +%T))"

# In monolithic mode, print the table. In lane mode, the operator runs
# `LANE=summary bash run_sampler_budget_probe.sh` once both lanes finish.
if [ "$LANE" = "all" ]; then print_summary; fi
