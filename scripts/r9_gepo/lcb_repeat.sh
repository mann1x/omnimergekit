#!/usr/bin/env bash
# R9.rpt -- repeat sampled LCB draw, qwen_suite basis, to measure the
# within-arm draw-noise floor.
#
# WHY: gepo2's sampled LCB cell sits -9.33pp below armJ on the
# both-untruncated set with an 11-vs-2 sampler-sensitivity asymmetry
# (p=0.0225). There is NO repeat draw anywhere in the cohort, so the
# noise floor of a single 77-problem sampled LCB draw is UNMEASURED.
# Without it the asymmetry cannot be told apart from one unlucky draw.
#
# The LCB runner sends NO seed (verified by grep in lcb_llama_server.py) --
# unlike the lm-eval path, whose payload carries a hardcoded seed=1234. So
# repeat draws on this bench are genuinely independent.
#
# BASIS: identical to the banked qwen_suite cells -- sampler=recommended
# (temp 0.6 / top_p 0.95 / top_k 20), max_gen_toks 32768, per-slot n_ctx
# 45056, n_slots 2. NEW served-names (_rpt2) so no banked cell is ever
# overwritten and the sqlite resume cache (keyed by task_id under
# <out_dir>/sqlite_cache) starts empty instead of replaying draw 1 verbatim.
set -uo pipefail

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/qwen_suite
BENCH=lcb_v6_77q
PORT=8099
PER_SLOT=45056
PAR=2
TOTAL=$(( PER_SLOT * PAR ))
PROFILE=qwen3_6
SAMPLER=recommended
LOG=/mnt/sdc/ml/brevity/gepo/lcb_repeat.log

export CUDA_VISIBLE_DEVICES=1
unset LLAMA_EXTRA
unset LLAMA_ARG_SPEC_TYPE

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# cells: served_name|gguf|tokenizer|reference_cell
CELLS=(
  "qwena3bgepo2_q6k_rpt2|/mnt/sdc/ml/brevity/gepo/gguf_gepo2/a3b-gepo2-Q6_K.gguf|/mnt/sdc/ml/brevity/gepo/a3b-gepo2|qwena3bgepo2_q6k"
)

jf() {  # jf <file> <dotted.path>
  python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    d = (d or {}).get(k) if isinstance(d,dict) else None
print(d)
' "$1" "$2" 2>/dev/null
}

say "=== R9.rpt LCB repeat draw :: bench=$BENCH per_slot=$PER_SLOT par=$PAR total=$TOTAL ==="

# ---------- GATE 0: host capacity ----------
root_avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_avail" -lt 200 ]; then
  say "REFUSE: root fs ${root_avail}G < 200G floor"; exit 2
fi
gpu1_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -dc '0-9')
if [ "${gpu1_mem:-9999}" -gt 500 ]; then
  say "REFUSE: GPU1 busy (${gpu1_mem} MiB used) -- not stealing it"; exit 2
fi
say "GATE 0 OK: root ${root_avail}G free, GPU1 ${gpu1_mem} MiB used"

for spec in "${CELLS[@]}"; do
  IFS='|' read -r NAME GGUF TOK REF <<< "$spec"
  out="$RES/$BENCH/$NAME"
  refsum="$RES/$BENCH/$REF/summary.json"

  say "--- cell $NAME (repeat of $REF) ---"

  # ---------- GATE 1: never overwrite, never resume a stale cache ----------
  if [ -e "$out/summary.json" ]; then
    say "REFUSE: $out/summary.json exists -- results are sacred, not overwriting"; exit 3
  fi
  if [ -d "$out/sqlite_cache" ]; then
    say "REFUSE: $out/sqlite_cache exists -- a warm cache would replay draw 1 verbatim"; exit 3
  fi

  # ---------- GATE 2: artifacts ----------
  [ -f "$GGUF" ] || { say "REFUSE: missing GGUF $GGUF"; exit 3; }
  [ -d "$TOK" ]  || { say "REFUSE: missing tokenizer $TOK"; exit 3; }

  # ---------- GATE 3: comparator basis (sampler + cap) must match intent ----------
  [ -f "$refsum" ] || { say "REFUSE: no reference summary $refsum"; exit 3; }
  ref_sampler=$(jf "$refsum" sampler.name)
  ref_cap=$(jf "$refsum" generation_caps.max_gen_toks)
  if [ "$ref_sampler" != "$SAMPLER" ]; then
    say "REFUSE: reference $REF sampler=$ref_sampler want=$SAMPLER"; exit 3
  fi
  say "GATE 3 OK: reference $REF sampler=$ref_sampler max_gen_toks=$ref_cap"

  # ---------- run ----------
  say "LAUNCH $NAME  gguf=$(basename "$GGUF")"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$BENCH" --quant q6_k \
      --model "$GGUF" --tokenizer "$TOK" --served-name "$NAME" --port "$PORT" \
      --results-dir "$RES" --parallel "$PAR" \
      --sampler-profile "$PROFILE" --sampler "$SAMPLER" \
      --metadata backend_args.llama_ctx=$TOTAL >>"$LOG" 2>&1
  rc=$?
  say "omk_eval rc=$rc"

  sum="$out/summary.json"
  if [ ! -f "$sum" ]; then
    say "FAIL: no summary.json banked for $NAME"; exit 4
  fi

  # ---------- POST GATES: geometry, sampler, cap ----------
  got_ctx=$(grep -aoE "n_ctx[ ]*=[ ]*[0-9]+" "$out/server.log" 2>/dev/null | tail -1 | tr -dc '0-9')
  got_slots=$(grep -aoE "n_slots[ ]*=[ ]*[0-9]+" "$out/server.log" 2>/dev/null | tail -1 | tr -dc '0-9')
  got_sampler=$(jf "$sum" sampler.name)
  got_cap=$(jf "$sum" generation_caps.max_gen_toks)
  got_score=$(jf "$sum" score)
  say "READBACK $NAME: n_ctx=$got_ctx n_slots=$got_slots sampler=$got_sampler cap=$got_cap score=$got_score"

  bad=0
  [ "$got_sampler" = "$SAMPLER" ] || { say "GATE-FAIL sampler=$got_sampler want=$SAMPLER"; bad=1; }
  [ "$got_cap" = "$ref_cap" ]     || { say "GATE-FAIL cap=$got_cap want=$ref_cap"; bad=1; }
  [ "$got_slots" = "$PAR" ]       || { say "GATE-FAIL n_slots=$got_slots want=$PAR"; bad=1; }
  if [ "$bad" = 1 ]; then say "R9RPT_BASIS_FAIL $NAME"; exit 5; fi
  say "GATES PASS $NAME"
done

say "R9RPT_DONE"
