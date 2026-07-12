#!/usr/bin/env bash
# gate_std16_he_mpe.sh — QUEUED deploy-sampler HE+164 / MultiPL-E-100 evals for the STD16
# (v7-coder force-keep) promotion cohort. Runs as the LAST stage, after every loop gate
# (plain cohort + 3 specialty tiers). Per the user rule, only tiers with 0 LOOPS are eligible,
# and the eval sampler is vendor_minp_rep at:
#     loops@t0.9==0 AND loops@t0.8==0  -> eval at t0.9   ("if at both temp they don't loop pick 0.9")
#     loops@t0.9==0 only               -> eval at t0.9
#     loops@t0.8==0 only               -> eval at t0.8
#     loops at BOTH temps              -> NOT eligible (skipped)
# Uses the humanevalplus_full_minprep0{9,8} + multipl_e_100_minprep0{9,8} shadow templates
# (deploy sampler baked in: temp 0.{9,8} / top_p 0.95 / top_k 64 / min_p 0.05 / repeat_penalty 1.1).
# 2 GPUs, atomic-mkdir claim pool. Loop counts come from each tier's 48-seed gate-log arm summary.
# User-skipped tiers (Q3_K_S Q2_K_L Q3_K_XL Q5_K_S Q4_0 Q4_1) are excluded from the candidate set.
set -uo pipefail
BM=/srv/ml
PY="$BM/envs/envs/omnimergekit/bin/python"
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
OMK="$BM/repos/omnimergekit/eval/omk_eval.py"
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
STEM=gemma-4-A4B-98e-v7-coder-it
GATE=/mnt/sdc/ml/std16_gate                 # plain gate logs/ + done/
SPEC=/mnt/sdc/ml/std16_gate/specialty       # specialty gate logs/
CDIQ2NL=/mnt/sdc/ml/std16_gate/cd_iq2nl     # CD-IQ2_NL gate logs/
WORK=/mnt/sdc/ml/std16_gate/he_mpe
mkdir -p "$WORK/results" "$WORK/logs" "$WORK/locks" "$WORK/done"
SUMMARY="$WORK/SUMMARY.tsv"
LOG="$WORK/orchestrator.log"
exec >>"$LOG" 2>&1
ts(){ date -u +%T; }

# Publish-candidate tiers only (skipped tiers excluded per user decision)
PLAIN=(Q3_K_M Q3_K_L IQ4_XS IQ4_NL Q4_K_S Q4_K_M Q4_K_L Q5_K_M Q5_K_L Q6_K Q6_K_L Q8_0)
SPECIAL=(CD-Q2_K qat-Q4_0 CD-qat-Q4_K_M)
CD_IQ2NL=(CD-IQ2_NL CD-IQ2_NL-q2k)
PLAIN_NEED="Q4_K_S Q4_K_M Q4_K_L Q5_K_M Q5_K_L Q6_K_L Q8_0"

echo "==================== STD16 HE+/MPE deploy-sampler orchestrator start $(ts) UTC ===================="

# [A] wait for plain loop gate AND specialty loop gate to finish (no GPU contention)
for i in $(seq 1 1200); do
  miss=""; for T in $PLAIN_NEED; do [ -f "$GATE/done/$T.done" ] || miss="$miss$T "; done
  spec_done=0; grep -q SPECIALTY_GATE_DONE "$SPEC/gate_specialty.log" 2>/dev/null && spec_done=1
  cd_done=0; grep -q CD_IQ2NL_GATE_DONE "$CDIQ2NL/gate_cd_iq2nl.log" 2>/dev/null && cd_done=1
  if [ -z "$miss" ] && [ "$spec_done" = 1 ] && [ "$cd_done" = 1 ]; then echo "[$(ts)] all loop gates done — proceeding"; break; fi
  sleep 30
done

# [B] classify each candidate tier from its 48-seed gate log -> eligible + eval temp
loops_of(){ # tier  temp(09|08)  logfile   -> echoes loop count (99 if unknown)
  local n
  n=$(grep -E "minp_t0\.${2#0}  *fails=" "$3" 2>/dev/null | grep -oE "loops=[0-9]+" | head -1 | grep -oE "[0-9]+")
  echo "${n:-99}"
}
declare -A TEMP_OF
ELIG=()
classify(){ # tier  logfile
  local T="$1" LF="$2" l9 l8
  l9=$(loops_of "$T" 09 "$LF"); l8=$(loops_of "$T" 08 "$LF")
  if   [ "$l9" = 0 ] && [ "$l8" = 0 ]; then TEMP_OF[$T]=09; ELIG+=("$T"); echo "[$(ts)] $T eligible -> t0.9 (loops 0/0)"
  elif [ "$l9" = 0 ];                  then TEMP_OF[$T]=09; ELIG+=("$T"); echo "[$(ts)] $T eligible -> t0.9 (loops t0.9=0 t0.8=$l8)"
  elif [ "$l8" = 0 ];                  then TEMP_OF[$T]=08; ELIG+=("$T"); echo "[$(ts)] $T eligible -> t0.8 (loops t0.9=$l9 t0.8=0)"
  else echo "[$(ts)] $T NOT eligible (loops t0.9=$l9 t0.8=$l8)"; fi
}
for T in "${PLAIN[@]}";   do classify "$T" "$GATE/logs/$T.log"; done
for T in "${SPECIAL[@]}"; do classify "$T" "$SPEC/logs/$T.log"; done
for T in "${CD_IQ2NL[@]}"; do classify "$T" "$CDIQ2NL/logs/$T.log"; done
echo "[$(ts)] ELIGIBLE (${#ELIG[@]}): ${ELIG[*]:-<none>}"
[ "${#ELIG[@]}" -eq 0 ] && { echo "[$(ts)] nothing eligible — exit"; echo "HE_MPE_ORCH_DONE"; exit 0; }

# [C] run HE+ + MPE per eligible tier across 2 GPUs (atomic-mkdir claim pool, resumable)
rm -rf "$WORK/locks"/*.lock 2>/dev/null   # clear stale locks (resume); per-bench .done still skips finished work
run_tier(){ # tier  gpu  port
  local T="$1" G="$2" P="$3" TP="${TEMP_OF[$T]}" gguf="$GG/$STEM-$T.gguf"
  [ -s "$gguf" ] || { echo "[$(ts)] SKIP $T (no gguf)"; return 0; }
  for BENCH in "humanevalplus_full_minprep$TP" "multipl_e_100_minprep$TP"; do
    [ -f "$WORK/done/${T}__${BENCH}.done" ] && { echo "[$(ts)] $T $BENCH already done"; continue; }
    local TD="$WORK/results/$T/$BENCH"
    echo "[$(ts)] [GPU$G:$P] $T $BENCH (t0.${TP#0}) start"
    CUDA_VISIBLE_DEVICES=$G "$PY" "$OMK" --model "$gguf" --template "$BENCH" \
      --backend llama --quant gguf --port "$P" --results-dir "$TD" --served-name "STD16-$T" \
      > "$WORK/logs/${T}__${BENCH}.log" 2>&1 || echo "[$(ts)] $T $BENCH rc=$?"
    touch "$WORK/done/${T}__${BENCH}.done"
    echo "[$(ts)] [GPU$G:$P] $T $BENCH done"
  done
}
worker(){ # gpu  port
  local G="$1" P="$2" T
  for T in "${ELIG[@]}"; do
    mkdir "$WORK/locks/$T.lock" 2>/dev/null || continue
    run_tier "$T" "$G" "$P"
  done
}
worker 0 8260 &
worker 1 8261 &
wait

# [D] collect canonical scores (summary.json .score — NEVER raw results_*.json)
echo "[$(ts)] ==================== orchestrator eval phase DONE — collecting scores ===================="
"$PY" - "$WORK" "${ELIG[@]}" <<'PYEOF' > "$SUMMARY"
import json, glob, sys, os
work = sys.argv[1]; tiers = sys.argv[2:]
def pick(d):
    fs = sorted(glob.glob(os.path.join(d, "**", "summary.json"), recursive=True), key=len)
    for f in fs:
        try:
            j = json.load(open(f))
            if isinstance(j, dict) and j.get("score") is not None:
                return j
        except Exception:
            pass
    return None
print("tier\ttemp\tHE+\tMPE100\tHE+_metric\tMPE_metric")
for T in tiers:
    rows = {}
    for tag, pat in (("HE+", "humanevalplus_full_minprep*"), ("MPE", "multipl_e_100_minprep*")):
        ds = glob.glob(os.path.join(work, "results", T, pat))
        j = pick(ds[0]) if ds else None
        rows[tag] = j
    he = rows["HE+"]; mpe = rows["MPE"]
    tp = "?"
    for j in (he, mpe):
        if j and j.get("sampler", {}).get("name"):
            tp = j["sampler"]["name"]
    def sc(j): return f"{round(j['score']*100,2)}" if j else "NA"
    def mt(j): return (j.get("metric","") + ("/"+j.get("filter","") if j.get("filter") else "")) if j else ""
    print(f"{T}\t{tp}\t{sc(he)}\t{sc(mpe)}\t{mt(he)}\t{mt(mpe)}")
PYEOF
echo "[$(ts)] ==================== STD16 HE+/MPE orchestrator DONE ===================="
echo "HE_MPE_ORCH_DONE"
echo "--- SUMMARY (deploy sampler vendor_minp_rep) ---"; cat "$SUMMARY"
