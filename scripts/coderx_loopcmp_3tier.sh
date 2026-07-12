#!/usr/bin/env bash
# coderx_loopcmp_3tier.sh — LOOP-TRUST comparison for the v7-coderx re-release.
#
# Builds code4/lcb3 (CX16c4l3) Q4_K_M / Q3_K_S / Q2_K_L from the bf16 reusing the
# PRESERVED imatrix (pre-placed so quantize_gguf does NOT recompute -> pure-CPU, no
# GPU contention with the running SACRED Q6 9-bench), then b9700 48-seed loop-gates
# each at vendor_minp_rep x {t0.9, t0.8} and compares per-temp vs the v7-coder = STD16
# baseline anchors already on disk. The 3 tiers are the discriminating set:
#   Q4_K_M  = clean control + the HF-complaint tier  (STD16: t0.9 0/0  t0.8 0/0)
#   Q3_K_S  = heavy looper                            (STD16: t0.9 19(18) t0.8 16(15))
#   Q2_K_L  = heaviest looper                         (STD16: t0.9 34(34) t0.8 35(34))
# MATCH  -> coderx inherits STD16's per-tier loop/temp guidance (skip full sweep).
# DIVERGE-> full 48-seed both-temp loop gate on EVERY coderx tier.
#
# Binary discipline: STD16 baseline + coderx-Q6 + the whole cohort used b9700, so we
# MUST gate with gate_sweep48_minp_p_b9700.sh (loop behaviour is binary-sensitive).
# PID/flock only — never pkill a pattern. Greedy is irrelevant here (loop gate uses
# the vendor_minp_rep deploy sampler by design, both temps).
set -uo pipefail

BF16=/mnt/sdc/ml/cx_std16/CX16c4l3-bf16
IMAT=/mnt/sdc/ml/cx_std16/CX16c4l3-imatrix.dat
OUTDIR=/mnt/sdc/ml/cx_std16/gguf_coderx
GDIR=/mnt/sdc/ml/cx_std16/loopcmp_gate
QG=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
PYB=/root/anaconda3/envs/omnimergekit/bin/python
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
LOG=/mnt/sdc/ml/coderx_loopcmp_3tier.log
LOCK=/mnt/sdc/ml/coderx_loopcmp_3tier.lock
TIERS=(Q4_K_M Q3_K_S Q2_K_L)

ts(){ date '+%T %Z'; }
magic_ok(){ [ -f "$1" ] && [ "$(head -c4 "$1" 2>/dev/null)" = "GGUF" ]; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" -lt 2000 ]; }
wait_gpu(){ local g=$1 i; for i in $(seq 1 1200); do gpu_free "$g" && return 0; sleep 20; done; return 1; }

exec >>"$LOG" 2>&1
exec 9>"$LOCK"; flock -n 9 || { echo "[$(ts)] already running (lock held) — abort"; exit 0; }
echo "################ coderx loop-cmp 3-tier $(ts) ################"

# ---- preflight ----
[ -d "$BF16" ] || { echo "FATAL no bf16 $BF16"; exit 2; }
[ -f "$IMAT" ] || { echo "FATAL no imatrix $IMAT"; exit 2; }
[ -f "$GATE" ] || { echo "FATAL no gate script $GATE"; exit 2; }
free=$(df -P /mnt/sdc | awk 'NR==2{print $4}'); [ "$free" -gt 120000000 ] || { echo "FATAL disk<120G ($free K)"; exit 2; }
mkdir -p "$OUTDIR" "$GDIR"

# ---- reuse the preserved imatrix: pre-place so compute_imatrix() reuses it (no GPU) ----
if [ ! -f "$OUTDIR/imatrix.dat" ]; then cp -v "$IMAT" "$OUTDIR/imatrix.dat"; fi

# ---- STAGE 1: CPU build of the 3 tiers, nice'd + thread-capped to yield to the eval ----
NEED=0
for t in "${TIERS[@]}"; do
  f=$(ls "$OUTDIR"/*-"$t".gguf 2>/dev/null | head -1)
  magic_ok "$f" || NEED=1
done
if [ "$NEED" = 1 ]; then
  THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4
  echo "[$(ts)] STAGE1 build Q4_K_M,Q3_K_S,Q2_K_L (CPU nice-19 threads=$THREADS, imatrix reused, no-upload)"
  # local-path --model needs --base-model-id (quantize_gguf.py:1732); OMK_NO_README=1
  # (quantize_gguf.py:1033) skips the README fetch so there is NO network dependency
  # on a gated repo. --no-upload means nothing is published regardless.
  OMK_NO_README=1 nice -n 19 "$PYB" "$QG" --model "$BF16" --only Q4_K_M,Q3_K_S,Q2_K_L \
      --output-dir "$OUTDIR" --base-precision f16 --no-upload \
      --base-model-id ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it --threads "$THREADS"
  rc=$?; echo "[$(ts)] STAGE1 quantize_gguf exit=$rc"
  [ "$rc" = 0 ] || { echo "FATAL quantize_gguf rc=$rc — see above"; exit 4; }
else
  echo "[$(ts)] STAGE1 skip — all 3 tiers already present"
fi

# ---- locate + verify the 3 GGUFs ----
declare -A GP
for t in "${TIERS[@]}"; do
  f=$(ls "$OUTDIR"/*-"$t".gguf 2>/dev/null | head -1)
  magic_ok "$f" || { echo "FATAL tier $t missing/invalid ($f)"; exit 3; }
  GP[$t]="$f"; echo "  $t -> $f  ($(stat -c %s "$f" | numfmt --to=iec))"
done

# ---- STAGE 2: b9700 48-seed loop gate each (both temps), shard across GPUs as they free ----
echo "[$(ts)] STAGE2 b9700 loop gate (waits for the eval GPUs to free; <2000 MiB)"
gate_tier(){ # tier gpu port
  local t=$1 g=$2 p=$3
  echo "[$(ts)] gate $t: wait GPU$g free ..."; wait_gpu "$g" || { echo "[$(ts)] GPU$g never freed for $t"; return 1; }
  echo "[$(ts)] gate $t on GPU$g:$p (b9700)"
  bash "$GATE" "${GP[$t]}" "$g" "$p" "$GDIR/${t}.json" "coderx-${t}"
}
( gate_tier Q4_K_M 0 8460 ) & A=$!
( gate_tier Q3_K_S 1 8461 ) & B=$!
wait "$A" "$B"
gate_tier Q2_K_L 0 8460

# ---- STAGE 3: compare vs STD16 (v7-coder) anchors + MATCH/DIVERGE verdict ----
echo "================ coderx vs v7-coder(STD16) per-tier loop comparison $(ts) ================"
"$PYB" - "$GDIR" <<'PYEOF'
import json, sys, os
gdir = sys.argv[1]
# STD16 (v7-coder) baseline anchors: tier -> {temp: (fails, loops)}
STD = {"Q4_K_M": {"0.9": (0, 0),  "0.8": (0, 0)},
       "Q3_K_S": {"0.9": (19, 18),"0.8": (16, 15)},
       "Q2_K_L": {"0.9": (34, 34),"0.8": (35, 34)}}
def load(t):
    p = os.path.join(gdir, f"{t}.json")
    try: d = json.load(open(p))
    except Exception: return None
    out = {}
    for r in d.get("results", []):
        cfg = r.get("config", "")
        temp = "0.9" if "0.9" in cfg else ("0.8" if "0.8" in cfg else cfg)
        loops = r.get("loops", round(r.get("loop_rate", 0) * r.get("seeds", 48)))
        out[temp] = (r.get("fails"), loops)
    return out
print("%-8s | %-26s | %-22s | verdict" % ("tier", "coderx code4/lcb3 (f/loops)", "v7-coder STD16 (f/loops)"))
print("-" * 86)
overall = "MATCH"
for t in ["Q4_K_M", "Q3_K_S", "Q2_K_L"]:
    cx = load(t)
    if not cx:
        print("%-8s | %-26s | %-22s | PENDING" % (t, "(no gate json yet)", ""))
        overall = "INCOMPLETE"; continue
    rcx  = "  ".join("t%s %s/%s" % (k, cx.get(k, ("?", "?"))[0], cx.get(k, ("?", "?"))[1]) for k in ["0.9", "0.8"])
    rstd = "  ".join("t%s %d/%d"  % (k, STD[t][k][0], STD[t][k][1]) for k in ["0.9", "0.8"])
    # loop-match heuristic: |loops_cx - loops_std| <= 3 per temp (runaway tail = noise).
    ok = all(abs((cx.get(k, (0, 0))[1] or 0) - STD[t][k][1]) <= 3 for k in ["0.9", "0.8"])
    v = "match" if ok else "DIVERGE"
    if not ok: overall = "DIVERGE"
    print("%-8s | %-26s | %-22s | %s" % (t, rcx, rstd, v))
print()
msg = {"MATCH":      "-> coderx INHERITS STD16 per-tier loop/temp guidance (skip full sweep)",
       "DIVERGE":    "-> full 48-seed both-temp loop gate on EVERY coderx tier",
       "INCOMPLETE": "-> gates still pending"}[overall]
print("OVERALL:", overall, msg)
PYEOF
echo "###### CODERX_LOOPCMP_DONE $(ts) ######"
