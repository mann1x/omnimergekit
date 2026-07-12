#!/usr/bin/env bash
# pp_smoke_driver.sh — GPU loss-equivalence + throughput smoke for the GPipe
# pipeline-parallel (--pp) path of phase1_train_yarn_lora.py, at 64k on bs2's
# 2x PRO 6000 (96GB, sm_120).
#
# WHY GATED + REPORT-ONLY: PP needs BOTH GPUs (the 26B is device_map-split across
# them). The flagship DDP@32k run holds both (75.7GB/GPU). This driver therefore
# does NOTHING if a trainer is already running — it prints who owns the GPUs and
# exits 0. Re-invoke once the GPUs are free; it never preempts the flagship run.
#
# WHAT IT PROVES:
#   (1) Loss-equivalence — STEP-0 loss is the pure base-model loss (LoRA-B is
#       zero-init so the adapter contributes nothing at step 0) and is therefore
#       deterministic: MP and PP MUST match to ~1e-3 on the same first packs.
#       A mismatch means the fill/drain restructure changed the math -> STOP.
#   (2) Throughput — STEP-1 tok/s (step 0 carries compile/warmup). PP should be
#       ~1.4-1.7x the MP baseline (GPipe 2M/(M+1), M=grad_accum).
#   (3) Memory — PP holds M micro-batch graphs at once; watch peak_vram for OOM.
#       If PP@M=4 OOMs, the driver retries PP@M=2 and reports both.
#
# Uses the SIDECAR trainer (phase1_train_yarn_lora_pp.py) so it never touches the
# file the flagship run was launched from. Greedy/deterministic; 2 steps each.
set -euo pipefail

PY=/srv/ml/envs/envs/omk-yarn/bin/python
TR=/srv/ml/scripts/phase1_train_yarn_lora_pp.py
CFG="${YARN_CFG_DIR:-/srv/ml/longctx/yarn_cfg_98e}"
DATA="${DATA_DIR:-/srv/ml/longctx/data_98e}"
WORK="${WORK_DIR:-/srv/ml/longctx}"
PACK="${PACK_LEN:-65536}"
GACC="${GRAD_ACCUM:-4}"
COMMON=(--gpus 0,1 --yarn-cfg-dir "$CFG" --data-dir "$DATA" --pack-len "$PACK"
        --grad-ckpt --attn memeff --ce-chunk 2048 --lr 1e-4 --rank 16 --alpha 32
        --max-steps 2 --resume never --log-every 1)

# --- preflight: refuse to run if a trainer already owns the GPUs -------------
if pgrep -af "phase1_train_yarn_lora" | grep -v "pp_smoke_driver" | grep -vq "$$"; then
  echo "[gate] a trainer is running — GPUs are busy. NOT preempting. Re-run when free:"
  pgrep -af "phase1_train_yarn_lora" | grep -v "pp_smoke_driver" | sed 's/^/[gate]   /' | cut -c1-110
  exit 0
fi
for g in 0 1; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
  if [ "$used" -gt 4000 ]; then
    echo "[gate] GPU$g has ${used} MiB used (>4GB) — not idle. NOT running. Free it first."
    exit 0
  fi
done
echo "[gate] both GPUs idle — running MP-vs-PP@${PACK} smoke (grad_accum=${GACC})."

# preflight inputs (FATAL-loud)
for p in "$PY" "$TR" "$CFG" "$DATA"; do
  [ -e "$p" ] || { echo "FATAL: missing $p" >&2; exit 1; }
done

run_one() {  # $1=tag  $2..=extra args
  local tag="$1"; shift
  local log="$WORK/pp_smoke_${tag}.log"
  local ckpt="$WORK/pp_smoke_${tag}_ckpt"
  rm -rf "$ckpt"
  echo "[run] $tag -> $log"
  if ! "$PY" "$TR" "${COMMON[@]}" --grad-accum "$GACC" --ckpt-dir "$ckpt" "$@" >"$log" 2>&1; then
    if grep -qiE "out of memory|CUDA error: out of memory" "$log"; then
      echo "[run] $tag OOM"; return 42
    fi
    echo "[run] $tag FAILED (see $log):"; tail -n 15 "$log" | sed 's/^/[run]   /'; return 1
  fi
}

# --- MP baseline (sequential micro-loop on the same split) -------------------
run_one mp || exit 1
# --- PP (GPipe fill/drain) ---------------------------------------------------
PP_TAG=pp_m${GACC}
if ! run_one "$PP_TAG" --pp; then
  rc=$?
  if [ "$rc" = 42 ] && [ "$GACC" -gt 2 ]; then
    echo "[run] retrying PP at grad_accum=2 (M=$GACC OOMs)"
    PP_TAG=pp_m2
    GACC=2 run_one "$PP_TAG" --pp || { echo "[run] PP@M=2 also failed"; exit 1; }
  else
    exit 1
  fi
fi

# --- compare: step-0 loss (equivalence) + step-1 tok/s (speedup) -------------
"$PY" - "$WORK/pp_smoke_mp.log" "$WORK/pp_smoke_${PP_TAG}.log" <<'PYEOF'
import json, sys, re
def rows(p):
    out={}
    for ln in open(p):
        m=re.search(r"\[train\] (\{.*\})", ln)
        if m:
            r=json.loads(m.group(1)); out[r["step"]]=r
    return out
mp, pp = rows(sys.argv[1]), rows(sys.argv[2])
print("\n=== PP@64k smoke — MP vs GPipe ===")
print(f"{'metric':22s} {'MP (seq)':>14s} {'PP (GPipe)':>14s} {'verdict':>12s}")
l0_mp, l0_pp = mp.get(0,{}).get('loss'), pp.get(0,{}).get('loss')
if l0_mp is not None and l0_pp is not None:
    d=abs(l0_mp-l0_pp); v="MATCH" if d<1e-3 else ("close" if d<1e-2 else "MISMATCH!")
    print(f"{'step0 loss (det.)':22s} {l0_mp:>14.4f} {l0_pp:>14.4f} {v:>12s}  (|d|={d:.2e})")
t1_mp, t1_pp = mp.get(1,{}).get('tok_per_s'), pp.get(1,{}).get('tok_per_s')
if t1_mp and t1_pp:
    sp=t1_pp/t1_mp
    print(f"{'step1 tok/s':22s} {t1_mp:>14.1f} {t1_pp:>14.1f} {sp:>11.2f}x")
v_mp = max((r.get('peak_vram_gb',0) for r in mp.values()), default=0)
v_pp = max((r.get('peak_vram_gb',0) for r in pp.values()), default=0)
print(f"{'peak_vram_gb':22s} {v_mp:>14.1f} {v_pp:>14.1f}")
print("\nGATE: step0 loss MUST be MATCH (<1e-3). Speedup expected ~1.4-1.7x. "
      "If MISMATCH -> the fill/drain restructure is wrong; do NOT use --pp.")
PYEOF
