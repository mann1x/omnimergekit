#!/bin/bash
# recreate_62e_on_pod.sh — recreate B1 (fc15_25-p8-it) + B2 (pes1_20-it) deterministically
# on the Linode Blackwell pod from base 128e + drop map. Run ON THE POD.
#
# Why: rsync of 54 GB BF16 over solidpc residential upstream takes ~3 hours at 5 MB/s.
# Recreation from base + drop map + 2 alpha steps is ~10 min CPU work on the pod's EPYC.

set -euo pipefail
BM=/srv/ml
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=$BM/repos/omnimergekit/scripts

BASE=$BM/google/gemma-4-26B-A4B-it
DROP_MAP=$SCR/v6coder_C6v3lcb_62e_fc15_25_p8_drop_map.json
B1=$BM/google/gemma-4-A4B-62e-fc15_25-p8-it
B2=$BM/google/gemma-4-A4B-62e-fc15_25-p8-pes1_20-it

LOG=$BM/logs/t141/recreate_62e_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$BM/logs/t141"
exec > >(tee "$LOG") 2>&1

echo "[$(date -Iseconds)] === recreate B1 + B2 on pod (T141) ==="
echo "  base       : $BASE"
echo "  drop map   : $DROP_MAP"
echo "  B1 target  : $B1"
echo "  B2 target  : $B2"

# ---- precond ----
[ -f "$BASE/config.json" ] || { echo "FATAL: base 128e missing at $BASE"; exit 1; }
[ -f "$DROP_MAP" ]         || { echo "FATAL: drop map missing at $DROP_MAP"; exit 1; }

# ---- 1. expert_drop to produce 62e (no shared upweight yet) ----
if [ ! -f "$B1/model.safetensors.index.json" ]; then
  echo "[$(date -Iseconds)] step 1: expert_drop -> $B1"
  "$PY" "$SCR/expert_drop.py" \
      --source-dir "$BASE" \
      --drop-map "$DROP_MAP" \
      --output-dir "$B1"
else
  echo "[$(date -Iseconds)] step 1 skipped — $B1 exists"
fi

# ---- 2. shared upweight alpha=1.2 on B1 (bake in the canonical recipe) ----
if [ ! -f "$B1/.shared_applied" ]; then
  echo "[$(date -Iseconds)] step 2: shared upweight alpha=1.2 on $B1"
  "$PY" "$SCR/router_shared_upweight.py" \
      --model-dir "$B1" \
      --target mlp.down_proj.weight \
      --alpha 1.2
  echo "shared_alpha=1.2 (canonical fc15_25-p8 recipe)" > "$B1/.shared_applied"
else
  echo "[$(date -Iseconds)] step 2 skipped — .shared_applied marker present"
fi

# ---- 3. cp -al B1 -> B2 ----
if [ ! -f "$B2/model.safetensors.index.json" ]; then
  echo "[$(date -Iseconds)] step 3: cp -al $B1 -> $B2"
  cp -al "$B1" "$B2"
fi

# ---- 4. per-expert-scale alpha=1.20 on B2 ----
if [ ! -f "$B2/.pes_applied" ]; then
  echo "[$(date -Iseconds)] step 4: PES alpha=1.20 on $B2"
  "$PY" "$SCR/router_per_expert_rescale.py" \
      --model-dir "$B2" \
      --alpha 1.20
  echo "pes_alpha=1.20" > "$B2/.pes_applied"
else
  echo "[$(date -Iseconds)] step 4 skipped — .pes_applied marker present"
fi

# ---- 5. verify ----
echo "[$(date -Iseconds)] === verification ==="
for d in "$B1" "$B2"; do
  if [ -f "$d/model.safetensors.index.json" ]; then
    sz=$(du -sh "$d" | cut -f1)
    cnt=$(ls "$d"/*.safetensors 2>/dev/null | wc -l)
    echo "  OK  $d  (size=$sz, shards=$cnt)"
  else
    echo "  FAIL $d (no index.json)"; exit 1
  fi
done

touch "$BM/logs/t141/recreate_62e_DONE"
echo "[$(date -Iseconds)] === recreate done ==="
