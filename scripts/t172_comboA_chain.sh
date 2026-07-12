#!/bin/bash
# Option A grid: shared x PES combos on pristine, both orderings, trio+loop audit.
# Sequential chain over phase2_combo_t172.sh (each pair = sf+pf orderings).
set -uo pipefail
exec > >(tee /srv/ml/logs/t172/comboA_chain.log) 2>&1
echo "[comboA START $(date -Iseconds)]"
df -h /mnt/sdc | tail -1
for pair in "1.30 1.10" "1.20 1.20"; do
  set -- $pair
  echo "==================== comboA pair shared=$1 pes=$2 $(date -Iseconds) ===================="
  bash /srv/ml/scripts/phase2_combo_t172.sh --shared-alpha "$1" --pes-alpha "$2" || echo "PAIR_FAIL shared=$1 pes=$2"
done
echo "==================== comboA chain DONE $(date +%H:%M:%S) ===================="
echo "=== combo TSV so far ==="
column -t -s $'\t' /srv/ml/logs/t172/phase2_combo_summary.tsv 2>/dev/null | tail -20
