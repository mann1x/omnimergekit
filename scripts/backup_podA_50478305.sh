#!/usr/bin/env bash
# Inventory + backup of vast.ai pod A (50478305) to solidpc backup_models.
# Runs ON solidpc. Pulls everything EXCEPT quant/base weights:
#   scripts, recipes, logs, eval results, toolbench results, calibration data,
#   imatrix.dat (mandatory-archival), model configs, git provenance.
# EXCLUDED by design: *.gguf, *.safetensors, /workspace/bf16, miniconda,
#   llama.cpp build tree, and the lm-evaluation-harness / MultiPL-E checkouts
#   (clean at pinned commits -- provenance recorded instead of 127 MB copied).
set -uo pipefail

H=ssh2.vast.ai; P=38304; POD=50478305
DST=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/backup_pods/pod${POD}
RS=(-az --info=stats2 -e "ssh -p $P -o StrictHostKeyChecking=no -o ConnectTimeout=30")
L="$DST/BACKUP.log"
mkdir -p "$DST"
exec > >(tee -a "$L") 2>&1
echo "=========================================================="
echo ">>> $(date -u +%FT%TZ) backup pod $POD -> $DST"

# --- 1. provenance + inventory FIRST (cheap, and it is the audit record) -----
ssh -p "$P" -o StrictHostKeyChecking=no root@"$H" '
  echo "### date_utc: $(date -u +%FT%TZ)"
  echo "### uname: $(uname -a)"
  echo "### gpu:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "### disk:"; df -h /workspace / | tail -3
  echo "### git checkouts:"
  for d in omnimergekit lm-evaluation-harness MultiPL-E; do
    if [ -d "/workspace/$d/.git" ]; then
      cd "/workspace/$d" || continue
      echo "--- $d"
      echo "    HEAD   $(git rev-parse HEAD)"
      echo "    subj   $(git log -1 --format=%s)"
      echo "    dirty  $(git status --porcelain | wc -l) file(s)"
      git status --porcelain | sed "s/^/      /"
    fi
  done
  echo "### imatrix inventory:"
  find /workspace -name "imatrix*.dat" -printf "    %10s  %TY-%Tm-%TdT%TH:%TM  %p\n" 2>/dev/null
  echo "### quant inventory (weights NOT copied):"
  find /workspace/quant -name "*.gguf" -printf "    %12s  %p\n" 2>/dev/null | sort -k2
  echo "### du:"; du -sh /workspace/* 2>/dev/null | sort -rh
' > "$DST/POD_INVENTORY.txt" 2>&1 || { echo "FATAL: inventory failed"; exit 1; }
echo "    inventory -> $DST/POD_INVENTORY.txt ($(wc -l < "$DST/POD_INVENTORY.txt") lines)"

# --- 2. top-level scripts / recipes / logs ----------------------------------
rsync "${RS[@]}" --include='*.sh' --include='*.py' --include='*.log' \
      --include='*.txt' --include='*.json' --include='*.yaml' --exclude='*' \
      root@"$H":/workspace/ "$DST/workspace_top/" || echo "WARN: top-level rsync rc=$?"

# --- 3. result + data trees (no weights) ------------------------------------
for d in eval_results toolbench_results calib models smoke; do
  rsync "${RS[@]}" --exclude='*.gguf' --exclude='*.safetensors' \
        root@"$H":/workspace/"$d"/ "$DST/$d/" || echo "WARN: $d rsync rc=$?"
done

# --- 4. the AC calibration corpus -------------------------------------------
rsync "${RS[@]}" root@"$H":/workspace/gguf/calib_train.txt "$DST/calib_train.txt" \
  || echo "WARN: calib_train rsync rc=$?"

# --- 5. imatrix.dat -- MANDATORY ARCHIVAL, one dir per quant ----------------
# Never batched behind the bulk trees: if anything in this script is going to
# survive, it is these.
mkdir -p "$DST/imatrix"
for q in $(ssh -p "$P" -o StrictHostKeyChecking=no root@"$H" 'ls /workspace/quant'); do
  ssh -p "$P" -o StrictHostKeyChecking=no root@"$H" "ls /workspace/quant/$q/imatrix*.dat /workspace/quant/$q/*.sha256 2>/dev/null" \
    | while read -r f; do
        [ -n "$f" ] || continue
        mkdir -p "$DST/imatrix/$q"
        rsync "${RS[@]}" root@"$H":"$f" "$DST/imatrix/$q/" || echo "WARN: $f rc=$?"
      done
done

# --- 6. omnimergekit run-host divergence (bidirectional rule) ---------------
mkdir -p "$DST/omnimergekit_divergence"
ssh -p "$P" -o StrictHostKeyChecking=no root@"$H" \
  'cd /workspace/omnimergekit && git diff' > "$DST/omnimergekit_divergence/tracked.diff" 2>&1
for f in $(ssh -p "$P" -o StrictHostKeyChecking=no root@"$H" \
             'cd /workspace/omnimergekit && git ls-files --others --exclude-standard'); do
  mkdir -p "$DST/omnimergekit_divergence/$(dirname "$f")"
  rsync "${RS[@]}" root@"$H":/workspace/omnimergekit/"$f" \
        "$DST/omnimergekit_divergence/$f" || echo "WARN: untracked $f rc=$?"
done

# --- 7. verify by artifact, not by exit code --------------------------------
echo ">>> result:"
du -sh "$DST"/* 2>/dev/null | sort -rh
N_IMAT=$(find "$DST/imatrix" -name 'imatrix*.dat' 2>/dev/null | wc -l)
N_SUM=$(find "$DST/eval_results" -name 'summary.json' 2>/dev/null | wc -l)
N_TB=$(find "$DST/toolbench_results" -name '*_seed*.json' 2>/dev/null | wc -l)
N_SH=$(find "$DST/workspace_top" -name '*.sh' 2>/dev/null | wc -l)
echo "    imatrix.dat      : $N_IMAT"
echo "    summary.json     : $N_SUM"
echo "    toolbench cells  : $N_TB"
echo "    scripts (.sh)    : $N_SH"
[ "$N_IMAT" -ge 1 ] && [ "$N_SUM" -ge 1 ] && [ "$N_SH" -ge 1 ] \
  && echo "PODA_BACKUP_OK $(date -u +%FT%TZ)" \
  || { echo "PODA_BACKUP_INCOMPLETE -- a census above is zero"; exit 1; }
