#!/usr/bin/env bash
# make_polyglot_subset.sh — build a harbor dataset dir from a polyglot manifest.
#
# harbor's local-dir loader (`--path`) takes a FLAT directory of task dirs with
# no manifest of its own, and `--n-tasks N` only takes the first N in order —
# so a NAMED selection has to be expressed as a directory. Symlinks keep it
# zero-copy and keep the real tasks single-sourced.
#
#   make_polyglot_subset.sh <manifest.tsv> <out-dir> [src-dataset-dir]
set -euo pipefail
MAN="${1:?manifest tsv}"; OUT="${2:?output dataset dir}"
SRC="${3:-/mnt/sdc/agentbench/datasets/aider-polyglot}"

[ -d "$SRC" ] || { echo "FATAL: source dataset not found: $SRC"; exit 1; }
mkdir -p "$OUT"

n=0; missing=0
while read -r lang ex _axis; do
  case "$lang" in ''|\#*) continue;; esac
  t="polyglot_${lang}_${ex}"
  if [ ! -d "$SRC/$t" ]; then
    echo "MISSING: $t"; missing=$((missing+1)); continue
  fi
  ln -sfn "$SRC/$t" "$OUT/$t"
  n=$((n+1))
done < "$MAN"

# A subset that silently dropped tasks is a different experiment than the one
# the manifest describes. Refuse rather than run a quietly smaller selection.
[ "$missing" -eq 0 ] || { echo "FATAL: $missing task(s) in $MAN not present in $SRC"; exit 1; }

echo "built $OUT: $n tasks"
ls "$OUT" | sed 's/^/  /'
