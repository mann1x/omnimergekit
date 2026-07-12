#!/bin/bash
# Watch matrix log for OMK_BENCH_FINISH events; run audit_full_bench.py per cell.
# Emits one AUDIT line per finished bench to stdout.
set -u
LOG=${1:-$(ls -t /srv/ml/logs/stage3_only_*.log /srv/ml/logs/post_track5_matrix_resume*.log 2>/dev/null | head -1)}
[ -z "$LOG" ] && { echo "FATAL no matrix log" >&2; exit 1; }
PY=/srv/ml/envs/envs/omnimergekit/bin/python
AUDIT=/srv/ml/scripts/audit_full_bench.py
RES=/srv/ml/eval_results_tracks_2_3

echo "WATCH log=$LOG" >&2

# Backfill: audit every existing cell once at startup
for bench in humanevalplus_full ifeval_100 multipl_e_100; do
    [ -d "$RES/$bench" ] || continue
    for v in "$RES/$bench"/*/; do
        variant=$(basename "$v")
        [ -f "$v/summary.json" ] || continue
        [ -f "$v/audit.json" ] && continue   # already audited
        "$PY" "$AUDIT" "$bench" "$variant" 2>/dev/null
    done
done

# Live tail
stdbuf -oL tail -F -n 0 "$LOG" 2>/dev/null | \
while IFS= read -r line; do
    case "$line" in
        *summary\ \\u2192*|*summary\ →*) ;;
        *) continue ;;
    esac
    path=$(echo "$line" | grep -oE '/srv/ml/eval_results_tracks_2_3/[^ ]+/summary\.json')
    [ -z "$path" ] && continue
    bench=$(echo "$path" | awk -F/ '{print $(NF-2)}')
    variant=$(echo "$path" | awk -F/ '{print $(NF-1)}')
    case "$bench" in humanevalplus_full|ifeval_100|multipl_e_100) ;;
      *) continue ;;
    esac
    # short settle window; the summary line precedes the rc by a fraction of a second
    sleep 1
    "$PY" "$AUDIT" "$bench" "$variant" 2>/dev/null
done
