#!/bin/bash
# Broad sweep across DARE/density/source-count axes. Each variant runs HE+MBPP only.
# Total 8 variants × ~20 min = ~2.7 hours.
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
RUNNER=$WS/scripts/local_4b_microcoder_variant.sh

# Each line: NAME|SOURCES|WEIGHTS|METHOD|DENSITY|EXTRA_ARGS
# Sources are subdirs of $HF/.
# v1b is already running (TIES, 2-source, 0.45/0.55, density 0.53). Skip here.
declare -a VARIANTS=(
    "v1c|jackrong-v2,coder_eval/continuum-code-forged|0.45,0.55|dare_linear|0.53|--pr682-turbo --m7-detector --m7-layer-aware"
    "v1d|jackrong-v2,coder_eval/continuum-code-forged,coder_eval/jackrong-python|0.40,0.35,0.25|dare_linear|0.53|--pr682-turbo --m7-detector --m7-layer-aware"
    "v1e|jackrong-v2,coder_eval/continuum-code-forged|0.45,0.55|omnimerge_v2|0.30|--pr682-turbo --m7-detector --m7-layer-aware --darex-q 0.75 --v2-features obim,darex,emr"
    "v1f|jackrong-v2,coder_eval/continuum-code-forged|0.30,0.70|omnimerge_v2|0.53|--pr682-turbo --m7-detector --m7-layer-aware --darex-q 0.75 --v2-features obim,darex,emr"
    "v1g|jackrong-v2,coder_eval/continuum-code-forged|0.50,0.50|omnimerge_v2|0.53|--pr682-turbo --m7-detector --m7-layer-aware --darex-q 0.75 --v2-features obim,darex,emr"
    "v1h|coder_eval/continuum-code-forged|1.0|omnimerge_v2|0.53|--pr682-turbo --m7-detector --m7-layer-aware --darex-q 0.75 --v2-features obim,darex,emr"
    "v1i|jackrong-v2,coder_eval/continuum-code-forged,coder_eval/jackrong-python|0.40,0.35,0.25|omnimerge_v2|0.30|--pr682-turbo --m7-detector --m7-layer-aware --darex-q 0.75 --v2-features obim,darex,emr"
    "v1j|jackrong-v2,coder_eval/continuum-code-forged|0.45,0.55|task_arithmetic|0.53|"
)

for line in "${VARIANTS[@]}"; do
    IFS='|' read -r NAME SOURCES WEIGHTS METHOD DENSITY EXTRA <<< "$line"
    echo ""
    echo "########################################"
    echo "### sweep: $NAME starting $(date -Iseconds)"
    echo "########################################"
    NAME="$NAME" SOURCES="$SOURCES" WEIGHTS="$WEIGHTS" \
        METHOD="$METHOD" DENSITY="$DENSITY" EXTRA_ARGS="$EXTRA" \
        WITH_LCB=0 \
        bash $RUNNER 2>&1
done

echo ""
echo "=== sweep done $(date -Iseconds) ==="

# Tabulate results
echo ""
echo "=== sweep results table ==="
EVAL_DIR=$WS/4b_phase1/eval_results
for v in v1b v1c v1d v1e v1f v1g v1h v1i v1j; do
    HE=$(python3 -c "import json,glob; f=glob.glob('$EVAL_DIR/humaneval_$v/$v/results_*.json'); print(round(json.load(open(f[0]))['results']['humaneval']['pass@1,create_test']*100,2)) if f else print('-')" 2>/dev/null)
    MB=$(python3 -c "import json,glob; f=glob.glob('$EVAL_DIR/mbpp_$v/$v/results_*.json'); print(round(json.load(open(f[0]))['results']['mbpp']['pass_at_1,none']*100,2)) if f else print('-')" 2>/dev/null)
    LC=$(python3 -c "import json,os; p='$EVAL_DIR/lcb_$v/lcb_results.json'; print(round(json.load(open(p))['pass_at_1']*100,2)) if os.path.exists(p) else print('-')" 2>/dev/null)
    printf "%-6s HE=%6s  MBPP=%6s  LCB=%6s\n" "$v" "$HE" "$MB" "$LC"
done
