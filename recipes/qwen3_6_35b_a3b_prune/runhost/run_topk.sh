#!/usr/bin/env bash
# top-8 vs top-10 routing matrix for the published Coder cut and armJ (CoderX cand).
#
# WHY: the published GGUF ships qwen35moe.expert_used_count=10, but every eval cell we
# have was served from a local file whose baked value is 8, with NO --override-kv. So no
# measurement of the SHIPPED routing exists for either model. Worse, the two "pub" cells
# in the b604 run were the SAME file+config run twice and scored 0.850 / 0.890 -- a 4pp
# same-config spread. At that band a single draw per config cannot decide top-8 vs top-10,
# so every config here gets TWO draws.
#
# The override is behaviourally verified (2026-08-20): with --jinja, top-8 and top-10
# produce different greedy text (1861 vs 2025 chars). This build never logs the expert
# count, so that behavioural check is the only available proof it took effect.
set -uo pipefail
OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results_topk
PORT=8099
B=multipl_e_100
TOK_BASE=/srv/ml/models/Qwen3.6-35B-A3B
GEN=$OMK/eval/multipl_e/multipl_e_generate.py
WANT_MD5=8a8e32c2728130ece233071cb858f6b2
PUB=/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf
ARMJ=$WORK/gguf/armJ_imat/armJ-Q6_K.gguf
LOG=$WORK/run_topk.log
say(){ echo "[topk $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# cell|gguf|topk
ARMS=(
 "pubcut_t10_a|$PUB|10"
 "armJ_t10_a|$ARMJ|10"
 "pubcut_t10_b|$PUB|10"
 "armJ_t10_b|$ARMJ|10"
 "armJ_t8_b|$ARMJ|8"
)

# GATE-I: the bug-604 extractor fix is the one installed (same gate as the b604 run)
got=$(md5sum "$GEN" | awk "{print \$1}")
[ "$got" = "$WANT_MD5" ] || { say "REFUSE GATE-I: $GEN md5=$got want=$WANT_MD5"; exit 3; }
"$OMKPY" "$GEN" --selftest > /tmp/topk_selftest.log 2>&1
grep -q "SELFTEST OK" /tmp/topk_selftest.log || { say "REFUSE GATE-I: golds failed"; exit 3; }
say "GATE-I ok (md5 $got)"

freeg=$(df -BG --output=avail / | tail -1 | tr -dc "0-9")
[ "${freeg:-0}" -ge 210 ] || { say "REFUSE: root fs ${freeg}G free (floor 200G)"; exit 1; }
for a in "${ARMS[@]}"; do IFS="|" read -r c g k <<<"$a"; [ -s "$g" ] || { say "REFUSE: missing $g"; exit 1; }; done
say "preflight ok: ${#ARMS[@]} cells, disk ${freeg}G"

for a in "${ARMS[@]}"; do
  IFS="|" read -r CELL G TOPK <<<"$a"
  out=$RES/qwen_suite/$B/$CELL
  [ -f "$out/summary.json" ] && { say "SKIP $CELL (done)"; continue; }
  [ -d "$out" ] && mv "$out" "${out}_PARTIAL_$(date -u +%Y%m%dT%H%M%SZ)"
  TOK=$TOK_BASE; case "$CELL" in armJ*) [ -d "$WORK/armJ" ] && TOK=$WORK/armJ;; esac
  EXTRA=""
  [ "$TOPK" = "10" ] && EXTRA="--metadata backend_args.llama_extra=[\"--override-kv\",\"qwen35moe.expert_used_count=int:10\"]"
  say "===== $CELL topk=$TOPK gguf=$(basename $G) tok=$TOK"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
      --model "$G" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
      --results-dir "$RES/qwen_suite" --parallel 2 \
      --sampler-profile qwen3_6 --sampler recommended \
      --metadata backend_args.llama_ctx=49152 $EXTRA 2>&1 | tail -5
  rc=$?
  # GATE: geometry readback (same as b604)
  L="$out/server.log"
  gs=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  gn=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $CELL per_slot=${gs:-?} slots=${gn:-?} (want 24576 / 2) rc=$rc"
  if [ -f "$out/summary.json" ]; then
    s=$("$OMKPY" -c "import json,sys;print(json.load(open(sys.argv[1]))[\"score\"])" "$out/summary.json" 2>/dev/null)
    say "SCORE $CELL topk=$TOPK = $s"
  else
    say "FAIL $CELL: no summary.json"
  fi
done
say "TOPK_MATRIX_DONE"
