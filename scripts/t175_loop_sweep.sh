#!/bin/bash
# T175 — 66e/68e/72e expert-count loop-floor SWEEP (council structural-wall test).
#
# Council csl-2026-05-31-1658-3956 prediction: the ~3% IFEval/constrained loop
# floor of the 62e prune (A2) is a STRUCTURAL routing-margin limit; a higher
# expert count (same fc15_25-p8 recipe) collapses it with NO post-hoc tuning.
# This is the cheap bf16 SCREEN that tests it: build PRISTINE 66e/68e/72e
# (expert_drop only — no PES, no GGUF), run a fixed 200-prompt loop-prone sample
# greedy, count detect_loop() (same gate as the published 3%). Reuses the
# already-built pristine-62e (PES-off floor control) and A2 (pes120, the 3%
# calibration anchor). Winner (smallest N trending <1%) is carried to a separate
# confirm stage: + PES1.20 + imatrix Q6_K + full Stage-B audit vs A2.
#
# Drop maps were generated + 62e-reproduce-verified bit-identical on solidpc
# (needs --outlier-wnorm-thresh 10000.0 --outlier-mode median). Maps are nested:
# keep(62)<keep(66)<keep(68)<keep(72), so each higher N = A2 keep set + extras
# (clean isolation of the single council variable).
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
G=/mnt/sdc/ml/google
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it          # 128e source (index OK)
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
CORPUS=/mnt/sdc/ml/corpora/antiloop_sft_corpus_v2.jsonl
RES=/srv/ml/eval_results_tracks_2_3/t175_loop_screen
LOG=/srv/ml/logs/t175
mkdir -p "$RES" "$LOG"

A2=$G/gemma-4-A4B-62e-fc15_25-p8-pes120-it
P62=$G/gemma-4-A4B-62e-fc15_25-p8-pristine-it

echo "==================== T175 loop-sweep $(date -Iseconds) ===================="

# 0 preflight
[ -d "$SRC" ] || { echo "FATAL: 128e source missing $SRC"; exit 2; }
[ -d "$A2" ] || { echo "FATAL: A2 bf16 missing $A2"; exit 2; }
[ -d "$P62" ] || { echo "FATAL: pristine-62e missing $P62"; exit 2; }
for N in 66 68 72; do
  [ -f "$SCR/v6coder_C6v3lcb_${N}e_fc15_25_p8_drop_map.json" ] || { echo "FATAL: ${N}e drop map missing"; exit 2; }
done

# 1 fixed 200-prompt stratified loop-prone sample (deterministic stride; seeds=4,
#   constrained=80, multilingual=60, openended=56). No unseeded RNG.
if [ ! -f "$SAMPLE" ]; then
  echo "[1 $(date +%H:%M:%S)] build loop sample -> $SAMPLE"
  $PY - "$CORPUS" "$SAMPLE" <<'PY'
import json, sys
corpus, out = sys.argv[1], sys.argv[2]
want = {"seeds": 4, "constrained": 80, "multilingual": 60, "openended": 56}
pools = {k: [] for k in want}
for x in open(corpus):
    r = json.loads(x); b = r.get("bucket")
    if b in pools:
        pools[b].append(r["prompt"])
sample = []
for b, k in want.items():
    p = pools[b]; p.sort()                      # stable order
    if len(p) <= k:
        sel = p
    else:
        stride = len(p) / k                     # evenly-spaced deterministic pick
        sel = [p[int(i * stride)] for i in range(k)]
    sample += [{"prompt": s, "bucket": b} for s in sel]
with open(out, "w") as f:
    for s in sample:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
from collections import Counter
print("sample built: %d  by_bucket=%s" % (len(sample), dict(Counter(s["bucket"] for s in sample))))
PY
else
  echo "[1] sample exists ($(wc -l < "$SAMPLE") prompts)"
fi

# 2 build pristine bf16 66e/68e/72e (expert_drop only)
for N in 66 68 72; do
  OUT=$G/gemma-4-A4B-${N}e-fc15_25-p8-pristine-it
  if [ -f "$OUT/model.safetensors.index.json" ]; then echo "[2] ${N}e pristine exists, skip"; continue; fi
  echo "[2 $(date +%H:%M:%S)] expert_drop ${N}e -> $OUT"
  $PY "$SCR/expert_drop.py" --source-dir "$SRC" \
    --drop-map "$SCR/v6coder_C6v3lcb_${N}e_fc15_25_p8_drop_map.json" \
    --output-dir "$OUT" 2>&1 | tee "$LOG/drop_${N}e.log" | tail -4
  [ -f "$OUT/model.safetensors.index.json" ] || { echo "FATAL: ${N}e build failed"; exit 4; }
done

# 3 screen: one model per GPU, sharded.  GPU0: A2, p62, p68 ; GPU1: p66, p72
run_gpu(){ local gpu=$1; shift; for spec in "$@"; do
  local name="${spec%%=*}" dir="${spec#*=}"
  [ -f "$RES/$name.json" ] && { echo "[g$gpu] SKIP $name"; continue; }
  echo "[g$gpu $(date +%H:%M:%S)] screen $name"
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error \
    $PY "$SCR/loop_screen.py" --model "$dir" --name "$name" --sample "$SAMPLE" \
    --out "$RES/$name.json" >>"$LOG/screen_g${gpu}.log" 2>&1 || echo "[g$gpu] FAIL $name"
done; echo "[g$gpu] STREAM DONE"; }

echo "==================== T175 SCREEN $(date -Iseconds) ===================="
run_gpu 0 "a2-pes120=$A2" "p62=$P62" "p68=$G/gemma-4-A4B-68e-fc15_25-p8-pristine-it" & q0=$!
run_gpu 1 "p66=$G/gemma-4-A4B-66e-fc15_25-p8-pristine-it" "p72=$G/gemma-4-A4B-72e-fc15_25-p8-pristine-it" & q1=$!
wait "$q0" "$q1"

# 4 summary table
echo "==================== T175 RESULT $(date +%H:%M:%S) ===================="
$PY - "$RES" <<'PY'
import json, glob, os, sys
res = sys.argv[1]
order = ["a2-pes120", "p62", "p66", "p68", "p72"]
rows = {}
for f in glob.glob(os.path.join(res, "*.json")):
    d = json.load(open(f)); rows[d["name"]] = d
print("%-12s %6s %8s   by-bucket (loops/n)" % ("variant", "loop%", "loops/n"))
for k in order:
    if k not in rows: continue
    d = rows[k]
    bb = " ".join("%s=%d/%d" % (b, v["loops"], v["n"]) for b, v in d["by_bucket"].items())
    print("%-12s %5.1f%% %8s   %s" % (k, d["loop_pct"], "%d/%d" % (d["loops"], d["n"]), bb))
a2 = rows.get("a2-pes120", {}).get("loop_pct")
if a2:
    thr = a2 / 3.0
    print("\nA2 sample-loop%%=%.1f  ~= 3%% full-bench  =>  <1%% full-bench proxy threshold ~%.1f%% sample" % (a2, thr))
    for k in ("p62", "p66", "p68", "p72"):
        if k in rows:
            lp = rows[k]["loop_pct"]
            print("  %-4s %.1f%%  %s" % (k, lp, "PASS (<~1%% proxy)" if lp <= thr else "above proxy"))
PY
echo "==================== T175 DONE $(date +%H:%M:%S) ===================="
