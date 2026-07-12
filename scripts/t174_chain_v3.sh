#!/usr/bin/env bash
# T174 v3 anti-loop chain (bs2, GPU0).
#
# User decision 2026-06-01: "validate v2 -> train v3", chained onto GPU0 to fire
# when the rp1.15 HE+ decode job frees the card (unattended overnight run).
#
# Flow (all pinned to GPU0):
#   0. wait for the GPU0 decode pid (arg $1) to exit, then settle VRAM
#   1. loop-screen A2 (base anchor)            -> loop_A2.json
#   2. loop-screen v2 merged (existing)        -> loop_v2.json
#   3. train v3 SFT LoRA, 3 epochs, balanced v2 corpus, r16/a32 (trainer default)
#   4. for each epoch: merge adapter -> 27G bf16 -> loop-screen -> rm merge (keep adapter)
#   5. write SUMMARY.txt (A2 vs v2 vs v3-ep1/2/3, with per-bucket loops)
#
# loop_screen.py is the canonical detect_loop() gate (T175); greedy, 200-prompt
# loop-prone sample, max-new 2048 -> the same measurement as A2's published ~3%.
# The Stage-B 9-bench quality audit on the winning epoch is a SEPARATE follow-up.
set -uo pipefail

PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/scripts
BASE=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it           # A2 (pes120-it)
V2IT=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-a2-antiloop-v2-it   # v2 merged (existing)
CORPUS=/mnt/sdc/ml/corpora/antiloop_sft_corpus_v2.jsonl               # balanced v2 (1448 rows loaded)
V3LORA=/mnt/sdc/ml/google/a2-antiloop-v3-lora
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
RES=/srv/ml/eval_results_tracks_2_3/t174_v3
WAITPID="${1:-}"

mkdir -p "$RES"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L(){ echo "[chain $(date -u +%H:%M:%S)] $*"; }

L "=== T174 v3 chain start (GPU0) ==="
if [ -n "$WAITPID" ]; then
  L "waiting for GPU0 decode pid $WAITPID to free the card ..."
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
  L "pid $WAITPID gone; settling 25s for VRAM release"; sleep 25
fi

screen(){ # $1 model_dir  $2 out_json  $3 name
  L ">>> SCREEN $3"
  if "$PY" "$SCR/loop_screen.py" --model "$1" --out "$2" --name "$3" \
        --sample "$SAMPLE" --bs 16 --max-new 2048; then
    L "<<< SCREEN $3 done -> $2"
  else
    L "!!! SCREEN $3 FAILED"
  fi
}

# 1+2) anchors
screen "$BASE" "$RES/loop_A2.json" "A2"
screen "$V2IT" "$RES/loop_v2.json" "antiloop-v2"

# 3) train v3 (3 epochs)
L ">>> TRAIN v3 (3 epochs, corpus=$(basename "$CORPUS"))"
if "$PY" "$SCR/lora_sft_antiloop.py" --base "$BASE" --corpus "$CORPUS" --out "$V3LORA" --epochs 3; then
  L "<<< TRAIN v3 done (adapters in $V3LORA/epoch{1,2,3})"
else
  L "!!! TRAIN v3 FAILED"
fi

# 4) merge + screen each epoch, purge the 27G merge after (disk-safe)
for e in 1 2 3; do
  AD="$V3LORA/epoch$e"
  if [ ! -d "$AD" ]; then L "skip epoch$e (no adapter dir)"; continue; fi
  MO="/mnt/sdc/ml/google/a2-antiloop-v3-it-ep$e"
  L ">>> MERGE epoch$e -> $MO"
  if "$PY" "$SCR/merge_adapter.py" "$BASE" "$AD" "$MO"; then
    L "<<< MERGE epoch$e done"
    screen "$MO" "$RES/loop_v3_ep$e.json" "antiloop-v3-ep$e"
    rm -rf "$MO" && L "  purged merged ep$e (kept adapter $AD)"
  else
    L "!!! MERGE epoch$e FAILED"
  fi
done

# 5) summary table
L ">>> SUMMARY"
"$PY" - "$RES" <<'PYEOF'
import json, glob, os, sys
res = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(res, "loop_*.json"))):
    try:
        d = json.load(open(f))
        rows.append((d.get("name", os.path.basename(f)), d.get("loop_pct"),
                     d.get("loops"), d.get("n"), d.get("by_bucket", {})))
    except Exception as e:
        rows.append((os.path.basename(f), "ERR", str(e), "", {}))
out = ["%-20s %8s %8s %6s" % ("model", "loop%", "loops", "n")]
for nm, pct, lp, n, bb in rows:
    out.append("%-20s %8s %8s %6s" % (nm, pct, lp, n))
out += ["", "per-bucket loops (constrained / multilingual are the loop-prone axes):"]
for nm, pct, lp, n, bb in rows:
    if isinstance(bb, dict) and bb:
        seg = "  ".join("%s=%s/%s" % (b, v.get("loops"), v.get("n")) for b, v in sorted(bb.items()))
        out.append("  %-18s %s" % (nm, seg))
txt = "\n".join(out)
print(txt)
open(os.path.join(res, "SUMMARY.txt"), "w").write(txt + "\n")
PYEOF
L "CHAIN_DONE -> $RES/SUMMARY.txt"
