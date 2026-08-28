#!/usr/bin/env bash
# Re-run MPE-100 on armJ with the CURRENT (post-bug-604) extractor, so the armJ row is
# comparable to the a3b_gepo1 row run 2026-08-24.
#
# WHY THIS EXISTS
# ---------------
# The banked armJ MPE cell (ream_arms/multipl_e_100/hybrid_p24_ourssal_reapfloor, 0.7100)
# ran 2026-08-19 13:11. Three scorer commits landed 2026-08-20:
#   3221739  bug-604  chat_to_body ate the first line of a body-only reply   <- CHANGES SCORES
#   8040564  bug-606  LCB opening-fence                                      (LCB only)
#   5e61099  bug-607  persist the RAW reply next to the extracted body       <- why no repair
# a3b_gepo1 ran 2026-08-24 on the fixed extractor. Two extractors, one table = invalid.
#
# It cannot be repaired offline: extraction happens at GENERATION time, so generations/*.json
# holds only post-extraction `completions`, and the armJ sqlite cache predates bug-607 —
# measured 0/300 rows carry `raw` (gepo1: 300/300). The raw replies are gone. Regeneration
# is the only route, which is why this needs the GGUF.
#
# THE GGUF IS THE PUBLISHED CoderX Q6_K, AND IT IS *THE SAME FILE* AS armJ-Q6_K.
# armJ, coderx_st_upload and publish/Qwen3.6-27B-A3B-CoderX are ONE inode (links=3), and the
# two retained imatrix files are byte-identical:
#   gguf/armJ_imat/imatrix.dat  == gguf_coderx/imatrix.dat   md5 b09b85da001623c8920adb4b4ddf98df
# Same weights + same imatrix + same tier + same quantize_gguf.py => bit-identical quant.
# Gated below against the retained receipt sha 92bfdc9dca32...
#
# THE OLD CELL IS NOT OVERWRITTEN. Results are sacred: the pre-b604 cell stays exactly where
# it is, and this writes a NEW cell suffixed _b604. Never pool the two.
#
# BASIS: copied from ream-work/eval_ream_arms.sh, identical to the a3b_gepo1 run --
# --parallel 4 PLUS --metadata llama_ctx=49152 (=12288/slot, NOT the template's bare 4096)
# and the fixed jinja chat template. Greedy, q6_k.
#
# GPU1 ONLY, chained behind the GPQA run.
set -uo pipefail

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/ream_arms
TMPL_FIX=/srv/ml/models/qwen36_chat_template_fixed.jinja
GGUF=/mnt/sdc/ream-work/gguf_armJ_hf/Qwen3.6-27B-A3B-CoderX-Q6_K.gguf
TOK=/mnt/sdc/ream-work/armJ
LOG=/mnt/sdc/ml/brevity/gepo/mpe_armj_b604.log
WANT_SHA=92bfdc9dca32f2ad81a85c6ba05d239cad6430fde112901245fb8b28f2ffa076

NAME=hybrid_p24_ourssal_reapfloor_b604
OLD=hybrid_p24_ourssal_reapfloor
B=multipl_e_100
PORT=8097

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export LLAMA_EXTRA="--jinja --chat-template-file $TMPL_FIX"
unset LLAMA_ARG_SPEC_TYPE

exec >>"$LOG" 2>&1
ts(){ date -u '+%F %T UTC'; }
say(){ echo ">>> [$(ts)] $*"; }
echo "=== mpe_armj_b604 start $(ts) ==="

# ---- wait for GPQA to release GPU1 -------------------------------------------
waited=0
while pgrep -f "gpqa_gepo1[.]sh" >/dev/null 2>&1; do
  sleep 60; waited=$((waited+60))
  [ $((waited % 900)) -eq 0 ] && say "gpqa still running (${waited}s)"
  [ "$waited" -ge 21600 ] && { say "ABORT: gpqa still running after 6h"; exit 3; }
done
say "gpqa finished (waited ${waited}s)"

# ---- wait for the download to land, then gate on the RECEIPT ----------------
for i in $(seq 1 120); do
  pgrep -f "hf download" >/dev/null 2>&1 || break
  sleep 30
done
[ -s "$GGUF" ] || { say "FATAL: $GGUF absent"; exit 2; }
say "hashing $(stat -c %s "$GGUF") bytes ..."
GOT=$(sha256sum "$GGUF" | cut -d' ' -f1)
say "sha256 got=$GOT"
say "        want=$WANT_SHA"
if [ "$GOT" != "$WANT_SHA" ]; then
  say "FATAL: published Q6_K does not match the retained armJ receipt -- NOT the same file"
  exit 4
fi
say "SHA_OK -- this is bit-identical to the deleted armJ-Q6_K"

for _ in $(seq 1 30); do
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 60000 ] && break
  say "GPU1 only ${free}MiB free, waiting"; sleep 60
done

[ -f "$RES/$B/$NAME/summary.json" ] && { say "SKIP: $NAME already exists"; exit 0; }

# ---- run ---------------------------------------------------------------------
say "===== $B $NAME (post-b604 extractor, greedy, parallel 4, ctx 49152 => 12288/slot)"
t0=$SECONDS
"$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
    --model "$GGUF" --tokenizer "$TOK" --served-name "$NAME" --port "$PORT" \
    --results-dir "$RES" --parallel 4 \
    --metadata backend_args.llama_ctx=49152 \
    --metadata backend_args.llama_content_headroom=8192
say "<<<< END rc=$? in $(( (SECONDS-t0)/60 ))m"

S=$RES/$B/$NAME/summary.json
[ -f "$S" ] || { say "FATAL: no summary.json"; exit 5; }

# ---- gates: geometry + sampler ----------------------------------------------
L=$RES/$B/$NAME/server.log
got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
say "GEOMETRY per_slot=${got:-unknown} (want 12288)"
[ "${got:-0}" = "12288" ] || say "!!! GEOMETRY MISMATCH -- not comparable to a3b_gepo1"
sn=$("$OMKPY" - "$S" <<'PY'
import json,sys
print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
PY
)
say "SAMPLER recorded=$sn (want template_default)"

# ---- the corrected table -----------------------------------------------------
say "=== MPE-100: armJ vs a3b-gepo1, BOTH on the post-b604 extractor ==="
"$OMKPY" - "$RES/$B/$OLD/summary.json" "$RES/$B/$NAME/summary.json" \
          "$RES/$B/a3b_gepo1/summary.json" <<'PY'
import json, sys

def load(p):
    try: return json.load(open(p))
    except Exception: return None

rows = [("armJ  (pre-b604, VOID)", sys.argv[1]),
        ("armJ  (post-b604)",      sys.argv[2]),
        ("gepo1 (post-b604)",      sys.argv[3])]
print(f"{'arm':<24}{'score':>9}{'pass':>10}{'p50':>7}{'p90':>7}{'sum tok':>10}{'empty':>7}{'sampler':>18}")
print("-" * 92)
v = {}
for tag, p in rows:
    d = load(p)
    if d is None:
        print(f"{tag:<24}{'(missing)':>9}"); continue
    t = d.get("token_stats") or {}; c = t.get("completion_tokens") or {}
    sm = d.get("sampler") or {}
    v[tag] = (d.get("score"), c.get("sum"), c.get("p50"))
    print(f"{tag:<24}{d.get('score'):>9.4f}{str(round(d.get('score')*300))+'/300':>10}"
          f"{c.get('p50') or 0:>7}{c.get('p90') or 0:>7}{c.get('sum') or 0:>10}"
          f"{t.get('empty_completions') or 0:>7}{sm.get('name'):>18}")

a = v.get("armJ  (post-b604)"); b = v.get("gepo1 (post-b604)")
old = v.get("armJ  (pre-b604, VOID)")
if a and old:
    print(f"\n  b604 extractor effect on armJ: {old[0]:.4f} -> {a[0]:.4f}  "
          f"({(a[0]-old[0])*100:+.2f} pp)  <- this is SCORER, not model")
if a and b:
    print(f"  VALID model delta:             {a[0]:.4f} -> {b[0]:.4f}  "
          f"({(b[0]-a[0])*100:+.2f} pp, {round(b[0]*300)-round(a[0]*300):+d} of 300)")
    print(f"  tokens:                        {a[1]} -> {b[1]}  "
          f"({(b[1]-a[1])/a[1]*100:+.1f}%)   p50 {a[2]} -> {b[2]}")
PY

echo "###### MPE_ARMJ_B604_DONE $(ts) ######"
