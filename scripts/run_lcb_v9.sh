#!/usr/bin/env bash
# run_lcb_v9.sh — v9-agentic 98e Q6_K on lcb_v6_77q at budget 24576, terminate-only message.
#
# This is a FAITHFUL COPY of run_lcb_v6_budget_probe.sh with only the model, output dir and
# cell list changed, so the result is flag-for-flag comparable to its b24_msg2 cell:
#     128e teacher Q6_K, b24_msg2, lcb_v6_77q = 69/77 = 0.8961   <-- the comparison arm
# Identical: MSG2 text (byte-for-byte), f16 KV forcing, MAXGEN 57344, parallel 4,
# llama_ctx 262144 => 65536/slot, greedy frozen template, PEG-degrade llama-server.
set -uo pipefail
GGUF="${1:?usage: run_lcb_v9.sh <q6k.gguf>}"
LC=/opt/llama.cpp/build-pegfix/bin/llama-server
OMK=/srv/ml/repos/omnimergekit
OMK_PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
OUT=/mnt/sdc/lcb_v6_v9
TPLS=( lcb_v6_77q )
export LLAMA_BIN=/opt/llama.cpp/build-pegfix/bin
GPU=0
NP=4
MAXGEN=57344

# byte-identical to the probe driver's MSG2
MSG2="I have used my thinking budget. I must stop analysing now and act on what I have and provide an answer."

say(){ echo "[v9lcb $(date -u +%H:%M:%SZ)] $*"; }

[ -f "$GGUF" ]   || { say "FATAL: model missing: $GGUF"; exit 1; }
[ -x "$LC" ]     || { say "FATAL: llama-server missing: $LC"; exit 1; }
[ -x "$OMK_PY" ] || { say "FATAL: python missing: $OMK_PY"; exit 1; }
mkdir -p "$OUT"
[ -f "${GGUF}.sha256" ] || { say "computing ${GGUF}.sha256"; sha256sum "$GGUF" > "${GGUF}.sha256"; }

# ── MANDATORY pre-launch gate: the frozen template must still be GREEDY ──
for _T in "${TPLS[@]}"; do
"$OMK_PY" - "$OMK/eval/templates/${_T}.yaml" <<'PY' || exit 1
import sys, yaml
g = (yaml.safe_load(open(sys.argv[1])) or {}).get("generation") or {}
bad = []
if float(g.get("temperature", -1)) != 0.0: bad.append("temperature=%r" % g.get("temperature"))
if float(g.get("top_p", -1)) != 1.0:       bad.append("top_p=%r" % g.get("top_p"))
if int(g.get("top_k", -1)) != 0:           bad.append("top_k=%r" % g.get("top_k"))
if g.get("do_sample") is not False:        bad.append("do_sample=%r" % g.get("do_sample"))
if bad:
    print("FATAL: template generation block is NOT greedy: " + ", ".join(bad)); sys.exit(1)
print("gate OK: template greedy (temp 0.0 / top_p 1.0 / top_k 0 / do_sample false)")
PY
done

# TAG|budget|message-kind (2 = terminate-only = MSG2)
CELLS=( "v9_b24_msg2|24576|2" )

for spec in "${CELLS[@]}"; do
  IFS='|' read -r TAG BUD KIND <<< "$spec"
  DEST="$OUT/$TAG"; mkdir -p "$DEST"
  USEMSG="$MSG2"; LABEL="terminate-only"

  EXTRA=$("$OMK_PY" - "$BUD" "$KIND" "$USEMSG" <<'PY'
import json,sys
bud,kind,msg = sys.argv[1], sys.argv[2], sys.argv[3]
# f16 KV: omk_eval hardcodes q8_0 into the serve cmd; these land last and win.
x = ["--cache-type-k","f16","--cache-type-v","f16",
     "--reasoning-format","deepseek","--reasoning-budget",bud]
if kind != "0":
    x += ["--reasoning-budget-message", msg]
print("backend_args.llama_extra=" + json.dumps(x))
PY
)
  TTB="generation.thinking_token_budget=$BUD"
  MGT="generation.max_gen_toks=$MAXGEN"

  for TPL in "${TPLS[@]}"; do
    SUM="$DEST/${TPL}/${TAG}/summary.json"
    if [ -f "$SUM" ] && "$OMK_PY" -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if isinstance(d.get("score"),(int,float)) else 1)' "$SUM" 2>/dev/null; then
      say "SKIP $TAG/$TPL — already has a numeric score"; continue
    fi
    say "=== CELL $TAG / $TPL  budget=$BUD  message=$LABEL  max_gen=$MAXGEN"
    {
      echo "cell:      $TAG"
      echo "model:     $GGUF"
      echo "sha256:    $(cut -d' ' -f1 < ${GGUF}.sha256)"
      echo "server:    $LC"
      echo "gpu:       $GPU   parallel: $NP   llama_ctx 262144 => 65536/slot (64k)"
      echo "kv:        f16 (forced; omk_eval hardcodes q8_0 at omk_eval.py:504)"
      echo "template:  $TPL (canonical, UNEDITED) sha=$(sha256sum "$OMK/eval/templates/${TPL}.yaml" | cut -c1-16)"
      echo "message:   $LABEL"
      echo "override:  $EXTRA"
      echo "override:  $TTB"
      echo "override:  $MGT"
      echo "sampler:   greedy (frozen: temp 0.0 / top_p 1.0 / top_k 0)"
      echo "compare:   128e b24_msg2/lcb_v6_77q = 69/77 = 0.8961 (same flags)"
      echo "started:   $(date -u +%FT%TZ)"
    } > "$DEST/STACK_${TPL}.txt"

    "$OMK_PY" "$OMK/eval/omk_eval.py" \
        --model "$GGUF" --template "$TPL" --backend llama \
        --served-name "$TAG" --results-dir "$DEST" \
        --gpus "$GPU" --parallel "$NP" \
        --metadata "$EXTRA" --metadata "$TTB" --metadata "$MGT" \
        >> "$OUT/${TAG}.log" 2>&1
    RC=$?
    say "CELL $TAG/$TPL rc=$RC"
    echo "finished:  $(date -u +%FT%TZ) rc=$RC" >> "$DEST/STACK_${TPL}.txt"
  done
done
say "V9_LCB_DONE"
