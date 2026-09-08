#!/usr/bin/env bash
# Ornith dose ladder p6/p12 -- run 3. Correct pipeline this time.
#
# RUN 1 died: arm shipped 0 mtp.* -> GGUF declared 41 blocks, shipped 40 ->
#   llama-imatrix: check_tensor_dims: tensor 'blk.40.attn_norm.weight' not found
# RUN 2 died (my wrong fix): passing --mtp-safetensors routes through ream's
#   build_mtp_layer, which crashes on this transformers version --
#   AttributeError: 'Qwen3_5MoeTextConfig' object has no attribute 'mlp_only_layers'
#   ...and it was the WRONG FIX anyway: it re-derives the MTP head per arm.
#
# CORRECT ROUTE (already in the repo, runhost/graft_mtp_ornith.py). Its policy:
#   "The MTP block is a speculative-decode drafter, skipped by llama.cpp's forward
#    graph and never enabled in our evals. Policy is therefore to hold it CONSTANT
#    and equal to the anchor's across arms, so it cannot explain any A-vs-B
#    difference. ANCHOR for Ornith = Arm A (Ornith-184e-Coder)."
# That is exactly how CoderX got its MTP (mtp.safetensors written 05:50, 29 min
# after its 05:21 shards). So: merge WITHOUT the mtp flag, then graft the anchor.
# Holding MTP constant is also what makes the dose ladder interpretable -- protect
# depth stays the only axis.
set -uo pipefail
L=/workspace/llama.cpp
G=/workspace/gguf
R=/workspace/ornith_prune/results
M=/workspace/models
W=/workspace/ornith_prune
SRC=$M/Ornith-1.5-35B-A3B
ANCHOR=$M/Ornith-184e-Coder          # the only 184e Ornith artifact carrying an MTP head
CAL=$W/calib_train.txt
PY=/workspace/venv-omk/bin/python
LOG=/workspace/logs/build_ladder3.log
mkdir -p /workspace/logs /workspace/ALERTS
say(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
refuse(){ say "REFUSE: $*"; echo "$*" > /workspace/ALERTS/NEEDS_DECISION_ladder3; exit 20; }
free_g(){ df --output=avail -BG /workspace | tail -1 | tr -dc '0-9'; }
need(){ [ "$(free_g)" -ge "$1" ] || refuse "need ${1}G free for $2, have $(free_g)G"; }
nmtp(){ "$PY" -c "import json,sys;w=json.load(open(sys.argv[1]+'/model.safetensors.index.json'))['weight_map'];print(sum(1 for k in w if k.startswith('mtp.')))" "$1" 2>/dev/null || echo 0; }

[ -s "$CAL" ] || refuse "no calib at $CAL"
grep -q "OMK_TOKENIZER_COPIED" "$W/ream/omk_ream_merge.py" || refuse "omk_ream_merge.py not tokenizer-fixed"
ANCHOR_MTP=$(nmtp "$ANCHOR")
[ "$ANCHOR_MTP" -gt 0 ] || refuse "anchor $ANCHOR carries NO mtp.* keys -- wrong anchor"
SRC_TOK=$(stat -c %s "$SRC/tokenizer_config.json")
say "anchor=$(basename $ANCHOR) mtp_keys=$ANCHOR_MTP ; src tokenizer_config=${SRC_TOK} B ; $(free_g)G free"
rm -f /workspace/ALERTS/NEEDS_DECISION_ladder /workspace/ALERTS/NEEDS_DECISION_ladder2

for DOSE in 6 12; do
  HM=$W/hybrid_maps/drop_map_184e_hybrid_p${DOSE}.json
  ARM=$M/Ornith-184e-P${DOSE}
  BF=$G/Ornith-184e-P${DOSE}-bf16.gguf
  IMA=$G/Ornith-184e-P${DOSE}.imatrix.gguf
  Q6=$G/Ornith-184e-P${DOSE}-Q6_K.gguf
  say "################ DOSE p${DOSE} ################"
  [ -s "$HM" ] || refuse "no hybrid map at $HM"

  # ---- merge (no --mtp-safetensors: that path crashes AND re-derives MTP) ----
  if [ ! -s "$ARM/config.json" ]; then
    need 90 "p${DOSE} merge"
    say "merge p${DOSE} (--merging none), map=$(basename "$HM")"
    "$PY" "$W/ream/omk_ream_merge.py" --model "$SRC" --merge-size 184 \
        --saliency reap --merging none --data-root "$W" --tokenizer-name ornith \
        --ream-dir /workspace/ream --dataset targeted --mix-ratio 1.0 \
        --save-path "$ARM" --seed 42 \
        --saliency-map "$R/competence_ornith35b_targeted.json" --drop-map "$HM" \
        --score tc --agg wmax --cat-weight corpus_targeted_lcb=2.0 \
        > "$W/p${DOSE}_build3.log" 2>&1
    grep -qa ">>> OMK_REAM_DONE" "$W/p${DOSE}_build3.log" \
      || { tail -10 "$W/p${DOSE}_build3.log" | tee -a "$LOG"; refuse "p${DOSE}: no OMK_REAM_DONE"; }
    say "p${DOSE} merge done"
  else
    say "p${DOSE} arm present, skipping merge"
  fi

  # ---- tokenizer + identity gates on the arm ----
  GOT=$(stat -c %s "$ARM/tokenizer_config.json" 2>/dev/null || echo 0)
  [ "$GOT" -eq "$SRC_TOK" ] || refuse "p${DOSE} tokenizer_config ${GOT} B vs src ${SRC_TOK} B -- STRIPPED"
  say "p${DOSE} tokenizer intact (${GOT} B)"
  say "p${DOSE} identity gate"
  "$PY" "$W/ream/verify_arm_identity.py" --base "$SRC" --built "$ARM" --drop-map "$HM" 2>&1 | tail -4
  [ "${PIPESTATUS[0]:-1}" -eq 0 ] || refuse "p${DOSE} identity gate FAILED"

  # ---- graft the anchor MTP (idempotent; SKIPs when already equal) ----
  BEFORE=$(nmtp "$ARM")
  if [ "$BEFORE" -ne "$ANCHOR_MTP" ]; then
    say "p${DOSE} grafting anchor MTP ($BEFORE -> expect $ANCHOR_MTP)"
    "$PY" "$W/runhost/graft_mtp_ornith.py" "$ANCHOR" "$ARM" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]:-1}" -eq 0 ] || refuse "p${DOSE}: MTP graft FAILED"
    # a grafted arm invalidates any bf16 built from the pre-graft arm
    if [ -s "$BF" ]; then rm -f "$BF" && say "p${DOSE}: removed pre-graft bf16 (declares 41 blocks, ships 40 -- unloadable, ~3.5 min to rebuild)"; fi
  else
    say "p${DOSE} MTP already present ($BEFORE keys)"
  fi
  AFTER=$(nmtp "$ARM")
  [ "$AFTER" -eq "$ANCHOR_MTP" ] || refuse "p${DOSE}: ${AFTER} mtp keys != anchor ${ANCHOR_MTP}"
  say "p${DOSE} MTP OK: ${AFTER} keys == anchor"

  # ---- bf16 + the block gate that would have caught run 1 in 3 seconds ----
  if [ ! -s "$BF" ]; then need 80 "p${DOSE} bf16"; say "p${DOSE} bf16 GGUF"
    "$PY" $L/convert_hf_to_gguf.py "$ARM" --outfile "$BF" --outtype bf16 2>&1 | tail -2; fi
  [ -s "$BF" ] || refuse "p${DOSE}: no bf16 GGUF"
  say "p${DOSE} block gate"
  "$PY" "$W/runhost/gguf_block_gate.py" "$BF" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]:-1}" -eq 0 ] || refuse "p${DOSE}: bf16 FAILED the block gate"

  # ---- imatrix (never deleted) + Q6_K ----
  if [ ! -s "$IMA" ]; then say "p${DOSE} imatrix (same corpus+flags as Coder/CoderX/256e)"
    $L/build/bin/llama-imatrix -m "$BF" -f "$CAL" -o "$IMA" -ngl 99 \
        --chunks -1 --parse-special --save-frequency 500 2>&1 | tail -3; fi
  [ -s "$IMA" ] || refuse "p${DOSE}: no imatrix"
  say "p${DOSE} imatrix $(stat -c %s "$IMA") B"

  if [ ! -s "$Q6" ]; then need 30 "p${DOSE} Q6_K"; say "p${DOSE} Q6_K"
    $L/build/bin/llama-quantize --imatrix "$IMA" "$BF" "$Q6" Q6_K 2>&1 | tail -3; fi
  [ -s "$Q6" ] || refuse "p${DOSE}: no Q6_K"
  sha256sum "$Q6" > "$Q6.sha256"
  say "p${DOSE} Q6_K $(du -h "$Q6" | cut -f1) -- ARTIFACT OK"
  rm -f "$BF" && say "reclaimed p${DOSE} bf16 ($(free_g)G free)"
done
say "LADDER3_BUILD_DONE p6 p12"
