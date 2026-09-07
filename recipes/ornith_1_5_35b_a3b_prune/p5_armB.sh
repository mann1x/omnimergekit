#!/usr/bin/env bash
# P4 -> P5 -> P6. Runs after armA_imatrix.sh lands Arm A's imatrix + quants.
#
#   A-eval : lcb_v6_77q + multipl_e_100, sampler `recommended`  (greedy is non-viable
#            for this family; basis matches the published A3B-Coder/CoderX cohort)
#   B-build: REAP saliency dump -> armE keep set DERIVED FROM THE DUMP (no armE build)
#            -> make_hybrid_dropmap --protect 24 -> omk_ream_merge --merging none
#            -> hard identity gate
#   B-imat : same AC corpus, same flags as Arm A and as the 256e run
#   B-eval : identical basis to A
#
# DISK: bf16 GGUFs are regenerable from the HF weights (~10 min) and are reclaimed
# after their quants exist. Every large step is preceded by a free-space gate that
# ABORTS rather than filling the volume. Nothing non-regenerable is ever removed.
set -uo pipefail
ts(){ date -u +%H:%M:%S; }
say(){ echo "[$(ts)] $*"; }
free_g(){ df --output=avail -BG /workspace | tail -1 | tr -dc '0-9'; }
need(){ local g=$1 what=$2; local f; f=$(free_g); [ "$f" -ge "$g" ] || { say "ABORT: need ${g}G for $what, have ${f}G"; exit 20; }; }

L=/workspace/llama.cpp
G=/workspace/gguf
R=/workspace/ornith_prune/results
M=/workspace/models
W=/workspace/ornith_prune
SRC=$M/Ornith-1.5-35B-A3B
CAL=$W/calib_train.txt
PY=/workspace/venv-omk/bin/python
RES=$W/eval_results
export REAP_BASE_MODEL="$SRC"
export PYTHONPATH=/workspace/omk/recipes/qwen3_6_35b_a3b_prune/ream:/workspace/ream:${PYTHONPATH:-}
exec 9>/var/lock/ornith_p5_armB.lock
flock -n 9 || { say "already running - refusing"; exit 1; }

run_eval(){   # $1=gguf  $2=served-name  $3=port
  export LLAMA_BIN=$L/build/bin OMK_PYTHON=/usr/bin/python3 HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES=0
  cd /workspace/omk
  for T in lcb_v6_77q multipl_e_100; do
    say "EVAL $2 / $T (sampler=recommended)"
    MPE_MODE=native MPE_HARNESS=/workspace/MultiPL-E \
    python3 eval/omk_eval.py --backend llama --template "$T" --quant q6_k \
      --model "$1" --tokenizer "$SRC" --served-name "$2" --port "$3" \
      --results-dir "$RES" --parallel 2 \
      --sampler-profile qwen3_6 --sampler recommended 2>&1 | tail -5
    say "EVAL $2 / $T finished"
  done
  cd "$W"
}

# ---------------- gate on Arm A imatrix+quants ----------------
say "gate: waiting for ARM_A_IMATRIX_QUANTS_DONE"
for i in $(seq 1 960); do
  grep -qa "ARM_A_IMATRIX_QUANTS_DONE" /workspace/logs/armA_imatrix.log 2>/dev/null && break
  sleep 30
done
grep -qa "ARM_A_IMATRIX_QUANTS_DONE" /workspace/logs/armA_imatrix.log 2>/dev/null \
  || { say "ABORT: Arm A imatrix/quants never completed"; exit 2; }
AQ6=$G/Ornith-184e-Coder-Q6_K.gguf
[ -s "$AQ6" ] || { say "ABORT: no Arm A Q6_K"; exit 3; }
say "Arm A ready: $(stat -c%s "$AQ6") bytes"

# reclaim Arm A bf16 (regenerable) now that its quants exist
[ -s "$G/Ornith-184e-Coder-bf16.gguf" ] && rm -f "$G/Ornith-184e-Coder-bf16.gguf" && say "reclaimed Arm A bf16 GGUF ($(free_g)G free)"

# ---------------- P5 step 1: REAP saliency dump ----------------
DUMP=$W/dump_reap.log
if [ ! -s "$W/reap_saliency.json" ] && ! grep -qa "OMK_REAP_DUMP_DONE" "$DUMP" 2>/dev/null; then
  need 20 "REAP dump"
  say "P5.1 REAP saliency dump"
  "$PY" "$W/runhost/dump_reap_saliency.py" --model "$SRC" --out "$W/reap_saliency.json" \
      --data-root "$W" --merge-size 184 --tokenizer-name ornith --seed 42 --ream-dir /workspace/ream > "$DUMP" 2>&1
  say "dump rc=$?"
fi
grep -qa "OMK_REAP_DUMP_DONE" "$DUMP" 2>/dev/null || { say "ABORT: no OMK_REAP_DUMP_DONE sentinel"; tail -5 "$DUMP"; exit 4; }
say "P5.1 done"

# ---------------- P5.1b: EOG terminator emit-map ----------------
# Terminator health is NOT visible in the score: an arm that sheds end-of-generation
# experts overruns, emits a malformed thinking channel, the chat parser 500s, and the
# eval records an EMPTY completion that scores as an ordinary wrong answer. Build the
# emit-map on the SAME corpus as the competence map so the two are commensurable.
EOG=$R/eog_emit_map_ornith.json
if [ ! -s "$EOG" ]; then
  need 10 "EOG emit map"
  say "P5.1b EOG emit-map (eog-ids 248046,248044; ~7 min)"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$W/runhost/build_eog_emit_map.py" \
      --model "$SRC" --corpus "$R/router_calib_corpus_ornith_full.jsonl" \
      --eog-ids 248046,248044 --limit 0 --out "$EOG" >>"$W/eog_build.log" 2>&1
  say "eog build rc=$?"
fi
[ -s "$EOG" ] || { say "ABORT: no EOG emit map at $EOG"; exit 14; }
say "P5.1b done"

# ---------------- Arm A eval: ALREADY DONE 2026-09-07 01:34 UTC ----------------
# lcb_v6_77q=0.7273 (CAPPED 4/77) multipl_e_100=0.8300 -- see eval_results/. Removed
# here so resuming for Arm B does not re-run a finished 2h13m eval.
say "Arm A eval: skipped (already complete)"

# ---------------- P5 steps 2-5: hybrid map + build + identity gate ----------------
HM=$W/hybrid_maps/drop_map_184e_hybrid_p24.json
ARMB=$M/Ornith-184e-CoderX
if [ ! -s "$ARMB/config.json" ]; then
  need 60 "Arm B build"
  say "P5.2 keepsets: armD RECOVERED from Arm A weights, armE DERIVED from the dump"
  KS=$W/keepsets.json
  if [ ! -s "$KS" ]; then
    "$PY" "$W/runhost/recover_keepsets.py" "$KS" armD "$M/Ornith-184e-Coder" 2>&1 | tail -4
    [ -s "$KS" ] || { say "ABORT: recover_keepsets produced nothing"; exit 12; }
    # armE = top-184 per layer by dumped REAP saliency. NOTE: this makes
    # make_hybrid_dropmap GATE 1 (dump reproduces armE) TAUTOLOGICAL -- there is no
    # shipped Ornith armE to check against. Gates 2 and 3 remain meaningful.
    "$PY" - "$KS" "$W/reap_saliency.json" <<'PYG'
import json, sys
ks_p, reap_p = sys.argv[1], sys.argv[2]
ks = json.load(open(ks_p)); reap = json.load(open(reap_p))
armE = {}
for layer, sal in reap.items():
    if isinstance(sal, dict):
        order = sorted(sal, key=lambda e: -float(sal[e]))
        armE[str(layer)] = sorted(int(e) for e in order[:184])
    elif isinstance(sal, list):
        order = sorted(range(len(sal)), key=lambda i: -float(sal[i]))
        armE[str(layer)] = sorted(order[:184])
if not armE:
    sys.exit("ABORT: could not derive armE from the dump (unexpected shape)")
bad = [l for l, v in armE.items() if len(v) != 184]
if bad:
    sys.exit(f"ABORT: layers without exactly 184 keeps: {bad[:5]}")
ks["armE"] = armE
json.dump(ks, open(ks_p, "w"))
print(f"armE derived from dump: {len(armE)} layers x 184 keeps  [GATE 1 is now tautological]")
PYG
    [ $? -eq 0 ] || { say "ABORT: armE derivation failed"; exit 13; }
  fi
  say "P5.3 hybrid map (protect 24)"
  "$PY" "$W/runhost/make_hybrid_dropmap.py" --reap "$W/reap_saliency.json" \
      --keepsets "$W/keepsets.json" \
      --map "$R/competence_ornith35b_targeted.json" \
      --drop-map "$R/drop_map_184e_coder_tc.json" \
      --score tc --agg wmax \
      --cat-weight corpus_targeted_lcb=1.5 --cat-weight corpus_targeted_mpe=1.5 \
      --keep 184 --protect 24 --out-dir "$W/hybrid_maps" 2>&1 | tail -8
  [ -s "$HM" ] || { say "ABORT: no hybrid map at $HM"; exit 5; }
  say "P5.3b EOG keepset gate (hybrid p24 vs the coder tc map)"
  "$PY" "$W/runhost/eog_keepset_gate.py" --eog-map "$EOG" \
      --arm "coder_tc=$R/drop_map_184e_coder_tc.json" \
      --arm "coderx_p24=$HM" \
      --reference coder_tc --max-drop 0.02 \
      --json-out "$R/eog_keepset_gate_p24.json" 2>&1 | tee -a "$W/eog_gate.log"
  # Report-only by default: on Qwen3.6 this cross-check found NO EOG signal, so a
  # regression here is a flag for a human, not an automatic build abort. Add
  # --fail-on-regression once the Ornith result justifies hard-gating.
  grep -qa "EOG_KEEPSET_GATE PASS" "$W/eog_gate.log" \
      || say "WARNING: EOG keepset gate reported a REGRESSION -- see $R/eog_keepset_gate_p24.json"
  say "P5.4 build Arm B (--merging none)"
  "$PY" "$W/ream/omk_ream_merge.py" --model "$SRC" --merge-size 184 \
      --saliency reap --merging none --data-root "$W" --tokenizer-name ornith --ream-dir /workspace/ream --dataset targeted --mix-ratio 1.0 \
      --save-path "$ARMB" --seed 42 \
      --saliency-map "$R/competence_ornith35b_targeted.json" --drop-map "$HM" \
      --score tc --agg wmax --cat-weight corpus_targeted_lcb=2.0 > "$W/armB_build.log" 2>&1
  grep -qa ">>> OMK_REAM_DONE" "$W/armB_build.log" || { say "ABORT: no OMK_REAM_DONE"; tail -12 "$W/armB_build.log"; exit 6; }
fi
[ -s "$ARMB/config.json" ] || { say "ABORT: Arm B not built"; exit 7; }
say "P5.5 identity gate"
"$PY" "$W/ream/verify_arm_identity.py" --base "$SRC" --built "$ARMB" --drop-map "$HM" 2>&1 | tail -8
[ ${PIPESTATUS[0]:-1} -eq 0 ] || { say "ABORT: identity gate FAILED"; exit 8; }
say "Arm B built + identity verified: $(du -sh "$ARMB" | cut -f1)"

# ---------------- Arm B bf16 + AC imatrix + quants ----------------
BBF=$G/Ornith-184e-CoderX-bf16.gguf
BIMA=$G/Ornith-184e-CoderX.imatrix.gguf
if [ ! -s "$BBF" ]; then need 60 "Arm B bf16"; say "Arm B bf16 GGUF"
  "$PY" $L/convert_hf_to_gguf.py "$ARMB" --outfile "$BBF" --outtype bf16 2>&1 | tail -3; fi
[ -s "$BBF" ] || { say "ABORT: no Arm B bf16"; exit 9; }
if [ ! -s "$BIMA" ]; then say "Arm B AC imatrix (same corpus + flags as Arm A / 256e)"
  $L/build/bin/llama-imatrix -m "$BBF" -f "$CAL" -o "$BIMA" -ngl 99 \
      --chunks -1 --parse-special --save-frequency 500 2>&1 | tail -5; fi
[ -s "$BIMA" ] || { say "ABORT: no Arm B imatrix"; exit 10; }
need 45 "Arm B quants"
for Q in Q6_K Q4_K_M; do
  OUT=$G/Ornith-184e-CoderX-$Q.gguf
  [ -s "$OUT" ] || { say "Arm B quantize $Q WITH imatrix"
    $L/build/bin/llama-quantize --imatrix "$BIMA" "$BBF" "$OUT" "$Q" 96 2>&1 | tail -3; }
done
BQ6=$G/Ornith-184e-CoderX-Q6_K.gguf
[ -s "$BQ6" ] || { say "ABORT: no Arm B Q6_K"; exit 11; }
rm -f "$BBF" && say "reclaimed Arm B bf16 GGUF ($(free_g)G free)"

# ---------------- Arm B eval ----------------
run_eval "$BQ6" ornith184e_coderx_q6k 8093
say "ARM_B_EVAL_DONE"
say "P4P5_PIPELINE_DONE"
