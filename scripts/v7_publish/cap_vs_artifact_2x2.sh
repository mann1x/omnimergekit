#!/usr/bin/env bash
# cap_vs_artifact_2x2.sh — is the low-quant rumination an attention artifact, an
# EXPERT i-quant artifact, or a genuine capability collapse? For each ruminating
# tier, build a 2x2: {attn protected?} x {experts k-quant?} on the same F16, all
# at the SAME base i-quant ftype, and termination-smoke each.
#   A plain        : attn=i-quant   experts=i-quant   (baseline ruminator)
#   B attn-protect : attn=Q5_K      experts=i-quant   (attention hypothesis)
#   C expert-kquant: attn=i-quant   experts=k-quant   ("it's the experts")
#   D both         : attn=Q5_K      experts=k-quant
# Verdict: B ruminates & C stops => expert i-quant artifact (NOT attn, NOT capability).
#          B stops               => attention was it.
#          all ruminate          => capability collapse at that bit-width.
# coder model, CPU build + GPU0 smoke, GGUFs deleted after each smoke.
set -uo pipefail
GPU=0
QUANT=/opt/llama.cpp/build/bin/llama-quantize
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SMOKE=/srv/ml/scripts/smoke_gguf.sh
GD=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF
F16="$GD/gemma-4-A4B-98e-v7-coder-it-F16.gguf"; IMAT="$GD/imatrix.dat"
ST=/mnt/sdc/ml/cd_fixed_v7/cap_probe; mkdir -p "$ST"
LOG=/srv/ml/logs/cap_vs_artifact_2x2.txt; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1

emit_map() { # $1=arm $2=kquant $3=outfile
  "$PY" - "$1" "$2" "$3" <<'PY'
import sys
arm,kq,out=sys.argv[1:4]
L=[]
for i in range(30):
    if arm in ("B","D"):
        for t in ("attn_q","attn_k","attn_v","attn_output"):
            L.append(f"blk.{i}.{t}.weight=Q5_K")
    if arm in ("C","D"):
        for t in ("ffn_down","ffn_gate","ffn_up","ffn_down_exps","ffn_gate_up_exps"):
            L.append(f"blk.{i}.{t}.weight={kq}")
open(out,"w").write("\n".join(L)+("\n" if L else ""))
PY
}

# tier:kquant   (full 2x2 for the two canonical ruminators; A/B for published IQ3_M)
declare -A ARMS=([IQ3_XS]="A B C D" [IQ2_S]="A B C D" [IQ3_M]="A B C")
declare -A KQ=([IQ3_XS]=Q3_K [IQ2_S]=Q2_K [IQ3_M]=Q3_K)
port=8300
declare -A V
for tier in IQ3_XS IQ3_M IQ2_S; do
  for arm in ${ARMS[$tier]}; do
    out="$ST/${tier}_${arm}.gguf"; map="$ST/${tier}_${arm}.map"
    echo "==== build $tier arm-$arm (kq=${KQ[$tier]}) $(date -u) ===="
    if [ "$arm" = "A" ]; then
      "$QUANT" --imatrix "$IMAT" "$F16" "$out" "$tier" >"$ST/.q_${tier}_${arm}.log" 2>&1
    else
      emit_map "$arm" "${KQ[$tier]}" "$map"
      "$QUANT" --imatrix "$IMAT" --tensor-type-file "$map" "$F16" "$out" "$tier" >"$ST/.q_${tier}_${arm}.log" 2>&1
    fi
    if [ $? -ne 0 ] || [ ! -f "$out" ]; then echo "[FATAL] build $tier $arm"; tail -4 "$ST/.q_${tier}_${arm}.log"; V[$tier/$arm]=BUILD-FAIL; continue; fi
    bpw=$(grep -oE "[0-9.]+ BPW" "$ST/.q_${tier}_${arm}.log" | tail -1)
    echo "  built $(du -h "$out"|cut -f1) ($bpw)"
    res=$(bash "$SMOKE" "$out" "$port" "$GPU" 2048 2>&1)
    echo "$res" | grep -E "STOP-ok|RUMINATE|500-ERR|RESULT:"
    V[$tier/$arm]="$(echo "$res" | grep -oE "RESULT: [0-9]+/5 STOP" | head -1) ($bpw)"
    rm -f "$out" "$map"
    port=$((port+1))
  done
done
echo "######## CAPABILITY-vs-ARTIFACT 2x2 SUMMARY ########"
echo "  arm A=plain  B=attn-protect  C=expert-kquant  D=both"
for k in "${!V[@]}"; do printf "  %-14s %s\n" "$k" "${V[$k]}"; done | sort
echo "[done] $(date -u)"
