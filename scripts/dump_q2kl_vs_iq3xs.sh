#!/usr/bin/env bash
# dump_q2kl_vs_iq3xs.sh — build Q2_K_L (clean) and IQ3_XS (ruminates) from the
# same F16, then dump the per-tensor quant assignment so we can see WHY Q2_K_L
# (fewer bpw) terminates while IQ3_XS does not. CPU-only, no GPU. Probe, deleted.
set -uo pipefail
QUANT=/opt/llama.cpp/build/bin/llama-quantize
PY=/srv/ml/envs/envs/omnimergekit/bin/python
GD=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF
F16="$GD/gemma-4-A4B-98e-v7-coder-it-F16.gguf"; IMAT="$GD/imatrix.dat"
ST=/mnt/sdc/ml/cd_fixed_v7/qkl_probe; mkdir -p "$ST"
LOG=/srv/ml/logs/dump_q2kl_vs_iq3xs.txt; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
for t in Q2_K_L IQ3_XS; do
  echo "== build $t $(date -u) =="
  "$QUANT" --imatrix "$IMAT" "$F16" "$ST/$t.gguf" "$t" >"$ST/.q_$t.log" 2>&1 || { echo "FATAL $t"; tail -3 "$ST/.q_$t.log"; }
done
"$PY" - <<'PYEOF'
from collections import Counter
from gguf import GGUFReader
ST="/mnt/sdc/ml/cd_fixed_v7/qkl_probe"
KEYS=["token_embd","output.weight","attn_v","attn_k","attn_q","attn_output",
      "ffn_down_exps","ffn_gate_up_exps","ffn_down.weight","ffn_gate.weight","ffn_up.weight"]
def types(p):
    return {t.name: t.tensor_type.name for t in GGUFReader(p).tensors}
A=types(f"{ST}/Q2_K_L.gguf"); B=types(f"{ST}/IQ3_XS.gguf")
print("\n=== histogram ===")
print("Q2_K_L :", dict(Counter(A.values())))
print("IQ3_XS :", dict(Counter(B.values())))
print("\n=== key tensor classes (Q2_K_L  vs  IQ3_XS) ===")
for k in KEYS:
    a=dict(Counter(v for n,v in A.items() if k in n))
    b=dict(Counter(v for n,v in B.items() if k in n))
    print(f"  {k:20s} {str(a):28s} | {b}")
PYEOF
echo "[done] $(date -u)"
