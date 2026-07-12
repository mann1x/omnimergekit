#!/usr/bin/env bash
# ruler_t87_chain.sh — bs2 autonomous T87 RULER chain: base anchor -> extended ladder.
# base ref (1 GPU) MUST finish before the extended TP=2 ladder (2 GPU); sequential.
# The extended driver ANCHOR_GATE validates the proportional_yarn->vLLM-yarn rope
# mapping against THIS base 256k anchor before spending on 384k/512k, so the chain
# is safe unattended: a bad rope stops the ladder at 256k, never wasting 512k.
set -uo pipefail
R=/srv/ml/repos/omnimergekit/recipes/gemma4/longctx_512k
LOG=/srv/ml/longctx/ruler_t87_chain.log
exec >>"$LOG" 2>&1
echo "=== T87 chain start $(date "+%F %T %Z") ==="
echo "--- [A] base RULER reference (anchor: VT 32k/256k + NIAH 256k) ---"
bash "$R/ruler_ref_base_a4b.sh"
echo "--- [A] base ref exited rc=$? $(date "+%F %T %Z") ---"
echo "--- [B] extended 512k merge->serve->ladder ---"
bash "$R/ruler_ext_512k.sh"
echo "=== T87 chain end rc=$? $(date "+%F %T %Z") ==="
