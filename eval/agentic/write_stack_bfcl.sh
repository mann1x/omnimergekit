#!/usr/bin/env bash
# write_stack_bfcl.sh — STACK.txt for one agentic-bench arm (EVAL_PROTOCOL §1.4.5).
# A cohort whose entries' STACK.txt differ on anything but the DECLARED VARIABLE
# is invalid. Diff the arms' STACK.txt before tabulating — that diff IS the audit.
set -uo pipefail
ARM="$1"; OUT="$2"; TPL="$3"; GGUF="$4"; LLAMA_BIN="$5"; ENVD="$6"
CTX="$7"; NP="$8"; MAXTOK="$9"; THINK="${10}"; CATS="${11}"; LIMIT="${12}"
# 13th arg = the --reasoning-format the caller ACTUALLY passed to llama-server.
# It used to be hardcoded to "deepseek" here, which made STACK.txt assert a flag
# the run never used (2026-09-13: the unbounded cell ran with "auto"). A
# provenance file that states an unobserved value is worse than one that admits
# the gap, so an absent arg prints a loud placeholder, never a guess.
FMT="${13:-}"
mkdir -p "$OUT"
{
echo "# STACK.txt — EVAL_PROTOCOL §1.4.5 (agentic bench, §8)"
echo "cohort            : ${OMK_COHORT:-unset}"
echo "arm               : $ARM"
echo "DECLARED VARIABLE : ${OMK_DECLARED_VARIABLE:-unset}"
echo "host              : $(hostname)  $(date -u +%FT%TZ)"
echo
echo "## harness"
"$ENVD/bin/python" -c "import importlib.metadata as m
for p in ['inspect_ai','inspect_evals','openai']:
    try: print('%-14s %s'%(p, m.version(p)))
    except Exception: print('%-14s MISSING'%p)"
echo "python         $("$ENVD/bin/python" -V 2>&1)"
D="$ENVD/lib/python3.12/site-packages"
for f in "$D"/inspect_evals-*.dist-info/direct_url.json; do
    [ -f "$f" ] && echo "inspect_evals commit $(sed -n 's/.*"commit_id": "\([a-f0-9]*\)".*/\1/p' "$f")"
done
echo "benchmark      inspect_evals/bfcl categories=$CATS limit=$LIMIT"
echo "scorer         bfcl_scorer (backend state + execution result; deterministic, NO LLM judge)"
echo
echo "## engine"
echo "llama.cpp bin  $LLAMA_BIN"
PEGN=$(strings "$LLAMA_BIN/libllama-common.so.0.0.1" | grep -c LLAMA_CHAT_PEG_STRICT || true)
echo "PEG-degrade    sentinel_hits=$PEGN  (>=1 REQUIRED)"
echo "libllama-common sha256 $(sha256sum "$LLAMA_BIN/libllama-common.so.0.0.1" | cut -d' ' -f1)"
echo
echo "## weights"
echo "gguf           $GGUF"
echo "gguf bytes     $(stat -c %s "$GGUF")"
echo "gguf sha256    $(sha256sum "$GGUF" | cut -d' ' -f1)"
echo
echo "## chat template"
echo "template       $TPL"
echo "template bytes $(stat -c %s "$TPL")"
echo "template sha256 $(sha256sum "$TPL" | cut -d' ' -f1)"
echo
echo "## serve + sampler"
echo "-c $CTX -np $NP -ngl 99 --no-warmup --jinja --reasoning-format ${FMT:-<NOT RECORDED: caller passed no format arg>} --reasoning-budget $THINK"
echo "per_slot_ctx   $(( CTX / NP ))  (required >= max(max_tokens $MAXTOK, thinking $THINK+4096))"
echo "sampler        GREEDY temperature=0.0  (EVAL_PROTOCOL §1.0 anchor)"
echo "max_tokens     $MAXTOK ; max_connections $NP"
echo
echo "## gpu"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
} > "$OUT/STACK.txt"
echo "wrote $OUT/STACK.txt"
