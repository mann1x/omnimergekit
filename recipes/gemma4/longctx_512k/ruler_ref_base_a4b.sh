#!/usr/bin/env bash
# ruler_ref_base_a4b.sh — BASE Gemma-4-26B-A4B-it RULER reference at 32k + 256k.
#
# WHY: the released base (max_position_embeddings=262144, NO rope_scaling) is the
# long-context CEILING the YaRN-extended model must MATCH at <=256k before we can
# trust its 512k numbers. This produces that anchor: native RULER (vt) at 32k and
# 256k on the untouched base. 512k is deliberately ABSENT — the base physically
# cannot serve it (that's the whole point of the extension).
#
# WHY native RULER (not lm-eval): there is no lm-eval RULER template above 256k,
# and we want one runner for both the base anchor here and the extended ladder
# later. omk_eval's `ruler_native` backend has a free-int --ctx-tokens and an
# inlined scorer (no nemo dep), reachable to 512k.
#
# GATED + SINGLE-GPU: refuses while any trainer owns the GPUs (the flagship DDP@32k
# run holds both). The base needs only ONE 96GB GPU — 26B bf16 ~52GB + 256k KV
# ~21GB ~= 73GB (sliding layers cache only a 1024 window, so KV is far below a
# dense estimate); NO tensor-parallel at 256k. Pin GPU via GPU=N, default 0.
#
# Smoke -> 32k -> 256k, in that order: the 4k smoke is a ~2-min serve/plumbing gate
# so we never discover a broken serve 10 minutes into the 256k prompt build.
set -uo pipefail

BASE_DIR="${BASE_DIR:-/srv/ml/google/gemma-4-26B-A4B-it}"
SERVED="${SERVED:-gemma-4-26B-A4B-it-base}"
TOK="${TOK:-$BASE_DIR}"
GPU="${GPU:-0}"
PORT="${PORT:-8196}"
RESULTS="${RESULTS:-/srv/ml/longctx/ruler_ref_base}"

OMK_PY="${OMK_PY:-/srv/ml/envs/envs/omnimergekit/bin/python}"
VLLM_PY="${VLLM_PY:-/srv/ml/envs/envs/vllm/bin/python}"
OMK="${OMK:-/srv/ml/repos/omnimergekit/eval/omk_eval.py}"
TMPL_DIR="${TMPL_DIR:-/srv/ml/repos/omnimergekit/eval/templates}"

# tier := "template:max_model_len:metadata".  max-model-len = ctx + output headroom.
# The 3rd field is an optional --metadata override (empty for smoke/32k).
#   smoke   4k x5  -> 8192  (plumbing gate)
#   vt_32k  32768  -> 33792 (32768 + 1024)
#   vt_256k served at native 262144, but the RULER prompt target is trimmed to
#           261120 via --metadata ctx_tokens=261120 so the 120-token answer fits
#           under the base's native ceiling. Uses the CANONICAL vt_256k template
#           (same one the extended model runs at full 262144) — no clone.
TIERS=(
  "ruler_native_smoke:8192:"
  "ruler_native_vt_32k:33792:"
  "ruler_native_vt_256k:262144:ctx_tokens=261120"
)

mkdir -p "$RESULTS"
LOG="$RESULTS/ruler_ref_base.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== ruler_ref_base_a4b $(date '+%F %T %Z') — GPU=$GPU port=$PORT ==="

# --- gate: no trainer may be running (it owns both GPUs) ----------------------
if pgrep -f "phase1_train_yarn_lora" >/dev/null 2>&1; then
  echo "[gate] a trainer (phase1_train_yarn_lora*) is running — GPUs busy. NOT preempting."
  pgrep -af "phase1_train_yarn_lora" | sed 's/^/[gate]   /' | cut -c1-110
  echo "[gate] re-run this script once it exits (post32k_chain handles that window)."
  exit 0
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
if [ -z "$used" ]; then echo "[gate] could not query GPU$GPU via nvidia-smi"; exit 1; fi
if [ "$used" -gt 4000 ]; then
  echo "[gate] GPU$GPU has ${used} MiB used (>4GB) — not idle. Pick a free one: GPU=N $0"
  exit 0
fi
echo "[gate] GPU$GPU idle (${used} MiB) — proceeding."

# --- preflight inputs (FATAL-loud) -------------------------------------------
for p in "$OMK_PY" "$VLLM_PY" "$OMK" "$BASE_DIR" "$TMPL_DIR"; do
  [ -e "$p" ] || { echo "FATAL: missing $p" >&2; exit 1; }
done

# --- synth preprocessor_config.json (vLLM 0.20.x requires it; base dir lacks it)
# Additive + idempotent: leaves an existing file untouched, never edits weights /
# tokenizer. Mirrors pod_eval_31b.sh: feature_extractor field, whole-proc fallback.
"$OMK_PY" - "$BASE_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
pp = d / "preprocessor_config.json"
if pp.exists():
    print(f"[synth] {pp} already present — leaving as-is"); sys.exit(0)
proc_f = d / "processor_config.json"
if not proc_f.exists():
    sys.exit(f"FATAL: no preprocessor_config.json AND no processor_config.json in {d}")
proc = json.loads(proc_f.read_text())
fe = proc.get("feature_extractor", {}) or proc
pp.write_text(json.dumps(fe, indent=2))
print(f"[synth] wrote {pp} from processor_config.json ({len(fe)} keys)")
PYEOF

# --- run the tiers (smoke gates the reals) -----------------------------------
for tier in "${TIERS[@]}"; do
  tmpl="${tier%%:*}"; rest="${tier#*:}"; mml="${rest%%:*}"; md="${rest#*:}"
  [ -f "$TMPL_DIR/$tmpl.yaml" ] || { echo "FATAL: template $TMPL_DIR/$tmpl.yaml missing" >&2; exit 1; }
  md_args=(); [ -n "$md" ] && md_args=(--metadata "$md")
  # Single-flight at >=200k ctx: two concurrent 256k contexts (~52GB model +
  # 2x ~21GB KV) can exceed 0.92x96GB. Deterministic over auto-clamp here.
  par_args=(); [ "$mml" -ge 200000 ] && par_args=(--parallel 1)
  echo "--- $tmpl (max-model-len=$mml${md:+, metadata=$md}${par_args:+, parallel=1}) $(date '+%T %Z') ---"
  if ! PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH" \
       VLLM_PYTHON="$VLLM_PY" \
       CUDA_VISIBLE_DEVICES="$GPU" \
       "$OMK_PY" "$OMK" \
         --model "$BASE_DIR" \
         --template "$tmpl" \
         --backend vllm \
         --max-model-len "$mml" \
         "${md_args[@]}" \
         "${par_args[@]}" \
         --served-name "$SERVED" \
         --tokenizer "$TOK" \
         --port "$PORT" \
         --results-dir "$RESULTS"; then
    echo "[run] $tmpl FAILED — stopping; later tiers need a healthy serve. See $LOG."
    break
  fi
  echo "[run] $tmpl done $(date '+%T %Z')"
done

# --- tabulate from summary.json .score (NEVER raw results_*.json) -------------
echo "=== BASE A4B RULER reference — scores (summary.json .score) ==="
"$OMK_PY" - "$RESULTS" <<'PYEOF'
import json, sys, glob, os
root = sys.argv[1]
rows = []
for sj in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
    try:
        d = json.load(open(sj))
    except Exception as e:
        rows.append((os.path.relpath(sj, root), "ERR", str(e)[:30])); continue
    score = d.get("score", d.get("headline", d.get("pass_at_1")))
    rows.append((os.path.relpath(sj, root), score, d.get("metric", "")))
if not rows:
    print("(no summary.json yet under", root, ")"); sys.exit(0)
print(f"{'result':58s} {'score':>8s}  metric")
for name, score, metric in rows:
    s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    print(f"{name:58s} {s:>8s}  {metric}")
PYEOF
echo "=== ruler_ref_base_a4b end $(date '+%F %T %Z') ==="
