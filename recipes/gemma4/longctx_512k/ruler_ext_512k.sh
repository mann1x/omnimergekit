#!/usr/bin/env bash
# ruler_ext_512k.sh — merge → serve → RULER-ladder driver for the YaRN-EXTENDED
# Gemma-4-26B-A4B 250M checkpoint. Runs ON bs2 (2× PRO 6000 96GB, NO NVLink).
#
# PIPELINE:
#   1. surgical-merge the 512k LoRA adapter (ckpt-000952) onto the 128e YaRN base
#      (yarn_cfg_98e) → gemma-4-26B-A4B-it-512k   [CPU, idempotent]
#   2. derive a vLLM-servable dir by translating the rope config
#      proportional_yarn → vLLM 'yarn'            [hardlinks, seconds, re-runnable]
#   3. launch ONE persistent TP=2 vLLM serve at max-model-len 524288 (the model's
#      max_position_embeddings ceiling). TP=2 is MANDATORY at 512k: 52GB weights +
#      ~42GB bf16 KV (5 full-attn layers × 80KB/tok) ≈ 94GB > one GPU's 0.92×96.
#      omk_eval's planner only tensor-parallels when the WEIGHTS don't fit one GPU
#      (they do), so it would OOM at KV alloc — hence the hand-launched TP=2 serve
#      + `omk_eval --no-server`. One 524288 serve covers every rung (smaller ctx
#      just uses fewer KV blocks).
#   4. RULER ladder via omk_eval --no-server, in order, with gates:
#        smoke(4k) → vt_32k → vt_256k → mk1_256k → [ANCHOR GATE] → vt_384k →
#        mk1_384k → vt_512k → mk1_512k
#
# THE ANCHOR GATE (rope-correctness): the proportional_yarn→'yarn' mapping is a
# HYPOTHESIS (vLLM 'yarn' = YaRN over the STANDARD base; training used YaRN over
# Gemma 4's proportional base — they *should* coincide but are not proven equal).
# So before spending ~3h on 384k/512k we REQUIRE the EXTENDED model's 256k RULER
# scores to match the BASE 256k anchor (ruler_ref_base) within GATE_TOL. If they
# diverge, the rope mapping is wrong: STOP, re-run make_serve_config.py with a
# different --rope-type (cheap), re-serve. Override with ANCHOR_OVERRIDE=1.
#
# GATED: refuses while any trainer owns the GPUs (TP=2 needs BOTH). Greedy/50.
set -uo pipefail

SCR="${SCR:-$(cd "$(dirname "$0")" && pwd)}"   # dir holding merge/serve helpers
LONGCTX="${LONGCTX:-/srv/ml/longctx}"

YARN_BASE="${YARN_BASE:-$LONGCTX/yarn_cfg_98e}"                 # 128e YaRN base (merge SRC)
ADAPTER="${ADAPTER:-$LONGCTX/ckpt_98e_ddp32k_fa2/ckpt-000952}"  # 250M LoRA ckpt
MERGED="${MERGED:-$LONGCTX/gemma-4-26B-A4B-it-512k}"            # canonical merged (proportional_yarn)
SERVE="${SERVE:-$LONGCTX/gemma-4-26B-A4B-it-512k-vllm-yarn}"    # vLLM serve dir ('yarn')
SERVED="${SERVED:-gemma-4-26B-A4B-it-512k}"
ROPE_TYPE="${ROPE_TYPE:-yarn}"

RESULTS="${RESULTS:-$LONGCTX/ruler_ext_512k}"
REF_RESULTS="${REF_RESULTS:-$LONGCTX/ruler_ref_base}"           # base anchor (ruler_ref_base_a4b.sh)
REF_SERVED="${REF_SERVED:-gemma-4-26B-A4B-it-base}"

PORT="${PORT:-8197}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAXLEN="${MAXLEN:-524288}"          # = max_position_embeddings ceiling
GATE_TOL="${GATE_TOL:-0.10}"        # abs RULER-score delta extended-vs-base @256k

OMK_PY="${OMK_PY:-/srv/ml/envs/envs/omnimergekit/bin/python}"
VLLM_PY="${VLLM_PY:-/srv/ml/envs/envs/vllm/bin/python}"
OMK="${OMK:-/srv/ml/repos/omnimergekit/eval/omk_eval.py}"
TMPL_DIR="${TMPL_DIR:-/srv/ml/repos/omnimergekit/eval/templates}"
OMK_BIN="${OMK_BIN:-/srv/ml/envs/envs/omnimergekit/bin}"

# tier := "template:metadata". The single persistent serve is at MAXLEN, so no
# per-tier max-model-len. metadata trims RULER's prompt target under the served
# ceiling where needed (256k matches the base anchor's 261120; 512k = 524288-1024
# so prompt+gen stays < 524288). "ANCHOR_GATE" is a control marker, not a tier.
TIERS=(
  "ruler_native_smoke:"
  "ruler_native_vt_32k:"
  "ruler_native_vt_256k:ctx_tokens=261120"
  "ruler_native_mk1_256k:ctx_tokens=261120"
  "ANCHOR_GATE"
  "ruler_native_vt_384k:"
  "ruler_native_mk1_384k:"
  "ruler_native_vt_512k:ctx_tokens=523264"
  "ruler_native_mk1_512k:ctx_tokens=523264"
)

mkdir -p "$RESULTS"
LOG="$RESULTS/ruler_ext_512k.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== ruler_ext_512k $(date '+%F %T %Z') — TP=2 serve port=$PORT maxlen=$MAXLEN ==="

# --- gate: no trainer (TP=2 needs BOTH GPUs) ---------------------------------
if pgrep -f "phase1_train_yarn_lora" >/dev/null 2>&1; then
  echo "[gate] a trainer (phase1_train_yarn_lora*) is running — GPUs busy. NOT preempting."
  pgrep -af "phase1_train_yarn_lora" | sed 's/^/[gate]   /' | cut -c1-110
  exit 0
fi

# --- preflight inputs (FATAL-loud) -------------------------------------------
for p in "$OMK_PY" "$VLLM_PY" "$OMK" "$TMPL_DIR" "$YARN_BASE" "$ADAPTER" \
         "$SCR/merge_512k_lora.py" "$SCR/make_serve_config.py"; do
  [ -e "$p" ] || { echo "FATAL: missing $p" >&2; exit 1; }
done

# --- step 1: surgical merge (CPU, idempotent) --------------------------------
echo "--- [1/4] surgical merge $(date '+%T %Z') ---"
if ! "$OMK_PY" "$SCR/merge_512k_lora.py" \
       --base "$YARN_BASE" --adapter "$ADAPTER" --out "$MERGED"; then
  echo "FATAL: merge failed — see above." >&2; exit 1
fi

# --- step 2: serve-config rope translation (cheap) ---------------------------
echo "--- [2/4] serve-config (rope → $ROPE_TYPE) $(date '+%T %Z') ---"
if ! "$OMK_PY" "$SCR/make_serve_config.py" \
       --merged "$MERGED" --out "$SERVE" --rope-type "$ROPE_TYPE"; then
  echo "FATAL: serve-config failed — see above." >&2; exit 1
fi

# --- re-gate GPUs right before serve (a trainer could have started) ----------
for g in 0 1; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  [ -n "$used" ] || { echo "[gate] cannot query GPU$g via nvidia-smi"; exit 1; }
  if [ "$used" -gt 4000 ]; then
    echo "[gate] GPU$g has ${used} MiB used (>4GB) — not idle. TP=2 needs both free. Aborting."
    exit 0
  fi
done
echo "[gate] both GPUs idle — launching TP=2 serve."

# --- step 3: persistent TP=2 vLLM serve --------------------------------------
SERVE_LOG="$RESULTS/serve_tp2.log"
echo "--- [3/4] vLLM TP=2 serve → $SERVE_LOG $(date '+%T %Z') ---"
SERVER_PID=""
cleanup() {
  echo "[cleanup] tearing down serve $(date '+%T %Z')"
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  pkill -f "api_server.*--port[ =]$PORT" 2>/dev/null
}
trap cleanup EXIT

# No reasoning parser: RULER uses base-template completions, not thinking-channel
# chat. --max-num-batched-tokens 8192 is the Gemma 4 MM-encoder floor (default
# 2048 < 2496 → boot crash). --max-num-seqs 4 trims the cudagraph capture set at
# 512k (templates run single-flight anyway). TP=2 is exact → does not change
# logits, so the 256k anchor comparison vs the base single-GPU serve stays valid.
setsid env CUDA_VISIBLE_DEVICES=0,1 "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$SERVE" \
    --served-model-name "$SERVED" \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAXLEN" \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --dtype bfloat16 \
    --trust-remote-code >"$SERVE_LOG" 2>&1 &
SERVER_PID=$!
echo "[serve] pid=$SERVER_PID — waiting for /health (TP=2 + 524288 KV boot is slow)..."
ready=0
for i in $(seq 1 120); do        # 120 × 15s = 30 min
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] FATAL: server process died before becoming healthy. Last 40 log lines:"
    tail -n 40 "$SERVE_LOG" | sed 's/^/[serve]   /'
    exit 1
  fi
  sleep 15
done
if [ "$ready" != 1 ]; then
  echo "[serve] FATAL: /health not ready after 30 min. Last 40 log lines:"
  tail -n 40 "$SERVE_LOG" | sed 's/^/[serve]   /'
  exit 1
fi
echo "[serve] healthy $(date '+%T %Z')"

# --- anchor-gate helper: extended@256k must match base@256k ------------------
anchor_gate() {
  "$OMK_PY" - "$RESULTS" "$SERVED" "$REF_RESULTS" "$REF_SERVED" "$GATE_TOL" <<'PYEOF'
import json, sys, os
res, served, ref, ref_served, tol = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
def score(root, tmpl, srv):
    p = os.path.join(root, tmpl, srv, "summary.json")
    try:
        d = json.load(open(p))
    except Exception as e:
        return None, f"{p}: {e}"
    return d.get("score", d.get("pass_at_1")), p
print("=== ANCHOR GATE — extended@256k vs base@256k ===")
ok = True; have_ref = True
for tmpl in ("ruler_native_vt_256k", "ruler_native_mk1_256k"):
    es, ep = score(res, tmpl, served)
    bs, bp = score(ref, tmpl, ref_served)
    if bs is None:
        print(f"  {tmpl:24s} base anchor MISSING ({bp})"); have_ref = False; continue
    if es is None:
        print(f"  {tmpl:24s} extended MISSING ({ep})"); ok = False; continue
    d = abs(float(es) - float(bs))
    verdict = "OK" if d <= tol else "DIVERGE"
    if d > tol: ok = False
    print(f"  {tmpl:24s} ext={es:.4f}  base={bs:.4f}  |d|={d:.4f}  [{verdict}]")
if not have_ref:
    print(f"RESULT: NO_ANCHOR (base ref not built — run ruler_ref_base_a4b.sh)")
    sys.exit(2)
print(f"RESULT: {'PASS' if ok else 'FAIL'} (tol={tol})")
sys.exit(0 if ok else 1)
PYEOF
}

# --- step 4: run the ladder via omk_eval --no-server -------------------------
echo "--- [4/4] RULER ladder $(date '+%T %Z') ---"
for tier in "${TIERS[@]}"; do
  if [ "$tier" = "ANCHOR_GATE" ]; then
    anchor_gate; rc=$?
    if [ "$rc" = 0 ]; then
      echo "[anchor] PASS — proceeding to 384k/512k."
    elif [ "$rc" = 2 ]; then
      if [ "${FORCE_NO_ANCHOR:-0}" = 1 ]; then
        echo "[anchor] NO_ANCHOR but FORCE_NO_ANCHOR=1 — proceeding (NOT recommended)."
      else
        echo "[anchor] NO base anchor to validate the rope mapping. STOPPING before 384k/512k."
        echo "[anchor] Build it first: bash $SCR/ruler_ref_base_a4b.sh  (then re-run)."
        echo "[anchor] Or set FORCE_NO_ANCHOR=1 to spend 512k compute unvalidated."
        break
      fi
    else
      if [ "${ANCHOR_OVERRIDE:-0}" = 1 ]; then
        echo "[anchor] FAIL but ANCHOR_OVERRIDE=1 — proceeding anyway."
      else
        echo "[anchor] FAIL — the proportional_yarn→$ROPE_TYPE mapping does not"
        echo "[anchor]   reproduce the base 256k score. STOPPING before 384k/512k."
        echo "[anchor]   Re-map: ROPE_TYPE=<other> bash $0  (re-derives serve cfg, no re-merge),"
        echo "[anchor]   or ANCHOR_OVERRIDE=1 to proceed regardless."
        break
      fi
    fi
    continue
  fi

  tmpl="${tier%%:*}"; md="${tier#*:}"
  [ -f "$TMPL_DIR/$tmpl.yaml" ] || { echo "FATAL: template $TMPL_DIR/$tmpl.yaml missing" >&2; break; }
  md_args=(); [ -n "$md" ] && md_args=(--metadata "$md")
  echo "--- $tmpl${md:+ (metadata=$md)} $(date '+%T %Z') ---"
  if ! PATH="$OMK_BIN:$PATH" \
       "$OMK_PY" "$OMK" \
         --model "$SERVE" \
         --template "$tmpl" \
         --backend vllm \
         --no-server \
         --max-model-len "$MAXLEN" \
         "${md_args[@]}" \
         --served-name "$SERVED" \
         --tokenizer "$SERVE" \
         --port "$PORT" \
         --results-dir "$RESULTS"; then
    echo "[run] $tmpl FAILED. Checking serve health before continuing..."
    if ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "[run] serve is DOWN — aborting ladder. Last 30 serve.log lines:"
      tail -n 30 "$SERVE_LOG" | sed 's/^/[serve]   /'
      break
    fi
    echo "[run] serve still healthy — continuing to next tier (this one is recorded as failed)."
  else
    echo "[run] $tmpl done $(date '+%T %Z')"
  fi
done

# --- tabulate (summary.json .score, NEVER raw results_*.json) ----------------
echo "=== EXTENDED 512k RULER ladder — scores ==="
"$OMK_PY" - "$RESULTS" "$REF_RESULTS" <<'PYEOF'
import json, sys, glob, os
ext, ref = sys.argv[1], sys.argv[2]
def rows(root):
    out = []
    for sj in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
        try:
            d = json.load(open(sj))
        except Exception as e:
            out.append((os.path.relpath(sj, root), "ERR", str(e)[:24])); continue
        s = d.get("score", d.get("pass_at_1"))
        out.append((os.path.relpath(sj, root), s, d.get("metric", "")))
    return out
for label, root in (("EXTENDED", ext), ("BASE-ANCHOR", ref)):
    rr = rows(root)
    print(f"\n--- {label}: {root} ---")
    if not rr:
        print("  (no summary.json yet)"); continue
    print(f"  {'result':56s} {'score':>8s}  metric")
    for name, s, m in rr:
        v = f"{s:.4f}" if isinstance(s, (int, float)) else str(s)
        print(f"  {name:56s} {v:>8s}  {m}")
PYEOF
echo "=== ruler_ext_512k end $(date '+%F %T %Z') ==="
