#!/usr/bin/env bash
# ruler_ext_512k.sh — merge → serve → RULER-ladder driver for the YaRN-EXTENDED
# Gemma-4-26B-A4B 250M checkpoint. Runs ON bs2 (2× PRO 6000 96GB, NO NVLink).
#
# PIPELINE:
#   1. surgical-merge the 512k LoRA adapter (ckpt-000952) onto the 128e YaRN base
#      (yarn_cfg_98e) → gemma-4-26B-A4B-it-512k   [CPU, idempotent]
#   2. derive a vLLM-servable dir by translating the rope config
#      proportional_yarn → vLLM 'yarn'            [hardlinks, seconds, re-runnable]
#   3. RULER ladder, with the sub-512k tiers PARALLELIZED across the two GPUs:
#        smoke(4k) + vt_32k + vt_256k  ∥  mk1_256k        (two single-GPU serves)
#          → [ANCHOR GATE]
#        vt_384k  ∥  mk1_384k                             (two single-GPU serves)
#          → vt_512k → mk1_512k                           (one TP=2 serve, sequential)
#
# WHY PARALLEL (T87, 2026-06-08): one serving copy is ~52GB weights; the global-KV
# is ~80KB/tok over the 5 full-attn layers. So a single 26B-A4B copy + its KV FITS
# one 96GB GPU at 256k (52+20=72GB) and at 384k (52+30=82GB), but NOT at 512k
# (52+40=92GB > 0.92×96). The original driver served EVERYTHING through one TP=2
# 524288 serve and ran all 9 tiers sequentially — leaving the 2nd GPU idle for the
# entire sub-512k portion. Now the 256k and 384k vt/mk1 pairs each run as two
# single-GPU serves (one per GPU) concurrently, halving those rungs; only the 512k
# pair stays TP=2 (it genuinely needs both GPUs for the 40GB KV). TP=2 is exact →
# does not change logits, so the 256k anchor comparison vs the base single-GPU
# serve stays valid whether the extended 256k ran single-GPU or TP=2.
#
# ROBUSTNESS: every parallel phase FALLS BACK to the proven TP=2 path on any
# serve-boot failure (e.g. a single-GPU 384k OOM) — worst case is exactly the old
# sequential behavior, never worse. NO_PARALLEL=1 forces the original single-serve
# path outright.
#
# THE ANCHOR GATE (rope-correctness): the proportional_yarn→'yarn' mapping is a
# HYPOTHESIS (vLLM 'yarn' = YaRN over the STANDARD base; training used YaRN over
# Gemma 4's proportional base — they *should* coincide but are not proven equal).
# So before spending ~3h on 384k/512k we REQUIRE the EXTENDED model's 256k RULER
# scores to match the BASE 256k anchor (ruler_ref_base) within GATE_TOL. If they
# diverge, the rope mapping is wrong: STOP, re-run make_serve_config.py with a
# different --rope-type (cheap), re-serve. Override with ANCHOR_OVERRIDE=1.
#
# GATED: refuses while any trainer owns the GPUs (the 512k rung needs BOTH). Greedy/50.
set -uo pipefail

SCR="${SCR:-$(cd "$(dirname "$0")" && pwd)}"   # dir holding merge/serve helpers
LONGCTX="${LONGCTX:-/srv/ml/longctx}"

YARN_BASE="${YARN_BASE:-$LONGCTX/yarn_cfg_98e}"                 # 128e YaRN base (merge SRC)
ADAPTER="${ADAPTER:-$LONGCTX/ckpt_98e_ddp32k_fa2/ckpt-000952}"  # 250M LoRA ckpt
MERGED="${MERGED:-$LONGCTX/gemma-4-26B-A4B-it-512k}"            # canonical merged (proportional_yarn)
SERVE="${SERVE:-$LONGCTX/gemma-4-26B-A4B-it-512k-vllm-yarn}"    # vLLM serve dir ('yarn')
SERVED="${SERVED:-gemma-4-26B-A4B-it-512k}"
ROPE_TYPE="${ROPE_TYPE:-yarn}"

# RULER's prepare.py loads a tokenizer with the omk env's transformers (5.5.0) to
# count tokens. It MUST be a CLEAN (non-yarn) config: transformers' YaRN validator
# rejects the serve dir's rope_scaling for lacking 'original_max_position_embeddings'
# (KeyError, 2026-06-08 — crashed every extended tier). The base model shares the
# extended model's tokenizer/vocab, so counts are identical — and it's exactly what
# the [A] base anchor tokenizes with, making [B]'s staged prompts token-for-token
# consistent with the anchor it's compared against. vLLM still SERVES the yarn dir
# (its env's transformers loads yarn fine); only RULER's prepare uses this.
RULER_TOK="${RULER_TOK:-/srv/ml/google/gemma-4-26B-A4B-it}"

RESULTS="${RESULTS:-$LONGCTX/ruler_ext_512k}"
REF_RESULTS="${REF_RESULTS:-$LONGCTX/ruler_ref_base}"           # base anchor (ruler_ref_base_a4b.sh)
REF_SERVED="${REF_SERVED:-gemma-4-26B-A4B-it-base}"

P0="${P0:-8197}"                    # GPU0 single-GPU serve (also the TP=2 port)
P1="${P1:-8198}"                    # GPU1 single-GPU serve
PTP="$P0"                           # TP=2 reuses the GPU0 port
GPU_UTIL="${GPU_UTIL:-0.92}"
MAXLEN="${MAXLEN:-524288}"          # = max_position_embeddings ceiling (TP=2 / 512k)
ML256="${ML256:-262144}"           # single-GPU 256k serve (prompt 261120 + gen < this)
ML384="${ML384:-397312}"           # single-GPU 384k serve (prompt 393216 + gen < this)
GATE_TOL="${GATE_TOL:-0.10}"        # abs RULER-score delta extended-vs-base @256k

OMK_PY="${OMK_PY:-/srv/ml/envs/envs/omnimergekit/bin/python}"
VLLM_PY="${VLLM_PY:-/srv/ml/envs/envs/vllm/bin/python}"
OMK="${OMK:-/srv/ml/repos/omnimergekit/eval/omk_eval.py}"
TMPL_DIR="${TMPL_DIR:-/srv/ml/repos/omnimergekit/eval/templates}"
OMK_BIN="${OMK_BIN:-/srv/ml/envs/envs/omnimergekit/bin}"

# Fallback-path tier list (original single-serve sequential ladder). tier :=
# "template:metadata"; metadata trims RULER's prompt target under the served
# ceiling. "ANCHOR_GATE" is a control marker, not a tier.
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
echo "=== ruler_ext_512k $(date '+%F %T %Z') — parallel sub-512k (P0=$P0 GPU0, P1=$P1 GPU1, TP=2 @512k) ==="

# --- gate: no trainer (the 512k rung needs BOTH GPUs) ------------------------
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
echo "--- [1/3] surgical merge $(date '+%T %Z') ---"
if ! "$OMK_PY" "$SCR/merge_512k_lora.py" \
       --base "$YARN_BASE" --adapter "$ADAPTER" --out "$MERGED"; then
  echo "FATAL: merge failed — see above." >&2; exit 1
fi

# --- step 2: serve-config rope translation (cheap) ---------------------------
echo "--- [2/3] serve-config (rope → $ROPE_TYPE) $(date '+%T %Z') ---"
if ! "$OMK_PY" "$SCR/make_serve_config.py" \
       --merged "$MERGED" --out "$SERVE" --rope-type "$ROPE_TYPE"; then
  echo "FATAL: serve-config failed — see above." >&2; exit 1
fi

# ── serve lifecycle helpers ──────────────────────────────────────────────────
declare -A SPID

serve_up() {  # $1=cuda_devices ("0"|"1"|"0,1")  $2=port  $3=maxlen  $4=serve_log
  local devs="$1" port="$2" maxlen="$3" slog="$4" ntp nseq pid i
  ntp=$(awk -F, '{print NF}' <<<"$devs")
  if [ "$ntp" -gt 1 ]; then nseq=4; else nseq=1; fi   # TP=2 keeps old nseq=4; single-flight is 1
  echo "[serve:$port] up devs=$devs tp=$ntp maxlen=$maxlen nseq=$nseq → $slog $(date '+%T %Z')"
  setsid env CUDA_VISIBLE_DEVICES="$devs" "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --model "$SERVE" \
      --served-model-name "$SERVED" \
      --port "$port" \
      --tensor-parallel-size "$ntp" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-model-len "$maxlen" \
      --max-num-batched-tokens 8192 \
      --max-num-seqs "$nseq" \
      --dtype bfloat16 \
      --trust-remote-code >"$slog" 2>&1 &
  pid=$!; SPID[$port]=$pid
  for i in $(seq 1 120); do        # 120 × 15s = 30 min
    if curl -fsS "http://localhost:$port/health" >/dev/null 2>&1; then
      echo "[serve:$port] healthy $(date '+%T %Z')"; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[serve:$port] FATAL: process died before healthy. Last 30 log lines:"
      tail -n 30 "$slog" | sed "s/^/[serve:$port]   /"
      unset 'SPID[$port]'; return 1
    fi
    sleep 15
  done
  echo "[serve:$port] FATAL: /health not ready after 30 min. Last 30 log lines:"
  tail -n 30 "$slog" | sed "s/^/[serve:$port]   /"
  kill "$pid" 2>/dev/null; unset 'SPID[$port]'; return 1
}

serve_down() {  # $1=port
  local port="$1"
  [ -n "${SPID[$port]:-}" ] && kill "${SPID[$port]}" 2>/dev/null
  pkill -f "api_server.*--port[ =]$port" 2>/dev/null
  unset 'SPID[$port]' 2>/dev/null || true
}

wait_gpus_free() {  # poll both GPUs down to <4GB (VRAM release lag between phases)
  local i u0 u1
  for i in $(seq 1 40); do        # 40 × 3s = 2 min
    u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
    [ "${u0:-9999}" -lt 4000 ] && [ "${u1:-9999}" -lt 4000 ] && return 0
    sleep 3
  done
  echo "[gpus] WARN: not fully freed after 2 min (GPU0=${u0:-?} GPU1=${u1:-?} MiB) — proceeding anyway"
  return 0
}

run_tier() {  # $1=port  $2=template  $3=metadata  $4=maxlen  $5=tag
  local port="$1" tmpl="$2" md="$3" maxlen="$4" tag="$5"
  [ -f "$TMPL_DIR/$tmpl.yaml" ] || { echo "FATAL: template $TMPL_DIR/$tmpl.yaml missing" >&2; return 2; }
  local md_args=(); [ -n "$md" ] && md_args=(--metadata "$md")
  echo "--- [$tag] $tmpl${md:+ (metadata=$md)} port=$port $(date '+%T %Z') ---"
  PATH="$OMK_BIN:$PATH" "$OMK_PY" "$OMK" \
      --model "$SERVE" \
      --template "$tmpl" \
      --backend vllm \
      --no-server \
      --max-model-len "$maxlen" \
      "${md_args[@]}" \
      --served-name "$SERVED" \
      --tokenizer "$RULER_TOK" \
      --port "$port" \
      --results-dir "$RESULTS"
}

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

# returns: 0 proceed | 1 stop-at-gate (NO_ANCHOR/FAIL, not overridden)
eval_anchor() {
  anchor_gate; local rc=$?
  if [ "$rc" = 0 ]; then echo "[anchor] PASS — proceeding to 384k/512k."; return 0; fi
  if [ "$rc" = 2 ]; then
    if [ "${FORCE_NO_ANCHOR:-0}" = 1 ]; then echo "[anchor] NO_ANCHOR but FORCE_NO_ANCHOR=1 — proceeding (NOT recommended)."; return 0; fi
    echo "[anchor] NO base anchor to validate the rope mapping. STOPPING before 384k/512k."
    echo "[anchor] Build it first: bash $SCR/ruler_ref_base_a4b.sh  (then re-run). Or FORCE_NO_ANCHOR=1."
    return 1
  fi
  if [ "${ANCHOR_OVERRIDE:-0}" = 1 ]; then echo "[anchor] FAIL but ANCHOR_OVERRIDE=1 — proceeding anyway."; return 0; fi
  echo "[anchor] FAIL — the proportional_yarn→$ROPE_TYPE mapping does not reproduce the base 256k score."
  echo "[anchor]   STOPPING. Re-map: ROPE_TYPE=<other> bash $0  (re-derives serve cfg), or ANCHOR_OVERRIDE=1."
  return 1
}

# --- FALLBACK: the original single persistent TP=2 serve + sequential ladder --
run_all_tp2() {
  echo "--- [fallback] single TP=2 serve @ $MAXLEN, sequential ladder $(date '+%T %Z') ---"
  wait_gpus_free
  serve_up "0,1" "$PTP" "$MAXLEN" "$RESULTS/serve_tp2.log" || { echo "FATAL: TP=2 serve failed to boot." >&2; return 1; }
  local tier tmpl md
  for tier in "${TIERS[@]}"; do
    if [ "$tier" = "ANCHOR_GATE" ]; then
      eval_anchor || break
      continue
    fi
    tmpl="${tier%%:*}"; md="${tier#*:}"
    if ! run_tier "$PTP" "$tmpl" "$md" "$MAXLEN" tp2; then
      echo "[run] $tmpl FAILED. Checking serve health..."
      if ! curl -fsS "http://localhost:$PTP/health" >/dev/null 2>&1; then
        echo "[run] serve is DOWN — aborting ladder. Last 30 serve.log lines:"
        tail -n 30 "$RESULTS/serve_tp2.log" | sed 's/^/[serve]   /'; break
      fi
      echo "[run] serve still healthy — continuing (this tier recorded as failed)."
    fi
  done
  serve_down "$PTP"
}

# --- FAST PATH: parallel single-GPU sub-512k pairs + TP=2 @ 512k -------------
# returns 10 → caller falls back to run_all_tp2 (256k serves unavailable).
run_fast() {
  # Phase 256k: GPU0 runs smoke+vt_32k+vt_256k, GPU1 runs mk1_256k — concurrently.
  echo "--- [3/3] phase 256k: parallel single-GPU (GPU0 vt-batch ∥ GPU1 mk1) $(date '+%T %Z') ---"
  wait_gpus_free
  if ! serve_up "0" "$P0" "$ML256" "$RESULTS/serve_g0_256k.log"; then return 10; fi
  if ! serve_up "1" "$P1" "$ML256" "$RESULTS/serve_g1_256k.log"; then serve_down "$P0"; return 10; fi
  ( run_tier "$P0" ruler_native_smoke   ""                  "$ML256" g0
    run_tier "$P0" ruler_native_vt_32k  ""                  "$ML256" g0
    run_tier "$P0" ruler_native_vt_256k "ctx_tokens=261120" "$ML256" g0
  ) >"$RESULTS/run_g0_256k.log" 2>&1 &
  local j0=$!
  ( run_tier "$P1" ruler_native_mk1_256k "ctx_tokens=261120" "$ML256" g1
  ) >"$RESULTS/run_g1_256k.log" 2>&1 &
  local j1=$!
  wait "$j0"; wait "$j1"
  sed 's/^/[g0] /' "$RESULTS/run_g0_256k.log"
  sed 's/^/[g1] /' "$RESULTS/run_g1_256k.log"
  serve_down "$P0"; serve_down "$P1"

  # ANCHOR GATE (extended@256k vs base@256k)
  eval_anchor || return 0

  # Phase 384k: GPU0 vt_384k ∥ GPU1 mk1_384k. Single-GPU is tight (~82GB); on any
  # boot failure, fall back to a TP=2 serve for just this pair.
  echo "--- phase 384k: parallel single-GPU (GPU0 vt ∥ GPU1 mk1) $(date '+%T %Z') ---"
  wait_gpus_free
  if serve_up "0" "$P0" "$ML384" "$RESULTS/serve_g0_384k.log" \
     && serve_up "1" "$P1" "$ML384" "$RESULTS/serve_g1_384k.log"; then
    ( run_tier "$P0" ruler_native_vt_384k  "" "$ML384" g0 ) >"$RESULTS/run_g0_384k.log" 2>&1 &
    local k0=$!
    ( run_tier "$P1" ruler_native_mk1_384k "" "$ML384" g1 ) >"$RESULTS/run_g1_384k.log" 2>&1 &
    local k1=$!
    wait "$k0"; wait "$k1"
    sed 's/^/[g0] /' "$RESULTS/run_g0_384k.log"
    sed 's/^/[g1] /' "$RESULTS/run_g1_384k.log"
    serve_down "$P0"; serve_down "$P1"
  else
    echo "[384k] single-GPU serve boot failed — TP=2 fallback for the 384k pair."
    serve_down "$P0"; serve_down "$P1"; wait_gpus_free
    if serve_up "0,1" "$PTP" "$MAXLEN" "$RESULTS/serve_tp2_384k.log"; then
      run_tier "$PTP" ruler_native_vt_384k  "" "$MAXLEN" tp2 || true
      run_tier "$PTP" ruler_native_mk1_384k "" "$MAXLEN" tp2 || true
      serve_down "$PTP"
    else
      echo "FATAL: TP=2 384k serve failed to boot — skipping 384k+512k." >&2; return 1
    fi
  fi

  # Phase 512k: TP=2 (52+40GB KV needs both GPUs), sequential.
  echo "--- phase 512k: TP=2 sequential $(date '+%T %Z') ---"
  wait_gpus_free
  if ! serve_up "0,1" "$PTP" "$MAXLEN" "$RESULTS/serve_tp2_512k.log"; then
    echo "FATAL: TP=2 512k serve failed to boot." >&2; return 1
  fi
  run_tier "$PTP" ruler_native_vt_512k  "ctx_tokens=523264" "$MAXLEN" tp2 || true
  run_tier "$PTP" ruler_native_mk1_512k "ctx_tokens=523264" "$MAXLEN" tp2 || true
  serve_down "$PTP"
  return 0
}

# --- teardown on any exit ----------------------------------------------------
cleanup() {
  echo "[cleanup] tearing down any live serves $(date '+%T %Z')"
  serve_down "$P0"; serve_down "$P1"
}
trap cleanup EXIT

# --- initial GPU gate (the 512k rung needs BOTH; let [A]'s serve release) -----
wait_gpus_free
for g in 0 1; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  [ -n "$used" ] || { echo "[gate] cannot query GPU$g via nvidia-smi"; exit 1; }
  if [ "$used" -gt 4000 ]; then
    echo "[gate] GPU$g has ${used} MiB used (>4GB) — not idle. The 512k rung needs both. Aborting."
    exit 0
  fi
done
echo "[gate] both GPUs idle — starting ladder."

# --- dispatch ----------------------------------------------------------------
if [ "${NO_PARALLEL:-0}" = 1 ]; then
  echo "[B] NO_PARALLEL=1 — original single TP=2 serve, sequential ladder."
  run_all_tp2
else
  run_fast; rc=$?
  if [ "$rc" = 10 ]; then
    echo "[B] 256k single-GPU serves unavailable — FULL TP=2 fallback (original path)."
    serve_down "$P0"; serve_down "$P1"
    run_all_tp2
  fi
fi

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
