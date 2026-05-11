#!/usr/bin/env bash
# T10 smoke — end-to-end validation of:
#   - competence_extract.py --structured + --resume / --ckpt-every (sidecar)
#   - competence_combine.py structured passthrough (--signal head_l1)
#   - omnimergekit.py --fisher-structured broadcast roundtrip
#
# Tiny scope (10 samples × max-len 256 on Qwen3.5-4B) to keep the run under 15 min
# on RTX 3090. No full merge — broadcast roundtrip is verified as a unit test using
# the actual produced combined safetensors.
#
# Outputs live under recipes/microcoder_4b/_t10_smoke/ (under-/shared/dev, NOT /tmp).

set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
PYBIN=/shared/dev/lightseek/.venv/bin/python
OMK=/shared/dev/omnimergekit
SMOKE_DIR=$OMK/recipes/microcoder_4b/_t10_smoke
LOG_DIR=$WS/logs
mkdir -p "$SMOKE_DIR" "$LOG_DIR"
LOG=$LOG_DIR/t10_smoke_$(date +%Y%m%d_%H%M%S).log

SRC=jackrong-v2
MODEL=$HF/$SRC
SAMPLES=$WS/4b_phase1/eval_results/humaneval_${SRC}/${SRC}/samples_humaneval_*.jsonl
OUT_ST=$SMOKE_DIR/${SRC}__he.safetensors
COMBINED_DIR=$SMOKE_DIR/combined
COMBINED_ST=$COMBINED_DIR/${SRC}.safetensors
SIDECAR=${OUT_ST}.ckpt.pt

[ -d "$MODEL" ] || { echo "ERROR: $MODEL not found"; exit 1; }
ls $SAMPLES 1>/dev/null 2>&1 || { echo "ERROR: no samples at $SAMPLES"; exit 1; }

# Clean slate so we exercise the fresh-run path
rm -f "$OUT_ST" "$SIDECAR" "${SIDECAR}.stale" "${SIDECAR}.tmp"
rm -f "$COMBINED_ST"
mkdir -p "$COMBINED_DIR"

echo "=== T10 smoke start $(date -Iseconds) ===" | tee -a "$LOG"

# ── Phase A: full extract with --structured + --ckpt-every 3 ─────────────────
echo "[A] full extract: 10 samples, --structured, --ckpt-every 3" | tee -a "$LOG"
PHASE_A_T0=$(date +%s)
"$PYBIN" "$OMK/competence/competence_extract.py" \
    --model "$MODEL" \
    --samples $SAMPLES \
    --task he \
    --output "$OUT_ST" \
    --max-samples 10 --max-len 256 \
    --structured --ckpt-every 3 --ckpt-time-sec 0 \
    --skip-grad-patterns embed_tokens,lm_head \
    2>&1 | tee -a "$LOG"
[ -f "$OUT_ST" ] || { echo "[A] FAIL: no $OUT_ST"; exit 2; }
[ ! -f "$SIDECAR" ] || { echo "[A] FAIL: sidecar should be deleted on clean finish"; exit 2; }
echo "[A] OK ($(($(date +%s) - PHASE_A_T0))s)" | tee -a "$LOG"

# Verify structured keys + structured_config metadata
"$PYBIN" - <<'PY' 2>&1 | tee -a "$LOG"
from safetensors import safe_open
import json, sys
p = "/shared/dev/omnimergekit/recipes/microcoder_4b/_t10_smoke/jackrong-v2__he.safetensors"
with safe_open(p, framework="pt") as f:
    md = f.metadata() or {}
    keys = list(f.keys())
sc = md.get("structured_config")
assert sc, f"no structured_config in metadata: {md}"
cfg = json.loads(sc)
print(f"[A] structured_config: {cfg}")
n_head = sum(1 for k in keys if ".head_compact_l1" in k)
n_neuron = sum(1 for k in keys if ".neuron_compact_l1" in k)
n_total = len(keys)
assert n_head > 0, "no head_compact_l1 keys"
assert n_neuron > 0, "no neuron_compact_l1 keys"
print(f"[A] keys: total={n_total} head_compact_l1={n_head} neuron_compact_l1={n_neuron}")
# Spot-check a head signal shape
sample = next(k for k in keys if k.endswith(".head_compact_l1"))
with safe_open(p, framework="pt") as f:
    t = f.get_tensor(sample)
assert t.dim() == 1, f"head signal should be 1D, got {t.dim()}D"
print(f"[A] {sample}: shape={tuple(t.shape)} expected=({cfg['num_heads']},) or ({cfg['num_kv_heads']},)")
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[A] structured-keys check failed"; exit 2; }

# ── Phase B: resume mid-flight ───────────────────────────────────────────────
# Re-run the same extract in the background, kill after enough time for
# at least one checkpoint, then re-launch with --resume.
echo "[B] mid-flight resume test" | tee -a "$LOG"
rm -f "$OUT_ST" "$SIDECAR"  # fresh start

# Launch in background with PIDFILE
"$PYBIN" "$OMK/competence/competence_extract.py" \
    --model "$MODEL" --samples $SAMPLES --task he \
    --output "$OUT_ST" --max-samples 10 --max-len 256 \
    --structured --ckpt-every 3 --ckpt-time-sec 0 \
    --skip-grad-patterns embed_tokens,lm_head \
    > "$SMOKE_DIR/_phaseB_run1.log" 2>&1 &
PID=$!
echo "[B] launched pid=$PID; waiting for first sidecar checkpoint…" | tee -a "$LOG"

# Poll for sidecar appearance (with a 5-min ceiling)
for i in $(seq 1 300); do
    sleep 1
    if [ -f "$SIDECAR" ]; then
        SIZE=$(stat -c %s "$SIDECAR")
        echo "[B] sidecar appeared at i=${i}s, size=${SIZE} bytes" | tee -a "$LOG"
        break
    fi
    if ! kill -0 $PID 2>/dev/null; then
        echo "[B] FAIL: process exited before sidecar appeared (check _phaseB_run1.log)"
        wait $PID; exit 2
    fi
done
[ -f "$SIDECAR" ] || { echo "[B] FAIL: no sidecar in 5 min"; kill $PID 2>/dev/null; exit 2; }

# Kill the run and wait
kill -TERM $PID 2>/dev/null
wait $PID 2>/dev/null
RC1=$?
echo "[B] killed run1 (rc=$RC1); sidecar preserved: $(ls -la $SIDECAR | awk '{print $5}') bytes" | tee -a "$LOG"

# Resume
echo "[B] resuming…" | tee -a "$LOG"
"$PYBIN" "$OMK/competence/competence_extract.py" \
    --model "$MODEL" --samples $SAMPLES --task he \
    --output "$OUT_ST" --max-samples 10 --max-len 256 \
    --structured --resume --ckpt-every 3 --ckpt-time-sec 0 \
    --skip-grad-patterns embed_tokens,lm_head \
    2>&1 | tee -a "$LOG"
RC2=${PIPESTATUS[0]}
[ "$RC2" -eq 0 ] || { echo "[B] FAIL: resume exit rc=$RC2"; exit 2; }
[ -f "$OUT_ST" ] || { echo "[B] FAIL: resume produced no output"; exit 2; }
[ ! -f "$SIDECAR" ] || { echo "[B] FAIL: sidecar not cleaned after resume completion"; exit 2; }
grep -E 'RESUMED from' "$LOG" >/dev/null || { echo "[B] FAIL: no 'RESUMED from' log line"; exit 2; }
echo "[B] OK — resume completed and cleaned sidecar" | tee -a "$LOG"

# ── Phase C: combine with --signal head_l1 ───────────────────────────────────
echo "[C] combine with --signal head_l1" | tee -a "$LOG"
"$PYBIN" "$OMK/competence/competence_combine.py" \
    --map "${SRC}:he:1.0:${OUT_ST}" \
    --signal head_l1 \
    --output-dir "$COMBINED_DIR" \
    --raw-rate \
    2>&1 | tee -a "$LOG"
[ -f "$COMBINED_ST" ] || { echo "[C] FAIL: no $COMBINED_ST"; exit 2; }

# Verify structured_config carried through; FFN keys absent (head_l1 only matches attn)
"$PYBIN" - <<PY 2>&1 | tee -a "$LOG"
from safetensors import safe_open
import json
p = "$COMBINED_ST"
with safe_open(p, framework="pt") as f:
    md = f.metadata() or {}
    keys = list(f.keys())
sc = md.get("structured_config")
assert sc, f"combined file missing structured_config: {md}"
print(f"[C] structured_config carried: {json.loads(sc)}")
print(f"[C] combined tensor count: {len(keys)}")
# Should contain attn names only — gate/up/down won't have head_compact_l1
n_attn = sum(1 for k in keys if any(s in k for s in (".q_proj", ".k_proj", ".v_proj", ".o_proj")))
n_ffn = sum(1 for k in keys if any(s in k for s in (".gate_proj", ".up_proj", ".down_proj")))
print(f"[C] attn={n_attn} ffn={n_ffn} (expect ffn=0 for head_l1)")
assert n_attn > 0
assert n_ffn == 0, f"head_l1 combined unexpectedly contains FFN keys: {n_ffn}"
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[C] combine validation failed"; exit 2; }
echo "[C] OK" | tee -a "$LOG"

# ── Phase D: broadcast roundtrip via _broadcast_structured_fisher ────────────
# Actual omnimergekit merge would consume the combined file with --fisher-structured;
# we unit-verify the broadcast helper here against the real combined file to keep
# the smoke under 15 min (full 4B merge is ~10-15 min on its own).
echo "[D] broadcast roundtrip on combined file" | tee -a "$LOG"
"$PYBIN" - <<'PY' 2>&1 | tee -a "$LOG"
# Validates broadcast against the actual produced safetensors. Reads structured_config
# from file metadata (so we don't have to re-parse possibly-nested model config.json).
import importlib.util, json, torch
from safetensors import safe_open
spec = importlib.util.spec_from_file_location("omk", "/shared/dev/omnimergekit/omnimergekit.py")
omk = importlib.util.module_from_spec(spec); spec.loader.exec_module(omk)

combined = "/shared/dev/omnimergekit/recipes/microcoder_4b/_t10_smoke/combined/jackrong-v2.safetensors"
src = "/shared/dev/omnimergekit/recipes/microcoder_4b/_t10_smoke/jackrong-v2__he.safetensors"
with safe_open(combined, framework="pt") as f:
    cfg = json.loads(f.metadata()["structured_config"])
    keys = list(f.keys())
print(f"[D] structured_config: {cfg}")
print(f"[D] {len(keys)} combined keys to validate against")

# Pick a k_proj (always aggregates because nkv*hd matches actual k_proj first dim)
kkey = next((k for k in keys if k.endswith(".k_proj.weight")), None)
assert kkey, "no k_proj.weight key in combined"
with safe_open(combined, framework="pt") as f:
    kcompact = f.get_tensor(kkey)
# Resolve actual k_proj shape from the per-element source file
with safe_open(src, framework="pt") as f:
    target_k = tuple(f.get_tensor(kkey).shape)
b = omk._broadcast_structured_fisher(kkey, kcompact, target_k, cfg)
assert b is not None, f"k_proj broadcast failed for shape {target_k}"
hd = cfg["head_dim"]
for h in range(cfg["num_kv_heads"]):
    rows = b[h*hd:(h+1)*hd]
    expected = kcompact[h].float().item()
    assert torch.allclose(rows, torch.full_like(rows, expected)), f"head {h} not constant"
print(f"[D] k_proj broadcast OK: compact{tuple(kcompact.shape)} -> {target_k}")

# o_proj: hidden axis is on dim 0; broadcast spreads along dim 1 within head
okey = next((k for k in keys if k.endswith(".o_proj.weight")), None)
if okey is not None:
    with safe_open(combined, framework="pt") as f:
        ocompact = f.get_tensor(okey)
    with safe_open(src, framework="pt") as f:
        target_o = tuple(f.get_tensor(okey).shape)
    b = omk._broadcast_structured_fisher(okey, ocompact, target_o, cfg)
    if b is not None:
        b_norm = b / (b.float().mean() + 1e-12)
        print(f"[D] o_proj broadcast OK: compact{tuple(ocompact.shape)} -> {target_o}; "
              f"post-renorm mean={b_norm.float().mean().item():.4f}")
    else:
        print(f"[D] o_proj broadcast skipped — model-specific layout (e.g. attn_output_gate)")

print("[D] OK — broadcast roundtrip verified against combined safetensors")
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[D] broadcast roundtrip failed"; exit 2; }
echo "[D] OK" | tee -a "$LOG"

echo "=== T10 smoke ALL GREEN $(date -Iseconds) ===" | tee -a "$LOG"
echo "  outputs:"
echo "    extract  : $OUT_ST"
echo "    combined : $COMBINED_ST"
echo "    log      : $LOG"
