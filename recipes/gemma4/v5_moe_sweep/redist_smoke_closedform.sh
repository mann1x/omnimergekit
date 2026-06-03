#!/usr/bin/env bash
# T191 closed-form plumbing smoke: fit -> emit -> verify a recovered 62e loads,
# has finite [62,...] expert tensors, and generates. Tiny calib => low quality is
# EXPECTED; this checks MECHANICAL validity (shapes/finite/loadable), not score.
#
# Usage: redist_smoke_closedform.sh [--run] <method> [capture.pt]
#   Dry-run by default (prints the plan, exits 0). Pass --run to execute.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$SCRIPT_DIR/redist_config.sh" ] && . "$SCRIPT_DIR/redist_config.sh"
DO_RUN=0; ARGS=()
for a in "$@"; do [ "$a" = "--run" ] && DO_RUN=1 || ARGS+=("$a"); done
set -- "${ARGS[@]+"${ARGS[@]}"}"

METHOD="${1:-hcsmoe}"
REDIST_PY="${REDIST_PY:-python}"
REDIST_SCRIPTS_DIR="${REDIST_SCRIPTS_DIR:-$REPO_ROOT/scripts}"
WORK="${REDIST_WORK:-$PWD/redist_work}"
CAP="${2:-$WORK/capture_multilingual_ream.pt}"
OUT="$WORK/smoke_${METHOD}_62e"

cat <<PLAN
=== redist_smoke_closedform plan (method=$METHOD) ===
  python      : $REDIST_PY
  scripts-dir : $REDIST_SCRIPTS_DIR
  teacher     : ${REDIST_TEACHER:-<unset REDIST_TEACHER>}
  student     : ${REDIST_STUDENT:-<unset REDIST_STUDENT>}
  keep-meta   : ${REDIST_KEEP_META:-<unset REDIST_KEEP_META>}
  capture     : $CAP
  emit ->     : $OUT   (purged after verify)
PLAN
[ "$DO_RUN" = 1 ] || { echo "(dry-run: pass --run to execute)"; exit 0; }

: "${REDIST_TEACHER:?set REDIST_TEACHER (128e teacher dir)}"
: "${REDIST_STUDENT:?set REDIST_STUDENT (pruned 62e dir)}"
: "${REDIST_KEEP_META:?set REDIST_KEEP_META (a2 keep metadata json)}"
[ -f "$CAP" ] || { echo "FAIL: capture not found: $CAP (run a capture first, or pass one)"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf "$OUT"
echo "=== [1/3] fit + emit ($METHOD) ==="
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/redist.py" redistribute --method "$METHOD" --driver multilingual \
  --capture "$CAP" --teacher "$REDIST_TEACHER" --student "$REDIST_STUDENT" --keep-meta "$REDIST_KEEP_META" \
  --emit "$OUT" --device cuda:0 --scripts-dir "$REDIST_SCRIPTS_DIR" || { echo "FIT/EMIT FAILED"; exit 1; }

echo "=== [2/3] verify shapes / finite / config ==="
"$REDIST_PY" - "$OUT" <<'PYEOF' || { echo "VERIFY FAILED"; exit 1; }
import json, os, sys, torch
from safetensors import safe_open
OUT = sys.argv[1]
cfg = json.load(open(OUT + "/config.json"))
ne = cfg["text_config"]["num_experts"]
print("num_experts:", ne); assert ne == 62, ne
idx = json.load(open(OUT + "/model.safetensors.index.json"))["weight_map"]
ok = True
for li in (0, 15, 29):
    for nm, exp in (("gate_up_proj", (62, 1408, 2816)), ("down_proj", (62, 2816, 704))):
        key = f"model.language_model.layers.{li}.experts.{nm}"
        with safe_open(os.path.join(OUT, idx[key]), framework="pt") as f:
            t = f.get_tensor(key)
        fin = bool(torch.isfinite(t.float()).all())
        good = tuple(t.shape) == exp and fin
        ok = ok and good
        print(f"  L{li} {nm}: {tuple(t.shape)} {t.dtype} finite={fin} {'OK' if good else 'BAD'}")
assert ok, "shape/finite check failed"
print("SHAPES_OK")
PYEOF

echo "=== [3/3] 1-prompt generate sanity (loads + emits tokens, no NaN crash) ==="
"$REDIST_PY" - "$OUT" <<'PYEOF' || { echo "GENERATE FAILED"; exit 1; }
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
OUT = sys.argv[1]
tok = AutoTokenizer.from_pretrained(OUT, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.bfloat16, trust_remote_code=True,
                                         attn_implementation="eager", device_map={"": 0}).eval()
msg = [{"role": "user", "content": "Say hello in one short sentence."}]
txt = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
enc = tok(txt, return_tensors="pt", add_special_tokens=False).to(0)
with torch.no_grad():
    o = m.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
print("GEN:", repr(tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)[:200]))
print("GENERATE_OK")
PYEOF

rm -rf "$OUT" && echo "PURGED $OUT"
echo "SMOKE_DONE $METHOD"
