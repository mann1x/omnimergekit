#!/usr/bin/env bash
# T191 closed-form plumbing smoke: fit -> emit -> verify a recovered 62e loads,
# has finite [62,...] expert tensors, and generates. Tiny calib => low quality is
# EXPECTED; this checks MECHANICAL validity (shapes/finite/loadable), not score.
# Usage: redist_smoke_closedform.sh <method> [capture.pt]
set -uo pipefail
METHOD="${1:-hcsmoe}"
CAP="${2:-/srv/ml/redist_work/capture_multilingual_ream.pt}"
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/scripts
TEACHER=/srv/ml/models/base/gemma-4-26B-A4B-it
STUDENT=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it
KM=/srv/ml/scripts/a2_keep_metadata.json
OUT=/srv/ml/redist_work/smoke_${METHOD}_62e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf "$OUT"
echo "=== [1/3] fit + emit ($METHOD) ==="
"$PY" "$SCR/redist.py" redistribute --method "$METHOD" --driver multilingual \
  --capture "$CAP" --teacher "$TEACHER" --student "$STUDENT" --keep-meta "$KM" \
  --emit "$OUT" --device cuda:0 --scripts-dir "$SCR" || { echo "FIT/EMIT FAILED"; exit 1; }

echo "=== [2/3] verify shapes / finite / config ==="
"$PY" - "$OUT" <<'PYEOF' || { echo "VERIFY FAILED"; exit 1; }
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
"$PY" - "$OUT" <<'PYEOF' || { echo "GENERATE FAILED"; exit 1; }
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
