#!/usr/bin/env bash
# Verify the PUBLISHED v7-coder GGUF tiers carry the FIXED 18051 chat template.
# Resumes from hf cache (download died at 7.4G earlier). Read-only audit.
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
PY=/srv/ml/envs/envs/omnimergekit/bin/python
"$PY" - <<'PYEOF'
from huggingface_hub import hf_hub_download
from gguf import GGUFReader
import os
repo="ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF"
fn="gemma-4-A4B-98e-v7-coder-it-CD-Q2_K.gguf"
p=hf_hub_download(repo, fn, local_dir="/mnt/sdc/ml/v7coder_verify")
r=GGUFReader(p); s=None
for f in r.fields.values():
    if f.name=="tokenizer.chat_template":
        s="".join(chr(b) for pi in f.data for b in bytes(f.parts[pi]))
n=len(s) if s else None
print("V7CODER_PUBLISHED_CD-Q2_K len=%s has_channel_thought=%s is_FIXED_18051=%s"
      % (n, ("channel>thought" in s) if s else "?", (n==18051) if s else False))
try: os.remove(p)
except Exception: pass
PYEOF
echo "V7CODER_VERIFY_DONE"
