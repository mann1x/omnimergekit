#!/usr/bin/env python
"""fa2_modelpath_probe.py — the FA2 *package* serves Gemma 4 hd=256 fine, but the
transformers model path raises 'head dimension at most 256'. transformers funnels
through the torch custom-op wrapper, so we patch the C++ chokepoint flash_attn_gpu.fwd
(and varlen_fwd) to print the EXACT head dim the model hands the kernel. Tiny seqlen,
1 forward, on the 128e base."""
import sys
import torch
from transformers import AutoModelForCausalLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/srv/ml/google/gemma-4-26B-A4B-it"

import flash_attn.flash_attn_interface as fi  # noqa: E402
flash_attn_gpu = fi.flash_attn_gpu

for name in ("fwd", "varlen_fwd"):
    if hasattr(flash_attn_gpu, name):
        _orig = getattr(flash_attn_gpu, name)

        def make(orig, nm):
            def traced(*a, **kw):
                # q is the first positional in both fwd and varlen_fwd
                q = a[0] if a else kw.get("q")
                try:
                    print("  [flash_attn_gpu.{}] q.shape={} hd={}".format(nm, tuple(q.shape), q.shape[-1]), flush=True)
                except Exception:
                    print("  [flash_attn_gpu.{}] (could not read q)".format(nm), flush=True)
                return orig(*a, **kw)
            return traced
        setattr(flash_attn_gpu, name, make(_orig, name))

m = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    trust_remote_code=True, low_cpu_mem_usage=True).to("cuda")
m.config.use_cache = False
tcfg = m.config.get_text_config()
print("config head_dim =", getattr(tcfg, "head_dim", None), "layer_types[0:6] =", getattr(tcfg, "layer_types", None)[:6])
ids = torch.randint(0, tcfg.vocab_size, (1, 2048), device="cuda")
try:
    with torch.no_grad():
        m(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
    print("FORWARD OK")
except Exception as e:
    print("FORWARD FAILED:", type(e).__name__, str(e)[:120])
