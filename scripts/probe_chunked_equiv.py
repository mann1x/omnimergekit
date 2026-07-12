#!/usr/bin/env python3
"""Isolate WHY chunked teacher-force diverges from full forward on Gemma4 globals.
3 arms on a short prefix (full forward fits):
  A full   : one forward, use_cache=False (reference)
  B chunkM : multi-chunk via DynamicCache(config) + cumulative attention_mask
  C single : ONE chunk (CHUNK>=T) via the cache path + attention_mask
Compare A-vs-B and A-vs-C per layer. If C matches A but B doesn't -> multi-chunk
accumulation bug. If C also diverges -> the cache-path forward itself differs."""
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

BASE = "/srv/ml/models/base/gemma-4-26B-A4B-it"
FX = "/srv/ml/agentic_loop/fixtures/solar_build_start.json"
N = 1536
CHUNK = 512
GLOBALS = (5, 11, 17, 23, 29)


def log(m):
    print("[equiv %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
fx = json.load(open(FX))
chat = tok.apply_chat_template(fx["messages"], tools=fx.get("tools"),
                               add_generation_prompt=True, tokenize=False)
ids = tok(chat, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :N].to("cuda:0")
T = ids.shape[1]
log("prefix tokens=%d chunk=%d" % (T, CHUNK))

base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0}).eval()
routers = sorted([(int(n.split("layers.")[1].split(".")[0]), m)
                  for n, m in base.named_modules() if n.endswith(".router")],
                 key=lambda t: t[0])
Lr = len(routers)
cap = {}


def mk(li):
    def hook(_m, _i, out):
        cap[li] = out[0].detach().float().cpu()
    return hook


handles = [routers[li][1].register_forward_hook(mk(li)) for li in range(Lr)]


def run_chunked(chunk_sz, cache):
    # mirror transformers _prefill chunked path: prepare_inputs_for_generation builds
    # the correct offset-causal mask + positions per chunk against the growing cache
    full_amask = torch.ones(1, T, device="cuda:0", dtype=torch.long)
    acc = {li: [] for li in range(Lr)}
    mk2 = {"past_key_values": cache, "use_cache": True}
    past_length = 0
    for input_chunk in torch.split(ids, chunk_sz, dim=-1):
        current_length = past_length + input_chunk.shape[-1]
        mk2["attention_mask"] = full_amask[:, :current_length]
        model_inputs = base.prepare_inputs_for_generation(input_chunk, **mk2)
        cap.clear()
        with torch.no_grad():
            out = base(**model_inputs, return_dict=True)
        for li in range(Lr):
            acc[li].append(cap[li][0] if cap[li].dim() == 3 else cap[li])
        mk2["past_key_values"] = out.past_key_values
        past_length = current_length
    # introspect per-layer cached key length (global vs sliding)
    try:
        lens = []
        for li in (0, 5, 11, 17, 29):
            kl = out.past_key_values.layers[li].keys.shape[-2]
            lens.append("L%d=%d" % (li, kl))
        log("    cache key-lens: %s (T=%d, global=5,11,17)" % (" ".join(lens), T))
    except Exception as e:  # noqa: BLE001
        log("    cache introspect failed: %s" % e)
    return {li: torch.cat(acc[li], 0) for li in range(Lr)}


# A: full no-cache
cap.clear()
with torch.no_grad():
    base(input_ids=ids, attention_mask=torch.ones_like(ids),
         mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
A = {li: (cap[li][0] if cap[li].dim() == 3 else cap[li]) for li in range(Lr)}
log("A full done")

B = run_chunked(CHUNK, DynamicCache(config=base.config))
log("B multi-chunk(%d) config-cache done" % CHUNK)
D = run_chunked(CHUNK, DynamicCache())
log("D multi-chunk(%d) plain-cache done" % CHUNK)
C = run_chunked(T + 8, DynamicCache(config=base.config))
log("C single-chunk done")

for h in handles:
    h.remove()


def cmp(ref, other, tag):
    gmax, worst = 0.0, None
    gl = 0.0
    for li in range(Lr):
        d = (ref[li] - other[li]).abs().max().item()
        if d > gmax:
            gmax, worst = d, li
        if li in GLOBALS:
            gl = max(gl, d)
    log("  %s: GLOBAL-max|delta|=%.3e (worst L%s)  globals-only-max=%.3e" % (tag, gmax, worst, gl))
    return gmax


log("=" * 56)
ab = cmp(A, B, "A-vs-B config-cache")
ad = cmp(A, D, "A-vs-D plain-cache ")
ac = cmp(A, C, "A-vs-C singlechunk ")
log("VERDICT B(config): %s | D(plain): %s | C(single): %s" % (
    "PASS" if ab < 2e-2 else "FAIL", "PASS" if ad < 2e-2 else "FAIL",
    "PASS" if ac < 2e-2 else "FAIL"))

# ---- statistic-impact: does the per-token divergence change the T223 ranking? ----
import numpy as np  # noqa: E402

DROP = "/srv/ml/scripts/v8coder_fkbroad_drop_map.json"
dm = {int(k): set(int(e) for e in v) for k, v in json.load(open(DROP)).items()}
E = A[0].shape[1]


def dropped_mass(probs):
    # summed-over-tokens mass on the DROPPED experts, per (layer, expert)
    out = []
    for li in range(Lr):
        col = probs[li].sum(0).numpy()  # [E]
        for e in sorted(dm.get(li, set())):
            if 0 <= e < E:
                out.append(((li, e), float(col[e])))
    return out


ma = dict(dropped_mass(A))
mb = dict(dropped_mass(B))
keys = list(ma.keys())
va = np.array([ma[k] for k in keys])
vb = np.array([mb[k] for k in keys])
corr = float(np.corrcoef(va, vb)[0, 1])
# top-K overlap by mass
ta = [k for k, _ in sorted(ma.items(), key=lambda x: -x[1])[:20]]
tb = [k for k, _ in sorted(mb.items(), key=lambda x: -x[1])[:20]]
ov = len(set(ta) & set(tb))
log("STAT IMPACT (dropped-mass over %d (layer,expert) pairs):" % len(keys))
log("  pearson(A,B)=%.4f   total_mass A=%.2f B=%.2f   top20-overlap=%d/20" % (
    corr, va.sum(), vb.sum(), ov))
log("  RANKING %s" % ("PRESERVED -> chunked usable" if corr > 0.98 and ov >= 18
                       else "DEGRADED -> chunked NOT safe for T223"))
