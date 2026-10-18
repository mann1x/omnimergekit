"""Sweep ALL 427 tensors of the remote OxCoder BF16 GGUF against local Qwen3.5-9B.
F32 tensors fetched in full; BF16 tensors sampled at a 64 KB prefix."""
import json, glob, re, urllib.request, numpy as np, torch, collections
from safetensors import safe_open

H = json.load(open("/mnt/sdc/oxopus/gguf_header.json"))
URL, DS = H["url"], H["data_start"]
R = "/mnt/sdc/oxopus/models"
FETCHED = [0]

def grab(i, nbytes=None):
    esz = 4 if i["type"] == 0 else 2
    n = int(np.prod(i["dims"])) * esz
    n = n if nbytes is None else min(nbytes, n)
    n -= n % esz
    s = DS + i["offset"]
    r = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{s+n-1}"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(r, timeout=180) as h: raw = h.read()
            break
        except Exception: raw = None
    if raw is None: return None
    FETCHED[0] += len(raw)
    if i["type"] == 0:
        return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).view(torch.bfloat16)

def load(d):
    t = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys(): t[k] = f
    return t
QB, OX = load(R + "/Qwen3.5-9B"), load(R + "/OxCoder-9B")
hs = {}
def get(t, k):
    if t[k] not in hs: hs[t[k]] = safe_open(t[k], "pt")
    return hs[t[k]].get_tensor(k)

L = "model.language_model.layers"
SUF = {"attn_norm.weight": "input_layernorm.weight",
       "post_attention_norm.weight": "post_attention_layernorm.weight",
       "ssm_norm.weight": "linear_attn.norm.weight",
       "ssm_in.weight": "linear_attn.in_proj.weight",
       "ssm_out.weight": "linear_attn.out_proj.weight",
       "ssm_conv1d.weight": "linear_attn.conv1d.weight",
       "ssm_a": "linear_attn.A_log", "ssm_dt.bias": "linear_attn.dt_bias",
       "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
       "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
       "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
       "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight",
       "ffn_gate.weight": "mlp.gate_proj.weight",
       "ffn_norm.weight": "post_attention_layernorm.weight"}
def hfname(g):
    if g == "token_embd.weight": return "model.language_model.embed_tokens.weight"
    if g == "output.weight": return "lm_head.weight"
    if g == "output_norm.weight": return "model.language_model.norm.weight"
    m = re.match(r"blk\.(\d+)\.(.+)$", g)
    if not m: return None
    n, suf = m.group(1), m.group(2)
    return f"{L}.{n}.{SUF[suf]}" if suf in SUF else None

NB = 64 * 1024
res = collections.Counter(); diff_names = []; unmapped = []
for i in H["infos"]:
    hn = hfname(i["name"])
    if hn is None or hn not in QB:
        unmapped.append(i["name"]); continue
    g = grab(i, None if i["type"] == 0 else NB)
    if g is None: res["fetch_fail"] += 1; continue
    q = get(QB, hn).flatten()[:g.numel()]
    if q.dtype != g.dtype: q = q.float() if g.dtype == torch.float32 else q
    if torch.equal(g, q): res["same_as_qwen"] += 1
    else:
        res["DIFFERS"] += 1
        o = get(OX, hn).flatten()[:g.numel()]
        if o.dtype != g.dtype: o = o.float() if g.dtype == torch.float32 else o
        diff_names.append((i["name"], bool(torch.equal(g, o)), float((g-q).abs().max())))
print("SWEEP of remote OxCoder BF16 GGUF vs LOCAL Qwen/Qwen3.5-9B")
print("  compared          : %d tensors" % (res["same_as_qwen"]+res["DIFFERS"]))
print("  BIT-IDENTICAL     : %d" % res["same_as_qwen"])
print("  DIFFERS           : %d" % res["DIFFERS"])
if unmapped: print("  unmapped (skipped): %d  %s" % (len(unmapped), unmapped[:6]))
if res["fetch_fail"]: print("  fetch failures    : %d" % res["fetch_fail"])
print("\n  every differing tensor (and whether it matches local OxCoder):")
for n, mo, mx in diff_names:
    print("    %-34s ==localOxCoder:%-5s max|gguf-qwen|=%.6f" % (n, mo, mx))
print("\n  bytes range-fetched: %.1f MB  (file is 17.92 GB)" % (FETCHED[0]/1e6))
