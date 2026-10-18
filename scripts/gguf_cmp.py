"""Compare prithivMLmods' OxCoder BF16 GGUF against LOCAL Qwen3.5-9B and LOCAL
OxCoder, by HTTP-range-fetching individual tensors. No full download."""
import json, glob, struct, urllib.request, numpy as np, torch
from safetensors import safe_open

H = json.load(open("/mnt/sdc/oxopus/gguf_header.json"))
URL, DS = H["url"], H["data_start"]
INFO = {i["name"]: i for i in H["infos"]}
R = "/mnt/sdc/oxopus/models"
FETCHED = [0]

def grab(name, nbytes=None):
    i = INFO[name]
    n = np.prod(i["dims"]) * (4 if i["type"] == 0 else 2)
    n = int(n if nbytes is None else min(nbytes, n))
    s = DS + i["offset"]
    r = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{s+n-1}"})
    with urllib.request.urlopen(r, timeout=120) as h:
        raw = h.read()
    FETCHED[0] += len(raw)
    dt = np.float32 if i["type"] == 0 else None
    if dt is np.float32:
        return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).view(torch.bfloat16)

def load(d):
    t = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys(): t[k] = f
    return t
OX, QB = load(R + "/OxCoder-9B"), load(R + "/Qwen3.5-9B")
hs = {}
def get(t, k):
    if t[k] not in hs: hs[t[k]] = safe_open(t[k], "pt")
    return hs[t[k]].get_tensor(k)

nssm = sorted(int(k.split(".")[1]) for k in INFO if k.endswith("ssm_norm.weight"))
print("GGUF has %d ssm_norm tensors (blocks: %s)" % (nssm.__len__(), nssm))
hf_ssm = sorted(int(k.split(".")[3]) for k in QB if k.endswith("linear_attn.norm.weight"))
print("local Qwen3.5-9B has %d linear_attn.norm.weight (layers: %s)\n" % (len(hf_ssm), hf_ssm))

print("=== ssm_norm: remote GGUF vs LOCAL Qwen  and  vs LOCAL OxCoder ===")
eq_q = eq_o = 0
for n in nssm:
    g = grab(f"blk.{n}.ssm_norm.weight")
    hk = f"model.language_model.layers.{n}.linear_attn.norm.weight"
    q = get(QB, hk).float(); o = get(OX, hk).float()
    sq = torch.equal(g, q); so = torch.equal(g, o)
    eq_q += sq; eq_o += so
    if n < 4 or not so:
        print("  blk.%-2d  ==Qwen:%-5s ==OxCoder:%-5s  max|g-q|=%.6f max|g-o|=%.8f"
              % (n, sq, so, (g-q).abs().max(), (g-o).abs().max()))
print("  TOTAL: identical to Qwen %d/%d   identical to OxCoder %d/%d"
      % (eq_q, len(nssm), eq_o, len(nssm)))

print("\n=== big BF16 matrices: first 128 KB of each vs LOCAL Qwen ===")
pairs = [("token_embd.weight", "model.language_model.embed_tokens.weight"),
         ("output.weight", "lm_head.weight"),
         ("blk.0.ffn_up.weight", "model.language_model.layers.0.mlp.up_proj.weight"),
         ("blk.5.ffn_down.weight", "model.language_model.layers.5.mlp.down_proj.weight"),
         ("blk.20.ffn_gate.weight", "model.language_model.layers.20.mlp.gate_proj.weight")]
NB = 128 * 1024
for gn, hn in pairs:
    if gn not in INFO or hn not in QB:
        print("  skip", gn, hn in QB); continue
    g = grab(gn, NB)
    q = get(QB, hn).flatten()[:g.numel()]
    o = get(OX, hn).flatten()[:g.numel()]
    print("  %-24s ==Qwen:%-5s ==OxCoder:%-5s  n=%d" %
          (gn, torch.equal(g, q), torch.equal(g, o), g.numel()))
print("\ntotal bytes range-fetched for tensor data: %.1f KB" % (FETCHED[0]/1024))
