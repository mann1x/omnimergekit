#!/usr/bin/env python3
"""bonsai_qat_pilot.py — does plain rotation-in-the-loop QAT reproduce PrismML's fingerprint?

First L+1 blocks of base Qwen3.8-27B. Student = same blocks with every Bonsai-contract linear
re-parametrised as a LATENT weight in PrismML's folded basis; the weight actually used is
   unfold( project(latent) ),  project = keep the 86 largest |w| per 128-group, sign, one scale
with a straight-through gradient. Every full-precision tensor (norms, ssm_alpha/beta, conv1d,
A_log, dt_bias) trains too, because PrismML's moved (prefix test). Loss: per-block hidden-state
distillation against the base blocks. The embedding stays full precision (pilot scope).

It is NOT trying to make a good model. It asks three fingerprint questions of a known recipe:
  1. of the trits training flips, what share land on PrismML's value?   (chance = base rate)
  2. does held-out block-output error approach the PrismML prefix's?
  3. do the full-precision tensors move the way PrismML's did (conv1d, beta, alpha >> norms)?
"""
import argparse
import json
import math

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForCausalLM

from bonsai_hadamard_fold import fwht_blocks, read_kv

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--prism-gguf", required=True)
ap.add_argument("--pack", required=True)
ap.add_argument("--train-text", required=True)
ap.add_argument("--eval-text", required=True)
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--nz", type=int, default=86)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
dev, L, G = a.device, a.layer, 128


class FWHT(torch.autograd.Function):
    """Orthonormal + symmetric, so the adjoint is the transform itself: no saved intermediates."""
    @staticmethod
    def forward(ctx, x):
        return fwht_blocks(x, 1024)

    @staticmethod
    def backward(ctx, g):
        return fwht_blocks(g, 1024)


c = read_kv(a.prism_gguf, "prism.hadamard.")
ak = read_kv(a.prism_gguf, "qwen35.ssm.")
signs, off = {}, 0
for w in c["prism.hadamard.sign_widths"]:
    signs[w] = torch.tensor(c["prism.hadamard.sign_values"][off:off + w], dtype=torch.float32, device=dev)
    off += w
n_v, n_k, width = ak["qwen35.ssm.time_step_rank"], ak["qwen35.ssm.group_count"], ak["qwen35.ssm.inner_size"]
hd, rp = width // n_v, n_v // n_k
gi = torch.arange(width, device=dev)
perm = gi % hd + hd * (((gi // hd) // rp) + n_k * ((gi // hd) % rp))
inv_perm = torch.argsort(perm)

cfg = json.load(open(f"{a.model_dir}/config.json"))
tc = cfg["text_config"]
tc["num_hidden_layers"] = L + 1
for k, v in list(tc.items()):
    if isinstance(v, list) and len(v) == 64:
        tc[k] = v[:L + 1]
config = Qwen3_5Config(**cfg).text_config
idx = json.load(open(f"{a.model_dir}/model.safetensors.index.json"))["weight_map"]


def build():
    m = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
    want, sd = m.state_dict(), {}
    for k, shard in idx.items():
        cand = k.replace("model.language_model.", "model.")
        if cand in want:
            with safe_open(f"{a.model_dir}/{shard}", "pt") as f:
                sd[cand] = f.get_tensor(k)
    m.load_state_dict(sd, strict=False)
    return m.float().to(dev).eval()


pack = torch.load(a.pack)
teacher = build()
for p_ in teacher.parameters():
    p_.requires_grad_(False)
base_sd = {k: v.clone() for k, v in teacher.state_dict().items()}

tern = {}                                     # hf key -> spec
for nm, e in pack["tern"].items():
    k = e["key"]
    ix = e["ix"].to(dev)
    g = base_sd[k].reshape(-1)[ix].reshape(e["gshape"])          # GGUF layout, primal
    is_out = ".ssm_out." in nm
    if is_out:
        g = g[:, perm]
    s = signs[g.shape[-1]]
    lat = torch.nn.Parameter(fwht_blocks(g * s, 1024))
    tern[k] = {"nm": nm, "lat": lat, "s": s, "is_out": is_out, "inv_ix": torch.argsort(ix),
               "hshape": base_sd[k].shape, "T": e["T"].to(dev).reshape(-1, G), "d": e["d"].to(dev)}
fp = {}
for nm, k in pack["fp_keys"].items():
    fp[k] = torch.nn.Parameter(base_sd[k].clone())
print(f"ternary latents: {len(tern)}  trainable FP tensors: {len(fp)}", flush=True)


def project(lat):
    g = lat.reshape(-1, G)
    a_ = g.abs()
    m = a_ >= a_.kthvalue(G - a.nz + 1, dim=1, keepdim=True).values
    d = (a_ * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
    return torch.sign(g) * m, d


def weights():
    out = dict(fp)
    for k, e in tern.items():
        T, d = project(e["lat"].detach())
        q = (T * d).reshape(e["lat"].shape)
        q = q + e["lat"] - e["lat"].detach()                      # straight-through
        u = FWHT.apply(q) * e["s"]
        if e["is_out"]:
            u = u[:, inv_perm]
        out[k] = u.reshape(-1)[e["inv_ix"]].reshape(e["hshape"])
    return out


def hiddens(model, ids, w=None):
    f = (lambda: model.model(input_ids=ids, output_hidden_states=True)) if w is None else \
        (lambda: torch.func.functional_call(model, {"model." + k[6:] if k.startswith("model.") else k: v for k, v in w.items()},
                                            kwargs={"input_ids": ids, "output_hidden_states": True}))
    o = f()
    hs = o.hidden_states if hasattr(o, "hidden_states") and o.hidden_states is not None else o[-1]
    return hs[1:]                                                 # after each block, pre final-norm


tok = AutoTokenizer.from_pretrained(a.model_dir)
tr = tok(open(a.train_text).read()[:8_000_000], return_tensors="pt").input_ids[0]
ev = tok(open(a.eval_text).read()[:400000], return_tensors="pt").input_ids[0][: 8 * a.ctx].reshape(8, a.ctx).to(dev)
print(f"train tokens {len(tr)}  eval tokens {ev.numel()}", flush=True)


@torch.no_grad()
def evaluate(w=None, model=None):
    model = model or teacher
    num = [0.0] * (L + 1)
    den = [0.0] * (L + 1)
    for i in range(ev.shape[0]):
        ht = hiddens(teacher, ev[i:i + 1])
        hs = hiddens(model, ev[i:i + 1], w)
        for j in range(L + 1):
            num[j] += float((hs[j] - ht[j]).pow(2).sum())
            den[j] += float(ht[j].pow(2).sum())
    return [math.sqrt(n / d_) for n, d_ in zip(num, den)]


def trit_stats(T0=None):
    r = {}
    for k, e in tern.items():
        T, d = project(e["lat"].detach())
        fam = e["nm"].split(".", 2)[2]
        x = r.setdefault(fam, {"agree": [], "scale_ratio": [], "flipped": [], "flip_to_prism": [], "flip_from_prism": []})
        x["agree"].append(float((T == e["T"]).float().mean()))
        x["scale_ratio"].append(float((e["d"] / d.reshape(-1)).median()))
        if T0 is not None:
            ch = T != T0[k]
            x["flipped"].append(float(ch.float().mean()))
            if ch.any():
                x["flip_to_prism"].append(float((T[ch] == e["T"][ch]).float().mean()))
                x["flip_from_prism"].append(float((T0[k][ch] == e["T"][ch]).float().mean()))
    return {f: {m: (sum(v) / len(v) if v else None) for m, v in x.items()} for f, x in r.items()}


R = {"args": vars(a)}
prism = build()
prism.load_state_dict({k: v for k, v in pack["prism_sd"].items()}, strict=False)
R["heldout_block_err_PRISM_prefix"] = evaluate(model=prism)
del prism
torch.cuda.empty_cache()
with torch.no_grad():
    T_init = {k: project(e["lat"].detach())[0].clone() for k, e in tern.items()}
    R["heldout_block_err_init_top86"] = evaluate(weights(), teacher)
R["trits_init"] = trit_stats()
print("PRISM prefix err per block:", R["heldout_block_err_PRISM_prefix"], flush=True)
print("init (top-86 RTN) err     :", R["heldout_block_err_init_top86"], flush=True)

params = [e["lat"] for e in tern.values()] + list(fp.values())
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, anneal_strategy="cos")
gen = torch.Generator().manual_seed(0)
R["curve"] = []
for step in range(1, a.steps + 1):
    st = torch.randint(0, len(tr) - a.ctx - 1, (a.bs,), generator=gen)
    ids = torch.stack([tr[s_:s_ + a.ctx] for s_ in st]).to(dev)
    with torch.no_grad():
        ht = hiddens(teacher, ids)
    hs = hiddens(teacher, ids, weights())
    loss = sum((hs[j] - ht[j]).pow(2).sum() / ht[j].pow(2).sum() for j in range(L + 1)) / (L + 1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    if step % 25 == 0 or step == 1:
        print(f"step {step:4d} loss {float(loss):.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
    if step % 150 == 0 or step == a.steps:
        with torch.no_grad():
            err = evaluate(weights(), teacher)
        ts = trit_stats(T_init)
        R["curve"].append({"step": step, "heldout_block_err": err, "trits": ts})
        k3 = ts.get("attn_k.weight", {})
        print(f"  [eval {step}] heldout err/block {['%.3f' % x for x in err]}  attn_k agree {k3.get('agree')}", flush=True)
        json.dump(R, open(a.out, "w"), indent=1)

fpm = {}
for nm, k in pack["fp_keys"].items():
    fam = nm.split(".", 2)[2]
    b = base_sd[k]
    fpm.setdefault(fam, {"ours": [], "prism": []})
    fpm[fam]["ours"].append(float((fp[k].detach() - b).norm() / b.norm()))
    fpm[fam]["prism"].append(float((pack["prism_sd"][k].to(dev) - b).norm() / b.norm()))
R["fp_rel_change_vs_base (HF param space)"] = {f: {m: sum(v) / len(v) for m, v in x.items()} for f, x in fpm.items()}
R["trits_final"] = trit_stats(T_init)
json.dump(R, open(a.out, "w"), indent=1)
print(json.dumps({"fp_rel_change": R["fp_rel_change_vs_base (HF param space)"], "trits_final": R["trits_final"]}, indent=1))
print("PILOT_DONE")
