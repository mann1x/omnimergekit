#!/usr/bin/env python3
"""bonsai_pt2_blocks.py — PT2-LLM's recipe adapted to the Bonsai 2 format, first L+1 blocks.

PT2-LLM (arXiv 2510.03267) = GPTQ sweep + iterative ternary fitting (ITF) + activation-aware grid
alignment (AGA) + column reordering (SSR), with a per-ROW asymmetric grid alpha*T + mu.
Bonsai's format has no offset, no column permutation, and one scale per 128 consecutive inputs in
the Hadamard-folded basis. The adaptation, all in that folded basis:
   GPTQ block      = one 128-column Bonsai scale group
   ITF             = alternate {T = round(w/alpha)} / {alpha = <w,T>/<T,T>} per (row, group), symmetric
   AGA             = one closed-form refit of all group scales of a row against H = X^T X, trits frozen
   mu, SSR         = dropped (not storable); the rotation is expected to do their job
Pure PTQ: full-precision tensors are untouched. Blocks are processed in order and each block is
calibrated on the outputs of the already-ternarized prefix.

Arms (cumulative), scored on the SAME held-out text and metric as bonsai_qat_pilot.py:
   itf        weight-space fit only, no calibration data
   itf+gptq   + error compensation
   itf+gptq+aga
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
ap.add_argument("--calib-text", required=True)
ap.add_argument("--eval-text", required=True)
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--n-calib", type=int, default=96)
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--arms", default="itf,itf+gptq,itf+gptq+aga")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
dev, L, G = a.device, a.layer, 128

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
base_sd = {k: v.clone() for k, v in teacher.state_dict().items()}
spec = {}
for nm, e in pack["tern"].items():
    k = e["key"]
    ix = e["ix"].to(dev)
    out_f, in_f = e["gshape"]
    colmap = ix.reshape(out_f, in_f)[0] % in_f            # GGUF column j reads HF input colmap[j]
    assert torch.equal(torch.sort(colmap).values, torch.arange(in_f, device=dev)), nm
    is_out = ".ssm_out." in nm
    spec[k] = {"nm": nm, "ix": ix, "inv_ix": torch.argsort(ix), "gshape": (out_f, in_f), "is_out": is_out,
               "xmap": colmap[perm] if is_out else colmap, "s": signs[in_f], "blk": int(nm.split(".")[1]),
               "T": e["T"].to(dev).reshape(out_f, in_f)}


def to_folded(k, w_hf):
    e = spec[k]
    g = w_hf.reshape(-1)[e["ix"]].reshape(e["gshape"])
    if e["is_out"]:
        g = g[:, perm]
    return fwht_blocks(g * e["s"], 1024)


def from_folded(k, q):
    e = spec[k]
    u = fwht_blocks(q, 1024) * e["s"]
    if e["is_out"]:
        u = u[:, inv_perm]
    return u.reshape(-1)[e["inv_ix"]].reshape(base_sd[k].shape)


def itf(w, iters=10):
    """symmetric iterative ternary fit of one 128-column group; w [rows, 128] -> alpha [rows, 1]"""
    aw = w.abs()
    m = aw > 0.75 * aw.mean(1, keepdim=True)
    al = (aw * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
    for _ in range(iters):
        T = torch.round(w / al.clamp(min=1e-12)).clamp(-1, 1)
        al = (w * T).sum(1, keepdim=True) / (T * T).sum(1, keepdim=True).clamp(min=1)
    return al.clamp(min=1e-12)


def quantize(Wf, H, arm):
    """Wf [out, in] folded (float64), H [in, in]. Returns Q = T*alpha (float64)."""
    n_out, n_in = Wf.shape
    use_gptq = "gptq" in arm
    W = Wf.clone()
    Q = torch.zeros_like(W)
    if use_gptq:
        Hd = H + a.damp * H.diag().mean() * torch.eye(n_in, device=dev, dtype=H.dtype)
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
    for i1 in range(0, n_in, G):
        i2 = i1 + G
        W1 = W[:, i1:i2].clone()
        al = itf(W1)[:, 0]
        if not use_gptq:
            Q[:, i1:i2] = torch.round(W1 / al[:, None]).clamp(-1, 1) * al[:, None]
            continue
        Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for j in range(G):
            w = W1[:, j]
            q = torch.round(w / al).clamp(-1, 1) * al
            Q[:, i1 + j] = q
            err = (w - q) / Hi[j, j]
            W1[:, j:] -= err[:, None] * Hi[j, j:][None]
            Err[:, j] = err
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    if "aga" in arm:
        # per row: min_alpha || (w - sum_g alpha_g T_g) X ||^2, T frozen  ->  (Tg H Tg^T) alpha = Tg H w
        T = torch.sign(Q)
        ng = n_in // G
        WH = Wf @ H
        new = torch.empty(n_out, ng, device=dev, dtype=H.dtype)
        chunk = max(1, int(2.0e9 // (ng * n_in * 8)))
        eye = torch.eye(ng, device=dev, dtype=H.dtype)
        for r0 in range(0, n_out, chunk):
            Tr = T[r0:r0 + chunk]                                    # [c, in]
            Tg = torch.zeros(Tr.shape[0], ng, n_in, device=dev, dtype=H.dtype)
            Tg.view(Tr.shape[0], ng, ng, G)[:, torch.arange(ng), torch.arange(ng)] = Tr.view(-1, ng, G)
            A = (Tg @ H) @ Tg.transpose(1, 2)                        # [c, ng, ng]
            b = (Tg * WH[r0:r0 + chunk, None, :]).sum(2)             # [c, ng]
            A = A + 1e-6 * A.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * eye
            new[r0:r0 + chunk] = torch.linalg.solve(A, b)
        old = Q.abs().reshape(n_out, ng, G).amax(2)
        new = torch.where(new > 0, new, old)                         # a non-positive scale is not storable
        Q = T * new.repeat_interleave(G, 1)
    return Q


def hiddens(model, ids):
    return model.model(input_ids=ids, output_hidden_states=True).hidden_states[1:]


tok = AutoTokenizer.from_pretrained(a.model_dir)
cal = tok(open(a.calib_text).read()[:4_000_000], return_tensors="pt").input_ids[0]
gen = torch.Generator().manual_seed(0)
st = torch.randint(0, len(cal) - a.ctx - 1, (a.n_calib,), generator=gen)
cal = torch.stack([cal[s_:s_ + a.ctx] for s_ in st])
ev = tok(open(a.eval_text).read()[:400000], return_tensors="pt").input_ids[0][: 8 * a.ctx].reshape(8, a.ctx).to(dev)
print(f"calibration {cal.numel()} tokens ({a.n_calib} x {a.ctx}); eval {ev.numel()} tokens", flush=True)


@torch.no_grad()
def evaluate(model):
    num, den = [0.0] * (L + 1), [0.0] * (L + 1)
    for i in range(ev.shape[0]):
        ht, hs = hiddens(teacher, ev[i:i + 1]), hiddens(model, ev[i:i + 1])
        for j in range(L + 1):
            num[j] += float((hs[j] - ht[j]).pow(2).sum())
            den[j] += float(ht[j].pow(2).sum())
    return [math.sqrt(n / d_) for n, d_ in zip(num, den)]


R = {"args": vars(a)}
prism = build()
prism.load_state_dict(pack["prism_sd"], strict=False)
with torch.no_grad():
    R["PRISM prefix"] = {"heldout_block_err": evaluate(prism)}
del prism
torch.cuda.empty_cache()
print("PRISM prefix          heldout err/block", ["%.3f" % x for x in R["PRISM prefix"]["heldout_block_err"]], flush=True)

for arm in a.arms.split(","):
    student = build()
    agree, zeros = [], []
    for b in range(L + 1):
        keys = [k for k, e in spec.items() if e["blk"] == b]
        Hs = {k: torch.zeros(spec[k]["gshape"][1], spec[k]["gshape"][1], device=dev, dtype=torch.float64) for k in keys}
        cnt = {k: 0 for k in keys}
        if arm != "itf":
            hooks = []
            for k in keys:
                def mk(k):
                    e = spec[k]

                    def hook(mod, inp):
                        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])[:, e["xmap"]]
                        xr = fwht_blocks(x * e["s"], 1024).double()
                        Hs[k].add_(xr.T @ xr)
                        cnt[k] += x.shape[0]
                    return hook
                hooks.append(student.get_submodule(k[: -len(".weight")]).register_forward_pre_hook(mk(k)))
            with torch.no_grad():
                for i in range(cal.shape[0]):
                    student.model(input_ids=cal[i:i + 1].to(dev))
            for h in hooks:
                h.remove()
        sd = student.state_dict()
        for k in keys:
            Wf = to_folded(k, base_sd[k]).double()
            H = Hs[k] / max(cnt[k], 1)
            Q = quantize(Wf, H, arm)
            sd[k].copy_(from_folded(k, Q.float()))
            Tq = torch.sign(Q)
            agree.append(float((Tq == spec[k]["T"]).double().mean()))
            zeros.append(float((Tq == 0).double().mean()))
            del Wf, H, Q
        Hs.clear()
        torch.cuda.empty_cache()
        print(f"  [{arm}] block {b} done", flush=True)
    with torch.no_grad():
        err = evaluate(student)
    R[arm] = {"heldout_block_err": err, "trit_agreement_with_prism": sum(agree) / len(agree), "zero_frac": sum(zeros) / len(zeros)}
    print(f"{arm:<21s} heldout err/block {['%.3f' % x for x in err]}  trit-agree {R[arm]['trit_agreement_with_prism']:.4f}  zeros {R[arm]['zero_frac']:.3f}", flush=True)
    json.dump(R, open(a.out, "w"), indent=1)
    del student
    torch.cuda.empty_cache()
print("PT2_BLOCKS_DONE")
