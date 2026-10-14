#!/usr/bin/env python
"""Ternary recovery on Qwen3-1.7B in the Bonsai-2 format: does a calibrated start beat a magnitude start?

Same rotated basis and training loop as the public alllwang/bonsai-repro-1.7B-ternary recipe (blockwise signed
Hadamard n=1024, groups of 128, straight-through KD against the full-precision teacher, KL + ce_weight*CE).
Two ways to start, selected by --init:

  top86  their start: latent = rotated weights, quantizer = keep the 86 largest |w| per group of 128.
  gptq86 calibrated start under PrismML's own quantizer: each group of 128 is projected to its top-86 with the
         mean-of-kept scale, and the group's whole error is pushed onto the not-yet-quantized groups through the
         inverse Hessian (block OBS update). The latent is that compensated weight; training uses the top-86 rule.
  gptq   ours: sequential GPTQ over the 128-column scale groups on activations from the already-ternarized
         prefix, then a closed-form refit of the group scales. The latent is the error-compensated weight,
         so its rounding reproduces the calibrated trits exactly; training continues from there with a
         round-to-nearest quantizer (fixed thresholds, trainable output scales).

--eval-only DIR scores any HF checkpoint (PrismML's unpacked 1.7B, a community reproduction) in the same
harness. Everything is local: text comes from a parquet shard and a raw eval file, no dataset streaming.
"""
import argparse
import json
import math
import os
import random
import time

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--train-parquet", default="")
ap.add_argument("--eval-text", required=True)
ap.add_argument("--eval-only", default="", help="HF checkpoint dir to score against the teacher, then exit")
ap.add_argument("--init", choices=["top86", "gptq", "gptq86"], default="top86")
ap.add_argument("--out", default="")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--lr", type=float, default=6e-5)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--ce-weight", type=float, default=0.1)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--save-every", type=int, default=1000)
ap.add_argument("--n-calib", type=int, default=96)
ap.add_argument("--calib-docs", type=int, default=2000, help="leading parquet rows reserved for calibration")
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--freeze-scales", action="store_true")
ap.add_argument("--lam-ramp", type=float, default=0.0, help="soft-to-hard: blend latent and ternary weight, hard after this fraction of steps")
ap.add_argument("--n-eval", type=int, default=32)
ap.add_argument("--eval-text-b", default="", help="second eval text (another domain); reported as kl_b / top1_b / ppl_b")
ap.add_argument("--load", default="", help="latent.pt of an earlier run with the same --init, loaded after the student is built")
ap.add_argument("--mix-text", default="", help="chat-format text (teacher's own traces) mixed into the training stream")
ap.add_argument("--mix-frac", type=float, default=0.0, help="share of training sequences drawn from --mix-text")
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = a.device
G, BLK, NZERO = 128, 1024, 42
LAM = [1.0]      # share of the ternary weight in the forward pass; 1.0 = hard (always used for eval)
LINEARS = [("self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + [("mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]
log = None


def say(*x):
    s = " ".join(str(v) for v in x)
    print(s, flush=True)
    if log:
        log.write(s + "\n")
        log.flush()


tok = AutoTokenizer.from_pretrained(a.base)
teacher = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)

# ---------------- eval (shared by every arm and every external checkpoint) ----------------
def eval_set(path):
    ids = tok(open(path).read(), return_tensors="pt").input_ids[0]
    assert len(ids) >= a.n_eval * 1024, f"{path}: {len(ids)} tokens < {a.n_eval} x 1024"
    ev = torch.stack([ids[i * 1024:(i + 1) * 1024] for i in range(a.n_eval)]).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ev_t = [teacher(x[None]).logits[0, :-1].float().cpu() for x in ev]
    t_nll = sum(-F.log_softmax(lt.to(dev), -1).gather(-1, x[1:, None]).mean().item() for lt, x in zip(ev_t, ev)) / len(ev)
    say(f"eval: {a.n_eval} x 1024 tokens of {os.path.basename(path)}; teacher ppl {math.exp(t_nll):.3f}")
    return ev, ev_t


EVSETS = [("", *eval_set(a.eval_text))] + ([("_b", *eval_set(a.eval_text_b))] if a.eval_text_b else [])


@torch.no_grad()
def evaluate(model, tag, extra=None):
    was, lam = model.training, LAM[0]
    LAM[0] = 1.0
    model.eval()
    res = dict(tag=tag)
    for sfx, ev, ev_t in EVSETS:
        nll = kl = top1 = 0.0
        for x, lt in zip(ev, ev_t):
            lt = lt.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls = model(x[None]).logits[0, :-1].float()
            v = min(lt.shape[-1], ls.shape[-1])
            lt, ls = lt[:, :v], ls[:, :v]
            lp = F.log_softmax(ls, -1)
            nll += -lp.gather(-1, x[1:, None]).mean().item()
            kl += (F.softmax(lt, -1) * (F.log_softmax(lt, -1) - lp)).sum(-1).mean().item()
            top1 += (ls.argmax(-1) == lt.argmax(-1)).float().mean().item()
        n = len(ev)
        res.update({"ppl" + sfx: math.exp(nll / n), "kl" + sfx: kl / n, "top1" + sfx: top1 / n})
    res.update(extra or {})
    say("EVAL", json.dumps(res))
    model.train(was)
    LAM[0] = lam
    return res


if a.eval_only:
    m = AutoModelForCausalLM.from_pretrained(a.eval_only, dtype=torch.bfloat16).to(dev)
    evaluate(m, os.path.basename(a.eval_only.rstrip("/")))
    raise SystemExit(0)

os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "log.txt"), "a")
say("args", json.dumps(vars(a)))

# ---------------- rotation ----------------
Hm = torch.tensor([[1.0]])
while Hm.shape[0] < BLK:
    Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
Hm = (Hm / math.sqrt(BLK)).to(dev)
SIGNS = {}


def signs_for(n):
    if n not in SIGNS:
        g = torch.Generator().manual_seed(1234 + n)
        SIGNS[n] = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(dev)
    return SIGNS[n]


def rot(x, s):  # x [..., n] -> (x * s) H   blockwise; used for weights (rows) and activations alike
    sh = x.shape
    return ((x.reshape(-1, sh[-1] // BLK, BLK) * s.reshape(1, -1, BLK).to(x.dtype)) @ Hm.to(x.dtype)).reshape(sh)


def unrot(x, s):
    sh = x.shape
    return ((x.reshape(-1, sh[-1] // BLK, BLK) @ Hm.to(x.dtype)) * s.reshape(1, -1, BLK).to(x.dtype)).reshape(sh)


# ---------------- quantizers ----------------
def top86(w):
    o, i = w.shape
    wg = w.reshape(o, i // G, G)
    aw = wg.abs()
    mask = aw > aw.kthvalue(NZERO, dim=-1, keepdim=True).values
    t = torch.sign(wg) * mask
    s = (aw * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
    return (s * t).reshape(o, i), t.reshape(o, i)


def itf(w, iters=10):
    aw = w.abs()
    m = aw > 0.75 * aw.mean(1, keepdim=True)
    al = (aw * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
    for _ in range(iters):
        t = torch.round(w / al.clamp(min=1e-12)).clamp(-1, 1)
        al = (w * t).sum(1, keepdim=True) / (t * t).sum(1, keepdim=True).clamp(min=1)
    return al.clamp(min=1e-12)


@torch.no_grad()
def calibrate(w32, h):
    """w32 [out, in] rotated. h [in, in] float64 or None. -> latent, threshold scale, output scale (all float32).
    round(latent / threshold).clamp(-1, 1) is the calibrated trit pattern."""
    n_out, n_in = w32.shape
    ng = n_in // G
    lat = torch.empty_like(w32)
    thr = torch.empty(n_out, ng, device=dev)
    sc = torch.empty(n_out, ng, device=dev)
    if h is not None:
        hd = h + a.damp * h.diag().mean() * torch.eye(n_in, device=dev, dtype=h.dtype)
        hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hd)), upper=True)
        eye = torch.eye(ng, device=dev, dtype=torch.float64)
    rows = max(256, int(4.0e9 // (n_in * 8 * 4)))
    for r0 in range(0, n_out, rows):
        wf = w32[r0:r0 + rows].double()
        w = wf.clone()
        nr = w.shape[0]
        lc = torch.empty_like(w)
        al_all = torch.empty(nr, ng, device=dev, dtype=torch.float64)
        t_all = torch.empty_like(w)
        for g, i1 in enumerate(range(0, n_in, G)):
            i2 = i1 + G
            w1 = w[:, i1:i2].clone()
            al = itf(w1)[:, 0]
            al_all[:, g] = al
            if h is None:
                lc[:, i1:i2] = w1
                t_all[:, i1:i2] = torch.round(w1 / al[:, None]).clamp(-1, 1)
                continue
            err_b = torch.zeros_like(w1)
            hi = hinv[i1:i2, i1:i2]
            for j in range(G):
                wj = w1[:, j]
                lc[:, i1 + j] = wj
                tj = torch.round(wj / al).clamp(-1, 1)
                t_all[:, i1 + j] = tj
                err = (wj - tj * al) / hi[j, j]
                w1[:, j:] -= err[:, None] * hi[j, j:][None]
                err_b[:, j] = err
            w[:, i2:] -= err_b @ hinv[i1:i2, i2:]
        scale = al_all.clone()
        if h is not None:                       # refit the group scales against the Hessian, trits frozen
            tv = t_all.view(nr, ng, G)
            am = torch.empty(nr, ng, ng, device=dev, dtype=torch.float64)
            for g in range(ng):
                p = t_all[:, g * G:(g + 1) * G] @ h[g * G:(g + 1) * G, :]
                am[:, g, :] = (p.view(nr, ng, G) * tv).sum(-1)
            b = ((wf @ h).view(nr, ng, G) * tv).sum(-1)
            am = am + 1e-6 * am.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * eye
            new = torch.linalg.solve(am, b)
            scale = torch.where(new > 0, new, scale)
        lat[r0:r0 + rows], thr[r0:r0 + rows], sc[r0:r0 + rows] = lc.float(), al_all.float(), scale.float()
    return lat, thr, sc


@torch.no_grad()
def calibrate86(w32, h):
    """Group-wise OBS with a hard top-86 projection. Returns the compensated latent (top86(latent) is the pattern)."""
    n_out, n_in = w32.shape
    hd = h + a.damp * h.diag().mean() * torch.eye(n_in, device=dev, dtype=h.dtype)
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hd)), upper=True)
    lat = torch.empty_like(w32)
    rows = max(256, int(4.0e9 // (n_in * 8 * 4)))
    for r0 in range(0, n_out, rows):
        w = w32[r0:r0 + rows].double()
        for i1 in range(0, n_in, G):
            i2 = i1 + G
            w1 = w[:, i1:i2]
            lat[r0:r0 + rows, i1:i2] = w1.float()
            q1 = top86(w1)[0]
            e = torch.linalg.solve_triangular(u[i1:i2, i1:i2], w1 - q1, upper=True, left=False)   # e @ U_bb = w1 - q1
            w[:, i2:] -= e @ u[i1:i2, i2:]
    return lat


class TernaryWeight(nn.Module):
    """A [out, in] weight held as a latent in the rotated basis and read through a ternary projection."""

    def __init__(self, w_rot, mode, h=None):
        super().__init__()
        self.mode = mode
        if mode == "top86":
            self.weight = nn.Parameter(w_rot.clone())
        elif mode == "gptq86":
            self.weight = nn.Parameter(w_rot.clone() if h is None else calibrate86(w_rot, h))
            self.mode = "top86"
        else:
            lat, thr, sc = calibrate(w_rot, h)
            self.weight = nn.Parameter(lat)
            self.register_buffer("thr", thr)
            self.scale = nn.Parameter(sc, requires_grad=not a.freeze_scales)
        with torch.no_grad():
            self.register_buffer("t0", self.trits().to(torch.int8))

    def trits(self):
        if self.mode == "top86":
            return top86(self.weight)[1]
        o, i = self.weight.shape
        return torch.round(self.weight.view(o, i // G, G) / self.thr[..., None]).clamp(-1, 1).view(o, i)

    def forward(self):
        w = self.weight
        if self.mode == "top86":
            wq = top86(w)[0]
            return w + LAM[0] * (wq - w).detach()
        o, i = w.shape
        wq = (self.trits().detach().view(o, i // G, G) * self.scale[..., None]).view(o, i)
        return w + LAM[0] * (wq - w.detach()) if LAM[0] < 1.0 else wq + (w - w.detach())

    @torch.no_grad()
    def drift(self):
        t, t0 = self.trits(), self.t0.float()
        nz = (t != 0) & (t0 != 0)
        return (((t != t0) & nz).float().sum() / nz.float().sum()).item(), ((t == 0) != (t0 == 0)).float().mean().item(), (t == 0).float().mean().item()


class QLinear(nn.Module):
    def __init__(self, lin, mode, h=None):
        super().__init__()
        self.s = signs_for(lin.weight.shape[1])
        self.tw = TernaryWeight(rot(lin.weight.data.float(), self.s), mode, h)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.float())

    def forward(self, x):
        return F.linear(rot(x, self.s), self.tw().to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class QEmbedding(nn.Module):
    """Tied embedding + head. Lookup: unrotate(Q[ids]); logits: rotate(x) @ Q^T."""

    def __init__(self, e, mode):
        super().__init__()
        self.s = signs_for(e.shape[1])
        self.tw = TernaryWeight(rot(e.float(), self.s), mode, None)

    def forward(self, ids):
        return unrot(F.embedding(ids, self.tw()), self.s)

    def head(self, x):
        return F.linear(rot(x, self.s), self.tw().to(x.dtype))


class QHead(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb

    def forward(self, x):
        return self.emb.head(x)


# ---------------- data ----------------
def docs(start):
    pf = pq.ParquetFile(a.train_parquet)
    k = 0
    for b in pf.iter_batches(batch_size=256, columns=["text"]):
        for t in b.column("text").to_pylist():
            if k >= start:
                yield t
            k += 1


def packed(it, seq):
    buf = []
    for text in it:
        buf.extend(tok(text).input_ids + [tok.eos_token_id])
        while len(buf) >= seq:
            yield torch.tensor(buf[:seq])
            buf = buf[seq:]


# ---------------- build the student ----------------
student = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32).to(dev)
assert student.config.tie_word_embeddings
t_b = time.time()
qe = QEmbedding(student.model.embed_tokens.weight.data, a.init)
student.model.embed_tokens = qe
student.lm_head = QHead(qe)
if a.init == "top86":
    for layer in student.model.layers:
        for par, nme in LINEARS:
            setattr(getattr(layer, par), nme, QLinear(getattr(getattr(layer, par), nme), "top86"))
else:
    cal_it = packed(docs(0), a.seq)
    CAL = torch.stack([next(cal_it) for _ in range(a.n_calib)]).to(dev)

    class Stop(Exception):
        pass

    def stop_hook(_m, _i, _o):
        raise Stop

    student.eval()
    for li, layer in enumerate(student.model.layers):
        hs, hooks = {}, []
        for par, nme in LINEARS:
            lin = getattr(getattr(layer, par), nme)
            n_in = lin.weight.shape[1]
            hs[(par, nme)] = torch.zeros(n_in, n_in, device=dev, dtype=torch.float64)

            def hook(_m, inp, _o, key=(par, nme), s=signs_for(n_in)):
                x = rot(inp[0].detach().float(), s).reshape(-1, inp[0].shape[-1]).double()
                hs[key] += x.T @ x

            hooks.append(lin.register_forward_hook(hook))
        hooks.append(layer.register_forward_hook(stop_hook))
        with torch.no_grad():
            for i in range(0, a.n_calib, 8):
                try:
                    student(CAL[i:i + 8])
                except Stop:
                    pass
        for hk in hooks:
            hk.remove()
        for par, nme in LINEARS:
            setattr(getattr(layer, par), nme, QLinear(getattr(getattr(layer, par), nme), a.init, hs[(par, nme)]))
        del hs
        if li % 7 == 0 or li == len(student.model.layers) - 1:
            say(f"  calibrated layer {li}  [{time.time() - t_b:.0f}s]")
    del CAL
    torch.cuda.empty_cache()                    # the float64 Hessian/solve blocks otherwise stay reserved through training
say(f"student built with init={a.init} in {time.time() - t_b:.0f}s")

for n, q in student.named_parameters():
    q.requires_grad_(bool((".tw." in n and not (n.endswith("scale") and a.freeze_scales)) or ("norm" in n and n.startswith("model."))))
trainable = [q for q in student.parameters() if q.requires_grad]
say(f"trainable params: {sum(q.numel() for q in trainable) / 1e6:.1f}M")
student.gradient_checkpointing_enable()
student.config.use_cache = False
QW = [m for m in student.modules() if isinstance(m, TernaryWeight)]


def drift():
    f, m, z = zip(*[w.drift() for w in QW])
    return dict(sign_flip=sum(f) / len(f), mask_change=sum(m) / len(m), zero_frac=sum(z) / len(z))


if a.load:
    ck = torch.load(a.load, map_location="cpu")
    miss, unexp = student.load_state_dict(ck["state"], strict=False)
    assert not unexp, unexp[:5]
    # lm_head.emb.* is the tied embedding under a second name; loading model.embed_tokens.* fills it
    assert not [k for k in miss if k.endswith(".tw.weight") and not k.startswith("lm_head.")], "checkpoint does not cover the ternary latents"
    say(f"loaded {len(ck['state'])} tensors from {a.load} (step {ck['step']})")
    del ck
results = [evaluate(student, "init", drift())]
if a.steps == 0:
    raise SystemExit(0)

# ---------------- train ----------------
gen = packed(docs(a.calib_docs), a.seq)
if a.mix_frac > 0:
    def convs():                                   # whole conversations, reshuffled every pass
        cs = [c + "<|im_end|>\n" for c in open(a.mix_text).read().split("<|im_end|>\n<|im_start|>system")]
        cs = [cs[0]] + ["<|im_start|>system" + c for c in cs[1:]]
        cs[-1] = cs[-1][: -len("<|im_end|>\n")]
        r, k = random.Random(a.seed), 0
        while True:
            r.shuffle(cs)
            say(f"mix pass {k}: {len(cs)} conversations")
            k += 1
            yield from cs

    mix_gen, mix_rng = packed(convs(), a.seq), random.Random(a.seed + 1)
    n_mix = [0, 0]


def batch():
    xs = []
    for _ in range(a.bs):
        m = a.mix_frac > 0 and mix_rng.random() < a.mix_frac
        xs.append(next(mix_gen if m else gen))
        if a.mix_frac > 0:
            n_mix[m] += 1
    return torch.stack(xs).to(dev)


opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
student.train()
t0 = time.time()
for step in range(1, a.steps + 1):
    opt.zero_grad(set_to_none=True)
    LAM[0] = min(1.0, step / (a.lam_ramp * a.steps)) if a.lam_ramp > 0 else 1.0
    tot = 0.0
    for _ in range(a.accum):
        x = batch()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lt = teacher(x).logits[:, :-1].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ls = student(x).logits[:, :-1].float()
        lps = F.log_softmax(ls, -1)
        loss = (F.softmax(lt, -1) * (F.log_softmax(lt, -1) - lps)).sum(-1).mean()
        if a.ce_weight > 0:
            loss = loss + a.ce_weight * (-lps.gather(-1, x[:, 1:, None]).mean())
        (loss / a.accum).backward()
        tot += loss.item() / a.accum
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    sched.step()
    if step % 25 == 0 or step == 1:
        say(f"step {step} loss={tot:.4f} lr={sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step"
            + (f" mix {n_mix[1]}/{sum(n_mix)}" if a.mix_frac > 0 else ""))
    if step % a.eval_every == 0 or step == a.steps:
        results.append(evaluate(student, f"step{step}", drift()))
        json.dump(dict(args=vars(a), results=results), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    if step % a.save_every == 0 or step == a.steps:
        sd = {n: q.detach().cpu() for n, q in student.named_parameters() if q.requires_grad}
        sd.update({n: b.cpu() for n, b in student.named_buffers() if n.endswith(".thr")})
        torch.save(dict(step=step, state=sd), os.path.join(a.out, "latent.pt.tmp"))
        os.replace(os.path.join(a.out, "latent.pt.tmp"), os.path.join(a.out, "latent.pt"))
        del sd
say(f"QAT17_DONE in {time.time() - t0:.0f}s")
