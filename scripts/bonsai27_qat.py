#!/usr/bin/env python3
"""Ternary recovery training in the Bonsai format that fits a 27B model on one 96 GB GPU.

bonsai4b_qat.py keeps float32 latents, gradients and two Adam moments (16 bytes per weight) next to the teacher. At
27B that is 430 GB. Here:
  * every ternary tensor is one float16 latent on the GPU (2 bytes per weight, stochastic rounding on write);
  * the served weight unfold(top86(latent)) is built when a layer runs, and the latent is updated the moment its
    gradient exists during the backward pass, so no gradient of the whole model is ever held;
  * the update is Adafactor without momentum: factored row/column second moments (a few KB per tensor), update
    clipped to unit RMS, straight-through to the latent;
  * the teacher is not on the GPU: --phase cache runs it once over the exact training stream and the eval rows and
    stores tokens plus top-K teacher log-probs; the loss is KL over those K classes plus one "rest" class (+ CE).
The small full-precision tensors (norms, in_proj_a/b, conv1d, A_log, dt_bias) train with ordinary AdamW.
MTP head and vision tower are never loaded.

  bonsai27_qat.py --phase cache --base B --cache C --train-parquet P [--mix-text ..] --eval-text wiki --eval-text-b a,b
  bonsai27_qat.py --phase train --base B --cache C --out O [--basis prism --prism-gguf F] [--eval-teacher online]
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

from bonsai_tern_lib import EMBED, FP_TRAIN, HEAD, QNAMES, Basis, sr16, ternary_keys, top86, trits

ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=["cache", "train"], required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--cache", required=True, help="directory of the teacher cache (written by --phase cache)")
ap.add_argument("--train-parquet", default="")
ap.add_argument("--eval-text", default="")
ap.add_argument("--eval-text-b", default="", help="extra eval texts, comma-separated (.txt or conversation .jsonl); suffixes _b, _c, ...")
ap.add_argument("--mix-text", default="", help="comma list of path[=weight]: conversation .jsonl files mixed into the stream")
ap.add_argument("--mix-frac", type=float, default=0.0)
ap.add_argument("--calib-docs", type=int, default=2000, help="leading parquet rows skipped (kept for calibration use)")
ap.add_argument("--n-seq", type=int, default=16000, help="cache phase: training sequences to prepare")
ap.add_argument("--n-eval", type=int, default=32)
ap.add_argument("--topk", type=int, default=64)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--basis", choices=["seeded", "prism"], default="seeded")
ap.add_argument("--prism-gguf", default="", help="Bonsai 2 GGUF giving signs, block and tensor list (--basis prism)")
ap.add_argument("--out", default="")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--stop-at", type=int, default=0, help="end the run after this step, schedule unchanged (probe runs)")
ap.add_argument("--bs", type=int, default=8, help="sequences per update")
ap.add_argument("--lr", type=float, default=6e-5)
ap.add_argument("--fp-lr", type=float, default=6e-5)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--beta2", type=float, default=0.999)
ap.add_argument("--clip", type=float, default=1.0, help="largest RMS of one tensor's normalized update")
ap.add_argument("--ce-weight", type=float, default=0.1)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--save-every", type=int, default=1000)
ap.add_argument("--load", default="")
ap.add_argument("--eval-teacher", choices=["cache", "online"], default="cache", help="online also loads the teacher and reports the full KL")
ap.add_argument("--drift-sample", type=int, default=12, help="ternary tensors whose trit drift is tracked")
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = a.device
log = None


def say(*x):
    s = " ".join(str(v) for v in x)
    print(s, flush=True)
    if log:
        log.write(s + "\n")
        log.flush()


tok = AutoTokenizer.from_pretrained(a.base)
AC = dict(device_type="cuda", dtype=torch.bfloat16)

# =====================================================================================================================
# phase cache: the training stream and the eval rows, with top-K teacher log-probs
# =====================================================================================================================
if a.phase == "cache":
    os.makedirs(a.cache, exist_ok=True)
    log = open(os.path.join(a.cache, "cache.log"), "a")
    say("args", json.dumps(vars(a)))
    teacher = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)

    @torch.no_grad()
    def topk_of(x):                              # x [n, seq] on dev -> idx int32 [n, seq-1, K], val float16, teacher nll sum
        with torch.autocast(**AC):
            lp = F.log_softmax(teacher(x).logits[:, :-1].float(), -1)
        v, i = lp.topk(a.topk, -1)
        nll = -lp.gather(-1, x[:, 1:, None]).sum().item()
        return i.to(torch.int32).cpu(), v.to(torch.float16).cpu(), nll

    def eval_rows(path):                         # same row choice as bonsai4b_qat.py
        if path.endswith(".jsonl"):
            cs = [tok(json.loads(line)["text"]).input_ids for line in open(path)]
            random.Random(0).shuffle(cs)
            rows, buf = [], []
            for k, c in enumerate(cs):
                if len(c) >= 1024:
                    rows.append(c[:1024] if k % 2 == 0 else c[-1024:])
                else:
                    buf.extend(c)
                    while len(buf) >= 1024:
                        rows.append(buf[:1024])
                        buf = buf[1024:]
            assert len(rows) >= a.n_eval, f"{path}: {len(rows)} rows < {a.n_eval}"
            return torch.tensor(rows[: a.n_eval])
        ids = tok(open(path).read(), return_tensors="pt").input_ids[0]
        assert len(ids) >= a.n_eval * 1024, f"{path}: {len(ids)} tokens < {a.n_eval} x 1024"
        return torch.stack([ids[i * 1024:(i + 1) * 1024] for i in range(a.n_eval)])

    ev = []
    for k, p in enumerate([a.eval_text] + [q for q in a.eval_text_b.split(",") if q]):
        x = eval_rows(p)
        parts = [topk_of(x[i:i + 4].to(dev)) for i in range(0, len(x), 4)]
        ppl = math.exp(sum(q[2] for q in parts) / (len(x) * 1023))
        ev.append(dict(sfx="" if k == 0 else "_" + "bcdefg"[k - 1], path=p, x=x.to(torch.int32), idx=torch.cat([q[0] for q in parts]),
                       val=torch.cat([q[1] for q in parts]), teacher_ppl=ppl))
        say(f"eval: {len(x)} x 1024 tokens of {os.path.basename(p)}; teacher ppl {ppl:.3f}; top-{a.topk} mass "
            f"{ev[-1]['val'].float().exp().sum(-1).mean().item():.4f}")
    torch.save(ev, os.path.join(a.cache, "eval.pt"))

    def docs(start):
        pf = pq.ParquetFile(a.train_parquet)
        k = 0
        for b in pf.iter_batches(batch_size=256, columns=["text"]):
            for t in b.column("text").to_pylist():
                if k >= start:
                    yield t
                k += 1

    def packed(it, seq, eos=True):
        buf = []
        for text in it:
            buf.extend(tok(text).input_ids + ([tok.eos_token_id] if eos else []))
            while len(buf) >= seq:
                yield torch.tensor(buf[:seq])
                buf = buf[seq:]

    gen = packed(docs(a.calib_docs), a.seq)
    MIX, n_src = [], []
    if a.mix_frac > 0:
        def convs(path):                          # whole conversations, reshuffled every pass
            cs = [json.loads(line)["text"] for line in open(path)]
            r, k = random.Random(a.seed), 0
            while True:
                r.shuffle(cs)
                say(f"mix pass {k} of {os.path.basename(path)}: {len(cs)} conversations")
                k += 1
                yield from cs

        MIX = [(q.split("=")[0], float(q.split("=")[1]) if "=" in q else 1.0) for q in a.mix_text.split(",")]
        mix_gens = [packed(convs(q), a.seq, eos=False) for q, _ in MIX]
        mix_rng, n_src = random.Random(a.seed + 1), [0] * len(MIX)

    def next_seq():                               # one sequence, in exactly the order bonsai4b_qat.py draws them
        if a.mix_frac > 0 and mix_rng.random() < a.mix_frac:
            k = mix_rng.choices(range(len(MIX)), [w for _, w in MIX])[0]
            n_src[k] += 1
            return next(mix_gens[k])
        return next(gen)

    SH, t0 = 1000, time.time()
    for s0 in range(0, a.n_seq, SH):
        fn = os.path.join(a.cache, f"train_{s0 // SH:04d}.pt")
        x = torch.stack([next_seq() for _ in range(min(SH, a.n_seq - s0))])     # always drawn, so a resume stays aligned
        if os.path.exists(fn):
            continue
        parts = [topk_of(x[i:i + 4].to(dev)) for i in range(0, len(x), 4)]
        torch.save(dict(x=x.to(torch.int32), idx=torch.cat([q[0] for q in parts]), val=torch.cat([q[1] for q in parts])), fn + ".tmp")
        os.replace(fn + ".tmp", fn)
        say(f"cached {s0 + len(x)}/{a.n_seq} sequences [{time.time() - t0:.0f}s] mix rows {n_src}")
    json.dump(dict(args=vars(a), n_seq=a.n_seq, shard=SH, topk=a.topk, mix_rows=n_src), open(os.path.join(a.cache, "cache.json"), "w"), indent=1)
    say("CACHE_DONE")
    raise SystemExit(0)

# =====================================================================================================================
# phase train
# =====================================================================================================================
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "log.txt"), "a")
say("args", json.dumps(vars(a)))
meta = json.load(open(os.path.join(a.cache, "cache.json")))
K, SH = meta["topk"], meta["shard"]
need = (a.stop_at or a.steps) * a.bs
assert meta["n_seq"] >= need, f"cache holds {meta['n_seq']} sequences, the run needs {need}"
EV = torch.load(os.path.join(a.cache, "eval.pt"))
import transformers.models.qwen3_5.modeling_qwen3_5 as _q35   # noqa: E402
say(f"linear-attention kernel: {'fla (triton)' if _q35.chunk_gated_delta_rule is not None else 'torch fallback'}")

student = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev)
basis = Basis(a.basis, dev, a.prism_gguf)
KEYS = ternary_keys(student, basis)
TIED = bool(student.config.tie_word_embeddings)
LR = [0.0]                                        # current learning rate of the ternary latents
LOSSLESS = [False]                                # True: bypass the quantizer (rotation gate)
DUMMY = torch.zeros((), device=dev, requires_grad=True)
STEP = [0]


class Store:
    """One ternary tensor: float16 latent in the folded basis + Adafactor state."""

    def __init__(self, key, w):                   # w: the model's own weight (bf16, on dev); replaced by its folded latent
        self.key, self.shape = key, tuple(w.shape)
        self.lat = torch.empty(self.shape, dtype=torch.float16, device=dev)
        for r0, r1 in self.chunks():
            self.lat[r0:r1] = self._fold(w[r0:r1].float(), r0, r1).to(torch.float16)
        self.r = torch.zeros(self.shape[0], device=dev)
        self.c = torch.zeros(self.shape[1], device=dev)
        self.t = 0
        self.vs, self.ts = 0.0, 0                 # scalar second moment of sparse (embedding-row) updates
        torch.cuda.synchronize(dev)               # bug-766: queued unsynchronized with expandable_segments this build faults

    def chunks(self, lim=96_000_000):
        if self.shape[0] * self.shape[1] <= lim or self.key not in (EMBED, HEAD):
            return [(0, self.shape[0])]
        n = max(1, lim // self.shape[1])
        return [(r, min(r + n, self.shape[0])) for r in range(0, self.shape[0], n)]

    def _fold(self, w, r0=0, r1=None):
        return basis.fold(self.key, w)

    def weight(self, rows=None):
        """Served weight (bf16) of the whole tensor, or of the given rows of a row-wise tensor."""
        if rows is not None:
            w = self.lat[rows].float()
            return basis.unfold(self.key, w if LOSSLESS[0] else top86(w)).to(torch.bfloat16)
        out = torch.empty(self.shape, dtype=torch.bfloat16, device=dev)
        for r0, r1 in self.chunks():
            w = self.lat[r0:r1].float()
            out[r0:r1] = basis.unfold(self.key, w if LOSSLESS[0] else top86(w)).to(torch.bfloat16)
        return out

    @torch.no_grad()
    def update(self, grad):
        """grad: gradient of the served weight (model basis). Straight-through Adafactor step on the latent."""
        self.t += 1
        b2, cs = a.beta2, self.chunks()
        gs = [basis.fold(self.key, grad[r0:r1].float()) for r0, r1 in cs] if len(cs) == 1 else None
        csum = torch.zeros_like(self.c)
        for k, (r0, r1) in enumerate(cs):         # pass 1: second-moment statistics
            g2 = (gs[k] if gs else basis.fold(self.key, grad[r0:r1].float())).square()
            self.r[r0:r1].mul_(b2).add_(g2.mean(1), alpha=1 - b2)
            csum += g2.sum(0)
        self.c.mul_(b2).add_(csum / self.shape[0], alpha=1 - b2)
        bc = 1 - b2 ** self.t
        rh, ch = self.r / bc, self.c / bc
        for k, (r0, r1) in enumerate(cs):         # pass 2: normalized update, clipped to unit RMS (per chunk for the big tables)
            g = gs[k] if gs else basis.fold(self.key, grad[r0:r1].float())
            u = g / (rh[r0:r1, None] * ch[None, :] / rh.mean().clamp(min=1e-30)).sqrt().clamp(min=1e-15)
            u /= max(1.0, u.square().mean().sqrt().item() / a.clip)
            self.lat[r0:r1] = sr16(self.lat[r0:r1].float() - LR[0] * u)

    @torch.no_grad()
    def update_rows(self, rows, grad):
        """Sparse step for looked-up embedding rows; rows unique."""
        g = basis.fold(self.key, grad.float())
        self.ts += 1
        self.vs = a.beta2 * self.vs + (1 - a.beta2) * g.square().mean().item()
        u = (g / math.sqrt(self.vs / (1 - a.beta2 ** self.ts) + 1e-30)).clamp(-3, 3)
        self.lat[rows] = sr16(self.lat[rows].float() - LR[0] * u)


class QuantW(torch.autograd.Function):
    """Served weight of a store. The backward pass applies the latent update at once and passes nothing on."""

    @staticmethod
    def forward(ctx, dummy, sid):
        ctx.sid = sid
        return STORES[sid].weight()

    @staticmethod
    def backward(ctx, gw):
        STORES[ctx.sid].update(gw)
        return torch.zeros_like(DUMMY), None


class QuantRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dummy, sid, rows):
        ctx.sid, ctx.rows = sid, rows
        return STORES[sid].weight(rows)

    @staticmethod
    def backward(ctx, gw):
        STORES[ctx.sid].update_rows(ctx.rows, gw)
        return torch.zeros_like(DUMMY), None, None


class QLinear(nn.Module):
    def __init__(self, sid, bias):
        super().__init__()
        self.sid, self.bias = sid, bias

    def forward(self, x):
        return F.linear(x, QuantW.apply(DUMMY, self.sid).to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class QEmbedding(nn.Module):
    def __init__(self, sid):
        super().__init__()
        self.sid = sid

    def forward(self, ids):
        u, inv = torch.unique(ids, return_inverse=True)
        return QuantRows.apply(DUMMY, self.sid, u)[inv]


# ---------------- build the student: every ternary weight becomes a float16 latent ----------------
STORES, t_b = {}, time.time()
emb = student.model.embed_tokens
STORES[EMBED] = Store(EMBED, emb.weight.data)
student.model.embed_tokens = QEmbedding(EMBED)
del emb
for i, layer in enumerate(student.model.layers):
    for par, ns in QNAMES.items():
        if getattr(layer, par, None) is None:
            continue
        for n in ns:
            lin, key = getattr(getattr(layer, par), n), f"model.layers.{i}.{par}.{n}.weight"
            STORES[key] = Store(key, lin.weight.data)
            setattr(getattr(layer, par), n, QLinear(key, lin.bias))
            del lin
    torch.cuda.empty_cache()                      # bug-769: free each layer's originals as they are folded
HEAD_SID = EMBED if TIED else HEAD
if not TIED:
    STORES[HEAD] = Store(HEAD, student.lm_head.weight.data)
student.lm_head = nn.Identity()                   # logits are built per sequence in loss_of(); the body returns hidden states
assert set(STORES) == set(KEYS), sorted(set(STORES) ^ set(KEYS))[:6]
torch.cuda.empty_cache()
fp_params = []
for n, q in student.named_parameters():
    tr = ("norm" in n and n.startswith("model.")) or any(k in n for k in FP_TRAIN)
    q.requires_grad_(bool(tr))
    if tr:
        q.data = q.data.float()
        fp_params.append(q)
n_tern = sum(s.lat.numel() for s in STORES.values())
say(f"student built in {time.time() - t_b:.0f}s: {len(STORES)} ternary tensors, {n_tern / 1e9:.3f}B weights as float16 latents "
    f"({'tied' if TIED else 'untied'} head, basis {a.basis}); full-precision trained {sum(q.numel() for q in fp_params) / 1e6:.2f}M; "
    f"GPU {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB")
student.gradient_checkpointing_enable()
student.config.use_cache = False
DR = [k for k in KEYS if k not in (EMBED, HEAD)]
DR = [DR[i] for i in sorted(set(int(j * (len(DR) - 1) / max(1, a.drift_sample - 1)) for j in range(a.drift_sample)))]
T0 = {k: trits(STORES[k].lat.float()).cpu() for k in DR}


def drift():
    f = m = z = 0.0
    for k in DR:
        t, t0 = trits(STORES[k].lat.float()), T0[k].to(dev)
        nz = (t != 0) & (t0 != 0)
        f += (((t != t0) & nz).float().sum() / nz.float().sum()).item()
        m += ((t == 0) != (t0 == 0)).float().mean().item()
        z += (t == 0).float().mean().item()
    return dict(sign_flip=f / len(DR), mask_change=m / len(DR), zero_frac=z / len(DR))


def hidden(x):
    with torch.autocast(**AC):
        return student.model(input_ids=x).last_hidden_state


def topk_terms(lp, idx, val, x_next):
    """lp [n, V] student log-probs. KL over the K cached classes + one rest class, CE, top-1 agreement (per-row sums)."""
    tl = val.float()
    pt = tl.exp()
    ls = lp.gather(-1, idx.long())
    pr = (1 - pt.sum(-1)).clamp(min=1e-8)
    sr = (-torch.expm1(torch.logsumexp(ls, -1).clamp(max=-1e-7))).clamp(min=1e-12)      # student mass outside the K classes
    kl = (pt * (tl - ls)).sum(-1) + pr * (pr.log() - sr.log())
    ce = -lp.gather(-1, x_next[:, None])[:, 0]
    return kl, ce, (lp.argmax(-1) == idx[:, 0].long()).float()


teacher = None
if a.eval_teacher == "online":
    teacher = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)


@torch.no_grad()
def evaluate(tag, extra=None):
    student.eval()
    res = dict(tag=tag)
    wh = STORES[HEAD_SID].weight()
    for e in EV:
        nll = klk = top1 = klf = 0.0
        for i0 in range(0, len(e["x"]), 4):          # 4 rows per forward: every forward rebuilds all served weights
            hb = hidden(e["x"][i0:i0 + 4].long().to(dev))
            for i in range(i0, i0 + len(hb)):
                x = e["x"][i].long().to(dev)
                with torch.autocast(**AC):
                    lg = F.linear(hb[i - i0, :-1], wh).float()
                lp = F.log_softmax(lg, -1)
                kl, ce, t1 = topk_terms(lp, e["idx"][i].to(dev), e["val"][i].to(dev), x[1:])
                nll, klk, top1 = nll + ce.mean().item(), klk + kl.mean().item(), top1 + t1.mean().item()
                if teacher is not None:
                    with torch.autocast(**AC):
                        lt = F.log_softmax(teacher(x[None]).logits[0, :-1].float(), -1)
                    klf += (lt.exp() * (lt - lp)).sum(-1).mean().item()
            del hb
        n = len(e["x"])
        res.update({"ppl" + e["sfx"]: math.exp(nll / n), "klk" + e["sfx"]: klk / n, "top1" + e["sfx"]: top1 / n})
        if teacher is not None:
            res["kl" + e["sfx"]] = klf / n
    del wh
    res.update(extra or {})
    say("EVAL", json.dumps(res))
    student.train()
    return res


def save(step):
    sdv = dict(step=step, basis=a.basis, latents={k: s.lat.cpu() for k, s in STORES.items()},
               opt={k: (s.r.cpu(), s.c.cpu(), s.t, s.vs, s.ts) for k, s in STORES.items()},
               fp={n: q.detach().cpu() for n, q in student.named_parameters() if q.requires_grad})
    torch.save(sdv, os.path.join(a.out, "latent.pt.tmp"))
    os.replace(os.path.join(a.out, "latent.pt.tmp"), os.path.join(a.out, "latent.pt"))
    del sdv


if a.load:
    ck = torch.load(a.load, map_location="cpu")
    assert ck["basis"] == a.basis and set(ck["latents"]) == set(STORES)
    for k, s in STORES.items():
        s.lat.copy_(ck["latents"][k])
        r, c, s.t, s.vs, s.ts = ck["opt"][k]
        s.r.copy_(r)
        s.c.copy_(c)
    named = dict(student.named_parameters())
    for n, v in ck["fp"].items():
        named[n].data.copy_(v)
    say(f"loaded {a.load} (step {ck['step']})")
    del ck
else:                                             # rotation gate: with the quantizer bypassed the student must BE the teacher
    LOSSLESS[0] = True
    g = evaluate("lossless")
    LOSSLESS[0] = False
    say(f"LOSSLESS_GATE klk = {g['klk']:.6f}")
    assert g["klk"] < 0.01, "folded, unquantized student does not reproduce the teacher"
results = [evaluate("init", drift())]
if a.steps == 0:
    raise SystemExit(0)

# ---------------- train ----------------
opt = torch.optim.AdamW(fp_params, lr=a.fp_lr, betas=(0.9, 0.95), weight_decay=0.0)
sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))   # noqa: E731
shard = (-1, None)
student.train()
t0, last = time.time(), a.stop_at or a.steps
for step in range(1, last + 1):
    STEP[0] = step
    LR[0] = a.lr * sched(step - 1)
    for gq in opt.param_groups:
        gq["lr"] = a.fp_lr * sched(step - 1)
    opt.zero_grad(set_to_none=True)
    s0 = (step - 1) * a.bs
    if s0 // SH != shard[0]:
        shard = (s0 // SH, torch.load(os.path.join(a.cache, f"train_{s0 // SH:04d}.pt")))
    assert s0 % SH + a.bs <= SH, "--bs must divide the cache shard size"
    j = s0 % SH
    x = shard[1]["x"][j:j + a.bs].long().to(dev)
    idx, val = shard[1]["idx"][j:j + a.bs].to(dev), shard[1]["val"][j:j + a.bs].to(dev)
    h = hidden(x)                                 # [bs, seq, d], graph through the body
    hd = h.detach().requires_grad_()
    wh = STORES[HEAD_SID].weight().requires_grad_()
    tot = 0.0
    for i in range(a.bs):                         # head + loss one sequence at a time: the logits never exist for the whole batch
        with torch.autocast(**AC):
            lg = F.linear(hd[i, :-1], wh).float()
        kl, ce, _ = topk_terms(F.log_softmax(lg, -1), idx[i], val[i], x[i, 1:])
        loss = (kl.mean() + a.ce_weight * ce.mean()) / a.bs
        loss.backward()
        tot += loss.item()
    STORES[HEAD_SID].update(wh.grad)
    gh = hd.grad
    del wh, lg, loss, kl, ce, hd
    h.backward(gh)                                # body: every QLinear updates its own latent as its gradient appears
    del h, gh
    torch.nn.utils.clip_grad_norm_(fp_params, 1.0)
    opt.step()
    if step % 25 == 0 or step <= 3:
        say(f"step {step} loss={tot:.4f} lr={LR[0]:.2e} {(time.time() - t0) / step:.2f}s/step {torch.cuda.max_memory_allocated(dev) / 2**30:.1f}GiB")
    if step % a.eval_every == 0 or step == last:
        opt.zero_grad(set_to_none=True)
        results.append(evaluate(f"step{step}", drift()))
        json.dump(dict(args=vars(a), results=results), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    if step % a.save_every == 0 or step == last:
        save(step)
say(f"QAT27_DONE in {time.time() - t0:.0f}s")
