#!/usr/bin/env python3
"""Add a text corpus to an existing bonsai27_qat.py teacher cache, spread evenly through the run.

The text is tokenized whole and packed into contiguous --seq rows (as llama-imatrix reads a calibration file),
the teacher's top-K is computed exactly as bonsai27_qat.py --phase cache does, and the new rows are interleaved
with the reference cache's rows (reference rows keep their order and their stored teacher values). eval.pt is
the reference's own file, so every EVAL line is on the same rows as the reference run.

Gate (fatal): this script's teacher code must reproduce the reference cache's stored top-K on its first rows.

    bonsai27_ac_cache.py --base B --ref-cache K --text corpus.txt --out K_ac
"""
import argparse
import json
import os
import shutil

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--ref-cache", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
dev = a.device
AC = dict(device_type="cuda", dtype=torch.bfloat16)
meta = json.load(open(os.path.join(a.ref_cache, "cache.json")))
K, SH, NREF = meta["topk"], meta["shard"], meta["n_seq"]
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "cache.log"), "a")


def say(*m):
    s = " ".join(str(x) for x in m)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


say("args", json.dumps(vars(a)))
tok = AutoTokenizer.from_pretrained(a.base)
teacher = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)


@torch.no_grad()
def topk_of(x):                                  # identical to bonsai27_qat.py --phase cache
    with torch.autocast(**AC):
        lp = F.log_softmax(teacher(x).logits[:, :-1].float(), -1)
    v, i = lp.topk(K, -1)
    return i.to(torch.int32).cpu(), v.to(torch.float16).cpu()


# ---- gate: reproduce the reference cache's stored values
r0 = torch.load(os.path.join(a.ref_cache, "train_0000.pt"))
gi, gv = topk_of(r0["x"][:4].long().to(dev))
same_top1 = (gi[..., 0] == r0["idx"][:4, :, 0]).float().mean().item()
dv = (gv[..., 0].float() - r0["val"][:4, :, 0].float()).abs().max().item()
say(f"GATE reproduce reference cache: top-1 index agreement {same_top1:.4f}, max |dlogp top-1| {dv:.4f}")
assert same_top1 > 0.995 and dv < 0.05, "this teacher code does not reproduce the reference cache"

# ---- the new rows
ids = tok(open(a.text).read()).input_ids
n_new = len(ids) // a.seq
X = torch.tensor(ids[: n_new * a.seq]).view(n_new, a.seq)
say(f"text {os.path.basename(a.text)}: {len(ids)} tokens -> {n_new} rows of {a.seq} (dropped {len(ids) - n_new * a.seq})")
parts = [topk_of(X[i:i + 4].to(dev)) for i in range(0, n_new, 4)]
NI, NV = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
say(f"teacher top-{K} done; mean top-{K} mass {NV.float().exp().sum(-1).mean().item():.4f}")
teacher = None                                    # free the GPU before the merge
torch.cuda.empty_cache()

# ---- interleave: new row k lands at combined position floor((k + 0.5) * total / n_new)
total = NREF + n_new
is_new = torch.zeros(total, dtype=torch.bool)
is_new[((torch.arange(n_new) + 0.5) * total / n_new).long()] = True
assert int(is_new.sum()) == n_new
order, ki, kn = [], 0, 0                         # (source, index) per combined row
for p in range(total):
    if is_new[p]:
        order.append(("new", kn))
        kn += 1
    else:
        order.append(("ref", ki))
        ki += 1
cur = (-1, None)


def ref_row(i):
    global cur
    if i // SH != cur[0]:
        cur = (i // SH, torch.load(os.path.join(a.ref_cache, f"train_{i // SH:04d}.pt")))
    s = cur[1]
    j = i % SH
    return s["x"][j], s["idx"][j], s["val"][j]


for s0 in range(0, total, SH):
    rows = [ref_row(i) if src == "ref" else (X[i].to(torch.int32), NI[i], NV[i]) for src, i in order[s0:s0 + SH]]
    fn = os.path.join(a.out, f"train_{s0 // SH:04d}.pt")
    torch.save(dict(x=torch.stack([r[0] for r in rows]), idx=torch.stack([r[1] for r in rows]),
                    val=torch.stack([r[2] for r in rows])), fn + ".tmp")
    os.replace(fn + ".tmp", fn)
    say(f"wrote shard {s0 // SH}: {len(rows)} rows, {sum(1 for o in order[s0:s0 + SH] if o[0] == 'new')} new")
shutil.copy2(os.path.join(a.ref_cache, "eval.pt"), os.path.join(a.out, "eval.pt"))
json.dump(dict(args=vars(a), n_seq=total, shard=SH, topk=K, ref_cache=a.ref_cache, n_ref=NREF, n_new=n_new,
               text_tokens=len(ids), mix_rows=meta.get("mix_rows")), open(os.path.join(a.out, "cache.json"), "w"), indent=1)
say(f"AC_CACHE_DONE n_seq {total} = {NREF} reference + {n_new} new")
