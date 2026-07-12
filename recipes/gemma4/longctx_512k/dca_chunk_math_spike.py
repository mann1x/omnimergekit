#!/usr/bin/env python3
"""T87 DCA Stage-1 — hd=512 softcap-aware 5-variant LSE-merge parity gate (CHEAPEST GO/NO-GO).

Standalone torch (no vLLM). Mirrors the v0 DENSE DCA prefill decomposition recovered to
`ref_v0_dual_chunk_flash_attn.py` (`_dual_chunk_flash_attn_prefill_func` lines 1062-1129 +
`_merge_attn_outputs` 1244-1264):

  For each query chunk i (chunks of chunk_len = chunk_size - local_size):
    intra : K/V = current chunk      , causal=True , query variant q
    succ  : K/V = previous chunk      , causal=False, query variant q_succ
    inter : K/V = everything before   , causal=False, query variant q_inter
  Each -> (output, softmax_lse); the (<=3) per-chunk results are LSE-merged
  (weight_v = exp(lse_v - max)/Σ exp; out = Σ weight_v · out_v).

The intra∪succ∪inter key coverage is EXACTLY full causal attention, so in the
in-distribution-position regime (positions in-range; the same RoPE applied to all variants —
RoPE-remap correctness is Stage 2, NOT this gate) the merge must reproduce full attention to
numerical precision. The research risk this gate exists to retire: **Gemma 4 logit soft-capping
(softcap=50, tanh)**. Softcap is elementwise PRE-softmax, so per-chunk softcapped logits equal the
corresponding slice of the full softcapped logits -> the LSE merge over already-softcapped logits
MUST equal softcap-over-full-attention. If FlashAttention's returned softmax_lse is computed from
RAW (pre-softcap) logits, the merge breaks. We prove the gate has teeth by also running a
softcap-UNAWARE merge (lse from raw logits) and showing it diverges.

GATE: softcap-aware chunked DCA == full softcapped attention within tol at hd=512.

Usage:  python dca_chunk_math_spike.py [--device cuda|cpu] [--dtype fp32|bf16]
"""
import argparse
import math

import torch

SOFTCAP = 50.0          # Gemma 4 global-layer attn logit soft-cap
CHUNK_SIZE = 256        # tiny for the spike; real serve uses 262144
LOCAL_SIZE = 64         # real serve uses 1024
HEAD_DIM = 512          # Gemma 4 head_dim — the kernel-coverage risk
N_HEADS = 8
N_CHUNKS = 3            # -> exercises intra (i>=0), succ (i>=1), inter (i>=2)


def softcap(x, cap):
    return cap * torch.tanh(x / cap) if cap and cap > 0 else x


def full_attention(q, k, v, scale, cap):
    """Reference: full causal attention with softcap, fp32."""
    L = q.shape[0]
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale     # (H,L,L)
    s = softcap(s, cap)
    causal = torch.triu(torch.full((L, L), float("-inf"), device=q.device), diagonal=1)
    s = s + causal.unsqueeze(0)
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, v.float())                 # (L,H,D)


def variant_attn(qc, kv_k, kv_v, scale, cap, causal, softcap_aware):
    """One DCA variant attention -> (out (Lq,H,D), lse (H,Lq)).
    softcap_aware=True : lse computed from SOFTCAPPED logits (correct).
    softcap_aware=False: lse computed from RAW logits (the FA-without-softcap-lse failure)."""
    Lq, Lk = qc.shape[0], kv_k.shape[0]
    raw = torch.einsum("qhd,khd->hqk", qc.float(), kv_k.float()) * scale   # (H,Lq,Lk)
    capped = softcap(raw, cap)
    if causal:  # intra: square lower-triangular (query & key both start at chunk base)
        m = torch.triu(torch.full((Lq, Lk), float("-inf"), device=qc.device),
                       diagonal=1).unsqueeze(0)
        capped = capped + m   # mask BOTH the softmax logits AND the lse source
        raw = raw + m
    p = torch.softmax(capped, dim=-1)
    out = torch.einsum("hqk,khd->qhd", p, kv_v.float())                   # (Lq,H,D)
    # softcap-aware -> lse over the SAME (softcapped, masked) logits the softmax used.
    # softcap-unaware -> lse over raw pre-softcap logits (the FA-without-softcap-lse failure).
    lse = torch.logsumexp(capped if softcap_aware else raw, dim=-1)       # (H,Lq)
    return out, lse


def merge(results):
    """LSE merge mirroring _merge_attn_outputs: weight each variant by exp(lse-max)/Σexp."""
    if len(results) == 1:
        return results[0][0]
    outs = torch.stack([r[0] for r in results])      # (n,Lq,H,D)
    lses = torch.stack([r[1] for r in results]).float()  # (n,H,Lq)
    w = torch.softmax(lses, dim=0)                   # (n,H,Lq)  == exp(l-max)/Σexp
    w = w.permute(0, 2, 1).unsqueeze(-1)             # (n,Lq,H,1)
    return (outs * w).sum(dim=0)                     # (Lq,H,D)


def dca_chunked(q, k, v, scale, cap, softcap_aware):
    chunk_len = CHUNK_SIZE - LOCAL_SIZE
    L = q.shape[0]
    out_chunks = []
    i = 0
    begin = 0
    while begin < L:
        prev = (begin // chunk_len) * chunk_len
        end = min(prev + chunk_len, L)
        qc = q[begin:end]
        res = []
        # intra (causal, current chunk)
        res.append(variant_attn(qc, k[prev:end], v[prev:end], scale, cap, True, softcap_aware))
        # succ (full, previous chunk)
        if prev - chunk_len >= 0:
            res.append(variant_attn(qc, k[prev - chunk_len:prev], v[prev - chunk_len:prev],
                                    scale, cap, False, softcap_aware))
        # inter (full, everything before previous chunk)
        if prev - 2 * chunk_len >= 0:
            res.append(variant_attn(qc, k[:prev - chunk_len], v[:prev - chunk_len],
                                    scale, cap, False, softcap_aware))
        out_chunks.append(merge(res))
        begin = end
        i += 1
    return torch.cat(out_chunks, dim=0)


def report(name, ref, got):
    err = (got - ref).abs()
    denom = ref.abs().clamp_min(1e-6)
    rel = (err / denom).mean().item()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten().float(), ref.flatten().float(), dim=0).item()
    print(f"  {name:34s} max_abs={err.max().item():.3e}  mean_rel={rel:.3e}  cos={cos:.8f}")
    return err.max().item(), cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logit-gain", type=float, default=1.0,
                    help="scale q so pre-softcap logits reach into the tanh nonlinearity; "
                         "default hd=512 gives logit std~1.1 (softcap near-linear). Use ~30 to "
                         "drive logits to +/-50+ (deep tanh saturation) — the regime softcap exists for.")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = a.device
    dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    chunk_len = CHUNK_SIZE - LOCAL_SIZE
    L = N_CHUNKS * chunk_len
    scale = 1.0 / math.sqrt(HEAD_DIM)
    # DCA log scaling factor (line 124 of v0): 0.1*ln(L/orig)+1, clamped >=1. orig=L here => 1.0,
    # but exercise a non-trivial value to confirm it commutes through the split.
    orig = chunk_len
    scaling_factor = max(1.0, 0.1 * math.log(L / orig) + 1.0)
    eff_scale = scale * scaling_factor

    q = torch.randn(L, N_HEADS, HEAD_DIM, device=dev, dtype=dt) * a.logit_gain
    k = torch.randn(L, N_HEADS, HEAD_DIM, device=dev, dtype=dt)
    v = torch.randn(L, N_HEADS, HEAD_DIM, device=dev, dtype=dt)

    # diagnostic: where do the pre-softcap logits actually land vs the cap?
    with torch.no_grad():
        _l = (torch.einsum("qhd,khd->hqk", q.float(), k.float()) * eff_scale)
        logit_std, logit_absmax = _l.std().item(), _l.abs().max().item()

    print(f"=== DCA Stage-1 parity gate ===  device={dev} dtype={a.dtype} "
          f"hd={HEAD_DIM} heads={N_HEADS} L={L} chunk_len={chunk_len} "
          f"softcap={SOFTCAP} scaling_factor={scaling_factor:.4f} logit_gain={a.logit_gain:g}")
    print(f"    pre-softcap logits: std={logit_std:.2f} |max|={logit_absmax:.2f} "
          f"(softcap engages meaningfully once |logit| ~ {SOFTCAP:g})")

    tol_abs = 1e-3 if a.dtype == "fp32" else 5e-2
    cos_gate = 0.99999 if a.dtype == "fp32" else 0.999

    ok = True
    for cap in (0.0, SOFTCAP):
        tag = f"softcap={cap}"
        ref = full_attention(q, k, v, eff_scale, cap)
        print(f"[{tag}]")
        # correct: softcap-aware LSE
        ea, ca = report("DCA softcap-AWARE merge", ref, dca_chunked(q, k, v, eff_scale, cap, True))
        passed = (ea <= tol_abs) and (ca >= cos_gate)
        print(f"      -> {'PASS' if passed else 'FAIL'} (tol_abs={tol_abs}, cos>={cos_gate})")
        ok = ok and passed
        if cap > 0:  # teeth: show the softcap-UNAWARE merge diverges
            eb, cb = report("DCA softcap-UNAWARE merge", ref,
                            dca_chunked(q, k, v, eff_scale, cap, False))
            teeth = (eb > tol_abs) or (cb < cos_gate)
            print(f"      -> unaware merge {'DIVERGES (gate has teeth)' if teeth else 'ALSO MATCHES (no teeth!)'}")

    print()
    print(f"GATE VERDICT: {'GO  — hd=512 softcap-aware chunked LSE merge reproduces full attention' if ok else 'NO-GO — parity failed; reassess before any vLLM plumbing'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
