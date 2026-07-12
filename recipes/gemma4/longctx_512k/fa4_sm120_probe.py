#!/usr/bin/env python3
"""T87 path-B probe — does vLLM's FA4 (CuTe-DSL) kernel actually RUN + compute correctly at
head_dim=512 + softcap=50 + return_lse on sm_120 (RTX PRO 6000 Blackwell)?

The vendored `_is_fa4_supported()` allowlist is {cc 9.x,10.x,11.x} and excludes sm_120 (cc 12.0),
so `supports_head_size(512)` returns False on bs2 — but that's a *Python gate*, not necessarily a
kernel limit. We bypass the gate by calling `flash_attn_varlen_func(..., fa_version=4)` directly
(the dispatch just branches to `cute.interface._flash_attn_fwd`, no support assertion inside).

If the CuTe kernel JIT-compiles + runs + matches a softcapped-SDPA reference on sm_120, then the
fork-FA DCA path is viable on bs2 and the ONLY fix needed is adding cc-family 120 to the allowlist
(a one-liner we upstream). If it errors/diverges, we need real CuTe/cutlass kernel work for sm_120.

Run in the bs2 `vllm` env on ONE GPU:  CUDA_VISIBLE_DEVICES=0 python fa4_sm120_probe.py
"""
import math
import torch


def softcap(x, cap):
    return cap * torch.tanh(x / cap) if cap and cap > 0 else x


def ref_attn(q, k, v, scale, cap, causal):
    # q,k,v: (S, H, D) single sequence. fp32 reference with softcap.
    S, H, D = q.shape
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    s = softcap(s, cap)
    if causal:
        m = torch.triu(torch.full((S, S), float("-inf"), device=q.device), diagonal=1)
        s = s + m.unsqueeze(0)
    p = torch.softmax(s, dim=-1)
    out = torch.einsum("hqk,khd->qhd", p, v.float())
    lse = torch.logsumexp(s, dim=-1)  # (H, S) natural-log
    return out, lse


def main():
    assert torch.cuda.is_available(), "no CUDA"
    dev = "cuda"
    cc = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}  cc={cc}")

    # report FA support flags (informational — we bypass the gate below)
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            is_fa_version_supported, fa_version_unsupported_reason, FA4_AVAILABLE)
        print(f"FA4_AVAILABLE={FA4_AVAILABLE}  is_fa_version_supported(4)="
              f"{is_fa_version_supported(4)}  reason={fa_version_unsupported_reason(4)}")
    except Exception as e:
        print("flag import err:", e)

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    H = 8
    # CAP=0 isolates the CORE sm_120 kernel (no score_mod); CAP=50 needs apply_score_mod (softcap).
    for CAP in (0.0, 50.0):
      for D in (256, 512):         # 256 = sanity; 512 = the real question (Gemma 4 head_dim)
        for S in (320,):           # > a chunk; single varlen sequence
            scale = 1.0 / math.sqrt(D)
            torch.manual_seed(0)
            q = torch.randn(S, H, D, device=dev, dtype=torch.bfloat16)
            k = torch.randn(S, H, D, device=dev, dtype=torch.bfloat16)
            v = torch.randn(S, H, D, device=dev, dtype=torch.bfloat16)
            cu = torch.tensor([0, S], device=dev, dtype=torch.int32)
            tag = f"D={D} S={S} causal softcap={CAP}"
            try:
                out, lse = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=S, max_seqlen_k=S,
                    softmax_scale=scale, causal=True, softcap=CAP,
                    return_softmax_lse=True, fa_version=4,
                    num_splits=1,   # force non-split path (sm_120 FA4 guards: no split-KV / paged / block-sparse)
                )
                rout, rlse = ref_attn(q, k, v, scale, CAP, causal=True)
                oerr = (out.float() - rout).abs().max().item()
                ocos = torch.nn.functional.cosine_similarity(
                    out.float().flatten(), rout.flatten(), dim=0).item()
                # FA lse shape is typically (H, S); align to ref (H,S)
                lf = lse.float()
                lerr = (lf - rlse).abs().max().item() if lf.shape == rlse.shape else float("nan")
                ok = oerr < 5e-2 and ocos > 0.999
                print(f"[{tag}] FA4 RAN ✓  out_max_abs={oerr:.3e} cos={ocos:.6f} "
                      f"lse_shape={tuple(lse.shape)} lse_max_abs={lerr:.3e}  -> "
                      f"{'CORRECT' if ok else 'DIVERGES'}")
            except Exception as e:
                import traceback
                print(f"[{tag}] FA4 FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()

    print("\nVERDICT: if D=512 RAN + CORRECT on sm_120 -> fork-FA DCA path viable on bs2; "
          "fix = add cc-family 120 to _is_fa4_supported allowlist (upstream one-liner). "
          "If FAILED -> real CuTe/cutlass sm_120 kernel work needed.")


if __name__ == "__main__":
    main()
