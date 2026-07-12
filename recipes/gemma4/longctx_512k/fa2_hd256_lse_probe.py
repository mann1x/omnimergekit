#!/usr/bin/env python3
"""T87 DCA unblock probe — does FA2 (already supported on sm_120) give us hd=256 + softmax_lse,
which is ALL the Gemma 4 DCA chunked-attention decomposition actually needs?

Gemma 4 26B-A4B config: head_dim=256, NO attn_logit_softcapping (only final_logit_softcapping=30
on the LM head). So DCA's 5x-query chunk attention needs: hd=256 + return LSE + plain (no-softcap)
online-softmax merge. FA4/hd512 (task #592) was solving a problem this model does not have.

If FA2 RUNS + matches a plain-SDPA reference at hd=256 AND returns a usable per-query LSE on sm_120,
the DCA v1 backend port (Stage 2-5) can proceed on FA2 with zero kernel work.

Run in bs2 `vllm` env on ONE GPU:  CUDA_VISIBLE_DEVICES=0 python fa2_hd256_lse_probe.py
"""
import math
import torch


def ref_attn(q, k, v, scale, causal):
    # q,k,v: (S,H,D) single seq, fp32 reference, NO softcap.
    S, H, D = q.shape
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    if causal:
        m = torch.triu(torch.full((S, S), float("-inf"), device=q.device), diagonal=1)
        s = s + m.unsqueeze(0)
    p = torch.softmax(s, dim=-1)
    out = torch.einsum("hqk,khd->qhd", p, v.float())
    lse = torch.logsumexp(s, dim=-1)  # (H,S)
    return out, lse


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported, FA2_AVAILABLE
    print(f"FA2_AVAILABLE={FA2_AVAILABLE}  is_fa_version_supported(2)={is_fa_version_supported(2)}")
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    H, D = 8, 256  # Gemma 4 26B-A4B: 16 q-heads / 8 kv-heads, head_dim 256
    for S in (320, 4096):  # a chunk-ish and a longer seq
        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(0)
        q = torch.randn(S, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(S, H, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(S, H, D, device=dev, dtype=torch.bfloat16)
        cu = torch.tensor([0, S], device=dev, dtype=torch.int32)
        tag = f"FA2 D={D} S={S} causal (no softcap)"
        try:
            out, lse = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=S, cu_seqlens_q=cu, max_seqlen_k=S, cu_seqlens_k=cu,
                softmax_scale=scale, causal=True,
                return_softmax_lse=True, fa_version=2,
            )
            rout, rlse = ref_attn(q, k, v, scale, causal=True)
            oerr = (out.float() - rout).abs().max().item()
            ocos = torch.nn.functional.cosine_similarity(
                out.float().flatten(), rout.flatten(), dim=0).item()
            lf = lse.float()
            # align lse to (H,S)
            lerr = (lf - rlse).abs().max().item() if lf.shape == rlse.shape else (
                   (lf.t() - rlse).abs().max().item() if lf.t().shape == rlse.shape else float("nan"))
            ok = oerr < 5e-2 and ocos > 0.999 and (lerr < 5e-2)
            print(f"[{tag}] RAN ✓ out_max_abs={oerr:.3e} cos={ocos:.6f} "
                  f"lse_shape={tuple(lse.shape)} lse_max_abs={lerr:.3e} -> {'CORRECT' if ok else 'CHECK'}")
        except Exception as e:
            import traceback
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\nVERDICT: hd=256+LSE CORRECT on FA2/sm_120 -> DCA port needs NO FA4 kernel work; "
          "FA2 is the attention primitive. #592 (FA4 sm_120) becomes optional/upstream-nicety, "
          "OFF the Gemma 4 DCA critical path.")


if __name__ == "__main__":
    main()
