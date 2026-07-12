#!/usr/bin/env python3
"""T87 path-B probe (DENSE variant) — isolates the sm_120 FA4 CORE kernel from the varlen epilogue.

The varlen probe (fa4_sm120_probe.py) failed at flash_fwd.py:402 in the *epilogue*:
`offset_batch_Q(mO, dim=3, ragged=ragged)[None,None,head_idx]` returns a rank-2 O view but is
indexed with a rank-3 coord — a layout congruence error. `ragged = use_tma_O and has_cu_seqlens_q`,
and TMA is sm_90+, so on the Sm80/Sm120 CpAsync lineage the *varlen* O writeback path appears
never to have been wired (production FA4 does varlen through the Sm90/Sm100 TMA epilogues).

This probe drives the SAME kernel in plain DENSE/batched mode (flash_attn_func, no cu_seqlens) so
`has_cu_seqlens_q=False` → the rank-4 batched O tensor → the epilogue's `[None,None,head_idx]`
indexing is rank-correct. If the sm_120 core RUNS + matches a softcapped-SDPA reference here, then
the MMA/softmax core is SOUND and the only gaps are (a) apply_score_mod for softcap, (b) a varlen
O-writeback epilogue. If it ALSO fails dense, the Sm80 path itself is non-functional on this DSL
and the scope is far larger.

Run in the bs2 `vllm` env on ONE GPU:  CUDA_VISIBLE_DEVICES=0 python fa4_sm120_probe_dense.py
"""
import math
import torch


def softcap(x, cap):
    return cap * torch.tanh(x / cap) if cap and cap > 0 else x


def ref_attn(q, k, v, scale, cap, causal):
    # q,k,v: (B, S, H, D). fp32 reference with softcap.
    B, S, H, D = q.shape
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    s = softcap(s, cap)
    if causal:
        m = torch.triu(torch.full((S, S), float("-inf"), device=q.device), diagonal=1)
        s = s + m
    p = torch.softmax(s, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", p, v.float())
    lse = torch.logsumexp(s, dim=-1)  # (B, H, S) natural-log
    return out, lse


def main():
    assert torch.cuda.is_available(), "no CUDA"
    dev = "cuda"
    cc = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)}  cc={cc}")

    from vllm.vllm_flash_attn import flash_attn_func

    B, H = 1, 8
    for CAP in (0.0, 50.0):
      for D in (256, 512):
        for S in (320,):
            scale = 1.0 / math.sqrt(D)
            torch.manual_seed(0)
            q = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
            k = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
            v = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
            tag = f"DENSE D={D} S={S} causal softcap={CAP}"
            try:
                res = flash_attn_func(
                    q, k, v,
                    softmax_scale=scale, causal=True, softcap=CAP,
                    return_softmax_lse=True, fa_version=4, num_splits=1,
                )
                out, lse = res if isinstance(res, (tuple, list)) else (res, None)
                rout, rlse = ref_attn(q, k, v, scale, CAP, causal=True)
                oerr = (out.float() - rout).abs().max().item()
                ocos = torch.nn.functional.cosine_similarity(
                    out.float().flatten(), rout.flatten(), dim=0).item()
                ok = oerr < 5e-2 and ocos > 0.999
                lse_s = tuple(lse.shape) if lse is not None else None
                print(f"[{tag}] FA4 RAN ✓  out_max_abs={oerr:.3e} cos={ocos:.6f} "
                      f"lse_shape={lse_s}  -> {'CORRECT' if ok else 'DIVERGES'}")
            except Exception as e:
                import traceback
                print(f"[{tag}] FA4 FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()

    print("\nVERDICT: DENSE D=512 CAP=0 RAN+CORRECT -> sm_120 CORE is sound; gaps = apply_score_mod "
          "(softcap) + varlen O-writeback epilogue. DENSE also FAILS -> Sm80 path broken on this DSL "
          "(scope = full kernel, not stubs).")


if __name__ == "__main__":
    main()
