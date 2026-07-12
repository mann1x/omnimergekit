#!/usr/bin/env python3
"""smoke_fold_equiv.py — single-variable guard for redist_dern_eq11_fold.py.

Numeric proof, on identical seeded synthetic tensors, that:
  (1) fold.fold_layer(fold_mode='average') == dern11.fold_layer  BITWISE  (no-op proof),
      for both hard (assign_topk=0) and soft top-2 (assign_topk=2, the v8 setting).
  (2) fold.fold_layer(fold_mode='mergemoe'):
        - keeps each survivor's gate_up BITWISE-identical to the teacher survivor slot,
        - produces FINITE down,
        - output-norm ratio (merged vs target) is ~1 (LS magnitude match),
        - its down DIFFERS from the average-mode down (the swap is actually live).

No model, no GPU build — pure tensor math. Run with the omk python on bs2.
"""
import sys

import torch

SCR = "/srv/ml/repos/omnimergekit/scripts"
sys.path.insert(0, SCR)

import redist_dern_eq11 as ref          # noqa: E402  (current working dern11)
import redist_dern_eq11_fold as new     # noqa: E402  (the candidate)


def make_inputs(seed, device, H=64, M=48, E=12, n_keep=9, T=160):
    g = torch.Generator(device="cpu").manual_seed(seed)
    twoM = 2 * M
    gu_t = (torch.randn(E, twoM, H, generator=g) * 0.05).to(torch.bfloat16).to(device)
    dn_t = (torch.randn(E, H, M, generator=g) * 0.05).to(torch.bfloat16).to(device)
    x = (torch.randn(T, H, generator=g)).to(device).float()
    freq_l = (torch.rand(E, generator=g) * 100 + 1).to(device)   # strictly positive
    ids = list(range(E))
    keep_ids = ids[:n_keep]
    drop_ids = ids[n_keep:]
    return x, gu_t, dn_t, keep_ids, drop_ids, freq_l


def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device} torch={torch.__version__}")
    ok = True

    for topk in (0, 2):                       # hard, then v8's soft top-2
        x, gu_t, dn_t, keep_ids, drop_ids, freq_l = make_inputs(1234, device)
        s_ref, s_new = [], []
        mg_r, md_r = ref.fold_layer(x, gu_t, dn_t, keep_ids, drop_ids, freq_l, device, s_ref,
                                    freq_exp=1.0, norm_anchor="survivor", assign_topk=topk)
        mg_n, md_n = new.fold_layer(x, gu_t, dn_t, keep_ids, drop_ids, freq_l, device, s_new,
                                    freq_exp=1.0, norm_anchor="survivor", assign_topk=topk,
                                    fold_mode="average", ridge=1e-2)
        eq_gu = torch.equal(mg_r, mg_n)
        eq_dn = torch.equal(md_r, md_n)
        eq_stats = (s_ref == s_new)
        tag = "soft-top2" if topk == 2 else "hard"
        print(f"[avg/{tag}] gate_up bitwise={eq_gu}  down bitwise={eq_dn}  "
              f"scale-stats==dern11={eq_stats}")
        ok = ok and eq_gu and eq_dn and eq_stats

    # ---- mergemoe branch (v8 soft top-2) ----
    x, gu_t, dn_t, keep_ids, drop_ids, freq_l = make_inputs(1234, device)
    s_avg, s_mm = [], []
    mg_avg, md_avg = new.fold_layer(x, gu_t, dn_t, keep_ids, drop_ids, freq_l, device, s_avg,
                                    freq_exp=1.0, norm_anchor="survivor", assign_topk=2,
                                    fold_mode="average", ridge=1e-2)
    mg_mm, md_mm = new.fold_layer(x, gu_t, dn_t, keep_ids, drop_ids, freq_l, device, s_mm,
                                  freq_exp=1.0, norm_anchor="survivor", assign_topk=2,
                                  fold_mode="mergemoe", ridge=1e-2)
    # gate_up must equal the teacher survivor slots EXACTLY (bf16 round-trips losslessly)
    surv_stack = torch.stack([gu_t[i] for i in keep_ids])
    gu_preserved = torch.equal(mg_mm, surv_stack)
    dn_finite = bool(torch.isfinite(md_mm).all())
    dn_changed = not torch.equal(md_mm, md_avg)            # the swap is live
    # scale stats[1:4] = (min,mean,max) output-norm ratio for mergemoe
    _, smn, smu, smx = s_mm[0]
    ratio_ok = 0.6 <= smn and smx <= 1.6
    print(f"[mergemoe] gate_up==survivor bitwise={gu_preserved}  down finite={dn_finite}  "
          f"down!=avg(swap live)={dn_changed}")
    print(f"[mergemoe] output-norm ratio min/mean/max = {smn:.3f}/{smu:.3f}/{smx:.3f}  "
          f"(want ~1) ok={ratio_ok}")
    ok = ok and gu_preserved and dn_finite and dn_changed and ratio_ok

    print("====", "PASS" if ok else "FAIL", "====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
