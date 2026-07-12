#!/usr/bin/env python3
"""T174.8 — verify router trainability on the ACTUAL A2 model before any
router-training GPU spend (council-endorsed de-risk step).

Checks:
  1. Discover the MoE router module + its real parameter names/shapes on A2
     (NOT the 31B unsloth cache the council cited).
  2. The live value of router.per_expert_scale (+ router.scale): is PES alpha=1.20
     a LIVE param, or was it baked into mlp.down_proj (so live scale is ~1.0)?
  3. Grads flow: set requires_grad on router.proj (+scale) ONLY, run one
     forward (with mm_token_type_ids) + backward, confirm finite non-zero grads
     and that NO non-router param accumulated grad.
  4. Router-swap reference mechanism: snapshot original router params to CPU,
     mutate live, restore, assert bit-identical (for the KL-anchor ref pass).

Run on bs2 GPU0 with the omk python.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"


def main():
    tok = AutoTokenizer.from_pretrained(A2, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        A2, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map={"": 0})
    model.eval()

    # ---- 1. discover router params (layer 0 only, for brevity) ----
    print("=== router-related params (layer 0) ===")
    router_names = []
    for n, p in model.named_parameters():
        low = n.lower()
        if "layers.0." in n and any(k in low for k in
                                    ("router", "gate", "per_expert", ".scale")):
            print("  %-70s shape=%s req_grad=%s" % (n, tuple(p.shape), p.requires_grad))
            if "router" in low or "per_expert" in low or low.endswith(".scale"):
                router_names.append(n)
    # broaden: any param name containing 'router' anywhere
    all_router = [n for n, _ in model.named_parameters() if "router" in n.lower()]
    print("total params with 'router' in name (all layers): %d" % len(all_router))
    if all_router:
        print("  example:", all_router[0])

    # ---- 2. live PES / scale values ----
    print("\n=== live scale values (layer 0) ===")
    sd = dict(model.named_parameters())
    for n in router_names:
        p = sd[n]
        if "scale" in n.lower() or "per_expert" in n.lower():
            f = p.detach().float()
            print("  %-70s min=%.4f max=%.4f mean=%.4f" % (
                n, f.min().item(), f.max().item(), f.mean().item()))
    # also check whether PES was baked into down_proj (heuristic: down_proj row-norm
    # spread). Just report mean abs of layer0 expert down_proj if present.
    for n, p in model.named_parameters():
        if "layers.0." in n and "down_proj" in n:
            print("  [bake-check] %-60s mean|w|=%.5f" % (n, p.detach().float().abs().mean().item()))
            break

    # ---- 3. grads flow on router only ----
    print("\n=== grad-flow test (router.proj + scale only) ===")
    trainable = []
    for n, p in model.named_parameters():
        want = ("router" in n.lower()) and (
            n.lower().endswith("proj.weight") or n.lower().endswith(".scale")
            or "per_expert" in n.lower())
        p.requires_grad_(bool(want))
        if want:
            trainable.append(n)
    print("trainable router params: %d" % len(trainable))
    print("  e.g.:", trainable[:4])
    model.config.use_cache = False

    txt = tok.apply_chat_template([{"role": "user", "content": "Write a haiku about the sea."}],
                                  add_generation_prompt=True, tokenize=False)
    enc = tok(txt, return_tensors="pt", add_special_tokens=False).to(model.device)
    ids = enc["input_ids"]
    out = model(input_ids=ids, attention_mask=enc["attention_mask"],
                mm_token_type_ids=torch.zeros_like(ids))
    loss = out.logits[:, :-1, :].float().log_softmax(-1).mean()  # dummy scalar
    loss.backward()
    g_router = sum(1 for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    g_leak = [n for n, p in model.named_parameters()
              if (not p.requires_grad) and p.grad is not None and p.grad.abs().sum() > 0]
    print("router params with non-zero grad: %d / %d" % (g_router, len(trainable)))
    print("NON-router params that leaked grad: %d %s" % (len(g_leak), g_leak[:2]))
    print("loss finite:", torch.isfinite(loss).item())

    # ---- 4. router-swap reference mechanism ----
    print("\n=== router-swap (KL-anchor reference) test ===")
    snap = {n: sd[n].detach().to("cpu").clone() for n in trainable}
    with torch.no_grad():
        for n in trainable:
            sd[n].add_(1.0)            # mutate live
        mutated_ok = all((sd[n].detach().cpu() - snap[n]).abs().mean() > 0.5 for n in trainable)
        for n in trainable:
            sd[n].copy_(snap[n].to(sd[n].device))   # restore
        restored_ok = all((sd[n].detach().cpu() - snap[n]).abs().max() == 0 for n in trainable)
    print("swap mutate ok:", mutated_ok, "| restore bit-identical:", restored_ok)
    cpu_mb = sum(t.numel() * t.element_size() for t in snap.values()) / 1e6
    print("router snapshot size (CPU): %.2f MB" % cpu_mb)

    print("\nVERIFY_DONE router_params=%d trainable=%d grad_ok=%s leak=%d swap_ok=%s" % (
        len(all_router), len(trainable), g_router == len(trainable),
        len(g_leak), restored_ok))


if __name__ == "__main__":
    main()
