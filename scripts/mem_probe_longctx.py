#!/usr/bin/env python
"""mem_probe_longctx.py — recipe-agnostic memory + throughput dry-run for the
Gemma 4 512k LoRA-extension plan (Pre-launch check #0).

It answers ONE question: does a target fit ONE 96GB GPU for 256k-seqlen LoRA
continued-pretrain, and at what step time — under
  * bf16 frozen base + LoRA (r/α) on Q/K/V/O of the full-attention layers only
  * gradient checkpointing
  * optional CPU activation offload (torch.autograd.graph.save_on_cpu) for the
    backbone forward (the ~169GB / ~44GB residual-snapshot term)
  * CHUNKED cross-entropy so the ~137GB full-logits tensor (seqlen*vocab) is
    never materialised — this is the real 256k wall and offload does NOT fix it.

It does NOT apply YaRN: peak VRAM is rope-independent, so this is safe to run
before the council validates the YaRN math. attn defaults to sdpa because
flash_attn is not installed in the omk env (note: the REAL trainer wants FA2).

Reports peak VRAM (GB) and per-step wall time. Pure synthetic random-token
input; no data, no checkpoints written.
"""
import argparse
import contextlib
import json
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM


def _repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def register_memeff():
    """Register a custom attn_implementation='memeff' that FORCES the SDPA
    mem-efficient backend. This is the ONLY backend that serves Gemma 4's
    full-attention layers (head_dim=512); FA2 / SDPA-FLASH / cuDNN all cap at 256
    (bug-436). repeat_kv instead of enable_gqa (mem-efficient rejects enable_gqa).
    Causal, mask ignored — fine for a random-token MEMORY probe (conservative:
    treats sliding layers as full attention)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def memeff(module, query, key, value, attention_mask=None, scaling=None,
               dropout=0.0, **kwargs):
        ng = getattr(module, "num_key_value_groups", 1) or 1
        key = _repeat_kv(key, ng)
        value = _repeat_kv(value, ng)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            o = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                               is_causal=True, scale=scaling)
        return o.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS["memeff"] = memeff


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seqlen", type=int, default=262144)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--ce-chunk", type=int, default=8192, help="seq tokens per logits chunk")
    ap.add_argument("--offload-activations", action="store_true")
    ap.add_argument("--attn", default="memeff", choices=["memeff", "sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--allow-math-sdpa", action="store_true",
                    help="permit the MATH SDPA backend (materialises the SxS matrix). "
                         "Default OFF: math+cudnn disabled, flash+mem_efficient forced, so a "
                         "non-dispatchable attn errors LOUDLY instead of allocating ~64GB at 256k.")
    ap.add_argument("--gpu", type=int, default=0)
    return ap.parse_args()


def find_backbone(base):
    """Return a callable that maps input_ids -> last_hidden_state for the text
    decoder, robust across Gemma4ForCausalLM and the multimodal wrapper."""
    for path in (("model",), ("model", "language_model"), ("language_model",)):
        obj = base
        ok = True
        for a in path:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and callable(obj):
            return obj, ".".join(path)
    raise RuntimeError("could not locate text backbone")


def main():
    a = parse()
    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"

    # Gemma 4 head_dim=256. The conda-forge flash_attn 2.8.3 sm_120 wheel's
    # transformers FA2 *model path* rejects hdim-256 ("forward only supports head
    # dimension at most 256"), even though torch's own fused FLASH/EFFICIENT SDPA
    # backends serve hdim-256 fine. So we run attn=sdpa and FORCE a memory-efficient
    # backend globally (applies inside grad-ckpt recompute too): flash + mem_efficient
    # ON, math + cudnn OFF. Disabling math means a non-dispatchable attn errors LOUDLY
    # instead of silently materialising the ~64GB SxS matrix at 256k (the real OOM trap).
    if a.attn in ("sdpa", "memeff"):
        torch.backends.cuda.enable_math_sdp(a.allow_math_sdpa)
        torch.backends.cuda.enable_cudnn_sdp(False)  # no sm_120 hdim-256/512 cuDNN engine
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print(f"[probe] sdpa backends: flash=on mem_efficient=on cudnn=off math={a.allow_math_sdpa}")
    if a.attn == "memeff":
        register_memeff()
        print("[probe] registered custom attn_implementation='memeff' (forced mem-efficient)")

    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    tcfg = getattr(cfg, "text_config", cfg)
    nlayers = tcfg.num_hidden_layers
    lt = getattr(tcfg, "layer_types", None)
    full = [i for i, x in enumerate(lt) if x == "full_attention"] if lt else None
    vocab = tcfg.vocab_size
    print(f"[probe] dir={a.model_dir}")
    print(f"[probe] layers={nlayers} full_attn={full} hidden={tcfg.hidden_size} vocab={vocab}")
    print(f"[probe] seqlen={a.seqlen} ce_chunk={a.ce_chunk} attn={a.attn} offload={a.offload_activations}")

    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, torch_dtype=torch.bfloat16, attn_implementation=a.attn,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(dev)
    model.config.use_cache = False

    suff = ("q_proj", "k_proj", "v_proj", "o_proj")
    targets = []
    for n, m in model.named_modules():
        if n.endswith(suff) and any(f".layers.{i}.self_attn." in (n + ".") for i in full):
            # Gemma 4 wraps the projection in Gemma4ClippableLinear(linear=Linear);
            # peft can only adapt the inner nn.Linear, so target that.
            inner = getattr(m, "linear", None)
            targets.append(n + ".linear" if isinstance(inner, torch.nn.Linear) else n)
    print(f"[probe] LoRA targets: {len(targets)} modules ({len(full)} layers x 4)")
    lc = LoraConfig(task_type="CAUSAL_LM", r=a.rank, lora_alpha=a.alpha,
                    lora_dropout=0.0, bias="none", target_modules=targets)
    model = get_peft_model(model, lc)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    base = model.get_base_model()
    backbone, bbname = find_backbone(base)
    head = base.get_output_embeddings()
    print(f"[probe] backbone={bbname} head={type(head).__name__} weight={tuple(head.weight.shape)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[probe] trainable params: {nparam/1e6:.2f}M")

    ctx = (torch.autograd.graph.save_on_cpu(pin_memory=True)
           if a.offload_activations else contextlib.nullcontext())

    torch.cuda.reset_peak_memory_stats(dev)
    result = {"model_dir": a.model_dir, "seqlen": a.seqlen, "attn": a.attn,
              "offload": a.offload_activations, "ce_chunk": a.ce_chunk}
    for s in range(a.steps):
        ids = torch.randint(0, vocab, (1, a.seqlen), device=dev)
        t0 = time.time()
        with ctx:
            # Gemma 4 hardens training forward (modeling_gemma4.py:2005) to require
            # mm_token_type_ids; text-only training passes all-zeros.
            hidden = backbone(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids),
                              use_cache=False)[0]  # [1,S,H], LoRA graph
        # cut-CE: detach a grad-leaf, backprop loss through head in chunks,
        # then push the accumulated grad back through the backbone once.
        hd = hidden.detach().requires_grad_(True)
        hs = hd[:, :-1, :].reshape(-1, hd.shape[-1])
        lb = ids[:, 1:].reshape(-1)
        ntok = lb.numel()
        Wt = head.weight  # [vocab, H], frozen
        for i in range(0, hs.shape[0], a.ce_chunk):
            lg = F.linear(hs[i:i + a.ce_chunk], Wt).float()
            (F.cross_entropy(lg, lb[i:i + a.ce_chunk], reduction="sum") / ntok).backward()
            del lg
        hidden.backward(hd.grad)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated(dev) / 1e9
        print(f"[probe] step {s}  dt={dt:.1f}s  peak_vram={peak:.1f} GB", flush=True)
        result[f"step{s}_s"] = round(dt, 1)
    result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 1)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
