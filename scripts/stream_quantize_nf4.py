#!/usr/bin/env python3
"""stream_quantize_nf4.py — memory-bounded, multi-GPU bf16 -> bitsandbytes NF4 quantizer.

WHY THIS EXISTS
---------------
`AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(4bit))`
on transformers 5.x routes through `core_model_loading._materialize_copy`, which
materializes the *full bf16 fragment* on each device-map target BEFORE bitsandbytes
packs it. So the bf16 footprint must physically fit the plan. When a model's bf16
size exceeds total GPU VRAM (e.g. Gemma-4 26B-A4B 128e = ~52 GB on 2x24 GB 3090s)
you get either:
  * CPU offload -> offloaded layers stay bf16 (`llm_int8_enable_fp32_cpu_offload`) ->
    a bloated ~45 GB "4-bit" dir, or
  * a forced all-GPU plan -> `torch.cuda.OutOfMemoryError` at `_materialize_copy`.

This script never materializes the whole bf16 model anywhere. It streams the source
checkpoint tensor-by-tensor and packs each quantizable Linear weight DIRECTLY on a
target GPU (peak host/GPU cost = one tensor's bf16 + the growing packed model). The
packed weights are kept RESIDENT, balanced across all visible GPUs (filling VRAM),
and packing runs CONCURRENTLY across GPUs via per-GPU worker threads. One
`save_pretrained` at the end writes a standard bnb-4bit checkpoint.

OUTPUT
------
A normal bnb NF4 dir: `config.json` carries `quantization_config`, weights are packed
uint8 + quant_state. Load resident with:
    AutoModelForCausalLM.from_pretrained(out, device_map={"": 0})
(no bf16 materialization spike — the on-disk weights are already 4-bit).

`--skip-modules` keeps modules in bf16 (default `lm_head`; add `router` to keep a
pruned-MoE gate bf16-trainable for Router-KD).

LIMITATION — bnb 4-bit only quantizes `nn.Linear`. Architectures that store experts
as FUSED 3D parameters (e.g. Gemma-4 MoE `experts.gate_up_proj`/`down_proj` of shape
[num_experts, ...], which are plain Parameters, NOT nn.Linear) are left in bf16 by
`replace_with_bnb_linear`. On such models this packs only attention/router/shared
(~1 GB of ~50 GB) and the output stays bf16-sized. Verified on Gemma-4 26B-A4B 128e:
47.2 GB bf16 + 1.14 GB U8. Use this only for DENSE / all-Linear models whose bf16
footprint exceeds VRAM (dense 31B, Qwen merges). For fused-MoE use NVFP4A16/GGUF.

Origin: T110b, 2026-05-24. See backup_models/memory/feedback_bnb_4bit_save_meta_crash.md.
"""
import argparse
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

# --- bnb monkey-patches (portable: solidpc uses a forked bnb; pods use stock 0.49.2) ---
# Must run BEFORE transformers imports the symbols.
import bitsandbytes as _bnb  # noqa: E402
from bitsandbytes.nn.modules import Params4bit as _P4b  # noqa: E402
_orig_p4b_new = _P4b.__new__
def _p4b_new(cls, *a, **k):
    k.pop("_is_hf_initialized", None)  # accelerate 1.13 passes it; bnb 0.49.2 rejects it
    return _orig_p4b_new(cls, *a, **k)
_P4b.__new__ = _p4b_new

import bitsandbytes.functional as _bnbF  # noqa: E402
_orig_as_dict = _bnbF.QuantState.as_dict
def _meta_safe_as_dict(self, packed=False):
    off = getattr(self, "offset", None)
    if getattr(self, "nested", False) and isinstance(off, torch.Tensor) and off.is_meta:
        return {}
    return _orig_as_dict(self, packed=packed)
_bnbF.QuantState.as_dict = _meta_safe_as_dict

from accelerate import init_empty_weights  # noqa: E402
from accelerate.utils import set_module_tensor_to_device  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from transformers.integrations.bitsandbytes import replace_with_bnb_linear  # noqa: E402
from transformers.quantizers import AutoHfQuantizer  # noqa: E402


def get_module_and_param(model, key):
    """key like 'model.layers.0.mlp.experts.0.gate_proj.weight' -> (module, 'weight')."""
    parent, _, leaf = key.rpartition(".")
    mod = model.get_submodule(parent) if parent else model
    return mod, leaf


def resolve_shards(src: Path):
    """Return list of safetensors shard paths covering the checkpoint."""
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return sorted({src / s for s in wm.values()})
    single = src / "model.safetensors"
    if single.exists():
        return [single]
    sh = sorted(src.glob("model-*-of-*.safetensors"))
    if sh:
        return sh
    raise FileNotFoundError(f"no safetensors found under {src}")


def should_skip(name: str, skip_substrings) -> bool:
    return any(s in name for s in skip_substrings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="source HF dir (bf16 safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-modules", default="lm_head",
                    help="comma-separated module-name substrings KEPT in bf16 "
                         "(llm_int8_skip_modules). 'lm_head' always added. Add 'router' "
                         "to keep a pruned-MoE gate bf16-trainable for Router-KD.")
    ap.add_argument("--quant-type", default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--no-double-quant", action="store_true",
                    help="disable nested/double quantization (default ON)")
    ap.add_argument("--compute-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-gpus", type=int, default=0,
                    help="cap GPUs used (0 = all visible). Packed weights are balanced "
                         "across them and kept resident (fills VRAM).")
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="concurrent pack threads per GPU. >1 overlaps disk->H2D->pack.")
    ap.add_argument("--bf16-device", default="cpu",
                    help="device for non-quantized (bf16) params: 'cpu' (keeps GPU VRAM "
                         "free for packed weights) or e.g. 'cuda:0'.")
    ap.add_argument("--max-shard-size", default="5GB")
    args = ap.parse_args()

    src, out = Path(args.model).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip_modules.split(",") if s.strip()} | {"lm_head"}
    compute_dtype = torch.bfloat16 if args.compute_dtype == "bfloat16" else torch.float16

    n_vis = torch.cuda.device_count()
    n_gpu = n_vis if args.max_gpus in (0, None) else min(args.max_gpus, n_vis)
    if n_gpu < 1:
        raise RuntimeError("no CUDA GPUs visible — bnb 4-bit packing requires CUDA")
    gpus = list(range(n_gpu))
    print(f"[stream-nf4] src={src}\n[stream-nf4] out={out}", flush=True)
    print(f"[stream-nf4] {n_gpu} GPU(s) {gpus} | skip(bf16)={sorted(skip)} | "
          f"quant={args.quant_type} double={'off' if args.no_double_quant else 'on'} | "
          f"workers/gpu={args.workers_per_gpu}", flush=True)

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=(not args.no_double_quant),
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=sorted(skip),
    )
    hf_quantizer = AutoHfQuantizer.from_config(qconfig)
    if not hf_quantizer.is_serializable():
        raise RuntimeError("bnb 4-bit quantizer reports not serializable — check bnb version")

    # --- build meta model + swap Linears to Linear4bit shells (zero memory) ---
    config = AutoConfig.from_pretrained(src, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = replace_with_bnb_linear(model, modules_to_not_convert=sorted(skip),
                                    quantization_config=qconfig)
    model.eval()

    from bitsandbytes.nn import Linear4bit  # after replace

    # --- balance quantizable Linear4bit weights across GPUs by numel ---
    quant_targets = {}   # weight-key -> gpu index
    loads = {g: 0 for g in gpus}
    for name, mod in model.named_modules():
        if isinstance(mod, Linear4bit) and not should_skip(name, skip):
            g = min(loads, key=loads.get)
            wkey = f"{name}.weight"
            quant_targets[wkey] = g
            loads[g] += mod.out_features * mod.in_features
    tot = sum(loads.values())
    print(f"[stream-nf4] {len(quant_targets)} Linear4bit weights to pack | "
          f"per-GPU load(M params): "
          + ", ".join(f"g{g}={loads[g]/1e6:.0f}" for g in gpus), flush=True)
    if tot == 0:
        raise RuntimeError("no quantizable Linear4bit modules found — wrong skip set?")

    # --- per-GPU thread pools so all GPUs pack simultaneously ---
    pools = {g: ThreadPoolExecutor(max_workers=args.workers_per_gpu,
                                   thread_name_prefix=f"pack-g{g}") for g in gpus}
    lock = threading.Lock()
    done = {"n": 0}

    def pack_job(wkey, value):
        g = quant_targets[wkey]
        mod, _ = get_module_and_param(model, wkey)
        old = mod.weight  # meta Params4bit carrying quant_type/blocksize/etc
        # mirror transformers' Bnb4bitQuantize.convert: pack on the target device
        new_p = _bnb.nn.Params4bit(
            value.to(compute_dtype), requires_grad=False, **old.__dict__
        ).to(f"cuda:{g}")
        with lock:
            mod.weight = new_p
            mod._is_hf_initialized = True
            done["n"] += 1
            if done["n"] % 200 == 0:
                print(f"[stream-nf4]   packed {done['n']}/{len(quant_targets)}", flush=True)

    # --- stream shards: pack quant weights (threaded), place bf16 params on bf16-device ---
    t0 = time.time()
    shards = resolve_shards(src)
    print(f"[stream-nf4] streaming {len(shards)} shard(s) ...", flush=True)
    seen = set()
    for sp in shards:
        futures = []
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for key in f.keys():
                seen.add(key)
                if key in quant_targets:
                    g = quant_targets[key]
                    futures.append(pools[g].submit(pack_job, key, f.get_tensor(key)))
                else:
                    # bf16 / non-Linear (embeddings, norms, router, lm_head, biases)
                    t = f.get_tensor(key)
                    val = t.to(compute_dtype) if t.is_floating_point() else t
                    try:
                        set_module_tensor_to_device(model, key, args.bf16_device, value=val)
                    except (KeyError, AttributeError):
                        # key not in this arch's module tree (e.g. tied lm_head) — skip
                        pass
        for fut in futures:
            fut.result()  # surface pack exceptions per shard
    for p in pools.values():
        p.shutdown(wait=True)
    print(f"[stream-nf4] packed all weights in {time.time()-t0:.0f}s", flush=True)

    # --- tie + finalize quantizer flags so save emits a valid bnb checkpoint ---
    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()
    model.hf_quantizer = hf_quantizer
    model.is_quantized = True
    model.is_loaded_in_4bit = True
    model.config.quantization_config = qconfig

    # sanity: nothing should remain on meta
    metas = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if metas:
        print(f"[stream-nf4] WARNING: {len(metas)} params still on meta "
              f"(first: {metas[:3]}) — likely tied/unused; continuing", flush=True)

    print(f"[stream-nf4] saving to {out} (max_shard_size={args.max_shard_size}) ...", flush=True)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size=args.max_shard_size)

    # tokenizer + aux files
    try:
        AutoTokenizer.from_pretrained(src, trust_remote_code=True).save_pretrained(str(out))
    except Exception as e:
        print(f"[stream-nf4] tokenizer copy skipped: {e}", flush=True)
    for fn in ("chat_template.jinja", "processor_config.json", "preprocessor_config.json",
               "expert_drop_metadata.json", "generation_config.json"):
        sp = src / fn
        if sp.exists() and not (out / fn).exists():
            shutil.copy2(sp, out / fn)

    # --- self-check: dtype histogram + size ---
    from collections import Counter
    files = list(out.rglob("*.safetensors"))
    sz = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024**3
    dt = Counter()
    for fp in files:
        with safe_open(str(fp), framework="pt", device="cpu") as f:
            for k in f.keys():
                dt[str(f.get_slice(k).get_dtype())] += 1
    print(f"[stream-nf4] DONE: {sz:.2f} GB | dtypes={dict(dt)} | shards={len(files)}", flush=True)
    if dt.get("U8", 0) == 0:
        raise SystemExit("[stream-nf4] FAIL: no uint8-packed tensors — nothing was quantized")
    print("[stream-nf4] OK (uint8-packed weights present + quantization_config written)", flush=True)


if __name__ == "__main__":
    main()
