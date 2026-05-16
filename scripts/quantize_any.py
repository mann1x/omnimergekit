#!/usr/bin/env python3
"""quantize_any — model-agnostic quant builder for the vLLM eval path.

Wraps three toolchains under one CLI:

  --method nvfp4a16    nvidia-modelopt NVFP4_DEFAULT_CFG  (env: modelopt)
  --method awq         autoawq                            (env: vllm)
  --method gptq        gptqmodel                          (env: modelopt)

Output is a HF-format directory the vLLM api server can load directly
(config.json has the right `quantization_config`, safetensors are sharded).

Why three: NVFP4A16 is preferred for 98e/MoE; AWQ is the broad fallback
when modelopt support for an architecture lags; GPTQ is the third option
when AWQ fails on a specific architecture or you need 3- or 8-bit.

Calibration data:
  - NVFP4A16: 128 samples from `tatsu-lab/alpaca` (default for modelopt)
  - AWQ / GPTQ: 128 samples from `pileval` / `c4` (autoawq/gptqmodel defaults)
  Override with `--calib-dataset <hf-id>` and `--calib-samples N`.

This script does NOT bring up vLLM or run any eval — quant only. The
output dir is consumed by `omk_eval.py --model <out> --backend vllm`.

Usage:
  ./quantize_any.py --src google/gemma-4-A4B-98e-v3-it \
                    --dst google/gemma-4-A4B-98e-v3-it-NVFP4A16 \
                    --method nvfp4a16

  ./quantize_any.py --src google/gemma-4-A4B-98e-v4-it \
                    --dst google/gemma-4-A4B-98e-v4-it-AWQ4 \
                    --method awq

Disk: NVFP4A16 of a 26B model is ~13 GB; AWQ-4bit ~14 GB; GPTQ-4bit ~13 GB.
VRAM: needs the full BF16 model loadable (24 GB for 26B → use device_map=auto
      to spill to CPU if needed).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

METHODS = {"nvfp4a16", "awq", "gptq",
           "w4a16_lc", "w4a8_lc", "nvfp4_lc", "int4_awq",
           "mxfp4", "nvfp4_awq_lite"}

# Per-method conda env to run the quantizer in.
METHOD_ENV = {
    "nvfp4a16": "/root/anaconda3/envs/modelopt",
    "awq": "/root/anaconda3/envs/vllm",  # legacy autoawq — DOES NOT support Gemma 4
    "gptq": "/root/anaconda3/envs/modelopt",
    # modelopt INT4_AWQ_CFG — works on Gemma 4 fused MoE experts (tensor-level).
    "int4_awq": "/root/anaconda3/envs/modelopt",
    "mxfp4":    "/root/anaconda3/envs/modelopt",
    "nvfp4_awq_lite": "/root/anaconda3/envs/modelopt",
    # llm-compressor (vLLM Project) successor to autoawq.
    "w4a16_lc": "/root/anaconda3/envs/llmcomp",
    "w4a8_lc":  "/root/anaconda3/envs/llmcomp",
    "nvfp4_lc": "/root/anaconda3/envs/llmcomp",
}


def quantize_nvfp4a16(src: str, dst: str, calib_dataset: str,
                      calib_samples: int) -> str:
    """Returns a Python snippet to run under the modelopt env."""
    return f"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained({src!r}, trust_remote_code=True)
import psutil as _ps
# Compute per-GPU max_memory dynamically: leave ~3.5 GiB headroom per GPU
# for export-time 4-bit packing scratch (avoids OOM in NVFP4QTensor.quantize).
# Scales to any VRAM (16/24/48/80 GB). Reserve ~85% of system RAM for offload.
_EXPORT_HEADROOM_BYTES = int(3.5 * 1024**3)
_max_memory = {{}}
for _i in range(torch.cuda.device_count()):
    _free, _total = torch.cuda.mem_get_info(_i)
    # Use the smaller of (free now) and (total - headroom) — accelerate
    # planning needs an absolute budget, not a fraction of remaining.
    _budget = max(2 * 1024**3, min(_free, _total - _EXPORT_HEADROOM_BYTES))
    _max_memory[_i] = f"{{int(_budget // (1024**2))}}MiB"
_cpu_bytes = int(_ps.virtual_memory().total * 0.85)
_max_memory['cpu'] = f"{{int(_cpu_bytes // (1024**2))}}MiB"
print(f'[quantize_any] max_memory={{_max_memory}} (per-GPU minus 3.5GiB export headroom)')
model = AutoModelForCausalLM.from_pretrained(
    {src!r}, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
    max_memory=_max_memory,
)
ds = load_dataset({calib_dataset!r}, split='train', streaming=False)
ds = ds.shuffle(seed=42).select(range({calib_samples}))
def calib_iter():
    for x in ds:
        text = x.get('text') or x.get('instruction') or x.get('prompt') or ''
        if not text:
            continue
        ids = tok(text, return_tensors='pt', max_length=512, truncation=True).input_ids
        yield ids.to(model.device)

# Gemma 4 multimodal exclude list. vLLM's gemma4_mm.py loader expects BF16
# for the vision_tower / embed_vision / embed_audio sub-modules — they
# don't go through the compressed-tensors weight loader, so loading FP4-
# packed shapes there triggers `AssertionError: torch.Size([1152, 2152]) vs
# torch.Size([1152, 4304])`. The working 128e quant excluded these too.
import copy
quant_cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
quant_cfg['quant_cfg'].extend([
    {{'quantizer_name': '*vision_tower*', 'enable': False}},
    {{'quantizer_name': '*embed_vision*', 'enable': False}},
    {{'quantizer_name': '*embed_audio*', 'enable': False}},
])
mtq.quantize(model, quant_cfg, calib_iter)
# Free GPU before export. 4-bit packing in `NVFP4QTensor.quantize` allocates
# scratch per tensor; on 26B models the GPU is nearly full post-calibration
# so export OOMs at `packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]`.
# Cannot `model.cpu()` here — accelerate's device_map="auto" has some params
# on meta device (offload hooks), and bulk `.cpu()` raises
# `NotImplementedError: Cannot copy out of meta tensor`. Just empty the cache;
# the export path itself handles per-tensor packing on GPU one tensor at a time
# so freeing reserved-but-unallocated memory is enough.
import gc as _gc
_gc.collect()
torch.cuda.empty_cache()
# Reduce per-tensor packing fragmentation
import os as _os
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
export_hf_checkpoint(model, export_dir={dst!r})
# Copy tokenizer + chat template alongside.
tok.save_pretrained({dst!r})
print('NVFP4A16 export done:', {dst!r})
"""


def quantize_int4_awq(src: str, dst: str, calib_dataset: str,
                      calib_samples: int) -> str:
    """Returns a Python snippet to run under the modelopt env.

    Uses nvidia-modelopt INT4_AWQ_CFG — same tensor-level engine as NVFP4A16,
    so it actually compresses Gemma 4 26B-A4B's fused MoE expert tensors
    (which llmcompressor + gptqmodel both miss because they walk for
    `nn.Linear` modules).
    """
    return f"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained({src!r}, trust_remote_code=True)
import psutil as _ps
# Compute per-GPU max_memory dynamically: leave ~3.5 GiB headroom per GPU
# for export-time 4-bit packing scratch (avoids OOM in NVFP4QTensor.quantize).
# Scales to any VRAM (16/24/48/80 GB). Reserve ~85% of system RAM for offload.
_EXPORT_HEADROOM_BYTES = int(3.5 * 1024**3)
_max_memory = {{}}
for _i in range(torch.cuda.device_count()):
    _free, _total = torch.cuda.mem_get_info(_i)
    # Use the smaller of (free now) and (total - headroom) — accelerate
    # planning needs an absolute budget, not a fraction of remaining.
    _budget = max(2 * 1024**3, min(_free, _total - _EXPORT_HEADROOM_BYTES))
    _max_memory[_i] = f"{{int(_budget // (1024**2))}}MiB"
_cpu_bytes = int(_ps.virtual_memory().total * 0.85)
_max_memory['cpu'] = f"{{int(_cpu_bytes // (1024**2))}}MiB"
print(f'[quantize_any] max_memory={{_max_memory}} (per-GPU minus 3.5GiB export headroom)')
model = AutoModelForCausalLM.from_pretrained(
    {src!r}, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
    max_memory=_max_memory,
)
ds = load_dataset({calib_dataset!r}, split='train', streaming=False)
ds = ds.shuffle(seed=42).select(range({calib_samples}))
def calib_iter():
    for x in ds:
        text = x.get('text') or x.get('instruction') or x.get('prompt') or ''
        if not text:
            continue
        ids = tok(text, return_tensors='pt', max_length=512, truncation=True).input_ids
        yield ids.to(model.device)
import copy
quant_cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
quant_cfg['quant_cfg'].extend([
    {{'quantizer_name': '*vision_tower*', 'enable': False}},
    {{'quantizer_name': '*embed_vision*', 'enable': False}},
    {{'quantizer_name': '*embed_audio*', 'enable': False}},
])
mtq.quantize(model, quant_cfg, calib_iter)
# Free GPU before export. 4-bit packing in `NVFP4QTensor.quantize` allocates
# scratch per tensor; on 26B models the GPU is nearly full post-calibration
# so export OOMs at `packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]`.
# Cannot `model.cpu()` here — accelerate's device_map="auto" has some params
# on meta device (offload hooks), and bulk `.cpu()` raises
# `NotImplementedError: Cannot copy out of meta tensor`. Just empty the cache;
# the export path itself handles per-tensor packing on GPU one tensor at a time
# so freeing reserved-but-unallocated memory is enough.
import gc as _gc
_gc.collect()
torch.cuda.empty_cache()
# Reduce per-tensor packing fragmentation
import os as _os
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
export_hf_checkpoint(model, export_dir={dst!r})
tok.save_pretrained({dst!r})
print('INT4_AWQ export done:', {dst!r})
"""


def quantize_modelopt_named(src: str, dst: str, calib_dataset: str,
                            calib_samples: int, cfg_name: str,
                            label: str) -> str:
    """Generic modelopt snippet that picks any *_CFG attribute by name.

    Used for MXFP4_DEFAULT_CFG, NVFP4_AWQ_LITE_CFG, etc. — same
    tensor-level engine as NVFP4A16, so Gemma 4 26B-A4B fused MoE
    expert tensors get compressed.
    """
    return f"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained({src!r}, trust_remote_code=True)
import psutil as _ps
# Compute per-GPU max_memory dynamically: leave ~3.5 GiB headroom per GPU
# for export-time 4-bit packing scratch (avoids OOM in NVFP4QTensor.quantize).
# Scales to any VRAM (16/24/48/80 GB). Reserve ~85% of system RAM for offload.
_EXPORT_HEADROOM_BYTES = int(3.5 * 1024**3)
_max_memory = {{}}
for _i in range(torch.cuda.device_count()):
    _free, _total = torch.cuda.mem_get_info(_i)
    # Use the smaller of (free now) and (total - headroom) — accelerate
    # planning needs an absolute budget, not a fraction of remaining.
    _budget = max(2 * 1024**3, min(_free, _total - _EXPORT_HEADROOM_BYTES))
    _max_memory[_i] = f"{{int(_budget // (1024**2))}}MiB"
_cpu_bytes = int(_ps.virtual_memory().total * 0.85)
_max_memory['cpu'] = f"{{int(_cpu_bytes // (1024**2))}}MiB"
print(f'[quantize_any] max_memory={{_max_memory}} (per-GPU minus 3.5GiB export headroom)')
model = AutoModelForCausalLM.from_pretrained(
    {src!r}, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
    max_memory=_max_memory,
)
ds = load_dataset({calib_dataset!r}, split='train', streaming=False)
ds = ds.shuffle(seed=42).select(range({calib_samples}))
def calib_iter():
    for x in ds:
        text = x.get('text') or x.get('instruction') or x.get('prompt') or ''
        if not text:
            continue
        ids = tok(text, return_tensors='pt', max_length=512, truncation=True).input_ids
        yield ids.to(model.device)
import copy
cfg = copy.deepcopy(getattr(mtq, {cfg_name!r}))
cfg['quant_cfg'].extend([
    {{'quantizer_name': '*vision_tower*', 'enable': False}},
    {{'quantizer_name': '*embed_vision*', 'enable': False}},
    {{'quantizer_name': '*embed_audio*', 'enable': False}},
])
mtq.quantize(model, cfg, calib_iter)
export_hf_checkpoint(model, export_dir={dst!r})
tok.save_pretrained({dst!r})
print('{label} export done:', {dst!r})
"""


def quantize_awq(src: str, dst: str, calib_samples: int) -> str:
    return f"""
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained({src!r}, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained({src!r}, device_map='auto', safetensors=True, trust_remote_code=True)
quant_cfg = dict(w_bit=4, q_group_size=128, zero_point=True, version='GEMM')
model.quantize(tok, quant_config=quant_cfg, max_calib_samples={calib_samples})
model.save_quantized({dst!r})
tok.save_pretrained({dst!r})
print('AWQ-4bit export done:', {dst!r})
"""


def quantize_gptq(src: str, dst: str, calib_dataset: str,
                  calib_samples: int) -> str:
    # gptqmodel's auto offload_to_disk_path lands in /tmp which is tmpfs
    # (RAM-backed, 64 GB) on solidPC. For a 26B-class model this can OOM
    # RAM. Pin it to a sibling of `dst` on persistent disk.
    import os as _os
    offload_dir = _os.path.join(_os.path.dirname(dst), "_gptq_offload")
    # gptqmodel 7.0 API: prepare_dataset() takes a list of strings or a
    # pre-loaded HF dataset; the older `n_samples=` kwarg is gone. We
    # load `calib_dataset` via HF datasets, sample `calib_samples` rows,
    # and pass the text list in.
    return f"""
import os
os.makedirs({offload_dir!r}, exist_ok=True)
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

# Pull a slice of the calibration corpus. allenai/c4 ships with multiple
# configs; use 'en' subset, train split, streaming-safe.
ds = load_dataset({calib_dataset!r}, 'en', split='train', streaming=True)
texts = []
for ex in ds:
    t = ex.get('text') or ''
    if len(t) >= 32:
        texts.append(t)
    if len(texts) >= {calib_samples}:
        break

qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                      offload_to_disk_path={offload_dir!r})
model = GPTQModel.load({src!r}, qcfg, trust_remote_code=True)
calib = model.prepare_dataset(texts)
model.quantize(calib)
model.save({dst!r})
import shutil; shutil.rmtree({offload_dir!r}, ignore_errors=True)
print('GPTQ-4bit export done:', {dst!r})
"""


def quantize_llmcomp(src: str, dst: str, calib_dataset: str,
                     calib_samples: int, scheme: str,
                     use_gptq: bool) -> str:
    """llm-compressor (vllm-project/llm-compressor) snippet.

    `scheme` is one of W4A16, W4A8, NVFP4, NVFP4A16. `use_gptq=True`
    swaps `QuantizationModifier` for `GPTQModifier` (round-to-nearest
    with second-order calibration) — the standard recipe for W4A16
    since it's what AWQ-replacement guides recommend.
    """
    modifier_cls = "GPTQModifier" if use_gptq else "QuantizationModifier"
    return f"""
import torch

# Compat-shim 1: llm-compressor 0.10 pins transformers <=4.57.6 (which has
# no `gemma4` model_type) AND compressed-tensors 0.14.0.1 (which still works
# with transformers 5.x once we patch a couple of missing symbols). We
# force transformers 5.x at install time and apply the following runtime
# shims to glue llm-compressor 0.10 against transformers 5.x:
#
#   (a) `TORCH_INIT_FUNCTIONS` removed from transformers.modeling_utils;
#       llmcompressor imports it. Re-add a minimal map.
#   (b) Gemma 4's tokenizer_config has `extra_special_tokens: ['<|video|>']`
#       (list); transformers 4.57.6 expected dict. Coerce list -> dict.
#       (Still applied even with transformers 5.x — safe no-op there.)
import transformers.modeling_utils as _mu
if not hasattr(_mu, "TORCH_INIT_FUNCTIONS"):
    _mu.TORCH_INIT_FUNCTIONS = {{n: getattr(torch.nn.init, n) for n in (
        "uniform_", "normal_", "trunc_normal_", "constant_",
        "xavier_uniform_", "xavier_normal_",
        "kaiming_uniform_", "kaiming_normal_",
    )}}

from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
_orig_set = _PTB._set_model_specific_special_tokens
def _patched_set(self, special_tokens):
    if isinstance(special_tokens, list):
        special_tokens = {{t: t for t in special_tokens}}
    return _orig_set(self, special_tokens)
_PTB._set_model_specific_special_tokens = _patched_set

# Compat-shim (c): transformers 5.x dropped/renamed PreTrainedModel._get_no_split_modules,
# but llmcompressor → accelerate.dispatch_model still calls it during oneshot
# (`infer_auto_device_map`). Provide a fallback that returns the model's own
# `_no_split_modules` (the canonical attribute in 5.x). Safe no-op when the
# original method is already present.
from transformers import PreTrainedModel as _PTM
if not hasattr(_PTM, "_get_no_split_modules"):
    def _get_no_split_modules_shim(self, device_map=None):
        cls_nsm = getattr(self, "_no_split_modules", None) or []
        # Walk submodules too — matches the behavior of the original method
        # which aggregates across nested PreTrainedModel children.
        out = list(cls_nsm)
        for m in self.modules():
            sub = getattr(m, "_no_split_modules", None)
            if sub:
                out.extend(sub)
        # de-dup, preserve order
        seen = set(); uniq = []
        for n in out:
            if n not in seen:
                seen.add(n); uniq.append(n)
        return uniq
    _PTM._get_no_split_modules = _get_no_split_modules_shim

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import {modifier_cls}

tok = AutoTokenizer.from_pretrained({src!r}, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    {src!r}, torch_dtype='auto', device_map='auto', trust_remote_code=True
)

# Calibration dataset — load slice and wrap as an in-memory HF Dataset.
# llm-compressor's `oneshot(dataset=...)` expects either a registered
# dataset name string or a Dataset object exposing `.column_names`. A
# Python list of strings crashes downstream in `get_columns()`.
from datasets import Dataset
ds = load_dataset({calib_dataset!r}, 'en', split='train', streaming=True)
rows = []
for ex in ds:
    t = ex.get('text') or ''
    if len(t) >= 32:
        rows.append({{'text': t}})
    if len(rows) >= {calib_samples}:
        break
calib_ds = Dataset.from_list(rows)

_kw = dict(targets='Linear', scheme={scheme!r}, ignore=['lm_head'])
# Gemma 4 expert mlp.down_proj has 2112 input cols (not divisible by 128).
# vLLM supports non-divisible group dims; tell the modifier to skip the check.
try:
    recipe = {modifier_cls}(**_kw, bypass_divisibility_checks=True)
except TypeError:
    # Older llmcompressor without the kwarg — fall back to group_size=64
    # which divides 2112 evenly (2112/64=33).
    from compressed_tensors.quantization import (
        QuantizationScheme, QuantizationArgs,
        QuantizationType, QuantizationStrategy,
    )
    weights = QuantizationArgs(num_bits=4, type=QuantizationType.INT,
                               strategy=QuantizationStrategy.GROUP,
                               group_size=64, symmetric=True)
    cfg = QuantizationScheme(targets=['Linear'], weights=weights)
    recipe = {modifier_cls}(config_groups={{'group_0': cfg}}, ignore=['lm_head'])
oneshot(
    model=model,
    dataset=calib_ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples={calib_samples},
    output_dir={dst!r},
)
tok.save_pretrained({dst!r})
print('{scheme} ({modifier_cls}) export done:', {dst!r})
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--calib-dataset", default="")
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the snippet that would run, but don't execute")
    args = ap.parse_args()

    # Default calibration set per method
    calib = args.calib_dataset or {
        "nvfp4a16": "tatsu-lab/alpaca",
        "int4_awq": "tatsu-lab/alpaca",
        "mxfp4":    "tatsu-lab/alpaca",
        "nvfp4_awq_lite": "tatsu-lab/alpaca",
        "awq": "mit-han-lab/pile-val-backup",
        "gptq": "allenai/c4",
        "w4a16_lc": "allenai/c4",
        "w4a8_lc":  "allenai/c4",
        "nvfp4_lc": "allenai/c4",
    }[args.method]

    if args.method == "nvfp4a16":
        snip = quantize_nvfp4a16(args.src, args.dst, calib, args.calib_samples)
    elif args.method == "int4_awq":
        snip = quantize_int4_awq(args.src, args.dst, calib, args.calib_samples)
    elif args.method == "mxfp4":
        snip = quantize_modelopt_named(args.src, args.dst, calib,
                                       args.calib_samples,
                                       cfg_name="MXFP4_DEFAULT_CFG",
                                       label="MXFP4")
    elif args.method == "nvfp4_awq_lite":
        snip = quantize_modelopt_named(args.src, args.dst, calib,
                                       args.calib_samples,
                                       cfg_name="NVFP4_AWQ_LITE_CFG",
                                       label="NVFP4_AWQ_LITE")
    elif args.method == "awq":
        snip = quantize_awq(args.src, args.dst, args.calib_samples)
    elif args.method == "gptq":
        snip = quantize_gptq(args.src, args.dst, calib, args.calib_samples)
    elif args.method == "w4a16_lc":
        # W4A16 via llm-compressor: standard GPTQ-style W4A16 (replacement for AWQ).
        snip = quantize_llmcomp(args.src, args.dst, calib, args.calib_samples,
                                scheme="W4A16", use_gptq=True)
    elif args.method == "w4a8_lc":
        # W4A8 — INT4 weights + INT8 activations. Activations need stat calibration.
        snip = quantize_llmcomp(args.src, args.dst, calib, args.calib_samples,
                                scheme="W4A8", use_gptq=True)
    elif args.method == "nvfp4_lc":
        # NVFP4 — full FP4 weights AND activations (faster than NVFP4A16 at
        # runtime; potentially lower quality on long traces).
        snip = quantize_llmcomp(args.src, args.dst, calib, args.calib_samples,
                                scheme="NVFP4", use_gptq=False)
    else:
        raise ValueError(f"unhandled method: {args.method}")

    env_path = METHOD_ENV[args.method]
    python_bin = f"{env_path}/bin/python"
    print(f"[quantize_any] method={args.method} env={env_path}")
    print(f"[quantize_any] src={args.src}")
    print(f"[quantize_any] dst={args.dst}")
    print(f"[quantize_any] calib={calib} n={args.calib_samples}")

    # ── Env canary checks ────────────────────────────────────────────────
    # Catch silent-regression bugs (output structurally valid but BF16-size,
    # because an upstream plugin disappeared). modelopt PyPI stable 0.43.0
    # dropped `_QuantFusedExperts`, which broke Gemma 4 MoE quantization
    # silently — 98e_v3/v4 produced 37GB single-file BF16 instead of the
    # expected ~10GB sharded NVFP4 packed. The dev/main branch has it.
    # Cost us 2.5h debug on 2026-05-12 before the regression was caught.
    # Fail loudly here rather than spend 12 min producing a garbage quant.
    canary_required = {
        # method → list of (python import path, attribute name) tuples
        "nvfp4a16": [
            ("modelopt.torch.quantization.plugins.huggingface",
             "_QuantFusedExperts"),
        ],
        "int4_awq": [
            ("modelopt.torch.quantization.plugins.huggingface",
             "_QuantFusedExperts"),
        ],
        "mxfp4": [
            ("modelopt.torch.quantization.plugins.huggingface",
             "_QuantFusedExperts"),
        ],
        "nvfp4_awq_lite": [
            ("modelopt.torch.quantization.plugins.huggingface",
             "_QuantFusedExperts"),
        ],
    }.get(args.method, [])
    if canary_required and not args.dry_run:
        canary_src = ";".join(
            f"from {mod} import {attr}"
            for mod, attr in canary_required
        )
        canary_cmd = [python_bin, "-c", canary_src]
        rc = subprocess.call(canary_cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)
        if rc != 0:
            names = ", ".join(f"{m}.{a}" for m, a in canary_required)
            sys.stderr.write(
                f"\n[quantize_any] ENV CANARY FAILED for method={args.method}\n"
                f"  required: {names}\n"
                f"  env:      {env_path}\n"
                f"  fix:      pip install --no-deps --force-reinstall "
                f"git+https://github.com/NVIDIA/Model-Optimizer.git\n"
                f"  context:  PyPI stable nvidia-modelopt 0.43.0 dropped "
                f"the fused-experts plugin needed for Gemma 4 MoE; the dev "
                f"main branch has it. Without this plugin, build will "
                f"silently produce a BF16-sized output that LOOKS valid.\n"
                f"Aborting before wasting GPU time.\n\n"
            )
            sys.exit(2)
        print(f"[quantize_any] canary OK ({len(canary_required)} symbol(s))")

    if args.dry_run:
        print("--- snippet ---")
        print(snip)
        return

    Path(args.dst).mkdir(parents=True, exist_ok=True)
    cmd = [python_bin, "-c", snip]
    # LD_PRELOAD harmless if not present
    env = dict(os.environ)
    env["LD_PRELOAD"] = env.get("LD_PRELOAD", f"{env_path}/lib/libstdc++.so.6")
    print(f"[quantize_any] $ {python_bin} -c <snippet>")
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
