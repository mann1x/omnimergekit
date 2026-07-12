#!/usr/bin/env python
"""phase1_train_yarn_lora.py — LoRA continued-pretrain for the Gemma 4 YaRN-2.0
(262144 -> 524288) context extension.

IMPLEMENTED 2026-06-07 after the dry-run (T87) overturned the original plan's
assumptions. Every choice below is source-verified against modeling_gemma4.py and
validated by mem_probe_longctx.py on bs2 (omk-yarn env). Read these before editing:

ATTENTION — PER-LAYER dispatch (attn_implementation='memeff', see register_memeff):
  Gemma 4 mixes two attention types. FA2 is OUT only for the GLOBAL ones:
  * full_attention (global) layers run at global_head_dim=512; NO FlashAttention build /
    SDPA-FLASH / cuDNN serves hd>256 (bug-436). They use SDPA EFFICIENT (mem-efficient),
    O(S) memory, full causal. MATH backend is globally FORBIDDEN so a non-dispatchable
    attn errors loudly instead of materialising the ~64GB SxS matrix.
  * sliding_attention layers (the MAJORITY: 25/30 on 26B, 50/60 on 31B) run at hd=256
    with a 1024 window -> FlashAttention-2 with native window_size=(1023,0). This is both
    a speed win (O(S*1024) not O(S^2)) and a CORRECTNESS fix: previously ALL layers were
    forced full-causal, so sliding layers saw the whole prefix (train/inference mismatch).
    FA2 serves hd<=256 incl. backward; falls back to full-causal SDPA if no FA2 kernel.

ROPE — rope_type='proportional_yarn' (registered by proportional_yarn_rope_init,
  imported below). That module now derives global_head_dim=512 for full_attention
  layers (bug-437) — a rope canary at startup asserts inv_freq length == 256.

LoRA TARGETS — {q_proj, k_proj, o_proj} of the FULL-ATTENTION layers ONLY.
  - NO v_proj: config.attention_k_eq_v=True -> v_proj is None on global layers
    (V reuses K). Adapting k_proj therefore also adapts V.
  - NO relative_k_proj: that module is audio-tower only (not in the text decoder).
  - Sliding layers are left frozen (their rope_type='default' is untouched by YaRN).

MEMORY — pack 32k..64k (NOT 256k: that OOMs one 96GB GPU — the live per-layer
  attention tensors are the wall, not the offloadable snapshots). 256k/512k is for
  EVALUATION only (NIAH/RULER). grad-ckpt + optional CPU activation offload +
  chunked cut-cross-entropy (Gemma vocab=262144 -> a naive logits tensor is 137GB
  at 256k; we never materialise it). Council csl-2026-06-07-1455-acde: train-short
  is sound AND max-quality for a YaRN reparam (validate at 256k via the probe-abort).

  MEASURED single-GPU peak (bs2 PRO 6000, 96GB), 26B-A4B, grad_accum=1:
    32k no-offload  -> 73.7 GB, ~1142 tok/s
    64k no-offload  -> 95.6 GB, ~684 tok/s   (0.4 GB headroom — UNSAFE for days)
    64k + offload   -> 85.3 GB, ~655 tok/s
  => 64k single-GPU is too tight for a multi-day run, and the dense 31B (62 GB of
     weights) cannot fit 64k on one card at all. MODEL-PARALLEL (--gpus 0,1) is the
     safe path: a balanced layer split halves per-GPU activation memory.

MULTI-GPU (2026-06-07, --gpus) — naive model-parallel via accelerate device_map
  ("balanced_low_0"): the decoder layers are split across the listed GPUs, one GPU
  active at a time (no DDP/FSDP — those replicate or shard *params*, which does NOT
  reduce the single-sequence *activation* memory that is our wall). Cross-device
  activation/rope movement is handled by accelerate's dispatch hooks. The cut-CE is
  device-robust: hidden chunks are moved to lm_head's device on the fly and autograd
  carries the grad back across the .to() — so the big [chunk, vocab] logits land on
  whichever GPU holds the (tied) lm_head, and grad still flows into the backbone.

RESUME (2026-06-07, --resume) — atomic, crash/stop/disk-full safe. Each checkpoint
  is {LoRA adapter, AdamW state, LR-sched state, step, seen_tokens, data position,
  wandb run id}. Each is written to a temp dir then atomically renamed to an
  immutable ckpt-NNNNNN/ dir; an atomic latest.json pointer commits it. Keeps the
  newest --keep-ckpts (default 5) for redundancy (~25 MB each, ~125 MB total) and
  prunes older ones. A startup preflight aborts if the ckpt-dir filesystem has less
  than --min-free-gb free. A SIGTERM/SIGINT checkpoints at the next step boundary and
  exits clean. Re-launching the identical command auto-resumes (token stream is
  fast-forwarded by the saved pack count; wandb run is reattached).

WANDB (2026-06-07, --wandb) — optional, non-fatal. Per-log-step metrics + a stable
  run id derived from --ckpt-dir so a resumed run continues the SAME wandb run.
"""
import argparse
import contextlib
import glob
import hashlib
import json
import os
import shutil
import signal
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# MANDATORY: register 'proportional_yarn' in ROPE_INIT_FUNCTIONS BEFORE any model
# load (the dispatch table is read at model __init__). Side-effect import.
import proportional_yarn_rope_init  # noqa: F401,E402
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
from transformers.optimization import get_cosine_schedule_with_warmup  # noqa: E402

# Full-attn LoRA target leaf names. v_proj intentionally absent (None on global
# layers: attention_k_eq_v -> V=K). relative_k_proj intentionally absent (audio only).
TARGET_MODULE_SUFFIXES = ("q_proj", "k_proj", "o_proj")

# Flipped by the SIGTERM/SIGINT handler so the loop checkpoints and exits cleanly.
_STOP_REQUESTED = False


def _install_signal_handlers():
    def _handler(signum, _frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        print(f"\n[signal] caught {signal.Signals(signum).name} — will checkpoint "
              f"and exit at the next step boundary", flush=True)
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _handler)


def _repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def register_memeff():
    """attn_implementation='memeff' — PER-LAYER dispatch (each Gemma 4 attention call
    receives sliding_window=self.sliding_window; None on global layers):

      * sliding_attention layers (hd=256, window=1024 — the MAJORITY, 25/30 or 50/60):
        FlashAttention-2 with NATIVE windowed causal attention, window_size=(window-1, 0).
        This is O(S*window) not O(S^2), AND it is the CORRECT computation: a sliding
        layer must attend only within its local window. The previous all-layers-causal
        approximation made sliding layers attend to the FULL prefix — a train/inference
        mismatch (at inference these layers ARE windowed) and ~window/S x wasted compute.
        FA2 serves hd<=256 incl. backward.

      * full_attention (global) layers (hd=512): NO FlashAttention kernel exists for
        hd>256 (bug-436), so SDPA mem-efficient, full causal (correct for global layers;
        repeat_kv because mem-efficient rejects enable_gqa=True).

    Packed batch=1, no padding -> causality + window come from the FA2 flags / is_causal,
    so NO materialised SxS mask is needed (the ~64GB wall we avoid). If flash_attn is
    unavailable / lacks a kernel for this GPU, sliding falls back to full-causal SDPA
    (memory-safe, the old behaviour) so the run still proceeds."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    try:
        from flash_attn import flash_attn_func
    except Exception:
        flash_attn_func = None

    def _sdpa_full_causal(query, key, value, ng, scaling):
        key = _repeat_kv(key, ng)
        value = _repeat_kv(value, ng)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            o = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                               is_causal=True, scale=scaling)
        return o.transpose(1, 2).contiguous()

    def memeff(module, query, key, value, attention_mask=None, scaling=None,
               dropout=0.0, sliding_window=None, **kwargs):
        ng = getattr(module, "num_key_value_groups", 1) or 1
        sw = sliding_window
        if sw is None and getattr(module, "is_sliding", False):
            sw = getattr(module, "sliding_window", None)
        if sw is not None and flash_attn_func is not None:
            # FA2 wants [B, S, H, hd]; GQA handled natively (q has H heads, kv has 8).
            o = flash_attn_func(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                dropout_p=float(dropout), softmax_scale=scaling, causal=True,
                window_size=(int(sw) - 1, 0))          # left=window-1 -> current + (window-1) prev
            return o.contiguous(), None                # already [B, S, H, hd]
        return _sdpa_full_causal(query, key, value, ng, scaling), None

    ALL_ATTENTION_FUNCTIONS["memeff"] = memeff


def force_mem_efficient_sdpa(allow_math=False):
    torch.backends.cuda.enable_math_sdp(allow_math)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def full_attn_indices(tcfg):
    lt = getattr(tcfg, "layer_types", None)
    if not lt:
        raise SystemExit("FATAL: config has no layer_types — cannot locate full-attn layers")
    return [i for i, x in enumerate(lt) if x == "full_attention"]


def build_lora_targets(model, full_idx):
    """Exact module names for {q,k,o}_proj on full-attn layers that EXIST and are
    nn.Linear. v_proj is None on global layers (attention_k_eq_v) so it never matches."""
    want_layers = {f".layers.{i}.self_attn." for i in full_idx}
    targets = []
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if not n.endswith(TARGET_MODULE_SUFFIXES):
            continue
        if any(w in (n + ".") for w in want_layers):
            targets.append(n)
    return targets


def find_backbone(base):
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
            return obj
    raise RuntimeError("could not locate text backbone")


def rope_canary(model, tcfg):
    """Guard bug-437: full_attention rotary inv_freq must have length
    global_head_dim//2 (256), not head_dim//2 (128). A length of 128 means the
    rope was built over the wrong head_dim -> YaRN ramp corrupt -> NIAH collapse."""
    ghd = getattr(tcfg, "global_head_dim", None)
    if not ghd:
        return
    want = ghd // 2
    for name, buf in model.named_buffers():
        if name.endswith("full_attention_inv_freq"):
            got = buf.shape[-1]
            if got != want:
                raise SystemExit(
                    f"FATAL rope canary (bug-437): {name} len={got}, expected "
                    f"{want} (global_head_dim//2). YaRN rope built over wrong "
                    f"head_dim — fix proportional_yarn_rope_init before training.")
            print(f"[rope-canary] OK: {name} len={got} == global_head_dim//2")
            return
    print("[rope-canary] WARN: no full_attention_inv_freq buffer found (skipped)")


def token_stream(data_dir, pack_len, world_size=1, rank=0, skip_packs=0, announce=False):
    """Yield input_id tensors of exactly pack_len by concatenating tokens from
    jsonl shards ({'input_ids': [...]} per line). Robust to varying row lengths.
    Loops shards forever (continued-pretrain runs to a token budget, not epochs).

    Data parallelism: packs are enumerated by a deterministic GLOBAL index; this
    rank yields only packs where global_idx % world_size == rank (disjoint shards,
    no overlap across ranks). For single-GPU/MP runs world_size=1, rank=0 -> every
    pack, identical to the original behaviour.

    skip_packs: fast-forward this many GLOBAL packs WITHOUT materialising tensors
    (used on resume to re-align the deterministic stream to seen_tokens // pack_len;
    the GLOBAL count, so it is world_size-consistent). The rank stride is applied
    AFTER the skip so resume lands each rank back on its own disjoint subset."""
    shards = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not shards:
        raise SystemExit(f"FATAL: no *.jsonl shards in {data_dir}")
    buf = []
    gidx = 0  # global pack index across ALL ranks
    while True:
        for shard in shards:
            with open(shard) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = json.loads(line).get("input_ids")
                    if not ids:
                        continue
                    buf.extend(ids)
                    while len(buf) >= pack_len:
                        chunk = buf[:pack_len]
                        buf = buf[pack_len:]
                        g = gidx
                        gidx += 1
                        if g < skip_packs:
                            if announce and g and g % 500 == 0:
                                print(f"[resume] fast-forwarded {g}/{skip_packs} packs",
                                      flush=True)
                            continue
                        if world_size > 1 and (g % world_size) != rank:
                            continue
                        yield torch.tensor(chunk, dtype=torch.long).unsqueeze(0)


# ---------------------------------------------------------------------------
# Checkpointing — atomic, keep-last-N, crash/stop/disk-full safe.
# ---------------------------------------------------------------------------
def disk_free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def _is_valid_ckpt(d):
    return (os.path.isdir(d)
            and os.path.isfile(os.path.join(d, "trainer_state.pt"))
            and os.path.isfile(os.path.join(d, "adapter_model.safetensors")))


def _list_ckpts(ckpt_dir):
    """[(step:int, path)] for valid ckpt-NNNNNN dirs, ascending by step."""
    out = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("ckpt-"):
            p = os.path.join(ckpt_dir, name)
            if _is_valid_ckpt(p):
                try:
                    out.append((int(name.split("-", 1)[1]), p))
                except ValueError:
                    pass
    return sorted(out)


def find_resume_ckpt(ckpt_dir):
    """Newest complete checkpoint: prefer latest.json, else highest valid dir."""
    ptr = os.path.join(ckpt_dir, "latest.json")
    if os.path.isfile(ptr):
        try:
            d = json.load(open(ptr)).get("dir")
            if d and not os.path.isabs(d):
                d = os.path.join(ckpt_dir, d)
            if d and _is_valid_ckpt(d):
                return d
        except Exception:
            pass
    cks = _list_ckpts(ckpt_dir)
    return cks[-1][1] if cks else None


def save_checkpoint(ckpt_dir, model, opt, sched, *, step, seen_tokens,
                    pack_len, wandb_id, keep, args):
    """Atomically write checkpoint ckpt-NNNNNN/, commit via latest.json, keep newest N.

    Each checkpoint dir is immutable once committed (write to _tmp-NNNNNN/, then
    atomic rename). The latest.json pointer (atomic small-file replace) is the
    commit point. Older dirs beyond `keep` are pruned. On ENOSPC we prune to the
    single newest existing checkpoint to free space and retry once; if that still
    fails it propagates with that newest checkpoint intact (resumable)."""
    tmp = os.path.join(ckpt_dir, f"_tmp-{step:06d}")
    final = os.path.join(ckpt_dir, f"ckpt-{step:06d}")

    def _attempt():
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        # LoRA adapter (-> adapter_model.safetensors + adapter_config.json)
        model.save_pretrained(tmp)
        # Trainer state (optimizer/sched/counters/data-position/wandb id)
        torch.save({
            "step": step,
            "seen_tokens": seen_tokens,
            "packs_yielded": seen_tokens // pack_len,
            "pack_len": pack_len,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "wandb_id": wandb_id,
            "args": vars(args),
        }, os.path.join(tmp, "trainer_state.pt"))
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump({"step": step, "seen_tokens": seen_tokens,
                       "packs_yielded": seen_tokens // pack_len}, f, indent=2)
        # Commit: atomic dir rename, then atomic pointer flip.
        if os.path.isdir(final):
            shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        ptmp = os.path.join(ckpt_dir, "latest.json.tmp")
        with open(ptmp, "w") as f:
            json.dump({"dir": f"ckpt-{step:06d}", "step": step,
                       "seen_tokens": seen_tokens}, f)
        os.replace(ptmp, os.path.join(ckpt_dir, "latest.json"))
        # Prune older, keep newest `keep`.
        if keep > 0:
            for _s, p in _list_ckpts(ckpt_dir)[:-keep]:
                shutil.rmtree(p, ignore_errors=True)

    try:
        _attempt()
    except OSError as e:
        print(f"[ckpt] WARN write failed ({e}); pruning to newest checkpoint and retrying once",
              flush=True)
        for _s, p in _list_ckpts(ckpt_dir)[:-1]:  # emergency: keep only newest valid
            shutil.rmtree(p, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
        _attempt()
    print(f"[ckpt] saved {final} @ step {step} / {seen_tokens:,} tokens "
          f"(keep last {keep})", flush=True)


def main():
    ap = argparse.ArgumentParser(description="LoRA continued-pretrain for Gemma 4 YaRN-2.0 extension")
    ap.add_argument("--yarn-cfg-dir", required=True, help="model dir with YaRN-patched config.json")
    ap.add_argument("--data-dir", required=True, help="jsonl shards of packed sequences")
    ap.add_argument("--ckpt-dir", required=True, help="output dir for LoRA adapter ckpts")
    ap.add_argument("--tokens", type=int, default=250_000_000, help="total target training tokens")
    ap.add_argument("--pack-len", type=int, default=65536, help="training sequence length (32k..64k; NOT 256k)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--ce-chunk", type=int, default=2048, help="seq tokens per cut-CE logits chunk")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--offload-activations", action="store_true",
                    help="CPU-offload grad-ckpt snapshots (single-GPU >=64k; rarely needed with --gpus)")
    ap.add_argument("--attn", default="memeff", choices=["memeff", "sdpa"])
    ap.add_argument("--gpu", type=int, default=0, help="single-GPU device index (ignored if --gpus given)")
    ap.add_argument("--gpus", default="", help="comma list e.g. '0,1' -> model-parallel (device_map balanced_low_0)")
    ap.add_argument("--ddp", action="store_true",
                    help="data-parallel via torchrun (one full model per GPU; manual LoRA-grad all-reduce). "
                         "Use when the model FITS one card (26B @ <=32k): ~world_size x throughput. "
                         "Launch with: torchrun --nproc_per_node=N phase1_train_yarn_lora.py --ddp --gpus 0,1 ...")
    ap.add_argument("--max-mem-gib", type=int, default=92, help="per-GPU cap for the balanced device_map (GiB)")
    ap.add_argument("--ckpt-every-steps", type=int, default=20, help="checkpoint every N optimizer steps")
    ap.add_argument("--ckpt-every", type=int, default=0, help="also checkpoint every N tokens (0=off)")
    ap.add_argument("--keep-ckpts", type=int, default=5, help="keep newest N checkpoints (redundancy; ~25 MB each)")
    ap.add_argument("--min-free-gb", type=float, default=5.0, help="abort at startup if ckpt-dir fs free < this")
    ap.add_argument("--resume", default="auto", choices=["auto", "never", "must"],
                    help="auto=resume if a checkpoint exists; never=fresh; must=require one")
    ap.add_argument("--log-every", type=int, default=20, help="log every N optimizer steps")
    ap.add_argument("--max-steps", type=int, default=0, help="cap optimizer steps (0=unbounded; >0 for smoke)")
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb-project", default="gemma4-longctx-512k")
    ap.add_argument("--wandb-run-name", default="", help="default: basename of --ckpt-dir")
    ap.add_argument("--wandb-entity", default="", help="optional W&B entity/team")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Resolve GPU set + parallelism mode. Three mutually-exclusive modes:
    #   single  : one GPU, one process (default).
    #   mp       : --gpus 0,1 -> naive model-parallel (device_map layer split), one
    #              process; for sequences too big to fit one card (64k MoE / 31B).
    #   ddp      : --ddp     -> data-parallel under torchrun, ONE process per GPU,
    #              full model replicated, LoRA grads all-reduced; for models that DO
    #              fit one card (26B @ <=32k) -> ~world_size x throughput.
    # Pin CUDA_VISIBLE_DEVICES BEFORE any cuda call so internal indices are 0..n-1.
    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""] or [args.gpu]
    ddp = args.ddp
    if ddp:
        # torchrun sets LOCAL_RANK/RANK/WORLD_SIZE. Bind THIS rank to its physical card
        # by DEVICE INDEX (gpu_list[local_rank]), leaving all GPUs visible. An
        # in-process CUDA_VISIBLE_DEVICES remap is unreliable under torchrun once
        # torch/NCCL touch CUDA — it left both ranks on GPU0 (OOM). The index bind is
        # the robust standard pattern.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        dev_idx = gpu_list[local_rank] if local_rank < len(gpu_list) else local_rank
        ndev = 1
        mp = False
    else:
        # Pin CUDA_VISIBLE_DEVICES so internal indices are 0..n-1 (MP device_map +
        # single-GPU both assume cuda:0 as the execution device).
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
        local_rank = rank = 0
        world_size = 1
        dev_idx = 0
        ndev = len(gpu_list)
        mp = ndev > 1
    is_main = (rank == 0)
    dev = f"cuda:{dev_idx}"                                   # this rank's exec device
    dev_indices = [dev_idx] if ddp else list(range(ndev))     # GPUs to poll for mem/sync

    if is_main:
        print("=== phase1_train_yarn_lora ===")
        for k in ("yarn_cfg_dir", "data_dir", "ckpt_dir", "tokens", "pack_len", "lr",
                  "rank", "alpha", "grad_accum", "ce_chunk", "grad_ckpt",
                  "offload_activations", "attn", "max_steps", "resume",
                  "ckpt_every_steps", "wandb"):
            print(f"  {k:20s}: {getattr(args, k)}")
        mode = ("data-parallel (DDP, %d ranks)" % world_size if ddp
                else "model-parallel" if mp else "single-GPU")
        print(f"  {'gpus':20s}: {gpu_list}  ({mode})")
        if args.pack_len > 131072:
            print(f"  WARN: pack_len={args.pack_len} exceeds the measured ceiling "
                  f"(MoE 64k single-GPU). Expect OOM without enough --gpus.")
        if mp and args.offload_activations:
            print("  NOTE: --offload-activations with --gpus is usually unnecessary "
                  "(the layer split already halves per-GPU activation memory).")
        print()
    if args.dry_run:
        if is_main:
            print("[dry-run] config OK; not loading model.")
        return 0

    _install_signal_handlers()

    # DDP: bring up the process group now (device already pinned to the sole visible
    # GPU -> internal cuda:0). NCCL across processes, each owning one physical card.
    if ddp:
        torch.cuda.set_device(dev_idx)
        # device_id binds the PG to this rank's GPU (mutes the barrier() device warning
        # and lets NCCL use the fast path).
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{dev_idx}"))
        if is_main:
            print(f"[ddp] nccl up: world_size={world_size} (one full model per GPU, "
                  f"LoRA grads all-reduced each step)")
        print(f"[ddp] rank={rank} local_rank={local_rank} phys_gpu={gpu_list[local_rank] if local_rank < len(gpu_list) else local_rank}",
              flush=True)

    # Disk preflight — fail fast before the multi-minute model load.
    os.makedirs(args.ckpt_dir, exist_ok=True)
    free = disk_free_gb(args.ckpt_dir)
    if is_main:
        print(f"[disk] free on ckpt-dir filesystem: {free:.1f} GB (min required {args.min_free_gb})")
    if free < args.min_free_gb:
        raise SystemExit(f"FATAL: only {free:.1f} GB free on {args.ckpt_dir} fs "
                         f"(< --min-free-gb {args.min_free_gb}); free space before launch")

    if args.attn in ("memeff", "sdpa"):
        force_mem_efficient_sdpa(allow_math=False)
    if args.attn == "memeff":
        register_memeff()
        print("[setup] attn_implementation='memeff' (forced SDPA mem-efficient, math forbidden)")

    cfg = AutoConfig.from_pretrained(args.yarn_cfg_dir, trust_remote_code=True)
    tcfg = getattr(cfg, "text_config", cfg)
    fidx = full_attn_indices(tcfg)
    vocab = tcfg.vocab_size
    print(f"[setup] layers={tcfg.num_hidden_layers} full_attn={fidx} "
          f"global_head_dim={getattr(tcfg, 'global_head_dim', None)} vocab={vocab}")

    if mp:
        # Naive model-parallel: build a BALANCED device_map explicitly so the
        # decoder layers are split EVENLY across the GPUs (the string
        # "balanced_low_0" empties GPU0, and since 26B fits on one card accelerate
        # then packs everything onto GPU1 — defeating the split). get_balanced_memory
        # forces an even division; infer_auto_device_map with no "cpu" key forbids
        # CPU offload.
        from accelerate import infer_auto_device_map, init_empty_weights
        with init_empty_weights():
            meta = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                                    torch_dtype=torch.bfloat16)
        # Discover the decoder-layer class name (clean list of strings — Gemma4's
        # multimodal `_no_split_modules` is a nested set that breaks accelerate's
        # get_balanced_memory). Keep the decoder layer atomic so a single layer is
        # never split across GPUs.
        nosplit = sorted({type(m).__name__ for _n, m in meta.named_modules()
                          if type(m).__name__.endswith("DecoderLayer")})
        # Force an EVEN split: per-GPU byte cap = ceil(total / ndev * 1.05). Caps sum
        # to >total (so nothing spills to CPU) yet each < total (so it cannot fit on
        # one card -> must divide). No "cpu" key -> CPU offload forbidden.
        total_bytes = sum(p.numel() * p.element_size() for p in meta.parameters())
        per = int(total_bytes / ndev * 1.05) + 1
        cap = {i: per for i in range(ndev)}
        device_map = infer_auto_device_map(meta, max_memory=cap, dtype=torch.bfloat16,
                                           no_split_module_classes=nosplit)
        del meta
        # Pin the embeddings / tied lm_head / rotary to cuda:0 (the backbone's
        # execution device). The backbone is the multimodal Gemma4Model: its
        # accelerate hook moves the input to cuda:0, so the first embedding lookup
        # must also be on cuda:0 (infer otherwise groups the tied embed+lm_head onto
        # cuda:1 -> cross-device index_select crash). Layers stay split; only these
        # small modules are re-pinned. Default device -> 0 so any internally-created
        # arange (cache_position) lands where the forward starts.
        for k in list(device_map.keys()):
            if any(t in k for t in ("embed_tokens", "lm_head", "rotary_emb")):
                device_map[k] = 0
        torch.cuda.set_device(0)
        print(f"[setup] forced even split: {total_bytes/1e9:.1f} GB weights, "
              f"cap {per/1e9:.1f} GB/GPU, no_split={nosplit}; embed/lm_head/rotary pinned to cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            args.yarn_cfg_dir, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
            trust_remote_code=True, device_map=device_map)
        # Summarise + ASSERT the layers really span >1 device (else silent OOM risk).
        per_dev = {}
        for mod, d in getattr(model, "hf_device_map", {}).items():
            if ".layers." in mod:
                per_dev[d] = per_dev.get(d, 0) + 1
        print(f"[setup] model-parallel over {ndev} GPUs; decoder-layer placement:")
        for d in sorted(per_dev, key=lambda x: str(x)):
            print(f"          cuda:{d}: {per_dev[d]} decoder layers")
        if len({str(d) for d in per_dev}) < 2:
            raise SystemExit(f"FATAL: device_map did not split layers across GPUs "
                             f"(placement={dict(getattr(model, 'hf_device_map', {}))}). "
                             f"Lower --max-mem-gib so the model cannot fit on one card.")
    else:
        torch.cuda.set_device(dev_idx)
        model = AutoModelForCausalLM.from_pretrained(
            args.yarn_cfg_dir, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
            trust_remote_code=True, low_cpu_mem_usage=True).to(dev)
    model.config.use_cache = False

    rope_canary(model, tcfg)  # bug-437 guard — abort before wasting hours

    targets = build_lora_targets(model, fidx)
    exp = len(fidx) * len(TARGET_MODULE_SUFFIXES)
    print(f"[lora] {len(targets)} target modules (expected {len(fidx)}x{len(TARGET_MODULE_SUFFIXES)}={exp}: "
          f"{','.join(TARGET_MODULE_SUFFIXES)} on full-attn layers)")
    if len(targets) != exp:
        print(f"[lora] NOTE: count != {exp} — verify v_proj absence (attention_k_eq_v) "
              f"and that no extra modules matched.")
    lc = LoraConfig(task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.alpha,
                    lora_dropout=0.0, bias="none", target_modules=targets)
    model = get_peft_model(model, lc)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    base = model.get_base_model()
    backbone = find_backbone(base)
    head = base.get_output_embeddings()
    Wt = head.weight  # [vocab, H] frozen; device may differ from hidden under MP
    in_dev = base.get_input_embeddings().weight.device  # where input_ids must live
    trainable = [p for p in model.parameters() if p.requires_grad]
    ntrain = sum(p.numel() for p in trainable)
    print(f"[lora] trainable params: {ntrain/1e6:.2f}M  (lm_head/Wt on {Wt.device}, inputs on {in_dev})")
    bad = [n for n, p in model.named_parameters() if p.requires_grad and p.device.type != "cuda"]
    if bad:
        raise SystemExit(f"FATAL: {len(bad)} trainable params not on CUDA (e.g. {bad[:2]}) — device_map misplacement")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    # tokens/step counts the GLOBAL batch: every rank consumes grad_accum packs per
    # optimizer step, so DDP multiplies the effective batch by world_size.
    tokens_per_step = args.pack_len * args.grad_accum * world_size
    total_steps = max(1, args.tokens // tokens_per_step)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(args.warmup_frac * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    if is_main:
        print(f"[sched] total_steps={total_steps} warmup={warmup} "
              f"tokens/step={tokens_per_step:,} (global; world_size={world_size})")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- Resume ----------------------------------------------------------
    wandb_id = "yarn-" + hashlib.md5(os.path.abspath(args.ckpt_dir).encode()).hexdigest()[:10]
    start_step = 0
    seen_tokens = 0
    resume_dir = None if args.resume == "never" else find_resume_ckpt(args.ckpt_dir)
    if args.resume == "must" and resume_dir is None:
        raise SystemExit(f"FATAL: --resume must, but no valid checkpoint in {args.ckpt_dir}")
    if resume_dir is not None:
        st = torch.load(os.path.join(resume_dir, "trainer_state.pt"), map_location="cpu",
                        weights_only=False)
        sd = load_file(os.path.join(resume_dir, "adapter_model.safetensors"))
        set_peft_model_state_dict(model, sd)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        seen_tokens = int(st["seen_tokens"])
        start_step = int(st["step"]) + 1
        wandb_id = st.get("wandb_id", wandb_id)
        if is_main:
            print(f"[resume] from {resume_dir}: step {st['step']} -> start {start_step}, "
                  f"seen_tokens={seen_tokens:,}")
    # GLOBAL packs already consumed (seen_tokens is the global count); the stream
    # fast-forwards by this many global packs, then applies the per-rank stride.
    skip_packs = seen_tokens // args.pack_len

    # DDP: sync trainable (LoRA) params across ranks before step 0. LoRA-A is randomly
    # initialised PER PROCESS (different seed per rank), so a fresh run would start from
    # divergent adapters; on resume each rank loaded the same adapter. Either way,
    # broadcast from rank 0 so every replica is bit-identical (the DDP wrapper would
    # normally do this at construction — we don't use it, so it's explicit here).
    if ddp:
        for p in trainable:
            dist.broadcast(p.data, src=0)
        if is_main:
            print("[ddp] broadcast step-0 LoRA params from rank 0 (replicas in sync)")

    # ---- wandb (rank 0 only — one run, resumed by stable id) --------------
    wb = None
    if args.wandb and is_main:
        try:
            import wandb
            wb = wandb.init(
                project=args.wandb_project,
                entity=(args.wandb_entity or None),
                name=(args.wandb_run_name or os.path.basename(os.path.normpath(args.ckpt_dir))),
                id=wandb_id, resume="allow", config=vars(args),
            )
            print(f"[wandb] run id={wandb_id} project={args.wandb_project} "
                  f"url={getattr(wb, 'url', '?')}")
        except Exception as e:  # telemetry must never kill a multi-day run
            print(f"[wandb] WARN init failed ({e}); continuing without wandb", flush=True)
            wb = None

    # Only rank 0 writes the train log / checkpoints (avoid concurrent writers).
    logf = open(os.path.join(args.ckpt_dir, "train_log.jsonl"), "a") if is_main else None
    offload_ctx = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if args.offload_activations else contextlib.nullcontext())
    stream = token_stream(args.data_dir, args.pack_len, world_size=world_size, rank=rank,
                          skip_packs=skip_packs, announce=is_main)
    if skip_packs and is_main:
        print(f"[resume] fast-forwarding token stream by {skip_packs} global packs...", flush=True)

    last_ckpt_step = start_step - 1
    last_ckpt_tokens = seen_tokens
    for i in dev_indices:
        torch.cuda.reset_peak_memory_stats(i)

    def _do_ckpt(step):
        save_checkpoint(args.ckpt_dir, model, opt, sched, step=step,
                        seen_tokens=seen_tokens, pack_len=args.pack_len,
                        wandb_id=wandb_id, keep=args.keep_ckpts, args=args)

    for step in range(start_step, total_steps):
        t0 = time.time()
        loss_acc = 0.0
        for micro in range(args.grad_accum):
            ids = next(stream).to(in_dev)
            with offload_ctx:
                hidden = backbone(input_ids=ids,
                                  mm_token_type_ids=torch.zeros_like(ids),
                                  use_cache=False)[0]
            # chunked cut-CE: never materialise [S, vocab] logits. Device-robust:
            # hidden chunks are moved to Wt.device (the lm_head GPU) on the fly;
            # autograd carries grad back across the .to() into the backbone.
            hd = hidden.detach().requires_grad_(True)            # hidden.device
            hs = hd[:, :-1, :].reshape(-1, hd.shape[-1])
            lb = ids[:, 1:].reshape(-1).to(Wt.device)
            ntok = lb.numel()
            micro_loss = 0.0
            for i in range(0, hs.shape[0], args.ce_chunk):
                c = hs[i:i + args.ce_chunk]
                if c.device != Wt.device:
                    c = c.to(Wt.device)
                lg = F.linear(c, Wt).float()
                cl = F.cross_entropy(lg, lb[i:i + args.ce_chunk], reduction="sum")
                (cl / ntok / args.grad_accum).backward()
                micro_loss += cl.item()
                del lg
            hidden.backward(hd.grad)
            loss_acc += micro_loss / ntok
            # GLOBAL tokens: every rank consumes an equal grad_accum packs/step, so
            # the global count advances by world_size * this rank's tokens.
            seen_tokens += int(ids.numel()) * world_size
        # DDP: average the (tiny ~8 MB) LoRA grads across ranks BEFORE clip+step, so
        # every rank applies the identical update and parameters stay bit-in-sync.
        # We drive backward manually (no DDP wrapper), so the reducer is explicit
        # here. Materialise a zero grad for any param that saw none, to keep the
        # all_reduce collective symmetric across ranks (else NCCL deadlocks).
        if ddp:
            for p in trainable:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        for i in dev_indices:
            torch.cuda.synchronize(i)
        dt = time.time() - t0
        loss = loss_acc / args.grad_accum

        if step % args.log_every == 0 or step == total_steps - 1:
            # DDP: report the GLOBAL mean loss (all ranks identical after this).
            if ddp:
                lt = torch.tensor([loss], device=dev)
                dist.all_reduce(lt, op=dist.ReduceOp.SUM)
                loss = lt.item() / world_size
            peak = max(torch.cuda.max_memory_allocated(i) for i in dev_indices) / 1e9
            tps = round(tokens_per_step / dt, 1) if dt > 0 else 0.0
            rec = {"step": step, "loss": round(loss, 4), "lr": sched.get_last_lr()[0],
                   "tokens": seen_tokens, "dt_s": round(dt, 1), "tok_per_s": tps,
                   "grad_norm": round(float(gnorm), 3), "peak_vram_gb": round(peak, 1)}
            if is_main:
                print("[train] " + json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if wb is not None:
                    wb.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=step)

        # Collective stop: a SIGTERM/SIGINT may be delivered to only one rank under
        # torchrun, so agree across ranks via all_reduce(MAX) before breaking — else
        # the stopping rank exits and the others deadlock in the next collective.
        stop = _STOP_REQUESTED
        if ddp:
            st = torch.tensor([1 if stop else 0], device=dev)
            dist.all_reduce(st, op=dist.ReduceOp.MAX)
            stop = bool(st.item())
        due = (step - last_ckpt_step) >= args.ckpt_every_steps
        if args.ckpt_every:
            due = due or (seen_tokens - last_ckpt_tokens) >= args.ckpt_every
        if due or step == total_steps - 1 or stop:
            if is_main:
                _do_ckpt(step)
            if ddp:
                dist.barrier()  # hold all ranks until the checkpoint is committed
            last_ckpt_step = step
            last_ckpt_tokens = seen_tokens
        if stop:
            if is_main:
                print("[signal] checkpointed; exiting cleanly (resume with the same command).",
                      flush=True)
            break

    if logf is not None:
        logf.close()
    if wb is not None:
        wb.finish()
    if ddp:
        dist.destroy_process_group()
    if is_main:
        print("=== training complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
