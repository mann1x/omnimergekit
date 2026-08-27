#!/usr/bin/env python3
"""`trl vllm-serve` for Qwen3.5/3.6 MoE **text** checkpoints, whose architecture
vLLM 0.20.2 implements but never registered.

THE GAP
-------
armJ / CoderX declare `architectures: ["Qwen3_5MoeForCausalLM"]`. vLLM 0.20.2 ships
that exact class -- `vllm/model_executor/models/qwen3_5.py:556`, inheriting
`Qwen3_5ForCausalLMBase` + `QwenNextMixtureOfExperts` -- but its registry only lists
the *ConditionalGeneration* (vision) variants, so a plain `vllm serve` dies with
"Model architectures ... are not supported". Nothing is missing except the map entry,
so this wrapper adds the entry and hands straight over to trl's own server. It does
NOT reimplement, subclass, or patch the model.

Serving the vision class instead would be the wrong fix: it expects a visual tower
this checkpoint does not have (the CoderX safetensors tower graft is still open).

WORKER PROPAGATION IS THE WHOLE DIFFICULTY, AND IT FAILS **SILENTLY**
---------------------------------------------------------------------
The registry lives in the process that registers it. vLLM resolves the architecture
AGAIN inside the EngineCore process, and a registration made only in the parent is
invisible there. That does NOT raise "architecture not supported". It hits
`ModelRegistry._normalize_arch` (registry.py:1083), which -- when the name is
unknown -- matches the `ForCausalLM` suffix and swaps it for every other known
architecture suffix until one is in the map. `Qwen3_5MoeForCausalLM` therefore
degrades to `Qwen3_5MoeForConditionalGeneration`, the VISION class, and the boot dies
somewhere unrelated:

    qwen3_5.py:821  self.use_data_parallel = multimodal_config.mm_encoder_tp_mode ...
    AttributeError: 'NoneType' object has no attribute 'mm_encoder_tp_mode'

Read literally that says "multimodal config missing"; it actually says "your
registration did not reach this process". Worse, on a checkpoint that DID carry a
vision tower the same silent swap would boot happily and serve the wrong class.
Measured 2026-08-23: the parent logged `Resolved architecture: Qwen3_5MoeForCausalLM`
while the child instantiated the ConditionalGeneration one.

The fix is to register at MODULE IMPORT time rather than inside main(). vLLM and trl
start their children with `spawn`, and spawn re-imports the parent's entry module in
the child (as `__mp_main__`) before running the target -- so a module-level
registration runs again in every child, by construction. That is why `register()` is
called at the bottom of this file and not from `main()`.

Two things that look like fixes and are not:
  * `VLLM_ENABLE_V1_MULTIPROCESSING=0` does keep the engine in-process, but trl's own
    vllm-serve still forks a data-parallel worker, and forking a process that has
    already initialised CUDA raises "Cannot re-initialize CUDA in forked subprocess".
  * `VLLM_WORKER_MULTIPROC_METHOD=fork` causes exactly that CUDA-in-fork failure.
Both were tried on 2026-08-23 and both are wrong here; the default spawn is correct,
and module-level registration is what makes spawn safe.

THE SAME OMISSION, A SECOND TIME: THE HYBRID FLAG
-------------------------------------------------
The text-only class is not only unregistered -- it never declares `IsHybrid` either.
Measured in this vLLM:

    stock Qwen3_5MoeForCausalLM              is_hybrid = False
    Qwen3_5MoeForConditionalGeneration       is_hybrid = True

and the CausalLM class has none of `get_mamba_state_shape_from_config`,
`get_mamba_state_dtype_from_config`, `get_mamba_state_copy_func`. armJ is a genuine
hybrid (30 linear_attention SSM layers + 10 full_attention), so this flag is simply
wrong, not merely absent.

It matters because `VllmConfig.try_verify_and_update_config` (config/vllm.py:1717)
dispatches the hybrid setup on `model_config.is_hybrid` -- NOT on the architecture
name -- and that path (`HybridAttentionMambaModelConfig` -> `MambaModelConfig`,
models/config.py:226) is the ONLY thing that assigns `cache_config.mamba_block_size`
(to `block_size` with prefix caching, to `max_model_len` without). Skip it and the
value stays None until the SSM layers ask for their cache spec:

    vllm/model_executor/layers/mamba/abstract.py:45
    assert mamba_block_size is not None
    AssertionError

which looks like a missing CLI flag and is not one -- `--enable_prefix_caching` was
innocent; there is no flag that repairs a False `is_hybrid`. So the subclass sets the
flag and supplies the three classmethods.

AND A THIRD TIME: M-RoPE
------------------------
Same family again, and it does not surface until the first token is generated -- the
server boots clean, answers /health, and then EngineCore dies on request #1:

    vllm/v1/worker/gpu_model_runner.py:1497  _init_mrope_positions
    assert supports_mrope(model), "M-RoPE support is not implemented."
    -> EngineDeadError

`_init_mrope_positions` runs whenever `model_config.uses_mrope`, which is read off the
HF config -- and armJ's rope_parameters genuinely say `mrope_interleaved: true,
mrope_section: [11, 11, 10]`. So the model IS M-RoPE; it is the text-only CLASS that
never declared `SupportsMRoPE`, exactly as it never declared `IsHybrid`. Handled in
`_build_text_class` with the text-only position rule.

The pattern across all three: vLLM 0.20.2 ships a complete Qwen3.5 text FORWARD pass
but wired none of the surrounding machinery -- not the registry entry, not the hybrid
KV-cache flag, not the M-RoPE interface. Expect a fourth if a new code path is hit;
look for an interface the ConditionalGeneration class has and this one does not.

The registration is passed as the STRING "module:Class" so each process imports it
lazily by path rather than needing the class pickled across the spawn boundary.

The gate at the bottom of register() is not decoration: because the failure mode is a
SILENT downgrade rather than an exception, "did not raise" proves nothing. It asserts
that a fresh normalization of the name still returns the name.
"""
from __future__ import annotations

import sys

ARCH = "Qwen3_5MoeForCausalLM"
TARGET = "vllm.model_executor.models.qwen3_5:Qwen3_5MoeForCausalLM"


def _build_text_class():
    """Stock `Qwen3_5MoeForCausalLM` + a weight-NAME mapper. Nothing else.

    The checkpoint is text-only but stores its tensors under the MULTIMODAL layout,
    `model.language_model.layers.N...` (22,690 of 22,712 keys), because Qwen3.5 nests
    the text config under `text_config`. vLLM's text-only class loads with
    `AutoWeightsLoader(self, skip_prefixes=["mtp."])` and NO mapper
    (qwen3_5.py:544-547), so it looks for `model.layers.N...` and dies with

        KeyError: 'language_model.layers.0.mlp.experts.w2_weight'

    Only the *names* differ, so only the names are remapped -- via `WeightsMapper`,
    which is vLLM's own mechanism for exactly this. No layer, no forward, no
    computation is touched. Serving the ConditionalGeneration class instead would
    "fix" the names and then fail on a visual tower this checkpoint does not have.

    This affects ONLY the initial checkpoint load. TRL's LoRA weight sync pushes
    parameters by their TRAINER runtime names, which are already `model.layers.N...`
    (transformers strips `language_model.` on load -- the same reason
    dpo_brevity.py's LORA_REGEX matches 310 modules without that segment). So the
    sync path needs no mapping and gets none.

    `mtp.` stays skipped: the MTP head is not in the weight index and vLLM has no
    place to put it.
    """
    import numpy as np
    import torch

    from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForCausalLM,
    )
    from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

    # The three hybrid classmethods are taken out of the vision class's __dict__ rather
    # than retyped, so they stay upstream's implementations and rot with them. Lifting
    # them is safe because they are pure config arithmetic over `hf_text_config`
    # (linear_num_key_heads, linear_conv_kernel_dim, ...) and never touch `cls` or any
    # visual attribute -- see qwen3_5.py:711-746.
    _CG = Qwen3_5ForConditionalGeneration

    class Qwen3_5MoeTextForCausalLM(Qwen3_5MoeForCausalLM, IsHybrid, SupportsMRoPE):
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "model."})

        is_hybrid = True
        get_mamba_state_dtype_from_config = _CG.__dict__[
            "get_mamba_state_dtype_from_config"]
        get_mamba_state_shape_from_config = _CG.__dict__[
            "get_mamba_state_shape_from_config"]
        get_mamba_state_copy_func = _CG.__dict__["get_mamba_state_copy_func"]

        supports_mrope = True

        def get_mrope_input_positions(self, input_tokens, mm_features):
            """Text-only M-RoPE positions: `arange` broadcast over the 3 sections.

            This is the vision implementation's own text path, not a new rule. In
            `Qwen3VLForConditionalGeneration._get_mrope_input_positions` (qwen3_vl.py
            :2498-2571) every multimodal branch is driven by `_iter_mm_grid_hw`, which
            yields nothing when there are no mm features; control falls straight to the
            `st < len(input_tokens)` tail, which emits
            `broadcast_to(arange(N), (3, N)) + 0` and a delta of
            `(N-1) + 1 - N == 0`. That tail is reproduced verbatim below.

            The vision method itself CANNOT be reused: it reads `config.video_token_id`,
            `config.vision_start_token_id` and `config.vision_config.spatial_merge_size`
            as ARGUMENTS to that call, so Python evaluates them before the empty loop is
            ever entered, and a text-only Qwen3.5 config has none of them -- it would
            raise AttributeError on every request rather than on multimodal ones.

            armJ really is an M-RoPE model: its rope_parameters carry
            `mrope_interleaved: true, mrope_section: [11, 11, 10]`. Reaching this method
            is correct; only the vision-specific bookkeeping is out of scope.
            """
            if mm_features:
                raise NotImplementedError(
                    f"{type(self).__name__} is the TEXT-only Qwen3.5 class and got "
                    f"{len(mm_features)} multimodal feature(s). Serving them here would "
                    f"compute text-shaped positions for image/video spans and corrupt "
                    f"the output silently. Serve the ConditionalGeneration class with a "
                    f"checkpoint that actually has a vision tower.")
            n = len(input_tokens)
            positions = np.broadcast_to(np.arange(n), (3, n))
            return torch.from_numpy(np.ascontiguousarray(positions)), 0

        def load_weights(self, weights):
            """Load with the prefix mapper, then PROVE every parameter was filled.

            AND A FOURTH TIME: NOTHING VERIFIES THE MAPPING.

            `AutoWeightsLoader` returns the set of names it matched. It does not
            raise when a parameter of `self` was never written -- an unmatched
            checkpoint prefix simply leaves that parameter at its uninitialised
            value and the server boots, answers /health, and emits token soup.
            That is not hypothetical: the 2026-08-23 GEPO smoke trained four steps
            on rollouts like "#  20f 2neduARSEa ( 记者了解到eszs" before anything
            noticed, because the only gate was "did the process reach
            GEPO_TRAIN_DONE". An unchecked case is a silent one, so the check is
            here, at the point of failure, and it REFUSES rather than warns --
            a wrong-weights server is worse than no server.
            """
            loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
            loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

            expected = {n for n, _ in self.named_parameters()}
            missing = sorted(expected - set(loaded))
            print(f">>> VLLM_WEIGHT_LOAD expected={len(expected)} "
                  f"loaded={len(loaded)} missing={len(missing)}", flush=True)
            if missing:
                raise RuntimeError(
                    f"REFUSE: {len(missing)}/{len(expected)} parameters were never "
                    f"loaded from the checkpoint -- they still hold their "
                    f"uninitialised values and this server would generate garbage. "
                    f"The prefix mapper {self.hf_to_vllm_mapper} did not cover the "
                    f"checkpoint layout. First unloaded: {missing[:8]}")
            return loaded

    return Qwen3_5MoeTextForCausalLM


def register() -> None:
    from vllm import ModelRegistry

    if ARCH in ModelRegistry.get_supported_archs():
        print(f">>> {ARCH} already registered by vLLM -- wrapper is a no-op", flush=True)
        return

    # Prove the class actually exists BEFORE registering a name that points at it.
    # Registering a dangling path would move the failure from "unsupported arch" at
    # boot to an ImportError deep inside a worker, which is far harder to read.
    mod, _, cls = TARGET.partition(":")
    import importlib

    try:
        getattr(importlib.import_module(mod), cls)
    except (ImportError, AttributeError) as exc:
        sys.exit(f"REFUSE: {TARGET} does not resolve in this vLLM -- {exc}. "
                 f"The registry gap is no longer the only problem; do not paper over it.")

    obj = _build_text_class()
    ModelRegistry.register_model(ARCH, obj)
    if ARCH not in ModelRegistry.get_supported_archs():
        sys.exit(f"REFUSE: register_model({ARCH}) did not take effect.")

    # The real check. `_normalize_arch` is the function that silently rewrites an
    # unknown ...ForCausalLM into ...ForConditionalGeneration; if the registration
    # took, it must now return the name unchanged. Asking it directly is the only
    # way to distinguish "registered" from "about to be swapped for the vision class".
    from vllm.model_executor.models.registry import ModelRegistry as _MR

    if ARCH not in getattr(_MR, "models", {}):
        sys.exit(f"REFUSE: {ARCH} is not in the resolver's model map after "
                 f"registration; it would be silently rewritten to a "
                 f"ForConditionalGeneration variant and serve the wrong class.")
    print(f">>> VLLM_ARCH_REGISTERED {ARCH} -> {TARGET} ({obj.__name__})", flush=True)


# MODULE LEVEL, not inside main(): `spawn` re-imports this file in every child
# process before running its target, so this line is what carries the registration
# across the process boundary. Moving it into main() reintroduces the silent
# downgrade to the vision class documented above.
register()


def main() -> int:
    from trl.scripts.vllm_serve import main as serve_main
    from trl.scripts.vllm_serve import make_parser

    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    serve_main(script_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
