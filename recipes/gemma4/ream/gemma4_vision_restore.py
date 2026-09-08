"""Restore Gemma-4's vision (and audio) towers onto a REAM-merged text model.

REAM saves only the trunk it merged. On Gemma-4 that silently drops 355
`visual.*` tensors, which is the failure the plan's census gate exists to catch.
This is the Gemma-4 analogue of upstream `qwen3_5.py`: load the original VLM,
swap in the merged language model, save the whole thing.

Gated: refuses to write unless the merged text model is actually smaller (the
merge happened) and the restored model carries the same vision tensor count as
the original.
"""
import argparse, json, sys
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from transformers import Gemma4ForConditionalGeneration


def count_vision(sd):
    return sum(1 for k in sd if k.startswith("visual.") or "vision_tower" in k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="the 128e base VLM dir")
    ap.add_argument("--merged", required=True, help="REAM output (text trunk only)")
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--expect-experts", type=int, required=True)
    ap.add_argument("--max-shard-size", default="4GB")
    a = ap.parse_args()

    print(f"loading original VLM: {a.original}")
    vlm = Gemma4ForConditionalGeneration.from_pretrained(
        a.original, dtype="auto", device_map="cpu")
    n_vis_before = count_vision(vlm.state_dict())
    print(f"  vision/audio tensors in original: {n_vis_before}")
    if n_vis_before == 0:
        sys.exit("REFUSING: original carries no vision tensors -- wrong source dir?")

    print(f"loading merged text model: {a.merged}")
    merged = AutoModelForCausalLM.from_pretrained(a.merged, dtype="auto", device_map="cpu")
    mcfg = merged.config
    # A Gemma4ForConditionalGeneration config nests the expert count under
    # text_config; only a bare text model exposes it at the top level. Reading
    # the top level unconditionally returns None and looks like "did not shrink".
    n_exp = getattr(mcfg, "num_experts", None)
    if n_exp is None:
        n_exp = getattr(getattr(mcfg, "text_config", None), "num_experts", None)
    print(f"  merged num_experts = {n_exp} (expect {a.expect_experts})")
    if n_exp != a.expect_experts:
        sys.exit(f"REFUSING: merged model has {n_exp} experts, expected {a.expect_experts}. "
                 f"A model that did not shrink is not a merged model.")

    # swap the trunk, mirroring upstream qwen3_5.py
    vlm.model.language_model = merged.model
    vlm.config.text_config.num_experts = n_exp
    vlm.config.text_config.top_k_experts = getattr(mcfg, "top_k_experts",
                                                   vlm.config.text_config.top_k_experts)
    if hasattr(mcfg, "merge_args"):
        vlm.config.text_config.merge_args = mcfg.merge_args

    sd = vlm.state_dict()
    n_vis_after = count_vision(sd)
    print(f"  vision/audio tensors after swap: {n_vis_after}")
    if n_vis_after != n_vis_before:
        sys.exit(f"REFUSING: vision census {n_vis_after} != {n_vis_before} -- "
                 f"the swap dropped tensors")

    print(f"saving -> {a.save_path}")
    vlm.save_pretrained(a.save_path, safe_serialization=True, max_shard_size=a.max_shard_size)
    try:
        AutoProcessor.from_pretrained(a.original).save_pretrained(a.save_path)
    except Exception as e:  # processor is nice-to-have, not load-bearing here
        print(f"  WARNING: processor not saved: {e}")

    print(json.dumps({"vision_tensors": n_vis_after, "num_experts": n_exp,
                      "save_path": a.save_path}, indent=2))
    print("VISION_RESTORE_OK")


if __name__ == "__main__":
    main()
