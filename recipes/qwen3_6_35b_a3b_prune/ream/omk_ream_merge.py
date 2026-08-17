#!/usr/bin/env python
"""Drive Samsung REAM's Merger with an optionally INJECTED saliency vector.

WHY NOT merge.py. Their merge.py imports config.py, which imports `vllm` and `lm_eval` at
module scope. vllm is not installed in the omnimergekit env (and installing it there would
disturb a pinned eval stack). We also need to replace the saliency vector, which merge.py
gives no hook for. So we drive ream.Merger directly.

WHAT REAM ACTUALLY DOES (read from the source, not the paper):
  - saliency  = REAP's gate x ||expert activation||           (ream/saliency.py:34)
  - grouping  = the top-k MOST SALIENT experts become centroids; everyone else is assigned
                to the nearest centroid by expert-output cosine (+ gate-logit) similarity,
                walked in descending-saliency order with a group-size cap C
                                                                (ream/ream.py:pseudo_group)
  - merging   = donors are PERMUTATION-ALIGNED to the centroid via linear_sum_assignment
                over (activation cdist + PCA'd weight cdist), THEN averaged with
                saliency-proportional weights                   (ream/merger.py:581-607)
  - sequential= after each merged layer the hidden states are recomputed, so layer L+1 is
                profiled against the already-merged upstream     (ream/merger.py:468)

THE HYBRID. pseudo_group consumes `saliency` for BOTH centroid selection
(`argsort(saliency)[::-1][:k]`) and merge weights (`w = saliency[group]`). So injecting our
own per-layer score makes REAM's centroids exactly our keep set, and the ONLY remaining
difference from our published cut is that dropped experts get merged in instead of deleted.

HOW THE INJECTED VECTOR IS BUILT. We do NOT re-derive the drop recipe from the competence
map -- re-implementing `--agg wmax --cat-weight ... --floor-count ...` risks silent drift
from the shipped map. Instead we read the SHIPPED drop map and construct:

    saliency[li][e] = rank01(importance[li][e]) + (1.0 if e in keep_set[li] else 0.0)

Kept experts land in [1,2), dropped in [0,1), so the top-k selection reproduces the shipped
keep set BY CONSTRUCTION, floor-clamp and all, while the within-group ordering still follows
the real importance score. The +1.0 offset does bias the merge weights toward the centroid
(it is a weight as well as a rank); that is a deliberate, recorded choice -- pass
--no-keep-offset to rank purely by importance and let the keep set fall out of the score
(then VERIFY it still matches, which is what the arm-D control is for). With --merging none
the weights are unused, so the control arm is unaffected either way.

NOTE ON PATHS: Merger builds its calibration filename as a RELATIVE 'data/...' path
(ream/merger.py:102), so we chdir to --data-root, which must contain data/<dset>_b<N>_
seq<L>_<sfx>_seed<S>.pt.
"""
import argparse
import json
import os
import sys
import time


def build_injected_saliency(competence_map, drop_map_path, score, agg, cat_weights,
                            keep_offset=True):
    """-> ({layer_ind: FloatTensor(E)}, meta dict). Reuses make_drop_map's own scoring."""
    import torch

    recipe_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(recipe_dir))  # .../recipes/qwen3_6_35b_a3b_prune/..
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    import make_drop_map as mdm

    cm = mdm.load_map(competence_map)
    L = int(cm["metadata"]["num_layers"])
    E = int(cm["metadata"]["num_experts"])
    cats = list(cm["categories"].keys())

    weights = mdm.parse_cat_weights(cat_weights)
    unknown = set(weights) - set(cats)
    if unknown:
        raise SystemExit(f"--cat-weight names categories not in the map: {sorted(unknown)}\n"
                         f"available: {cats}")
    weighted = agg in mdm.WEIGHTED_AGGS
    if weights and not weighted:
        raise SystemExit(f"--cat-weight is only meaningful with --agg wmax|wsum (got {agg})")

    w = {c: float(weights.get(c, 1.0)) for c in cats}
    raw = mdm.raw_per_cat_layer(cm, cats, L, E, score)

    importance = {}
    if weighted:
        norm = {c: {li: mdm.rank_normalize(raw[c][li], E) for li in range(L)} for c in cats}
        wsum_all = sum(w[c] for c in cats) or 1e-12
        for li in range(L):
            if agg == "wmax":
                importance[li] = {e: max(w[c] * norm[c][li][e] for c in cats) for e in range(E)}
            else:
                importance[li] = {e: sum(w[c] * norm[c][li][e] for c in cats) / wsum_all
                                  for e in range(E)}
    else:
        for li in range(L):
            importance[li] = {e: mdm.aggregate([raw[c][li][e] for c in cats], agg)
                              for e in range(E)}

    with open(drop_map_path) as fh:
        drop_map = json.load(fh)

    out, meta = {}, {"layers": L, "experts": E, "keep_per_layer": {}}
    for li in range(L):
        dropped = set(int(x) for x in drop_map[str(li)])
        keep = [e for e in range(E) if e not in dropped]
        meta["keep_per_layer"][li] = len(keep)
        rank01 = mdm.rank_normalize(importance[li], E)
        vec = torch.zeros(E, dtype=torch.float32)
        for e in range(E):
            vec[e] = rank01[e] + (1.0 if (keep_offset and e not in dropped) else 0.0)
        out[li] = vec

        if keep_offset:
            # by construction the top-|keep| must be exactly the keep set; assert, never assume
            top = set(int(i) for i in vec.argsort(descending=True)[:len(keep)].tolist())
            if top != set(keep):
                raise SystemExit(f"INTERNAL: injected saliency top-k != keep set at L{li}")

    # MTP layer (index L) reuses the map's "mtp" entry when present, else layer 0's.
    mtp_dropped = drop_map.get("mtp")
    if mtp_dropped is not None:
        dropped = set(int(x) for x in mtp_dropped)
        rank01 = mdm.rank_normalize(importance[0], E)
        vec = torch.zeros(E, dtype=torch.float32)
        for e in range(E):
            vec[e] = rank01[e] + (1.0 if (keep_offset and e not in dropped) else 0.0)
        out[L] = vec
        meta["mtp_keep"] = E - len(dropped)

    return out, meta


def make_injected_merger_cls(base_cls):
    class InjectedMerger(base_cls):
        """Merger that replaces the profiled saliency with a supplied per-layer vector.

        Surgical override: run the real forward pass (we still need expert_logits,
        expert_act and gate_logits for grouping and permutation alignment) and swap ONLY
        the saliency term.
        """

        def __init__(self, *a, injected_saliency=None, **kw):
            self._injected = injected_saliency
            super().__init__(*a, **kw)

        def _forward_pass(self, states, layer_ind, collect_outputs=True, upd_hid=False,
                          inputs_embeds=None, verbose=False):
            outs, states = super()._forward_pass(
                states, layer_ind, collect_outputs=collect_outputs, upd_hid=upd_hid,
                inputs_embeds=inputs_embeds, verbose=verbose)
            if collect_outputs and self._injected is not None:
                if layer_ind not in self._injected:
                    raise SystemExit(
                        f"refusing: no injected saliency for layer {layer_ind} "
                        f"(have {sorted(self._injected)[:5]}... n={len(self._injected)})")
                vec = self._injected[layer_ind]
                prof = outs.get("saliency")
                try:
                    n_prof = len(prof)
                except TypeError:
                    n_prof = -1
                if n_prof != len(vec):
                    raise SystemExit(
                        f"refusing: injected saliency len {len(vec)} != profiled {n_prof} "
                        f"at layer {layer_ind}")
                print(f">>> OMK_SALIENCY_INJECTED layer={layer_ind} n={len(vec)} "
                      f"min={float(vec.min()):.4f} max={float(vec.max()):.4f}", flush=True)
                outs["saliency"] = vec.clone()
            return outs, states

    return InjectedMerger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--merge-size", type=int, required=True)
    ap.add_argument("--ream-dir", default="/shared/dev/ream")
    ap.add_argument("--data-root", default="/mnt/sdc/ream-work",
                    help="dir containing data/<dset>_b<N>_seq<L>_<sfx>_seed<S>.pt "
                         "(Merger resolves calib files relative to cwd)")
    # REAM knobs
    ap.add_argument("--saliency", default="reap", choices=["freq", "reap"])
    ap.add_argument("--grouping", default="ream", choices=["ream", "hcsmoe"])
    ap.add_argument("--merging", default="logits+weights",
                    choices=["avg", "avg_freq", "weights", "logits", "logits+weights", "none"])
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--dataset", default="math+code")
    ap.add_argument("--mix-ratio", default="0.3,0.7")
    ap.add_argument("--tokenizer-name", default="qwen36")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--calib-size", type=int, default=3072)
    ap.add_argument("--calib-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calib-limit", type=int, default=0,
                    help="After loading, subsample the calib batch to N sequences (0=all). "
                         "NOTE: --calib-size is part of the .pt FILENAME, so it must keep "
                         "matching the file on disk; this trims post-load instead. Uses a "
                         "seeded permutation because the batch is math-then-code "
                         "concatenated -- a head slice would be all math.")
    ap.add_argument("--no-sequential", action="store_true")
    ap.add_argument("--no-gate-output", action="store_true")
    ap.add_argument("--no-gated-sim", action="store_true")
    ap.add_argument("--mtp-safetensors", default=None)
    # injection
    ap.add_argument("--saliency-map", default=None, help="our competence map JSON")
    ap.add_argument("--drop-map", default=None, help="our shipped drop map JSON")
    ap.add_argument("--score", default="tc", choices=["tc", "wnorm", "wnorm_tc"])
    ap.add_argument("--agg", default="wmax", choices=["sum", "max", "mean", "wmax", "wsum"])
    ap.add_argument("--cat-weight", action="append", default=[], metavar="CAT=W")
    ap.add_argument("--no-keep-offset", action="store_true",
                    help="rank purely by importance instead of forcing the shipped keep set")
    args = ap.parse_args()

    if bool(args.saliency_map) != bool(args.drop_map):
        raise SystemExit("--saliency-map and --drop-map must be given together")

    sys.path.insert(0, args.ream_dir)
    os.chdir(args.data_root)
    print(f"cwd -> {os.getcwd()} (Merger resolves 'data/*.pt' relative to this)", flush=True)

    injected = meta = None
    if args.saliency_map:
        injected, meta = build_injected_saliency(
            args.saliency_map, args.drop_map, args.score, args.agg, args.cat_weight,
            keep_offset=not args.no_keep_offset)
        keeps = set(meta["keep_per_layer"].values())
        print(f">>> OMK_INJECT_READY layers={meta['layers']} experts={meta['experts']} "
              f"keep_per_layer={sorted(keeps)} mtp_keep={meta.get('mtp_keep')}", flush=True)
        if keeps != {args.merge_size}:
            raise SystemExit(
                f"refusing: drop map keeps {sorted(keeps)} per layer but --merge-size is "
                f"{args.merge_size}; REAM emits a uniform merge_size, so these must agree")

    from ream import Merger
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"loading {args.model} on cpu ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="cpu",
        local_files_only=True, low_cpu_mem_usage=False).eval()
    print(f"loaded in {time.time()-t0:.0f}s, params={model.num_parameters()}", flush=True)

    mtp_state_dict = None
    if args.mtp_safetensors:
        from safetensors.torch import load_file
        mtp_state_dict = {}
        for f in args.mtp_safetensors.split(","):
            mtp_state_dict.update(load_file(f))
        print(f"loaded MTP state dict: {len(mtp_state_dict)} tensors", flush=True)

    cls = make_injected_merger_cls(Merger) if injected is not None else Merger
    kw = dict(
        mtp_state_dict=mtp_state_dict,
        merge_size=args.merge_size,
        grouping=args.grouping,
        merging=args.merging,
        saliency=args.saliency,
        dataset=args.dataset,
        mix_ratio=args.mix_ratio,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
        group_size=args.group_size,
        sequential=not args.no_sequential,
        use_gate_output=not args.no_gate_output,
        gated_sim=not args.no_gated_sim,
        calibration_data_size=args.calib_size,
        calibration_data_seq_len=args.calib_seq_len,
        seed=args.seed,
    )
    if injected is not None:
        kw["injected_saliency"] = injected

    merger = cls(model, **kw)

    if args.calib_limit:
        import torch
        n = merger.batch["input_ids"].shape[0]
        if args.calib_limit >= n:
            print(f">>> OMK_CALIB_LIMIT skipped ({args.calib_limit} >= loaded {n})", flush=True)
        else:
            g = torch.Generator().manual_seed(args.seed)
            idx = torch.randperm(n, generator=g)[:args.calib_limit].to(
                merger.batch["input_ids"].device)
            for k in ("input_ids", "attention_mask"):
                merger.batch[k] = merger.batch[k][idx]
            print(f">>> OMK_CALIB_LIMIT {n} -> {merger.batch['input_ids'].shape[0]} "
                  f"(seeded permutation, seed={args.seed})", flush=True)

    model = merger.fit()
    print(f"params after merging: {model.num_parameters()}", flush=True)

    model.config.merge_args = {k: v for k, v in vars(args).items()}
    model.config.merge_args["omk_injected_saliency"] = injected is not None
    os.makedirs(args.save_path, exist_ok=True)
    tok.save_pretrained(args.save_path)
    if merger.mtp_state_dict is not None:
        from safetensors.torch import save_file
        save_file(merger.mtp_state_dict, os.path.join(args.save_path, "mtp.safetensors"))
        print("saved mtp.safetensors (needs manual index rename, see ream/qwen3_5.py)",
              flush=True)
    model.save_pretrained(args.save_path, safe_serialization=True, max_shard_size="4GB")
    print(f">>> OMK_REAM_SAVED {args.save_path}", flush=True)
    print(">>> OMK_REAM_DONE", flush=True)


if __name__ == "__main__":
    main()
