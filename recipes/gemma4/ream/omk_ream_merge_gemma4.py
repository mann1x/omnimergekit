"""Drive the ported REAM merger on Gemma-4. Builds arms B' and C'.

  B'  stock REAM      --saliency reap --merging logits+weights            (no injection)
  C'  ours + REAP sal --saliency reap --merging logits+weights --drop-map ...
                      injection makes REAM's centroids exactly OUR keep set;
                      the REAP dump supplies the WITHIN-set ordering, which is
                      what sets the merge weights (w = saliency[group]).

CALIBRATION IS A BASIS. Every merge arm must consume the same .pt. The basis
sidecar (…​.basis.json) records corpus + batch sha256; check it before tabulating.
"""
import argparse, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/shared/dev/ream")
sys.path.insert(0, HERE)

from transformers import AutoTokenizer, Gemma4ForConditionalGeneration  # noqa: E402
from gemma4_merger import Merger  # noqa: E402  (the ported merger)


def build_injected_saliency(reap_dump_path, drop_map_path, n_experts, keep_offset=True,
                            variant="published"):
    """{layer: FloatTensor(E)} whose top-(E-|drop|) is EXACTLY our keep set.

    Ordering inside each of the kept/dropped sets comes from the REAP dump, so
    the merge weights are REAP's while the keep set is ours. The +1.0 offset is
    what forces the split; --no-keep-offset ranks purely by REAP and lets the
    keep set fall out of the score instead.
    """
    reap = json.load(open(reap_dump_path))
    drop = json.load(open(drop_map_path))

    # The P0 dump carries BOTH per_expert_scale variants, and they disagree on
    # ~5.5 experts/layer at the top-98 boundary -- 18% of the drop decision, so
    # picking one is a basis choice, never a default.
    #   published = softmax(logits), i.e. REAP/REAM's own definition
    #   effective = the model's top_k_weights, INCLUDING per_expert_scale
    # B' computes REAP live through Gemma4GateShim, which is softmax-of-logits
    # with no per_expert_scale. C' must therefore inject 'published', or the
    # matched pair B' vs C' would differ in the saliency DEFINITION as well as
    # the keep set, and would isolate nothing.
    if variant in reap:
        per_layer = reap[variant]
        meta = reap.get("meta", {})
        print(f"  REAP variant: {variant!r}  corpus={meta.get('corpus')} "
              f"sha={str(meta.get('corpus_sha256'))[:12]} tokens={meta.get('tokens_used')}")
        print(f"  NOTE: the injected ordering comes from THIS corpus, which is not the "
              f"merge calibration basis; the keep set is ours and is fixed either way.")
    elif "per_layer" in reap:
        per_layer = reap["per_layer"]
    else:
        avail = [k for k in reap if k != "meta"]
        raise SystemExit(f"REFUSING: REAP dump has no variant {variant!r}; available: {avail}")
    out, meta = {}, {}
    for li_s, dropped in drop.items():
        if not li_s.isdigit():
            continue
        li = int(li_s)
        dropped = set(int(x) for x in dropped)
        raw = per_layer.get(str(li)) or per_layer.get(li)
        if raw is None:
            raise SystemExit(f"REFUSING: REAP dump has no layer {li}")
        if isinstance(raw, dict):
            raw = [raw[str(e)] if str(e) in raw else raw[e] for e in range(n_experts)]
        if len(raw) != n_experts:
            raise SystemExit(f"REFUSING: layer {li} REAP vector len {len(raw)} != {n_experts}")
        t = torch.tensor(raw, dtype=torch.float)
        # rank-normalise to [0,1] so the +1.0 offset always dominates
        # Divide by n_experts, NOT n_experts-1: with /(n-1) the top-ranked expert
        # scores exactly 1.0, which collides with the +1.0 keep offset applied to
        # the LOWEST-ranked kept expert (rank 0.0 + 1.0 = 1.0). topk then breaks
        # that tie arbitrarily and can select a dropped expert. Caught by the
        # top-k==keep-set assertion below, at layer 12.
        order = torch.argsort(torch.argsort(t)).float()
        rank01 = order / n_experts   # in [0, (n-1)/n], strictly below 1.0
        vec = rank01.clone()
        if keep_offset:
            for e in range(n_experts):
                if e not in dropped:
                    vec[e] += 1.0
        out[li] = vec
        if keep_offset:
            k = n_experts - len(dropped)
            top = set(torch.topk(vec, k).indices.tolist())
            if top != set(range(n_experts)) - dropped:
                raise SystemExit(f"INTERNAL: injected top-k != keep set at layer {li}")
        meta[li] = {"dropped": len(dropped), "kept": n_experts - len(dropped)}
    if not out:
        raise SystemExit("REFUSING: drop map produced no layers")
    return out, meta


def make_injected(base_cls):
    class InjectedMerger(base_cls):
        def __init__(self, *a, injected_saliency=None, **kw):
            self._injected = injected_saliency
            super().__init__(*a, **kw)

        def _forward_pass(self, states, layer_ind, collect_outputs=True, **kw):
            expert_outs, states = super()._forward_pass(
                states, layer_ind, collect_outputs=collect_outputs, **kw)
            if collect_outputs and self._injected is not None:
                if layer_ind not in self._injected:
                    raise SystemExit(
                        f"refusing: no injected saliency for layer {layer_ind} "
                        f"(have n={len(self._injected)})")
                sal = self._injected[layer_ind]
                cur = expert_outs["saliency"]
                sal = sal.to(cur.device if torch.is_tensor(cur) else "cpu")
                expert_outs["saliency"] = sal
            return expert_outs, states
    return InjectedMerger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--merge-size", type=int, required=True)
    ap.add_argument("--saliency", default="reap", choices=["reap", "freq"])
    ap.add_argument("--merging", default="logits+weights")
    ap.add_argument("--grouping", default="ream")
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--dataset", default="acblk")
    ap.add_argument("--mix-ratio", default="1.0")
    ap.add_argument("--tokenizer-name", default="gemma4")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--calib-size", type=int, default=3072)
    ap.add_argument("--calib-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", required=True, help="dir containing data/<batch>.pt")
    ap.add_argument("--saliency-map", help="REAP dump json (C' injection)")
    ap.add_argument("--saliency-variant", default="published",
                    choices=["published", "effective"],
                    help="which per_expert_scale variant of the REAP dump to inject. "
                         "'published' (softmax of logits) matches what REAM computes "
                         "live, so it is the only choice that keeps B' vs C' a "
                         "one-variable comparison.")
    ap.add_argument("--drop-map", help="our drop map json (C' injection)")
    ap.add_argument("--no-keep-offset", action="store_true")
    ap.add_argument("--no-sequential", action="store_true")
    a = ap.parse_args()

    if bool(a.saliency_map) != bool(a.drop_map):
        sys.exit("REFUSING: --saliency-map and --drop-map must be given together")

    os.chdir(a.data_root)  # merger resolves data/<batch>.pt relative to cwd
    bf = f"data/{a.dataset}_b{a.calib_size}_seq{a.calib_seq_len}_{a.tokenizer_name}_seed{a.seed}.pt"
    if not os.path.exists(bf):
        sys.exit(f"REFUSING: calibration basis missing: {os.path.join(a.data_root, bf)}")
    side = bf + ".basis.json"
    if os.path.exists(side):
        print("calib basis:", json.dumps(json.load(open(side)), indent=2)[:600])

    print(f"loading {a.model}")
    vlm = Gemma4ForConditionalGeneration.from_pretrained(a.model, dtype="auto", device_map="cpu")
    n_exp = vlm.config.text_config.num_experts
    print(f"  num_experts={n_exp} top_k={vlm.config.text_config.top_k_experts}")

    injected = None
    if a.saliency_map:
        injected, meta = build_injected_saliency(
            a.saliency_map, a.drop_map, n_exp, keep_offset=not a.no_keep_offset,
            variant=a.saliency_variant)
        kept = {m["kept"] for m in meta.values()}
        print(f"  injection: {len(injected)} layers, kept-per-layer={sorted(kept)}")
        if kept != {a.merge_size}:
            sys.exit(f"REFUSING: drop map keeps {sorted(kept)} but --merge-size={a.merge_size}")

    cls = make_injected(Merger) if injected is not None else Merger
    kw = dict(model=vlm, merge_size=a.merge_size, grouping=a.grouping, merging=a.merging,
              saliency=a.saliency, dataset=a.dataset, mix_ratio=a.mix_ratio,
              tokenizer_name=a.tokenizer_name, batch_size=a.batch_size,
              group_size=a.group_size, sequential=not a.no_sequential,
              calibration_data_size=a.calib_size,
              calibration_data_seq_len=a.calib_seq_len, seed=a.seed)
    if injected is not None:
        kw["injected_saliency"] = injected
    merger = cls(**kw)
    model = merger.fit()

    os.makedirs(a.save_path, exist_ok=True)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.save_path)
    model.save_pretrained(a.save_path, safe_serialization=True, max_shard_size="4GB")
    print(">>> OMK_REAM_DONE", a.save_path)


if __name__ == "__main__":
    main()
