#!/usr/bin/env python
"""Dump REAM's REAP saliency vector per layer, WITHOUT building a model.

WHY. The hybrid arm needs REAP's per-expert saliency as NUMBERS so it can be combined with
our competence ranking. Nothing on disk has them: the arm dirs store no keep metadata, the
armE build log prints the grouping for layer 0 only, and data-root holds only the calibration
tensors. So we re-run REAM's profiler with armE's EXACT arguments and capture the saliency
term at the point the Merger computes it -- the same hook point omk_ream_merge.py uses to
INJECT saliency, used here to READ it instead.

FAITHFULNESS GATE. Running the profiler again is only useful if it reproduces the selection
armE actually shipped. It is checked, not assumed: verify_against_armE() requires that the
top-`merge_size` experts by dumped saliency equal armE's keep set for all 40 layers, where
armE's keep set was recovered independently by byte-matching its expert tensors against the
base (keepset_diff.py). If the hook grabbed the wrong forward pass, or the profile is not
deterministic under seed 42, this fails loudly and nothing downstream runs.

Deliberately does NOT call save_pretrained -- we want the profile, not another 60 GB arm.
"""
import argparse
import json
import os
import sys
import time

RECIPE = "/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune/ream"


def make_dump_merger_cls(base_cls, sink):
    class DumpMerger(base_cls):
        """Records outs['saliency'] the first time each layer is profiled.

        Signature mirrors omk_ream_merge.make_injected_merger_cls exactly; a positional
        mismatch here would silently bypass the hook.
        """

        def _forward_pass(self, states, layer_ind, collect_outputs=True, upd_hid=False,
                          inputs_embeds=None, verbose=False):
            outs, states = super()._forward_pass(
                states, layer_ind, collect_outputs=collect_outputs, upd_hid=upd_hid,
                inputs_embeds=inputs_embeds, verbose=verbose)
            if collect_outputs:
                sal = outs.get("saliency")
                if sal is not None and layer_ind not in sink:
                    v = [float(x) for x in sal]
                    sink[layer_ind] = v
                    print(f">>> OMK_REAP_DUMP layer={layer_ind} n={len(v)} "
                          f"min={min(v):.6g} max={max(v):.6g} "
                          f"nonzero={sum(1 for x in v if x > 0)}", flush=True)
                    # checkpoint every layer: a crash at layer 39 must not cost the run
                    with open(sink["_path"], "w") as fh:
                        json.dump({str(k): x for k, x in sink.items()
                                   if isinstance(k, int)}, fh)
            return outs, states

    return DumpMerger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/ml/models/Qwen3.6-35B-A3B")
    ap.add_argument("--out", default="/mnt/sdc/ream-work/reap_saliency.json")
    ap.add_argument("--ream-dir", default="/shared/dev/ream")
    ap.add_argument("--data-root", default="/mnt/sdc/ream-work")
    ap.add_argument("--merge-size", type=int, default=184)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--dataset", default="math+code")
    ap.add_argument("--mix-ratio", default="0.3,0.7")
    ap.add_argument("--tokenizer-name", default="qwen36")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--calib-size", type=int, default=3072)
    ap.add_argument("--calib-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, args.ream_dir)
    sys.path.insert(0, RECIPE)
    os.chdir(args.data_root)
    print(f"cwd -> {os.getcwd()} (Merger resolves 'data/*.pt' relative to this)", flush=True)

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
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    sink = {"_path": args.out}
    cls = make_dump_merger_cls(Merger, sink)
    merger = cls(
        model,
        mtp_state_dict=None,
        merge_size=args.merge_size,
        grouping="ream",
        merging="none",          # armE: selection only, no averaging
        saliency="reap",
        dataset=args.dataset,
        mix_ratio=args.mix_ratio,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
        group_size=args.group_size,
        sequential=True,
        use_gate_output=True,
        gated_sim=True,
        calibration_data_size=args.calib_size,
        calibration_data_seq_len=args.calib_seq_len,
        seed=args.seed,
    )
    merger.fit()

    layers = {str(k): v for k, v in sink.items() if isinstance(k, int)}
    with open(args.out, "w") as fh:
        json.dump(layers, fh)
    print(f">>> OMK_REAP_DUMP_DONE layers={len(layers)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
