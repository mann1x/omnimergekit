#!/usr/bin/env python
# phase1_probe_watcher.py — GPU-1 watcher: on each new ckpt under
# --ckpt-dir, merge LoRA into base, run a quick RULER NIAH single-needle
# probe @ 256k (n=10 questions), log score. If 3 consecutive ckpts score
# below --threshold (default 0.80), signal abort by killing --kill-pid.
#
# This implements the early-abort safety net from plan v2 §"Phase 1" so a
# botched YaRN+LoRA recipe doesn't burn the full 10-14h training budget.
#
# ### COUNCIL — read brief §"What good council output looks like":
#  - Q5a/b: are these thresholds (NIAH<80% × 3 ckpts = abort) right?
#  - Quick-probe n=10 vs full RULER n=500: noise risk on n=10. Council may
#    recommend n=20 minimum, or a different probe (MRCR sub-sample, etc.).
#
# Inputs:
#   --ckpt-dir   trainer's adapter-output dir; watches for new step-NNNNN/ subdirs
#   --base-dir   YaRN-patched base model dir (where LoRA merges into)
#   --abort-on   N consecutive bad probes that trigger kill (default 3)
#   --threshold  minimum acceptable NIAH score (default 0.80)
#   --kill-pid   trainer PID to SIGTERM if abort fires
#   --probe-n    NIAH questions per probe (default 10)
#
# Outputs:
#   <ckpt-dir>/probe_log.jsonl  one JSON line per ckpt: step, niah_256k, mscale, ts
#
# ### Not yet implemented:
#   - watchdog or polling for new ckpt-dirs
#   - merge LoRA into base on GPU 1 (peft.PeftModel.from_pretrained + merge_and_unload)
#   - launch lightweight vLLM serving on GPU 1 (gpu_memory_utilization=0.7)
#   - fire 10 NIAH questions via /v1/completions, score, log

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--abort-on", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--kill-pid", type=int, required=True)
    ap.add_argument("--probe-n", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=== phase1_probe_watcher ===")
    print(f"  ckpt_dir   : {args.ckpt_dir}")
    print(f"  base_dir   : {args.base_dir}")
    print(f"  abort_on   : {args.abort_on} consecutive bad probes")
    print(f"  threshold  : {args.threshold}")
    print(f"  kill_pid   : {args.kill_pid}")
    print(f"  probe_n    : {args.probe_n}")

    if args.dry_run:
        print("[dry-run] no-op.")
        return 0

    # TODO COUNCIL-APPROVED — implement watch + merge + probe + abort.
    print("FATAL: probe watcher not yet implemented (council review pending).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
