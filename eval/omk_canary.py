"""omk_canary.py — orchestrator for the EVAL_PROTOCOL v3 canary regime.

Two-layer canary:
  1. Structural rules (model-agnostic) — applied to samples_*.jsonl after the
     anchor bench runs. Catches parser/generation breakage regardless of model.
  2. Reference anchor (model-specific) — compares scored result to recorded
     expectations in stack_anchors.yaml. Catches stack-level scoring drift.

Usage:
    omk_canary.py --stack eval/stack.lock.yaml \
                  --anchor-model google/gemma-4-26B-A4B-it \
                  --family gemma-4-26B-A4B \
                  --out eval_results/canary/<stack>_<model>_<ts>/

Exit codes: 0 ALL_PASS, 2 ANY_FAIL, 3 SETUP_ERROR.
Runs the 3 sub-benches via omk_eval.py + applies structural_canary to each
samples file + diffs the score against stack_anchors.yaml.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required (pip install pyyaml)", file=sys.stderr)
    sys.exit(3)

OMK_EVAL = Path(__file__).parent / "omk_eval.py"
STRUCT   = Path(__file__).parent / "structural_canary.py"
ANCHOR   = Path(__file__).parent / "templates" / "anchor30.yaml"
ANCHORS  = Path(__file__).parent / "stack_anchors.yaml"

def load_yaml(p): return yaml.safe_load(Path(p).read_text())

def run_subbench(*, sub_name, parent_template, indices, model_dir, served_name,
                 out_dir: Path, backend="vllm"):
    """Run one anchor sub-bench by invoking omk_eval.py with a parent template
    override (--limit-indices). Writes results under out_dir/<sub_name>/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(OMK_EVAL),
        "--template", parent_template,
        "--model-dir", str(model_dir),
        "--served-name", served_name,
        "--results-dir", str(out_dir / sub_name),
        "--backend", backend,
        "--limit-indices", ",".join(map(str, indices)),
    ]
    print(f"  → {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    return rc

def find_summary(d: Path):
    cands = list(d.rglob("summary.json"))
    return cands[0] if cands else None

def find_samples(d: Path):
    cands = list(d.rglob("samples_*.jsonl"))
    return cands[0] if cands else None

def structural_check(samples_path: Path):
    """Invoke structural_canary.py; returns (passed, report_dict)."""
    cmd = [sys.executable, str(STRUCT), str(samples_path), "--json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout.strip():
        rep = json.loads(p.stdout)[0]
        return rep["passed"], rep
    return False, {"error": p.stderr.strip()}

def anchor_check(score, expected, tolerance):
    if score is None or expected is None:
        return False, "missing score"
    delta = abs(score - expected)
    return delta <= tolerance, f"score={score:.4f} expected={expected:.4f} Δ={delta:.4f} tol={tolerance:.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack",          required=True, help="path to stack.lock.yaml")
    ap.add_argument("--anchor-model",   required=True, help="path to NVFP4A16 (or BF16) model dir")
    ap.add_argument("--served-name",    required=True, help="vLLM served-model name")
    ap.add_argument("--family",         required=True, help="key into stack_anchors.yaml (e.g. gemma-4-26B-A4B)")
    ap.add_argument("--out",            required=True, help="output dir for this canary run")
    ap.add_argument("--backend",        default="vllm", choices=["vllm", "llama_cpp"])
    ap.add_argument("--anchors-file",   default=str(ANCHORS))
    ap.add_argument("--anchor-template",default=str(ANCHOR))
    ap.add_argument("--skip-structural", action="store_true")
    ap.add_argument("--skip-anchor",     action="store_true")
    args = ap.parse_args()

    stack = load_yaml(args.stack)
    stack_key = f"{stack['name']}@{stack['version']}"
    anchors = load_yaml(args.anchors_file)
    family_block = anchors["stacks"].get(stack_key, {}).get(args.family)
    if not family_block:
        print(f"ERROR: no anchor expectations for {stack_key} × {args.family}", file=sys.stderr)
        print(f"  add them to {args.anchors_file} before promoting this stack", file=sys.stderr)
        sys.exit(3)

    bench_cfg = load_yaml(args.anchor_template)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    started = datetime.now().isoformat()
    print(f"== omk_canary.py == {started}")
    print(f"   stack:        {stack_key}")
    print(f"   anchor model: {args.anchor_model}")
    print(f"   family:       {args.family}")
    print(f"   out:          {out_root}")

    # Phase 1: run each sub-bench
    sub_results = {}
    for sb in bench_cfg["sub_benches"]:
        print(f"\n=== sub-bench: {sb['name']} (n={len(sb['selection']['indices'])}) ===")
        rc = run_subbench(
            sub_name        = sb["name"],
            parent_template = sb["parent_template"],
            indices         = sb["selection"]["indices"],
            model_dir       = args.anchor_model,
            served_name     = args.served_name,
            out_dir         = out_root,
            backend         = args.backend,
        )
        sub_results[sb["name"]] = {"rc": rc, "parent": sb["parent_template"]}

    # Phase 2: structural canary on every samples file produced
    if not args.skip_structural:
        print("\n=== structural canary ===")
        for name, info in sub_results.items():
            sp = find_samples(out_root / name)
            if not sp:
                info["structural"] = {"passed": False, "error": "no samples file"}
                continue
            passed, rep = structural_check(sp)
            info["structural"] = {"passed": passed, "report": rep}
            print(f"  {name}: structural {'PASS' if passed else 'FAIL'}")

    # Phase 3: anchor expectation check
    if not args.skip_anchor:
        print("\n=== anchor expectation check ===")
        # Map sub-bench → anchors key (strip 'anchor_' prefix, strip trailing '_10', map)
        ANCHOR_KEY = {
            "anchor_gpqa_10":   "gpqa_diamond",
            "anchor_aime_10":   "aime24",
            "anchor_ifeval_10": "ifeval",
        }
        for name, info in sub_results.items():
            ak = ANCHOR_KEY.get(name)
            if not ak or ak not in family_block["expected"]:
                info["anchor"] = {"passed": True, "note": "no anchor recorded — skipped"}
                continue
            sp = find_summary(out_root / name)
            if not sp:
                info["anchor"] = {"passed": False, "error": "no summary.json"}
                continue
            summary = json.loads(sp.read_text())
            score = summary.get("score")
            exp_block = family_block["expected"][ak]
            passed, msg = anchor_check(score, exp_block["score"], exp_block["tolerance"])
            info["anchor"] = {"passed": passed, "msg": msg}
            print(f"  {name}: {msg} → {'PASS' if passed else 'FAIL'}")

    # Verdict + report
    overall = all(
        info.get("rc") == 0
        and info.get("structural", {}).get("passed", True)
        and info.get("anchor", {}).get("passed", True)
        for info in sub_results.values()
    )

    report = {
        "stack": stack_key,
        "anchor_model": str(args.anchor_model),
        "family": args.family,
        "started": started,
        "finished": datetime.now().isoformat(),
        "overall_pass": overall,
        "sub_results": sub_results,
    }
    (out_root / "canary_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n=== VERDICT: {'ALL_PASS' if overall else 'FAIL'} ===")
    print(f"   report: {out_root / 'canary_report.json'}")
    sys.exit(0 if overall else 2)

if __name__ == "__main__":
    main()
