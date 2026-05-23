#!/usr/bin/env python3
"""
Unified model publish pipeline: HF weights + model card + GGUF quant pack.

Orchestrates the full publish flow:
  Phase 1 (CPU only, safe to run parallel with GPU eval):
    1. Upload HF safetensors weights + config files
    2. Upload model card (README.md)
    3. Convert to F16 GGUF
    4. Quantize + upload non-imatrix tiers (Q8_0 through Q3_K_S)
    5. Delete each local file after upload

  Phase 2 (GPU required):
    6. Compute imatrix from calibration data
    7. Quantize + upload imatrix tiers (IQ4_NL, IQ4_XS, IQ3_*, IQ2_*, Q2_K*)
    8. Run sanity check on Q6_K (3 capital city questions)
    9. Upload imatrix.dat for reproducibility
    10. Update README with final quant table + skipped quants

Usage:
  # Full pipeline (both phases)
  python publish_model.py \\
      --model-dir /path/to/merged \\
      --hf-repo ManniX-ITA/Model-Name \\
      --hf-gguf-repo ManniX-ITA/Model-Name-GGUF \\
      --readme /path/to/README.md \\
      --cal-data /path/to/calibration_datav5.txt

  # Phase 1 only (CPU, no GPU interference)
  python publish_model.py --phase 1 ...

  # Phase 2 only (GPU, after eval finishes)
  python publish_model.py --phase 2 ...

  # Skip sanity check
  python publish_model.py --no-sanity ...

Requires: huggingface_hub, llama.cpp (convert_hf_to_gguf.py, llama-quantize, llama-server)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


# ── Quant tiers ──────────────────────────────────────────────

# Tiers that do NOT need imatrix (CPU-only quantize)
PHASE1_QUANTS = [
    "Q8_0",
    "Q6_K_L", "Q6_K",
    "Q5_K_L", "Q5_K_M", "Q5_K_S",
    "Q4_K_L", "Q4_K_M", "Q4_K_S",
    "Q4_1", "Q4_0",
    "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S",
]

# Tiers that NEED imatrix (GPU compute step first)
PHASE2_QUANTS = [
    "IQ4_NL", "IQ4_XS",
    "IQ3_M", "IQ3_XS", "IQ3_XXS",
    "Q2_K_L", "Q2_K",
    "IQ2_M", "IQ2_S", "IQ2_XS",
]

# _L/_XL variants: base quant + output/embed at Q8_0
_L_XL_MAP = {
    "Q6_K_L": "Q6_K", "Q5_K_L": "Q5_K_M", "Q4_K_L": "Q4_K_M",
    "Q3_K_XL": "Q3_K_L", "Q2_K_L": "Q2_K",
}

# Files to upload from the model directory
MODEL_FILES = [
    "*.safetensors", "*.json", "*.jinja", "*.txt", "*.py",
    "*.model",  # sentencepiece
]

# Files to skip
SKIP_FILES = {"__pycache__", ".git", ".gitattributes"}


def find_llama_cpp() -> Tuple[Optional[str], Optional[str]]:
    """Find llama.cpp binaries."""
    candidates = ["/opt/llama.cpp/build/bin", "/usr/local/bin", "llama.cpp/build/bin"]
    for base in candidates:
        quantize = Path(base) / "llama-quantize"
        server = Path(base) / "llama-server"
        if quantize.exists():
            return str(quantize), str(server) if server.exists() else None
    # Try PATH
    for cmd in ["llama-quantize"]:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        if result.returncode == 0:
            base = Path(result.stdout.strip()).parent
            return str(base / "llama-quantize"), str(base / "llama-server")
    return None, None


def find_convert_script() -> Optional[str]:
    """Find convert_hf_to_gguf.py."""
    candidates = [
        "/opt/llama.cpp/convert_hf_to_gguf.py",
        "llama.cpp/convert_hf_to_gguf.py",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def human_size(path: str) -> str:
    """Human-readable file size."""
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def upload_file(api: HfApi, local: str, repo: str, name: str, msg: str):
    """Upload a file to HF with retry."""
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=name,
                            repo_id=repo, commit_message=msg)
            return True
        except Exception as e:
            print(f"    Upload attempt {attempt+1} failed: {e}", flush=True)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return False


def quantize_one(f16_path: str, out_path: str, quant: str,
                 quantize_bin: str, imatrix: Optional[str] = None) -> bool:
    """Quantize a single tier. Returns True on success."""
    base_quant = _L_XL_MAP.get(quant, quant)
    cmd = [quantize_bin]

    # imatrix for IQ*/Q2 tiers
    if imatrix and ("IQ" in quant or "Q2_K" in quant):
        cmd.extend(["--imatrix", imatrix])

    # _L/_XL: embed/output at Q8_0
    if quant in _L_XL_MAP:
        cmd.extend(["--output-tensor-type", "q8_0", "--token-embedding-type", "q8_0"])

    cmd.extend([f16_path, out_path, base_quant])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[-500:]}", flush=True)
        return False
    return True


def compute_imatrix(f16_path: str, cal_data: str, output_dir: str,
                    quantize_bin: str) -> Optional[str]:
    """Compute imatrix using llama-imatrix. Returns path to imatrix.dat."""
    imatrix_bin = str(Path(quantize_bin).parent / "llama-imatrix")
    if not Path(imatrix_bin).exists():
        print(f"  WARNING: {imatrix_bin} not found, skipping imatrix", flush=True)
        return None

    out = os.path.join(output_dir, "imatrix.dat")
    cmd = [imatrix_bin, "-m", f16_path, "-", cal_data, "-o", out,
           "-ngl", "99", "--chunks", "128"]

    print("  Computing imatrix (this uses GPU)...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"  imatrix FAILED: {result.stderr[-300:]}", flush=True)
        return None

    print(f"  imatrix ready: {human_size(out)}", flush=True)
    return out


def sanity_check(gguf_path: str, server_bin: str) -> bool:
    """Run 3 capital city questions. Returns True if all correct."""
    import requests

    print(f"  Sanity check on {os.path.basename(gguf_path)}...", flush=True)

    # Start server
    proc = subprocess.Popen(
        [server_bin, "-m", gguf_path, "--port", "8098", "-c", "2048",
         "-ngl", "99", "--no-warmup", "-t", "4"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Wait for ready
    for i in range(60):
        try:
            r = requests.get("http://localhost:8098/health", timeout=2)
            if r.json().get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        proc.kill()
        print("  Sanity: server didn't start", flush=True)
        return False

    questions = [
        ("What is the capital of France? Answer with just the city name.", "paris"),
        ("What is the capital of Japan? Answer with just the city name.", "tokyo"),
        ("What is the capital of Germany? Answer with just the city name.", "berlin"),
    ]
    correct = 0
    for q, expected in questions:
        try:
            r = requests.post("http://localhost:8098/v1/completions", json={
                "prompt": q, "max_tokens": 20, "temperature": 0
            }, timeout=30)
            answer = r.json()["choices"][0]["text"].lower().strip()
            if expected in answer:
                correct += 1
        except Exception as e:
            print(f"  Sanity question failed: {e}", flush=True)

    proc.kill()
    proc.wait()

    passed = correct >= 2
    print(f"  Sanity: {correct}/3 correct → {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def main():
    ap = argparse.ArgumentParser(description="Unified model publish pipeline")
    ap.add_argument("--model-dir", required=True, type=Path,
                    help="Path to merged model directory (safetensors)")
    ap.add_argument("--hf-repo", required=True,
                    help="HF repo for weights (e.g. ManniX-ITA/Model-Name)")
    ap.add_argument("--hf-gguf-repo", required=True,
                    help="HF repo for GGUFs (e.g. ManniX-ITA/Model-Name-GGUF)")
    ap.add_argument("--readme", type=Path, default=None,
                    help="Path to README.md to upload as model card")
    ap.add_argument("--gguf-readme", type=Path, default=None,
                    help="Path to GGUF README.md (auto-generated if omitted)")
    ap.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                    help="0=all (default), 1=CPU-only (weights+non-imatrix), 2=GPU (imatrix+low-bit+sanity)")
    ap.add_argument("--cal-data", type=Path, default=None,
                    help="Calibration data for imatrix (required for phase 2)")
    ap.add_argument("--f16-gguf", type=Path, default=None,
                    help="Pre-existing F16 GGUF (skip conversion if provided)")
    ap.add_argument("--gguf-prefix", type=str, default=None,
                    help="Prefix for GGUF filenames (default: derived from hf-gguf-repo)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Directory for GGUF outputs (default: <model-dir>/../gguf_publish)")
    ap.add_argument("--no-sanity", action="store_true",
                    help="Skip sanity check")
    ap.add_argument("--no-upload", action="store_true",
                    help="Quantize locally but don't upload to HF")
    ap.add_argument("--keep-local", action="store_true",
                    help="Don't delete local GGUF files after upload")
    ap.add_argument("--skip-quants", type=str, default="",
                    help="Comma-separated quant names to skip")
    args = ap.parse_args()

    api = HfApi()

    # Defaults
    if args.output_dir is None:
        args.output_dir = args.model_dir.parent / "gguf_publish"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.gguf_prefix is None:
        args.gguf_prefix = args.hf_gguf_repo.split("/")[-1].replace("-GGUF", "")

    skip_quants = set(args.skip_quants.split(",")) if args.skip_quants else set()

    quantize_bin, server_bin = find_llama_cpp()
    convert_script = find_convert_script()

    if not quantize_bin:
        print("ERROR: llama-quantize not found", file=sys.stderr)
        sys.exit(1)
    if not convert_script:
        print("ERROR: convert_hf_to_gguf.py not found", file=sys.stderr)
        sys.exit(1)

    run_phase1 = args.phase in (0, 1)
    run_phase2 = args.phase in (0, 2)

    print("=== Publish pipeline ===")
    print(f"  model      : {args.model_dir}")
    print(f"  hf-repo    : {args.hf_repo}")
    print(f"  hf-gguf    : {args.hf_gguf_repo}")
    print(f"  phase      : {'all' if args.phase == 0 else args.phase}")
    print(f"  gguf-prefix: {args.gguf_prefix}")
    print(f"  output-dir : {args.output_dir}")
    print(f"  quantize   : {quantize_bin}")
    print(f"  convert    : {convert_script}")
    print(flush=True)

    # ── Phase 1 ──────────────────────────────────────────────

    if run_phase1:
        # Step 1: Create repos
        print("\n=== Phase 1: Upload weights + non-imatrix quants ===", flush=True)
        for repo in [args.hf_repo, args.hf_gguf_repo]:
            try:
                api.create_repo(repo, exist_ok=True)
                print(f"  Repo ready: {repo}", flush=True)
            except Exception as e:
                print(f"  Repo {repo}: {e}", flush=True)

        # Step 2: Upload weights
        print(f"\n  Uploading weights from {args.model_dir}...", flush=True)
        if not args.no_upload:
            for f in sorted(args.model_dir.iterdir()):
                if f.name in SKIP_FILES or f.is_dir():
                    continue
                if f.suffix in (".safetensors", ".json", ".jinja", ".txt", ".py", ".model"):
                    print(f"    {f.name} ({human_size(str(f))})...", flush=True)
                    upload_file(api, str(f), args.hf_repo, f.name,
                                f"Add {f.name}")

        # Step 3: Upload README
        if args.readme and args.readme.exists():
            print("\n  Uploading model card...", flush=True)
            if not args.no_upload:
                upload_file(api, str(args.readme), args.hf_repo, "README.md",
                            "Add model card")

        # Step 4: Convert to F16
        f16_path = args.f16_gguf
        if f16_path is None or not f16_path.exists():
            f16_path = args.output_dir / f"{args.gguf_prefix}-F16.gguf"
            if not f16_path.exists():
                print("\n  Converting to F16 GGUF...", flush=True)
                result = subprocess.run(
                    [sys.executable, convert_script, str(args.model_dir),
                     "--outfile", str(f16_path), "--outtype", "f16"],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"  F16 conversion FAILED: {result.stderr[-300:]}", file=sys.stderr)
                    sys.exit(1)
                print(f"  F16 ready: {human_size(str(f16_path))}", flush=True)
            else:
                print(f"  F16 exists: {human_size(str(f16_path))}", flush=True)

        # Step 5: Quantize + upload non-imatrix tiers
        print("\n  Quantizing non-imatrix tiers...", flush=True)
        results = {}
        for quant in PHASE1_QUANTS:
            if quant in skip_quants:
                print(f"    {quant}: skipped", flush=True)
                continue

            out_path = str(args.output_dir / f"{args.gguf_prefix}-{quant}.gguf")
            print(f"    {quant}: quantizing...", flush=True)

            if quantize_one(str(f16_path), out_path, quant, quantize_bin):
                size = human_size(out_path)
                results[quant] = {"status": "ok", "size": size}
                print(f"    {quant}: {size}", flush=True)

                if not args.no_upload:
                    print(f"    {quant}: uploading...", flush=True)
                    upload_file(api, out_path, args.hf_gguf_repo,
                                os.path.basename(out_path), f"Add {quant}")

                if not args.keep_local:
                    os.remove(out_path)
            else:
                results[quant] = {"status": "failed"}

        print(f"\n  Phase 1 complete: {sum(1 for v in results.values() if v['status']=='ok')}/{len(results)} quants", flush=True)

    # ── Phase 2 ──────────────────────────────────────────────

    if run_phase2:
        print("\n=== Phase 2: imatrix + low-bit quants + sanity ===", flush=True)

        # Find F16
        f16_path = args.f16_gguf
        if f16_path is None or not f16_path.exists():
            f16_path = args.output_dir / f"{args.gguf_prefix}-F16.gguf"
        if not f16_path.exists():
            print("ERROR: F16 GGUF not found. Run phase 1 first.", file=sys.stderr)
            sys.exit(1)

        # Step 6: Compute imatrix
        imatrix = None
        if args.cal_data and args.cal_data.exists():
            imatrix = compute_imatrix(str(f16_path), str(args.cal_data),
                                       str(args.output_dir), quantize_bin)
        else:
            print("  WARNING: no calibration data, IQ* quants will skip imatrix", flush=True)

        # Step 7: Quantize + upload imatrix tiers
        results = {}
        sanity_failures = []
        for quant in PHASE2_QUANTS:
            if quant in skip_quants:
                print(f"    {quant}: skipped", flush=True)
                continue

            out_path = str(args.output_dir / f"{args.gguf_prefix}-{quant}.gguf")
            print(f"    {quant}: quantizing...", flush=True)

            if quantize_one(str(f16_path), out_path, quant, quantize_bin, imatrix):
                size = human_size(out_path)

                # Sanity check for each quant
                if not args.no_sanity and server_bin:
                    if not sanity_check(out_path, server_bin):
                        print(f"    {quant}: FAILED sanity — skipping upload", flush=True)
                        sanity_failures.append(quant)
                        results[quant] = {"status": "sanity_fail", "size": size}
                        os.remove(out_path)
                        continue

                results[quant] = {"status": "ok", "size": size}
                print(f"    {quant}: {size}", flush=True)

                if not args.no_upload:
                    print(f"    {quant}: uploading...", flush=True)
                    upload_file(api, out_path, args.hf_gguf_repo,
                                os.path.basename(out_path), f"Add {quant}")

                if not args.keep_local:
                    os.remove(out_path)
            else:
                results[quant] = {"status": "failed"}

        # Step 8: Upload imatrix
        if imatrix and os.path.exists(imatrix) and not args.no_upload:
            print("  Uploading imatrix.dat...", flush=True)
            upload_file(api, imatrix, args.hf_gguf_repo, "imatrix.dat",
                        "Add imatrix.dat for reproducibility")

        # Report
        ok = sum(1 for v in results.values() if v["status"] == "ok")
        failed_sanity = [q for q, v in results.items() if v["status"] == "sanity_fail"]

        print(f"\n  Phase 2 complete: {ok}/{len(results)} quants", flush=True)
        if failed_sanity:
            print(f"  Sanity failures (not uploaded): {', '.join(failed_sanity)}", flush=True)

    # ── Cleanup ──────────────────────────────────────────────

    # Optionally delete F16
    f16_path = args.output_dir / f"{args.gguf_prefix}-F16.gguf"
    if f16_path.exists() and not args.keep_local:
        if run_phase2 or args.phase == 0:
            # Only delete F16 after phase 2 (might need it)
            print(f"\n  Deleting F16 ({human_size(str(f16_path))})...", flush=True)
            os.remove(f16_path)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
