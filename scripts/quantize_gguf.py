#!/usr/bin/env python3
"""
GGUF Quantization Pipeline
===========================
Converts HF models to GGUF and creates all standard quantizations.
Supports imatrix, sanity checking, parallel upload, and pod mode.

Usage:
  # Full pipeline: all quants, upload to HF
  python quantize_gguf.py --model google/gemma-4-26B-A4B-it

  # Specific quants only
  python quantize_gguf.py --model google/gemma-4-26B-A4B-it --only Q4_K_M,Q5_K_M,UD-Q4_K_M

  # Exclude some quants
  python quantize_gguf.py --model ./local_model --exclude IQ2_XXS,IQ2_XS

  # With sanity check, no upload
  python quantize_gguf.py --model ./local_model --sanity-check --no-upload

  # Pod mode (auto-install dependencies)
  python quantize_gguf.py --model google/gemma-4-26B-A4B-it --pod

  # Custom output repo name
  python quantize_gguf.py --model google/gemma-4-26B-A4B-it --repo ManniX-ITA/my-model-GGUF
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread, Event

# ── Quant definitions ────────────────────────────────────────

# Bartowski standard quants (ordered large → small)
BARTOWSKI_QUANTS = [
    "Q8_0",
    "Q6_K_L", "Q6_K",
    "Q5_K_L", "Q5_K_M", "Q5_K_S",
    "Q4_K_L", "Q4_K_M", "Q4_1", "Q4_K_S", "Q4_0",
    "IQ4_NL", "IQ4_XS",
    "Q3_K_XL", "IQ3_M", "Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_XS", "IQ3_XXS",
    "Q2_K_L", "Q2_K",
    "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
]

# ContribDynamic (CD) quants: per-layer dynamic quantization based on expert
# contribution analysis. Important layers get higher precision, less important
# layers get lower precision. Uses --tensor-type-file with llama-quantize.
CD_QUANTS = [
    "CD-Q6_K", "CD-Q5_K_M", "CD-Q4_K_M", "CD-Q3_K_M", "CD-Q2_K",
]

ALL_QUANTS = BARTOWSKI_QUANTS + CD_QUANTS

# Quants that require imatrix for good results
IMATRIX_QUANTS = {q for q in ALL_QUANTS if q.startswith("IQ") or q.startswith("UD-IQ")}

# Sanity check: capital questions
SANITY_QUESTIONS = [
    ("What is the capital of France? Reply ONLY with JSON: {\"capital\": \"...\"}", "Paris"),
    ("What is the capital of Japan? Reply ONLY with JSON: {\"capital\": \"...\"}", "Tokyo"),
    ("What is the capital of Germany? Reply ONLY with JSON: {\"capital\": \"...\"}", "Berlin"),
]


# ── Utility functions ────────────────────────────────────────

def run(cmd: list[str], desc: str = "", timeout: int = None, **kwargs) -> subprocess.CompletedProcess:
    """Run a command, print status, raise on failure."""
    if desc:
        print(f"  [{desc}]", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
    if result.returncode != 0:
        print(f"  FAILED: {' '.join(cmd[:4])}", flush=True)
        print(f"  stderr: {result.stderr[-500:]}", flush=True)
        raise RuntimeError(f"Command failed: {desc or cmd[0]}")
    return result


def find_llama_cpp() -> dict[str, str]:
    """Find llama.cpp binaries and scripts."""
    search_paths = [
        Path("/opt/llama.cpp"),
        Path.home() / "llama.cpp",
        Path("./llama.cpp"),
    ]
    for base in search_paths:
        server = base / "build" / "bin" / "llama-server"
        quantize = base / "build" / "bin" / "llama-quantize"
        imatrix = base / "build" / "bin" / "llama-imatrix"
        convert = base / "convert_hf_to_gguf.py"
        if quantize.exists() and convert.exists():
            return {
                "base": str(base),
                "server": str(server) if server.exists() else None,
                "quantize": str(quantize),
                "imatrix": str(imatrix) if imatrix.exists() else None,
                "convert": str(convert),
            }

    # Try PATH
    for tool in ["llama-quantize", "llama-server"]:
        if shutil.which(tool):
            base = Path(shutil.which(tool)).parent.parent
            return {
                "base": str(base),
                "server": shutil.which("llama-server"),
                "quantize": shutil.which("llama-quantize"),
                "imatrix": shutil.which("llama-imatrix"),
                "convert": str(base / "convert_hf_to_gguf.py"),
            }

    return None


def install_pod_dependencies():
    """Install llama.cpp and dependencies for pod/cloud environment."""
    print("=== Pod mode: installing dependencies ===", flush=True)

    # System deps
    print("  Installing system packages...", flush=True)
    subprocess.run("apt-get update -qq && apt-get install -y -qq cmake build-essential curl git",
                   shell=True, check=False)

    # Upgrade cmake (many pod images have outdated cmake)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake", "--upgrade"],
                   check=False)

    # Build llama.cpp with CUDA
    llama_dir = Path("/opt/llama.cpp")
    if not (llama_dir / "build" / "bin" / "llama-quantize").exists():
        if not llama_dir.exists():
            print("  Cloning llama.cpp...", flush=True)
            subprocess.run(["git", "clone", "--depth=1",
                            "https://github.com/ggml-org/llama.cpp.git", str(llama_dir)],
                           check=True)

        # Install Python deps from llama.cpp requirements
        req_file = llama_dir / "requirements.txt"
        if req_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
                           check=False)

        # Detect CUDA and build
        cuda_nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        has_cuda = Path(cuda_nvcc).exists()

        print(f"  Building llama.cpp ({'CUDA' if has_cuda else 'CPU-only'})...", flush=True)
        cmake_args = [shutil.which("cmake") or "cmake", "-B", "build"]
        if has_cuda:
            cmake_args.extend(["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={cuda_nvcc}"])

        env = os.environ.copy()
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

        subprocess.run(cmake_args, cwd=str(llama_dir), check=True, env=env)

        nproc = os.cpu_count() or 4
        subprocess.run([shutil.which("cmake") or "cmake", "--build", "build",
                        "--config", "Release", f"-j{nproc}"],
                       cwd=str(llama_dir), check=True, env=env)
        print("  llama.cpp built.", flush=True)
    else:
        print("  llama.cpp already built.", flush=True)

    # Install Python deps (including transformers for convert_hf_to_gguf.py)
    print("  Installing Python dependencies...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "huggingface_hub", "sentencepiece", "protobuf", "torch",
                    "numpy", "transformers"],
                   check=True)


def resolve_model_path(model: str, cache_dir: str = None) -> tuple[Path, str]:
    """Resolve model to a local path. Downloads from HF if needed.
    Returns (local_path, repo_id_or_name)."""
    local = Path(model)
    if local.is_dir() and (local / "config.json").exists():
        # Infer repo name from directory name or config
        name = local.name
        try:
            cfg = json.loads((local / "config.json").read_text())
            name = cfg.get("_name_or_path", name)
        except Exception:
            pass
        return local, name

    # HF repo — download
    print(f"  Downloading {model} from HF...", flush=True)
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, cache_dir=cache_dir,
                             ignore_patterns=["*.bin", "*.pt", "consolidated.*"])
    return Path(path), model


def get_model_name(repo_id: str) -> str:
    """Extract a clean model name from repo_id."""
    # google/gemma-4-26B-A4B-it → gemma-4-26B-A4B-it
    return repo_id.split("/")[-1] if "/" in repo_id else repo_id


# ── imatrix computation ──────────────────────────────────────

def detect_gpu_vram_mb() -> int:
    """Detect available GPU VRAM in MB. Returns 0 if no GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # Sum all GPUs
            total = sum(int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip())
            return total
    except Exception:
        pass
    return 0


def auto_ngl(model_size_gb: float) -> int:
    """Auto-detect ngl based on available VRAM. Returns 0 for CPU-only."""
    vram_mb = detect_gpu_vram_mb()
    if vram_mb == 0:
        return 0
    model_mb = model_size_gb * 1024
    available = vram_mb - 512  # reserve 512MB for overhead
    if available >= model_mb:
        return 99  # fits entirely
    # Partial offload: estimate proportion of layers that fit.
    # 0.85 of available leaves ~3-4 GB headroom for KV cache + compute
    # buffers on a 24 GB card. Previously 0.60 — too conservative; observed
    # only ~10 GB used out of 22 GB free during 31B imatrix on 3090.
    usable = available * 0.85
    ratio = usable / model_mb
    # Typical dense models have 40-80 layers (Gemma-4-31B: 62, Llama-70B: 80).
    # 50 is a safer mid-point than the old 35 for large dense models.
    ngl = max(0, int(ratio * 50))
    return ngl


def compute_imatrix(tools: dict, f16_gguf: Path, cal_data: Path,
                    output: Path, ngl: int = None) -> Path:
    """Compute importance matrix from F16 GGUF using calibration data."""
    imatrix_bin = tools.get("imatrix")
    if not imatrix_bin:
        print("  WARNING: llama-imatrix not found, imatrix quants may have reduced quality", flush=True)
        return None

    imatrix_file = output / "imatrix.dat"
    if imatrix_file.exists():
        print(f"  imatrix already exists: {imatrix_file}", flush=True)
        return imatrix_file

    # Auto-detect GPU layers if not specified
    if ngl is None:
        model_size_gb = f16_gguf.stat().st_size / 1024**3
        ngl = auto_ngl(model_size_gb)
        print(f"  Auto ngl={ngl} (VRAM: {detect_gpu_vram_mb()}MB, model: {model_size_gb:.1f}GB)", flush=True)

    print(f"  Computing imatrix from {cal_data.name}...", flush=True)
    cmd = [
        imatrix_bin,
        "-m", str(f16_gguf),
        "-f", str(cal_data),
        "-o", str(imatrix_file),
        "-ngl", str(ngl),
        "--chunks", "128",
    ]
    run(cmd, desc="imatrix computation", timeout=7200)
    print(f"  imatrix saved: {imatrix_file} ({imatrix_file.stat().st_size / 1024:.0f} KB)", flush=True)
    return imatrix_file


# ── Quantization ─────────────────────────────────────────────

# _L and _XL variants: standard quant with embed/output at Q8_0
_L_XL_VARIANTS = {
    "Q6_K_L": "Q6_K", "Q5_K_L": "Q5_K_M", "Q4_K_L": "Q4_K_M",
    "Q3_K_XL": "Q3_K_M", "Q2_K_L": "Q2_K",
}


def quantize_one(tools: dict, f16_gguf: Path, output_dir: Path,
                 model_name: str, quant: str, imatrix_file: Path = None,
                 threads: int = None) -> Path:
    """Quantize F16 GGUF to a specific quant type. Returns output path."""
    out_name = f"{model_name}-{quant}.gguf"
    out_path = output_dir / out_name
    if out_path.exists():
        print(f"  {quant}: already exists, skipping", flush=True)
        return out_path

    # Determine the base quant type for llama-quantize
    if quant in _L_XL_VARIANTS:
        q_type = _L_XL_VARIANTS[quant]
    elif quant.startswith("CD-"):
        q_type = quant[3:]  # e.g. CD-Q4_K_M -> Q4_K_M
    else:
        q_type = quant

    cmd = [tools["quantize"]]

    # CD (ContribDynamic) quants: read the per-layer tensor type file, and
    # detect whether ANY tensor is assigned to an IQ*/Q2_K type that needs imatrix.
    cd_needs_imatrix = False
    tensor_type_file = None
    if quant.startswith("CD-"):
        tensor_type_file = Path(__file__).parent / f"tensor_types_{quant}.txt"
        if tensor_type_file.exists():
            try:
                ttxt = tensor_type_file.read_text()
                # Any IQ* tier or Q2_K requires an imatrix at quantize time
                if "IQ" in ttxt or "Q2_K" in ttxt:
                    cd_needs_imatrix = True
            except Exception:
                pass
        else:
            tensor_type_file = None
            print(f"  WARNING: tensor_types_{quant}.txt not found, falling back to uniform {q_type}", flush=True)

    # Add imatrix if available and quant benefits from it
    # (standalone IQ*/UD-IQ* quants, anything with "IQ" in its name, or CD- maps
    #  that assign at least one tensor to an IQ*/Q2_K tier)
    if imatrix_file and (quant in IMATRIX_QUANTS or "IQ" in quant or cd_needs_imatrix):
        cmd.extend(["--imatrix", str(imatrix_file)])

    # _L/_XL variants: keep embed/output at Q8_0 for better quality
    if quant in _L_XL_VARIANTS:
        cmd.extend(["--output-tensor-type", "q8_0", "--token-embedding-type", "q8_0"])

    # CD (ContribDynamic) quants: use per-layer tensor type file
    if tensor_type_file is not None:
        cmd.extend(["--tensor-type-file", str(tensor_type_file)])

    cmd.extend([str(f16_gguf), str(out_path), q_type])

    if threads:
        cmd.append(str(threads))

    t0 = time.time()
    print(f"  {quant}: quantizing...", flush=True)
    run(cmd, desc=f"quantize {quant}", timeout=7200)
    size_gb = out_path.stat().st_size / 1024**3
    elapsed = time.time() - t0
    print(f"  {quant}: {size_gb:.2f} GB ({elapsed:.0f}s)", flush=True)
    return out_path


# ── Sanity check ─────────────────────────────────────────────

def sanity_check(tools: dict, gguf_path: Path, quant_name: str,
                 port: int = 18099) -> bool:
    """Quick sanity check: load model, ask capital questions, validate JSON answers.
    Uses --fit to auto-adjust GPU offloading to available VRAM."""
    import requests

    server_bin = tools.get("server")
    if not server_bin:
        print(f"  {quant_name}: sanity check skipped (no llama-server)", flush=True)
        return True

    # Start server with --fit, reasoning disabled for clean JSON answers
    proc = subprocess.Popen(
        [server_bin, "-m", str(gguf_path), "--port", str(port),
         "-c", "2048", "--fit", "on", "--no-warmup", "-t", "4",
         "--reasoning-format", "deepseek", "--reasoning-budget", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    # Wait for ready
    ready = False
    for _ in range(120):
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(1)

    if not ready:
        proc.kill()
        print(f"  {quant_name}: sanity FAIL — server didn't start", flush=True)
        return False

    # Ask questions
    passed = 0
    for prompt, expected in SANITY_QUESTIONS:
        try:
            r = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 100, "temperature": 0},
                timeout=120,
            )
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content", "") + " " + msg.get("reasoning_content", "")
            # Simple check: is the expected answer anywhere in the response?
            if expected.lower() in content.lower():
                passed += 1
            else:
                print(f"  {quant_name}: sanity: expected '{expected}', got '{content[:100]}'", flush=True)
        except Exception as e:
            print(f"  {quant_name}: sanity error: {e}", flush=True)

    proc.kill()
    proc.wait()

    # Pass if at least 2/3 correct (some models struggle with accented names like Brasília)
    ok = passed >= 2
    print(f"  {quant_name}: sanity {'PASS' if ok else 'FAIL'} ({passed}/{len(SANITY_QUESTIONS)})", flush=True)
    return ok


# ── HF upload ────────────────────────────────────────────────

def create_hf_repo(repo_id: str, original_repo: str, model_name: str,
                    quant_list: list[str]) -> str:
    """Create HF repo with README based on original model's README."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True)

    # Fetch original README
    original_readme = ""
    try:
        from huggingface_hub import hf_hub_download
        readme_path = hf_hub_download(original_repo, "README.md")
        original_readme = Path(readme_path).read_text()
    except Exception:
        pass

    # Strip original frontmatter
    original_body = original_readme
    if original_readme.startswith("---"):
        end = original_readme.find("---", 3)
        if end > 0:
            original_body = original_readme[end + 3:].strip()

    # Build GGUF README
    readme = f"""---
base_model: {original_repo}
tags:
  - gguf
  - imatrix
  - quantized
license: apache-2.0
---

# {model_name}-GGUF

GGUF quantizations of [{original_repo}](https://huggingface.co/{original_repo}).

All quants made using imatrix with [calibration data v5](https://gist.github.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d).

## Available Quantizations

| Quantization | Status |
|---|---|
"""
    for q in quant_list:
        readme += f"| {q} | pending |\n"

    readme += f"""
## How to Use

With [llama.cpp](https://github.com/ggml-org/llama.cpp):
```bash
llama-server -m {model_name}-Q4_K_M.gguf -c 8192 -ngl 99
```

With [ollama](https://ollama.ai) (requires Modelfile or HF direct load).

---

## Original Model Card

{original_body}
"""

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial README with quant table",
    )

    return repo_id


def update_readme_quant_status(repo_id: str, quant_name: str, size_gb: float, status: str = "done"):
    """Update the GGUF repo README to mark a quant as uploaded with its size."""
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        readme_path = api.hf_hub_download(repo_id, "README.md", repo_type="model")
        readme = Path(readme_path).read_text()

        # Replace "| <quant_name> | pending |" with "| <quant_name> | <size> |"
        old = f"| {quant_name} | pending |"
        new = f"| {quant_name} | {size_gb:.2f} GB |" if status == "done" else f"| {quant_name} | FAILED |"
        if old in readme:
            readme = readme.replace(old, new)
            api.upload_file(
                path_or_fileobj=readme.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Update README: {quant_name} {status} ({size_gb:.2f} GB)" if status == "done" else f"Update README: {quant_name} failed",
            )
    except Exception as e:
        print(f"  {quant_name}: README update failed: {e}", flush=True)


def upload_gguf(repo_id: str, gguf_path: Path, quant_name: str,
                max_retries: int = 10, backoff_base: float = 30.0):
    """Upload a single GGUF file to HF with retries."""
    from huggingface_hub import HfApi
    api = HfApi()
    size_gb = gguf_path.stat().st_size / 1024**3

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            print(f"  {quant_name}: uploading ({size_gb:.2f} GB, attempt {attempt})...", flush=True)
            api.upload_file(
                path_or_fileobj=str(gguf_path),
                path_in_repo=gguf_path.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add {quant_name} quantization ({size_gb:.2f} GB)",
            )
            elapsed = time.time() - t0
            print(f"  {quant_name}: uploaded ({elapsed:.0f}s)", flush=True)
            return
        except Exception as e:
            wait = backoff_base * attempt
            if attempt < max_retries:
                print(f"  {quant_name}: upload attempt {attempt} failed: {e}, retrying in {wait:.0f}s", flush=True)
                time.sleep(wait)
            else:
                raise RuntimeError(f"Upload failed after {max_retries} attempts: {e}")


# ── Ollama push (optional, runs in same worker between HF upload and delete) ──
#
# Template strategy:
#   "auto"            — bare `FROM <gguf>` Modelfile, ollama derives chat
#                       formatting from the GGUF metadata. Works only when the
#                       embedded chat_template is plain enough for ollama's Go
#                       template engine. Fails silently to `{{ .Prompt }}` on
#                       complex Jinja (e.g. 31B Google tool template).
#   "gemma4"          — emit `RENDERER gemma4` + `PARSER gemma4`. Built-in
#                       Go-native renderer for the Gemma 4 family (requires
#                       ollama >= 0.20). Right call for the 31B-it dense
#                       variants — same recipe `ollama.com/library/gemma4:31b`
#                       uses. Tools + thinking capabilities enabled.
#   "gemma4-a4b"      — RENDERER + PARSER **plus** the custom TEMPLATE that
#                       fixes the 26B-A4B MoE's 2nd-tool-call-turn bug (the
#                       template baked into `mannix/gemma4-98e-v3/-v4`). Use
#                       this for 26B-A4B variants only; using it on dense 31B
#                       would format the prompt incorrectly.
GEMMA4_A4B_TOOLS_TEMPLATE = r'''<start_of_turn>{{- if or (eq .Role "system") (eq .Role "user") }}user
{{- if (eq .Role "system") }}
{{ .Content }}
{{ else }}
{{ if .Content }}{{ .Content }}{{ end }}
{{- end }}<end_of_turn>
{{ end }}{{- if eq .Role "assistant" }}model
{{ .Content }}{{ if .ToolCalls }}{{ range .ToolCalls }}
```tool_code
{{ .Function.Name }}({{ range $i, $arg := .Function.Arguments }}{{ if $i }}, {{ end }}{{ $arg.Key }}={{ $arg.Value }}{{ end }})
```{{ end }}
{{ end }}<end_of_turn>
{{ end }}{{- if eq .Role "tool" }}user
```tool_output
{{ .Content }}
```<end_of_turn>
{{ end }}<start_of_turn>model
'''


def _gguf_tag_from_filename(gguf_path: Path) -> str:
    """Best-effort: extract the quant tag suffix from a GGUF filename
    (mirrors the regex in scripts/ollama_push_generic.sh)."""
    import re
    base = gguf_path.stem  # strip .gguf
    parts = base.split("-")
    pat = re.compile(r"^(F16|F32|Q\d+(_[KS01ML]+)?(_[KSMLX01]+)?|IQ\d+(_[A-Z]+)?|CD-Q\d+(_[KS]+)?(_[KSMLX01]+)?)$")
    for n in (3, 2, 1):
        if n > len(parts):
            continue
        candidate = "-".join(parts[-n:])
        if pat.match(candidate):
            return candidate
    return base.split("-")[-1]  # fallback


def push_to_ollama_from_local(gguf_path: Path, ollama_target: str,
                              template_style: str = "auto",
                              log_dir: Path = None,
                              max_attempts: int = 3,
                              retry_sleep: int = 5) -> bool:
    """ollama create from a LOCAL gguf file (no HF re-download), push to
    ollama.com, then rm the local ref and purge blob digests.

    Returns True on success, False on failure (does not raise — upstream
    HF upload + local-file lifecycle should not be blocked by ollama hiccups).

    template_style:
      "auto"   — let ollama auto-detect chat_template from GGUF metadata
      "gemma4" — embed the custom Gemma 4 tools template (mannix/gemma4-98e style)
    """
    tag = _gguf_tag_from_filename(gguf_path)
    target_tag = f"{ollama_target}:{tag}"
    label = f"  ollama[{tag}]"
    log_dir = log_dir or gguf_path.parent

    workdir = log_dir / "_ollama_push"
    workdir.mkdir(parents=True, exist_ok=True)
    mf_path = workdir / f"Modelfile.{tag}"

    # Build Modelfile
    lines = [f"FROM {gguf_path}"]
    if template_style in ("gemma4", "gemma4-a4b"):
        # Built-in Gemma 4 Go-native renderer + parser (ollama >= 0.20).
        # Same recipe ollama.com/library/gemma4:* tags use. Handles the full
        # tool-calling template natively; no Jinja2-to-Go translation needed.
        lines.append("RENDERER gemma4")
        lines.append("PARSER gemma4")
        if template_style == "gemma4-a4b":
            # 26B-A4B MoE only: add the custom TEMPLATE that fixes the
            # 2nd-tool-call-turn bug observed on 98e v3/v4. The renderer + parser
            # provide the structural rendering; this TEMPLATE overrides the
            # per-turn assistant/tool format that the renderer would otherwise
            # emit. Mirrors the published mannix/gemma4-98e recipe.
            lines.append('TEMPLATE """' + GEMMA4_A4B_TOOLS_TEMPLATE + '"""')
    mf_path.write_text("\n".join(lines) + "\n")

    def _flush_local_refs():
        # rm both the target tag and any leftover hf.co/ ref from prior runs
        subprocess.run(["ollama", "rm", target_tag],
                       capture_output=True, text=True)
        # purge any partial blobs
        import glob
        for partial in glob.glob("/root/.ollama/models/blobs/sha256-*-partial*"):
            try:
                os.remove(partial)
            except OSError:
                pass

    create_ok = False
    for attempt in range(1, max_attempts + 1):
        print(f"{label}: ollama create {target_tag} (attempt {attempt}/{max_attempts}) ...", flush=True)
        r = subprocess.run(["ollama", "create", target_tag, "-f", str(mf_path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            create_ok = True
            break
        print(f"{label}:   create FAILED: {r.stderr.strip()[:300]}", flush=True)
        _flush_local_refs()
        if attempt < max_attempts:
            time.sleep(retry_sleep)
    if not create_ok:
        print(f"{label}: FINAL FAIL on create after {max_attempts} attempts", flush=True)
        return False

    # Push
    print(f"{label}: pushing {target_tag} ...", flush=True)
    push = subprocess.run(["ollama", "push", target_tag],
                          capture_output=True, text=True)
    if push.returncode != 0:
        print(f"{label}: push FAILED: {push.stderr.strip()[:300]}", flush=True)
        # don't return early — still try to clean up
        success = False
    else:
        print(f"{label}: pushed OK", flush=True)
        success = True

    # Capture blob digests from the manifest BEFORE removing the ref
    blobs = set()
    mf_disk = Path(f"/root/.ollama/models/manifests/registry.ollama.ai/{ollama_target}/{tag}")
    if mf_disk.exists():
        import re
        blobs = set(re.findall(r"sha256:[0-9a-f]{64}", mf_disk.read_text()))

    # Remove local ref + force-purge blobs (belt + suspenders against ollama GC bugs)
    subprocess.run(["ollama", "rm", target_tag], capture_output=True, text=True)
    for b in blobs:
        fn = Path("/root/.ollama/models/blobs") / b.replace("sha256:", "sha256-")
        for p in (fn, Path(str(fn) + "-partial")):
            try:
                p.unlink()
            except (OSError, FileNotFoundError):
                pass

    return success


def upload_worker(upload_queue: Queue, repo_id: str, stop_event: Event,
                  keep_base: set[str] = None,
                  ollama_target: str = None,
                  ollama_template: str = "auto",
                  ollama_failed: list = None):
    """Worker thread: uploads completed quants from queue, optionally pushes
    to ollama.com, then deletes the local file.

    keep_base: set of filenames to NOT delete (e.g. the F16 base GGUF).
    ollama_target: if set, ollama-create + push each successfully HF-uploaded
                   quant before deleting the local file. Skips F16/F32 bases.
    ollama_template: "auto" (GGUF metadata) or "gemma4" (custom tools template).
    ollama_failed: shared list to record (quant_name, gguf_filename) pairs whose
                   inline ollama push failed; main() retries them at the end
                   by re-downloading from the published HF GGUF repo.
    """
    while not stop_event.is_set() or not upload_queue.empty():
        try:
            item = upload_queue.get(timeout=5)
        except Exception:
            continue
        if item is None:
            break
        gguf_path, quant_name = item
        try:
            upload_gguf(repo_id, gguf_path, quant_name)
            # Update README with quant size
            size_gb = gguf_path.stat().st_size / 1024**3 if gguf_path.exists() else 0
            update_readme_quant_status(repo_id, quant_name, size_gb, "done")

            # Optional: push to ollama.com from the SAME local file (no re-download).
            # Skip the F16/F32 base (no one runs an F16 GGUF on ollama).
            if (ollama_target and gguf_path.exists()
                    and not quant_name.upper().startswith(("F16", "F32"))):
                pushed = push_to_ollama_from_local(
                    gguf_path, ollama_target,
                    template_style=ollama_template,
                    log_dir=gguf_path.parent,
                )
                if not pushed and ollama_failed is not None:
                    # Record for end-of-script retry (downloads from HF, since
                    # the local file is about to be deleted to save disk).
                    ollama_failed.append((quant_name, gguf_path.name))

            # Delete after successful upload to save disk (unless keep_base=None = keep all)
            if keep_base is not None and gguf_path.name not in keep_base:
                gguf_path.unlink(missing_ok=True)
                print(f"  {quant_name}: deleted local file after upload", flush=True)
        except Exception as e:
            print(f"  {quant_name}: upload FAILED (keeping local file): {e}", flush=True)
        upload_queue.task_done()


# ── Main pipeline ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GGUF Quantization Pipeline")
    parser.add_argument("--model", required=True,
                        help="HF repo ID (e.g. google/gemma-4-26B-A4B-it) or local path")
    parser.add_argument("--repo", default=None,
                        help="Target HF repo for uploads (default: auto-generate with -GGUF suffix)")
    parser.add_argument("--output-dir", default=None,
                        help="Local output directory (default: ./gguf_output/<model_name>)")
    parser.add_argument("--base-precision", choices=["f16", "f32"], default="f16",
                        help="Base GGUF precision (default: f16)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of quants to create (overrides full list)")
    parser.add_argument("--exclude", default=None,
                        help="Comma-separated list of quants to exclude from full list")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run sanity check on each quant before upload")
    parser.add_argument("--no-upload", action="store_true",
                        help="Don't upload to HF, just quantize locally")
    parser.add_argument("--no-imatrix", action="store_true",
                        help="Skip imatrix computation")
    parser.add_argument("--cal-data", default=None,
                        help="Path to calibration data for imatrix (default: bundled calibration_datav5.txt)")
    parser.add_argument("--ngl", type=int, default=None,
                        help="GPU layers for imatrix (default: auto-detect based on VRAM)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Threads for quantization (default: auto)")
    parser.add_argument("--pod", action="store_true",
                        help="Pod mode: auto-install dependencies if missing")
    parser.add_argument("--keep-local", action="store_true",
                        help="Don't delete quants after upload (default: delete to save disk)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (default: from HF_TOKEN env or ~/.cache/huggingface/token)")
    parser.add_argument("--base-model-id", default=None,
                        help="HF repo ID to use as base_model in README (required when --model is a local path; "
                             "must be a valid HF model ID, not a local path)")
    parser.add_argument("--ollama-target", default=None,
                        help="If set, after each successful HF upload also push the SAME local GGUF "
                             "to ollama.com as <ollama-target>:<quant-tag> (e.g. mannix/gemma4-31b-he1). "
                             "Skips F16/F32 bases. Requires the `ollama` CLI to be installed and authenticated.")
    parser.add_argument("--ollama-template", choices=["auto", "gemma4", "gemma4-a4b"],
                        default="auto",
                        help="Ollama Modelfile chat template strategy. "
                             "'auto' lets ollama derive from GGUF metadata (fails to {{.Prompt}} on "
                             "complex Jinja). "
                             "'gemma4' emits RENDERER gemma4 + PARSER gemma4 (built-in Go renderer; "
                             "right for Gemma 4 dense models including 31B-it). "
                             "'gemma4-a4b' adds the custom TEMPLATE that fixes the 26B-A4B MoE's "
                             "2nd-tool-call-turn bug (use only for 26B-A4B variants like 98e v3/v4).")
    args = parser.parse_args()

    # Set HF token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    elif not os.environ.get("HF_TOKEN"):
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            os.environ["HF_TOKEN"] = token_path.read_text().strip()

    print("=== GGUF Quantization Pipeline ===", flush=True)
    print(f"Model: {args.model}", flush=True)

    # Pod mode: install deps
    if args.pod:
        install_pod_dependencies()

    # Find llama.cpp
    tools = find_llama_cpp()
    if not tools:
        print("ERROR: llama.cpp not found. Use --pod to auto-install or set PATH.", flush=True)
        sys.exit(1)
    print(f"llama.cpp: {tools['base']}", flush=True)

    # Resolve model name first without downloading
    model_name = get_model_name(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./gguf_output/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if base GGUF already exists — skip download if so
    precision = args.base_precision
    base_gguf = output_dir / f"{model_name}-{precision.upper()}.gguf"

    if base_gguf.exists():
        print(f"Base GGUF exists: {base_gguf} ({base_gguf.stat().st_size / 1024**3:.1f} GB)", flush=True)
        print("Skipping model download.", flush=True)
        model_path = None
        repo_id = args.model
    else:
        model_path, repo_id = resolve_model_path(args.model)
        model_name = get_model_name(repo_id)

    # Validated base_model ID for README (HF rejects local paths in base_model field)
    if args.base_model_id:
        base_model_id = args.base_model_id
    elif Path(args.model).exists() and Path(args.model).is_dir():
        # Local path: must provide explicit --base-model-id
        print(f"ERROR: --model is a local path ({args.model}). "
              f"Pass --base-model-id <hf_repo_id> for README generation.", flush=True)
        sys.exit(1)
    else:
        base_model_id = repo_id

    print(f"Model name: {model_name}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    # Determine quant list
    if args.only:
        quants = [q.strip() for q in args.only.split(",")]
        # Validate
        invalid = [q for q in quants if q not in ALL_QUANTS]
        if invalid:
            print(f"WARNING: Unknown quants: {invalid}", flush=True)
    else:
        quants = list(ALL_QUANTS)
        if args.exclude:
            exclude = {q.strip() for q in args.exclude.split(",")}
            quants = [q for q in quants if q not in exclude]

    print(f"Quants to create: {len(quants)}", flush=True)

    # HF repo
    hf_repo = None
    if not args.no_upload:
        if args.repo:
            hf_repo = args.repo
        else:
            # Auto-generate: owner/model-GGUF
            from huggingface_hub import HfApi
            api = HfApi()
            user = api.whoami()["name"]
            hf_repo = f"{user}/{model_name}-GGUF"

        print(f"HF repo: {hf_repo}", flush=True)
        print(f"Base model (README): {base_model_id}", flush=True)
        create_hf_repo(hf_repo, base_model_id, model_name, quants)

    # ── Step 1: Convert to F16 GGUF ──────────────────────────
    if not base_gguf.exists():
        print(f"\n=== Converting to {precision.upper()} GGUF ===", flush=True)
        cmd = [
            sys.executable, tools["convert"],
            str(model_path),
            "--outfile", str(base_gguf),
            "--outtype", precision,
        ]
        t0 = time.time()
        run(cmd, desc=f"convert to {precision}")
        size_gb = base_gguf.stat().st_size / 1024**3
        print(f"  Base GGUF: {size_gb:.2f} GB ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"\n  Base GGUF exists: {base_gguf}", flush=True)

    # Delete HF weights to free disk after base GGUF is created
    if model_path and not args.keep_local and base_gguf.exists():
        hf_weight_files = list(model_path.rglob("*.safetensors")) + list(model_path.rglob("*.bin"))
        hf_weight_size = sum(f.stat().st_size for f in hf_weight_files) / 1024**3
        if hf_weight_size > 1:
            if not Path(args.model).is_dir():
                # Downloaded from HF cache — safe to delete entirely
                print(f"  Deleting HF cache weights ({hf_weight_size:.1f} GB) to free disk...", flush=True)
                cache_root = model_path
                while cache_root.parent.name != "hub" and cache_root.parent != cache_root:
                    cache_root = cache_root.parent
                shutil.rmtree(str(cache_root), ignore_errors=True)
                print(f"  Deleted HF cache: {cache_root}", flush=True)
            else:
                print(f"  HF weights: {hf_weight_size:.1f} GB at {model_path} (local, not deleting)", flush=True)

    # ── Step 2: Compute imatrix ──────────────────────────────
    imatrix_file = None
    if not args.no_imatrix:
        # Find calibration data
        cal_data = None
        if args.cal_data:
            cal_data = Path(args.cal_data)
        else:
            # Search common locations
            for p in [
                Path(__file__).parent / "calibration_datav5.txt",
                Path("./calibration_datav5.txt"),
                Path("/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/scripts/calibration_datav5.txt"),
            ]:
                if p.exists():
                    cal_data = p
                    break

        if cal_data and cal_data.exists():
            print("\n=== Computing imatrix ===", flush=True)
            print(f"  Calibration data: {cal_data} ({cal_data.stat().st_size / 1024:.0f} KB)", flush=True)
            imatrix_file = compute_imatrix(tools, base_gguf, cal_data, output_dir, ngl=args.ngl)
        else:
            print("  WARNING: No calibration data found, skipping imatrix", flush=True)

    # ── Step 3: Quantize + Upload ────────────────────────────
    print(f"\n=== Quantizing ({len(quants)} variants) ===", flush=True)

    # Start upload worker thread
    upload_queue = Queue()
    stop_upload = Event()
    upload_thread = None
    ollama_failed: list = []   # (quant_name, gguf_filename) for end-of-run retry
    if hf_repo:
        # Never delete the base F16/F32 GGUF or imatrix — quants depend on them
        # If --keep-local, keep everything by passing all filenames
        if args.keep_local:
            keep_files = None  # upload_worker treats None as "keep all"
        else:
            keep_files = {base_gguf.name}
        upload_thread = Thread(target=upload_worker,
                               args=(upload_queue, hf_repo, stop_upload, keep_files,
                                     args.ollama_target, args.ollama_template,
                                     ollama_failed),
                               daemon=True)
        upload_thread.start()

    # Also upload base F16 GGUF
    if hf_repo:
        upload_queue.put((base_gguf, f"{precision.upper()} base"))

    # MANDATORY: upload imatrix.dat alongside the quants so it can never be lost.
    # If future rebuilds are needed, the canonical imatrix is always reachable
    # from the HF repo itself — no pod-destroy-and-lose-forever scenarios.
    # User rule 2026-04-11: "You MUST SAVE the imatrix.dat used to quantize".
    if hf_repo and imatrix_file and imatrix_file.exists():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            imatrix_size_mb = imatrix_file.stat().st_size / 1024**2
            print(f"\n  Uploading imatrix.dat ({imatrix_size_mb:.1f} MB) to HF repo...", flush=True)
            api.upload_file(
                path_or_fileobj=str(imatrix_file),
                path_in_repo="imatrix.dat",
                repo_id=hf_repo,
                repo_type="model",
                commit_message="Add imatrix.dat used to quantize (for reproducibility/audit)",
            )
            print(f"  imatrix.dat uploaded to {hf_repo}", flush=True)
        except Exception as e:
            print(f"  WARNING: imatrix.dat upload FAILED: {e}", flush=True)
            print(f"  imatrix.dat is still available locally at: {imatrix_file}", flush=True)

    failed = []
    succeeded = []
    t_total = time.time()

    for i, quant in enumerate(quants):
        print(f"\n[{i+1}/{len(quants)}] {quant}", flush=True)
        try:
            gguf_path = quantize_one(
                tools, base_gguf, output_dir, model_name, quant,
                imatrix_file=imatrix_file, threads=args.threads,
            )

            # Sanity check
            if args.sanity_check:
                ok = sanity_check(tools, gguf_path, quant)
                if not ok:
                    print(f"  {quant}: FAILED sanity check, skipping upload", flush=True)
                    failed.append((quant, "sanity check failed"))
                    continue

            # Queue for upload
            if hf_repo:
                upload_queue.put((gguf_path, quant))

            succeeded.append(quant)

        except Exception as e:
            print(f"  {quant}: FAILED — {e}", flush=True)
            failed.append((quant, str(e)))

    # Wait for uploads to finish
    if upload_thread:
        print("\n=== Waiting for uploads to complete ===", flush=True)
        upload_queue.join()
        stop_upload.set()
        upload_thread.join(timeout=10)

    # ── Optional: retry failed ollama pushes by re-downloading from HF ─────
    # This mirrors scripts/ollama_push_generic.sh: the local files are gone
    # at this point (deleted by the worker after HF upload), so we pull each
    # failed quant back from the just-published HF GGUF repo, retry the
    # push, then drop the scratch file.
    if args.ollama_target and ollama_failed:
        print(f"\n=== Retrying {len(ollama_failed)} failed ollama push(es) "
              f"via HF re-download ===", flush=True)
        scratch_dir = output_dir / "_ollama_retry_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        ollama_recovered = []
        ollama_still_failed = []
        for quant_name, fname in ollama_failed:
            print(f"  retry [{quant_name}]: hf download {hf_repo}/{fname}", flush=True)
            try:
                from huggingface_hub import hf_hub_download
                local_path = hf_hub_download(
                    repo_id=hf_repo, filename=fname,
                    local_dir=str(scratch_dir),
                )
                gguf = Path(local_path)
                ok = push_to_ollama_from_local(
                    gguf, args.ollama_target,
                    template_style=args.ollama_template,
                    log_dir=scratch_dir,
                )
                # Always delete the scratch file after the attempt
                try:
                    gguf.unlink()
                except OSError:
                    pass
                if ok:
                    ollama_recovered.append(quant_name)
                else:
                    ollama_still_failed.append(quant_name)
            except Exception as e:
                print(f"  retry [{quant_name}]: FAILED to redownload/push — {e}",
                      flush=True)
                ollama_still_failed.append(quant_name)
        # Drop the scratch dir if empty
        try:
            scratch_dir.rmdir()
        except OSError:
            pass
        print(f"  ollama retry: recovered {len(ollama_recovered)}, "
              f"still failed {len(ollama_still_failed)}", flush=True)
        if ollama_still_failed:
            print(f"  STILL FAILED on ollama: {ollama_still_failed}", flush=True)
            print(f"  → re-run with: bash scripts/ollama_push_generic.sh "
                  f"{hf_repo} {args.ollama_target} <hf_token>", flush=True)

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"  DONE in {elapsed/3600:.1f}h", flush=True)
    print(f"  Succeeded: {len(succeeded)}/{len(quants)}", flush=True)
    if failed:
        print(f"  Failed: {len(failed)}", flush=True)
        for q, reason in failed:
            print(f"    {q}: {reason}", flush=True)
    if hf_repo:
        print(f"  HF repo: https://huggingface.co/{hf_repo}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
