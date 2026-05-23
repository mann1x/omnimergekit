#!/usr/bin/env python3
"""Inspect an Ollama-published model: layers, GGUF metadata, runtime params.

Usage:
    ollama_inspect_model.py <ns/model:tag> [--mode scrape|registry] [--json]

Default mode `scrape` reads ollama.com's HTML at `/<ns>/<model>:<tag>/blobs/<digest>`
which carries the parsed GGUF metadata table (general.*, <arch>.*, tokenizer.*).
Zero deps beyond stdlib.

Mode `registry` pulls the manifest from registry.ollama.ai/v2/ and reports
layer digests/sizes/types only — does NOT parse the GGUF blob (use `gguf-py`
locally for that; this script stays dep-free).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request

REGISTRY = "https://registry.ollama.ai/v2"
WEB = "https://ollama.com"


def _http_get(url: str, accept: str = "*/*") -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "ollama-inspect/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _parse_target(target: str) -> tuple[str, str, str]:
    """Parse `ns/model:tag` → (ns, model, tag). Tag defaults to `latest`."""
    if ":" in target:
        repo, tag = target.rsplit(":", 1)
    else:
        repo, tag = target, "latest"
    if "/" not in repo:
        raise ValueError(f"need ns/model form, got {target!r}")
    ns, model = repo.split("/", 1)
    return ns, model, tag


def fetch_manifest(ns: str, model: str, tag: str) -> dict:
    url = f"{REGISTRY}/{ns}/{model}/manifests/{tag}"
    raw = _http_get(url, accept="application/vnd.docker.distribution.manifest.v2+json")
    return json.loads(raw)


def fetch_config_blob(ns: str, model: str, digest: str) -> dict:
    raw = _http_get(f"{REGISTRY}/{ns}/{model}/blobs/{digest}")
    return json.loads(raw)


def fetch_text_blob(ns: str, model: str, digest: str) -> str:
    return _http_get(f"{REGISTRY}/{ns}/{model}/blobs/{digest}").decode("utf-8", errors="replace")


_META_PAIR = re.compile(
    r'<div class="text-neutral-600 sm:text-black">([^<]+)</div>\s*<div[^>]*>([^<]+)</div>'
)

# Per-tensor quant type rows. ollama.com renders each tensor as:
#   <div class="text-neutral-600 sm:text-black">blk.0.attn_k.weight</div>
#   <div class="col-span-1 font-mono hidden sm:block">Q4_K</div>
# The tensor-name div uses the same `text-neutral-600 sm:text-black` class as
# metadata keys, so we filter by name pattern (blk.* / token_embd / output_norm
# / output / rope_freqs).
_TENSOR_NAME_RE = re.compile(r"^(blk\.\d+\.|token_embd|output_norm|output\.|rope_freqs)")


def scrape_gguf_metadata(ns: str, model: str, tag: str, gguf_digest: str) -> dict[str, str]:
    """Pull the parsed metadata table that ollama.com renders for a GGUF blob.

    `gguf_digest` may be the short form (12 hex) or full sha256:..., either works
    because ollama.com routes both. We pass the short form to match the URLs
    the web UI uses.
    """
    short = gguf_digest.split(":")[-1][:12]
    url = f"{WEB}/{ns}/{model}:{tag}/blobs/{short}"
    page = _http_get(url, accept="text/html").decode("utf-8", errors="replace")
    out: dict[str, str] = {}
    for k, v in _META_PAIR.findall(page):
        k = html.unescape(k).strip()
        v = html.unescape(v).strip()
        if not k or not v or k == v:
            continue
        # Some pairs repeat (key, then value div using same class); first wins.
        out.setdefault(k, v)
    return out


def scrape_tensor_types(ns: str, model: str, tag: str, gguf_digest: str) -> dict[str, str]:
    """Pull per-tensor quant types from ollama.com.

    Returns {tensor_name: quant_type}. Quant types are like Q4_K, Q5_K,
    Q6_K, Q8_0, F32, F16, BF16, IQ2_S, etc.
    """
    short = gguf_digest.split(":")[-1][:12]
    url = f"{WEB}/{ns}/{model}:{tag}/blobs/{short}"
    page = _http_get(url, accept="text/html").decode("utf-8", errors="replace")
    out: dict[str, str] = {}
    for k, v in _META_PAIR.findall(page):
        k = html.unescape(k).strip()
        v = html.unescape(v).strip()
        if not _TENSOR_NAME_RE.match(k):
            continue
        out.setdefault(k, v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="ns/model[:tag]  (e.g. mannix/gemma4-31b-he1:IQ2_S)")
    ap.add_argument("--mode", choices=("scrape", "registry"), default="scrape",
                    help="scrape (default): web HTML + parsed GGUF metadata. "
                    "registry: v2 manifest + layer types/sizes only.")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of pretty output")
    args = ap.parse_args()

    try:
        ns, model, tag = _parse_target(args.target)
    except ValueError as e:
        print(f"ERR: {e}", file=sys.stderr)
        return 2

    manifest = fetch_manifest(ns, model, tag)
    cfg = fetch_config_blob(ns, model, manifest["config"]["digest"])

    layers_info = []
    gguf_digest = None
    template_text = None
    params_text = None
    for layer in manifest.get("layers", []):
        media = layer["mediaType"]
        info = {"mediaType": media, "digest": layer["digest"], "size": layer["size"]}
        if media == "application/vnd.ollama.image.model":
            gguf_digest = layer["digest"]
        elif media == "application/vnd.ollama.image.template":
            template_text = fetch_text_blob(ns, model, layer["digest"])
        elif media == "application/vnd.ollama.image.params":
            params_text = fetch_text_blob(ns, model, layer["digest"])
        layers_info.append(info)

    metadata: dict[str, str] = {}
    tensors: dict[str, str] = {}
    if args.mode == "scrape" and gguf_digest:
        try:
            metadata = scrape_gguf_metadata(ns, model, tag, gguf_digest)
            tensors = scrape_tensor_types(ns, model, tag, gguf_digest)
        except Exception as e:
            print(f"WARN: scrape failed: {e}", file=sys.stderr)

    payload = {
        "target": f"{ns}/{model}:{tag}",
        "config": cfg,
        "layers": layers_info,
        "template": template_text,
        "params": json.loads(params_text) if params_text else None,
        "gguf_metadata": metadata,
        "tensors": tensors,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    # Pretty render
    print(f"=== {ns}/{model}:{tag} ===")
    print(f"  family       {cfg.get('model_family')}")
    print(f"  type         {cfg.get('model_type')}")
    print(f"  quant        {cfg.get('file_type')}")
    print(f"  renderer     {cfg.get('renderer')}")
    print(f"  parser       {cfg.get('parser')}")
    print()
    print("=== Layers ===")
    for layer in layers_info:
        mt = layer["mediaType"].split(".")[-1]
        size_mb = layer["size"] / (1024 * 1024)
        unit = f"{size_mb / 1024:.2f} GiB" if size_mb >= 1024 else f"{size_mb:.2f} MiB"
        print(f"  {mt:12s} {layer['digest'][:19]}...  {unit:>12s}")
    if payload["params"]:
        print()
        print("=== Runtime params ===")
        for k, v in payload["params"].items():
            print(f"  {k:20s} {v}")
    if metadata:
        print()
        print("=== GGUF metadata ===")
        for k, v in metadata.items():
            v_disp = v if len(v) <= 80 else v[:77] + "..."
            print(f"  {k:55s} {v_disp}")
    if template_text:
        print()
        print(f"=== Chat template ({len(template_text)} chars) ===")
        print(template_text[:400] + ("..." if len(template_text) > 400 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
