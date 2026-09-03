#!/usr/bin/env python3
"""Republish already-published ollama tags with the correct serving identity.

A tag created from a bare `FROM <gguf>` Modelfile has an EMPTY manifest config:
no renderer, no parser. ollama honours them only when the config names them
(server/images.go), so such a tag silently falls back to the GGUF-derived chat
template -- which degrades to `{{ .Prompt }}` on complex Jinja. The tag loads,
looks healthy in `ollama ps`, and has no chat structure and no thinking channel.

This walks a namespace's published tags and, for each one still bare, pulls it,
re-creates it with RENDERER / PARSER / PARAMETER copied verbatim from a reference
published model, verifies the config that was actually written, pushes, and then
reclaims the disk.

Idempotent and resumable: a tag that already carries an identity is skipped, so
re-running after an interruption costs only the checks.

    ollama_republish_identity.py --namespace mannix/omnimerge-v4 \
        --inherit library/qwen3.6:27b

Requires an ollama daemon whose version knows the renderer (see --probe).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REG = "https://registry.ollama.ai/v2"
ACCEPT = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
# Directives we regenerate; anything else in the derived Modelfile is preserved.
_STRIP = re.compile(r"^\s*(RENDERER|PARSER|PARAMETER|REQUIRES)\b", re.I)
# TEMPLATE is dropped too, and that is deliberate. `ollama show --modelfile` on a
# tag with no template layer renders the GGUF-derived fallback as a literal
# TEMPLATE directive -- and on complex Jinja that fallback is the degenerate
# `{{ .Prompt }}`. Preserving it BAKES a no-formatting template into the
# republished tag, which then overrides the renderer we are adding. Measured on
# mannix/omnimerge-v4:vision-Q4_K_M (2026-09-03): a 13-byte template layer
# containing exactly `{{ .Prompt }}`, inherited the same way by
# build_vision_ollama_tiers.sh. The vendor tags carry NO template layer at all
# (library/qwen3.6:27b, library/qwen3.8:27b) -- the renderer supplies formatting.
_TEMPLATE = re.compile(r"^\s*TEMPLATE\b", re.I)


def strip_template(lines: list[str]) -> list[str]:
    """Drop TEMPLATE directives, including multi-line triple-quoted blocks."""
    out, i = [], 0
    while i < len(lines):
        if _TEMPLATE.match(lines[i]):
            rest = lines[i].split(None, 1)[1] if len(lines[i].split(None, 1)) > 1 else ""
            if rest.startswith('"""'):
                # consume until the closing """ (which may be on this same line)
                if rest.count('"""') < 2:
                    i += 1
                    while i < len(lines) and '"""' not in lines[i]:
                        i += 1
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def _get(url: str, headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_identity(ref: str) -> dict:
    """renderer / parser / params / requires of a published model, verbatim.

    `requires` matters as much as the renderer: it is the guard that makes an
    ollama too old for the renderer REFUSE the model instead of loading it and
    degrading silently at render time (the qwen3.8 renderer needs >= 0.32.12).
    A tag that carries the renderer but not the requires floor still fails on
    an old client -- just later, and invisibly.
    """
    if "/" not in ref:
        ref = "library/" + ref
    repo, _, tag = ref.partition(":")
    man = _get(f"{REG}/{repo}/manifests/{tag or 'latest'}", ACCEPT)
    out = {"renderer": "", "parser": "", "params": {}, "requires": ""}
    cfg = (man.get("config") or {}).get("digest")
    if cfg:
        c = _get(f"{REG}/{repo}/blobs/{cfg}")
        out["renderer"] = c.get("renderer") or ""
        out["parser"] = c.get("parser") or ""
        out["requires"] = c.get("requires") or ""
    for layer in man.get("layers", []):
        if str(layer.get("mediaType", "")).endswith("params"):
            out["params"] = _get(f"{REG}/{repo}/blobs/{layer['digest']}")
    return out


def published_config(namespace: str, tag: str) -> dict | None:
    try:
        man = _get(f"{REG}/{namespace}/manifests/{tag}", ACCEPT)
        return _get(f"{REG}/{namespace}/blobs/{man['config']['digest']}")
    except Exception:
        return None


def published_identity(namespace: str, tag: str) -> dict | None:
    """Everything this tool WRITES, read back off the registry.

    The skip test has to compare the full intended identity, not merely ask
    whether a renderer exists. A tag carrying renderer+parser but missing
    `requires` is NOT done -- it is missing the guard that stops an old ollama
    loading it and degrading at render time. Comparing only "has something"
    makes the tool accept partial state and never converge.
    """
    try:
        man = _get(f"{REG}/{namespace}/manifests/{tag}", ACCEPT)
    except Exception:
        return None
    out = {"renderer": "", "parser": "", "requires": "", "params": {}}
    cfg = (man.get("config") or {}).get("digest")
    if cfg:
        try:
            c = _get(f"{REG}/{namespace}/blobs/{cfg}")
            out["renderer"] = c.get("renderer") or ""
            out["parser"] = c.get("parser") or ""
            out["requires"] = c.get("requires") or ""
        except Exception:
            return None
    for layer in man.get("layers", []):
        if str(layer.get("mediaType", "")).endswith("params"):
            try:
                out["params"] = _get(f"{REG}/{namespace}/blobs/{layer['digest']}")
            except Exception:
                return None
    return out


def list_tags(namespace: str) -> list[str]:
    """Tag list off the public page. The registry v2 /tags/list is not open here."""
    model = namespace.split("/")[-1]
    html = subprocess.run(
        ["curl", "-s", f"https://ollama.com/{namespace}/tags",
         "-H", "User-Agent: Mozilla/5.0"],
        capture_output=True, text=True, timeout=120).stdout
    return sorted(set(re.findall(rf"{re.escape(model)}:([A-Za-z0-9_.-]+)", html)))


def ollama_store() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env)
    try:
        user = subprocess.run(["systemctl", "show", "ollama", "-p", "Environment", "--value"],
                              capture_output=True, text=True, timeout=10).stdout
        for tok in user.split():
            if tok.startswith("OLLAMA_MODELS="):
                return Path(tok.split("=", 1)[1])
    except Exception:
        pass
    return Path("/root/.ollama/models")


def local_config(store: Path, ref: str) -> dict | None:
    repo, _, tag = ref.partition(":")
    mf = store / "manifests" / "registry.ollama.ai" / repo / (tag or "latest")
    try:
        man = json.loads(mf.read_text())
        blob = store / "blobs" / man["config"]["digest"].replace(":", "-")
        return json.loads(blob.read_text())
    except Exception:
        return None


def param_lines(params: dict) -> list[str]:
    out = []
    for k, v in sorted(params.items()):
        if isinstance(v, bool):
            out.append(f"PARAMETER {k} {str(v).lower()}")
        elif isinstance(v, (int, float)):
            out.append(f"PARAMETER {k} {v}")
        elif isinstance(v, list):
            out.extend(f'PARAMETER {k} "{i}"' for i in v)
        else:
            out.append(f'PARAMETER {k} "{v}"')
    return out


def run(cmd: list[str], timeout: int = 7200):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def has_vision(ref: str) -> bool:
    """True if the local tag reports the vision capability.

    Read via capture_output, never `ollama show | grep -q`: grep exits on match,
    SIGPIPEs ollama show (141), and under `set -o pipefail` the caller inherits
    that -- a model that HAS vision then reports FALSE (bug-666, 13 lost tiers).
    """
    r = run(["ollama", "show", ref], timeout=300)
    return "vision" in (r.stdout or "").lower()


def free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def purge_orphans(store: Path) -> float:
    """Delete blobs no manifest references. Returns GB freed."""
    referenced = set()
    for mf in (store / "manifests").rglob("*"):
        if mf.is_file():
            try:
                referenced.update(
                    d.replace(":", "-")
                    for d in re.findall(r"sha256[:-][0-9a-f]{64}", mf.read_text()))
            except OSError:
                pass
    freed = 0
    bd = store / "blobs"
    if not bd.is_dir():
        return 0.0
    for b in bd.iterdir():
        if b.is_file() and b.name.startswith("sha256-") and b.name not in referenced:
            try:
                sz = b.stat().st_size
                b.unlink()
                freed += sz
            except OSError:
                pass
    return freed / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", required=True, help="e.g. mannix/omnimerge-v4")
    ap.add_argument("--inherit", required=True, help="reference model, e.g. library/qwen3.6:27b")
    ap.add_argument("--only", default=None, help="comma-separated subset of tags")
    ap.add_argument("--license", type=Path, default=None,
                    help="Embed this licence file into any tag that lacks a licence layer. "
                         "Tags that already carry one are left byte-identical.")
    ap.add_argument("--min-free-gb", type=float, default=120.0,
                    help="refuse to pull when the store filesystem is below this (default 120)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    if not shutil.which("ollama"):
        print("FATAL: no ollama on PATH", flush=True)
        return 2

    store = ollama_store()
    if not store.is_dir():
        print(f"FATAL: ollama store {store} does not exist. Export OLLAMA_MODELS.", flush=True)
        return 2
    print(f">>> store: {store}  free={free_gb(store):.0f} GB", flush=True)

    ident = fetch_identity(args.inherit)
    if not ident["renderer"] and not ident["parser"]:
        print(f"FATAL: {args.inherit} declares no renderer/parser — nothing to inherit", flush=True)
        return 2
    print(f">>> identity from {args.inherit}: renderer={ident['renderer']!r} "
          f"parser={ident['parser']!r} requires={ident['requires']!r} "
          f"params={json.dumps(ident['params'], sort_keys=True)}",
          flush=True)

    tags = args.only.split(",") if args.only else list_tags(args.namespace)
    if not tags:
        print(f"FATAL: no tags found for {args.namespace}", flush=True)
        return 2
    print(f">>> {len(tags)} tag(s) in {args.namespace}", flush=True)

    done = skipped = failed = 0
    for i, tag in enumerate(tags, 1):
        ref = f"{args.namespace}:{tag}"
        cur = published_identity(args.namespace, tag)
        if cur is not None and all(
                cur.get(k) == ident.get(k)
                for k in ("renderer", "parser", "requires", "params")):
            print(f"[{i}/{len(tags)}] {tag}: identity already matches — skip",
                  flush=True)
            skipped += 1
            continue
        if cur is not None and (cur.get("renderer") or cur.get("parser")):
            diff = [k for k in ("renderer", "parser", "requires", "params")
                    if cur.get(k) != ident.get(k)]
            print(f"[{i}/{len(tags)}] {tag}: PARTIAL identity, differs on "
                  f"{'+'.join(diff)} — republishing", flush=True)
        if args.dry_run:
            print(f"[{i}/{len(tags)}] {tag}: WOULD republish", flush=True)
            done += 1
            continue

        avail = free_gb(store)
        if avail < args.min_free_gb:
            freed = purge_orphans(store)
            avail = free_gb(store)
            print(f"[{i}/{len(tags)}] {tag}: purged {freed:.0f} GB, free now {avail:.0f} GB",
                  flush=True)
            if avail < args.min_free_gb:
                print(f"[{i}/{len(tags)}] ABORT: only {avail:.0f} GB free, "
                      f"need {args.min_free_gb:.0f}", flush=True)
                return 3

        t0 = time.time()
        print(f"[{i}/{len(tags)}] {tag}: pulling ...", flush=True)
        r = run(["ollama", "pull", ref])
        if r.returncode != 0:
            print(f"[{i}/{len(tags)}] {tag}: PULL FAILED: {r.stderr.strip()[-200:]}", flush=True)
            failed += 1
            continue

        # A vision tag must still be a vision tag after the rebuild. The Modelfile
        # is regenerated from `ollama show --modelfile`, and that command is already
        # known not to round-trip everything (it drops REQUIRES outright). If it also
        # omits the projector FROM, the recreated tag silently loses its mmproj layer
        # and the renderer/parser-only verify below would happily push a text-only
        # model over a good vision tag -- 19 v6 + 45 v4 tags of irreversible damage.
        had_vision = has_vision(ref)

        show = run(["ollama", "show", ref, "--modelfile"])
        if show.returncode != 0:
            print(f"[{i}/{len(tags)}] {tag}: SHOW FAILED", flush=True)
            failed += 1
            continue
        # Preserve FROM / SYSTEM / LICENSE etc; regenerate the identity and drop
        # any derived TEMPLATE (see _TEMPLATE above -- it would shadow the renderer).
        body = strip_template(
            [ln for ln in show.stdout.splitlines() if not _STRIP.match(ln)])
        lines = body + [f"RENDERER {ident['renderer']}" if ident["renderer"] else "",
                        f"PARSER {ident['parser']}" if ident["parser"] else "",
                        f"REQUIRES {ident['requires']}" if ident["requires"] else ""]
        lines = [x for x in lines if x] + param_lines(ident["params"])
        # Add a licence only when the tag has none. `show --modelfile` DOES
        # round-trip LICENSE, so an existing one is already in `body`; appending a
        # second would create a duplicate layer.
        if args.license and not any(ln.lstrip().upper().startswith("LICENSE")
                                    for ln in body):
            lines.append('LICENSE """' + args.license.read_text() + '"""')
        mf = store.parent / f"_republish_{tag}.Modelfile"
        mf.write_text("\n".join(lines) + "\n")

        ok = False
        for attempt in range(1, args.retries + 1):
            c = run(["ollama", "create", ref, "-f", str(mf)])
            if c.returncode == 0:
                ok = True
                break
            print(f"[{i}/{len(tags)}] {tag}: create attempt {attempt} failed: "
                  f"{c.stderr.strip()[-200:]}", flush=True)
            time.sleep(5)
        mf.unlink(missing_ok=True)
        if not ok:
            failed += 1
            continue

        if had_vision and not has_vision(ref):
            nfrom = sum(1 for ln in body if ln.strip().upper().startswith("FROM"))
            print(f"[{i}/{len(tags)}] {tag}: VISION LOST in rebuild "
                  f"({nfrom} FROM line(s) carried over) — NOT pushing; "
                  f"registry copy untouched, re-pull to restore locally", flush=True)
            failed += 1
            continue

        wrote = local_config(store, ref) or {}
        if wrote.get("renderer") != ident["renderer"] or wrote.get("parser") != ident["parser"]:
            print(f"[{i}/{len(tags)}] {tag}: VERIFY FAILED — wrote "
                  f"{wrote.get('renderer')!r}/{wrote.get('parser')!r}; not pushing", flush=True)
            failed += 1
            continue

        p = run(["ollama", "push", ref])
        if p.returncode != 0:
            print(f"[{i}/{len(tags)}] {tag}: PUSH FAILED: {p.stderr.strip()[-200:]}", flush=True)
            failed += 1
            continue

        run(["ollama", "rm", ref])
        freed = purge_orphans(store)
        print(f"[{i}/{len(tags)}] {tag}: REPUBLISHED ok "
              f"({time.time()-t0:.0f}s, reclaimed {freed:.0f} GB, "
              f"free {free_gb(store):.0f} GB)", flush=True)
        done += 1

    print(f">>> {args.namespace}: {done} republished, {skipped} already ok, {failed} failed",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
