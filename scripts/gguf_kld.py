#!/usr/bin/env python3
"""KL-divergence / perplexity stats per GGUF tier, against an unquantized reference.

Same basis as bartowski's "PPL / Mean KLD / Same top-p" tables: llama.cpp
`llama-perplexity` over `wiki.test.raw` (wikitext-2), 100 chunks, reference
logits saved once with `--kl-divergence-base`, every tier then scored with
`--kl-divergence` against that file. Defaults: 100 chunks x 512 ctx.

Two entry points share this one implementation:

  1. quantize_gguf.py calls ref_pass() once before the tier loop and measure()
     for each tier in the upload worker, on the same local bytes that were just
     shipped and before they are deleted (default ON, --no-kld to opt out).

  2. This CLI measures a repo that is ALREADY published: it downloads each tier,
     checks its sha256 against the Hub, measures it, deletes it, and resumes
     from kld_results.jsonl if interrupted.

     python scripts/gguf_kld.py --repo ManniX-ITA/<model>-GGUF \\
         --ref /path/<model>-BF16.gguf --work /workspace/kld [--upload]

Outputs (in the work/output dir): <tier>.gguf.kld.log per tier,
kld_results.jsonl (one row per tier, append-only), perplexity.md (bartowski
layout: summary table + KLD percentile ladder) and kld_results.json.

The reference file is ~n_chunks * ctx/2 * n_vocab * 2 bytes (Gemma-4, 262k
vocab: ~13 GB at 100x512). It is reused only when its .meta.json matches the
reference GGUF, ctx, chunks and dataset; anything else is a different basis.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT_MEMBER = "wikitext-2-raw/wiki.test.raw"
WIKITEXT_SHA256 = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
DEFAULT_CHUNKS = 100
DEFAULT_CTX = 512

# Gemma-4 A4B (the 26B-A4B MoE and every expert-pruned derivative) never yields a usable
# perplexity under llama-perplexity: the v9-agentic BF16 reference scored PPL 32682 on
# wiki.test.raw and Q8_0-vs-BF16 KLD was 0.80 (2026-10-15). Its KLD ladder is meaningless, so
# it is skipped. The dense Gemma-4 models (E2B, E4B, 31B) and every other arch keep it.
A4B_SKIP_REASON = ("Gemma-4 A4B (MoE) has no working llama-perplexity PPL "
                   "(v9 BF16 reference PPL 32682 on wiki.test.raw)")


def is_gemma4_a4b(gguf_path) -> bool:
    """True for a Gemma-4 MoE GGUF: general.architecture == gemma4 AND gemma4.expert_count > 0.
    E2B / E4B / 31B are dense and carry no expert_count, so they return False."""
    from gguf import GGUFReader
    f = GGUFReader(str(gguf_path)).fields
    arch = f.get("general.architecture")
    if arch is None or bytes(arch.parts[arch.data[0]]).decode("utf-8", "replace") != "gemma4":
        return False
    ec = f.get("gemma4.expert_count")
    return ec is not None and int(ec.parts[ec.data[0]][0]) > 0


# Percentile rows of the KLD ladder, in llama-perplexity's print order.
LADDER = ["Maximum", "99.9%", "99.0%", "95.0%", "90.0%", "Median",
          "10.0%", "5.0%", "1.0%", "0.1%", "Minimum"]


def _sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_wikitext(dest_dir: Path) -> Path:
    """wiki.test.raw, verified by sha256. Uses zipfile, not `unzip` (pods often lack it)."""
    out = dest_dir / WIKITEXT_MEMBER
    if out.exists() and _sha256(out) == WIKITEXT_SHA256:
        return out
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(WIKITEXT_URL, timeout=120) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extract(WIKITEXT_MEMBER, dest_dir)
    got = _sha256(out)
    if got != WIKITEXT_SHA256:
        raise RuntimeError(f"wiki.test.raw sha256 {got} != expected {WIKITEXT_SHA256}")
    return out


def find_perplexity(tools: dict | None = None, build_if_missing: bool = True) -> str:
    """llama-perplexity next to the llama-quantize the run uses (same build, same basis)."""
    cand = []
    if tools and tools.get("quantize"):
        cand.append(Path(tools["quantize"]).with_name("llama-perplexity"))
    if os.environ.get("LLAMA_CPP_HOME"):
        cand.append(Path(os.environ["LLAMA_CPP_HOME"]) / "build/bin/llama-perplexity")
    cand += [Path("/opt/llama.cpp/build/bin/llama-perplexity"),
             Path("/workspace/llama.cpp/build/bin/llama-perplexity")]
    for c in cand:
        if c.exists():
            return str(c)
    if build_if_missing and tools and tools.get("base"):
        build = Path(tools["base"]) / "build"
        if build.is_dir():
            print(f"  kld: building llama-perplexity in {build}", flush=True)
            subprocess.run(["cmake", "--build", str(build), "--target", "llama-perplexity",
                            "-j", "8"], check=True)
            p = build / "bin/llama-perplexity"
            if p.exists():
                return str(p)
    raise FileNotFoundError("llama-perplexity not found (build it: "
                            "cmake --build <llama.cpp>/build --target llama-perplexity)")


def _build_version(ppl_bin: str) -> str:
    try:
        out = subprocess.run([ppl_bin, "--version"], capture_output=True, text=True,
                             timeout=60).stderr
        m = re.search(r"version:\s*(\S+)\s*\((\w+)\)", out)
        return f"b{m.group(1)} ({m.group(2)})" if m else out.strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return f"unknown ({e})"


def _run(argv: list[str], log_path: Path, timeout: int) -> str:
    with open(log_path, "w") as f:
        f.write("$ " + " ".join(argv) + "\n")
        f.flush()
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    text = log_path.read_text(errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"llama-perplexity rc={p.returncode} (see {log_path})")
    return text


def _common_args(ctx: int, chunks: int, threads: int | None) -> list[str]:
    a = ["-c", str(ctx), "--chunks", str(chunks), "-fit", "on"]
    if threads:
        a += ["-t", str(threads)]
    # OMK_KLD_EXTRA_ARGS is appended last, so it overrides the defaults above,
    # e.g. "-fit off -ngl 0" (CPU only on a shared host) or "-dev CUDA1".
    return a + shlex.split(os.environ.get("OMK_KLD_EXTRA_ARGS", ""))


def ref_pass(ppl_bin: str, ref_gguf: Path, data: Path, out_dir: Path,
             ctx: int = DEFAULT_CTX, chunks: int = DEFAULT_CHUNKS,
             threads: int | None = None, timeout: int = 6 * 3600) -> dict:
    """Save the reference log-probs once. Returns the basis dict (incl. base PPL)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"kld_base_{ref_gguf.stem}_c{ctx}_n{chunks}.bin"
    meta_p = base.with_suffix(".meta.json")
    st = ref_gguf.stat()
    want = {"ref": ref_gguf.name, "ref_bytes": st.st_size, "ref_mtime": int(st.st_mtime),
            "ctx": ctx, "chunks": chunks, "dataset": "wiki.test.raw",
            "dataset_sha256": _sha256(data)}
    if base.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if all(meta.get(k) == v for k, v in want.items()):
            print(f"  kld: reusing reference logits {base.name} (basis matches)", flush=True)
            meta["base_file"] = str(base)
            return meta
        print(f"  kld: {base.name} exists but its basis differs -> recomputing", flush=True)
    log = out_dir / f"{ref_gguf.name}.kld_base.log"
    print(f"  kld: reference pass on {ref_gguf.name} ({chunks} chunks x {ctx} ctx)", flush=True)
    t0 = time.time()
    text = _run([ppl_bin, "-m", str(ref_gguf), "-f", str(data),
                 *_common_args(ctx, chunks, threads), "--kl-divergence-base", str(base)],
                log, timeout)
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", text)
    if not m or not base.exists():
        raise RuntimeError(f"reference pass produced no PPL/base file (see {log})")
    meta = {**want, "base_ppl": float(m.group(1)), "base_ppl_err": float(m.group(2)),
            "llama_build": _build_version(ppl_bin), "base_bytes": base.stat().st_size,
            "seconds": round(time.time() - t0, 1)}
    meta_p.write_text(json.dumps(meta, indent=2))
    meta["base_file"] = str(base)
    print(f"  kld: base PPL {meta['base_ppl']:.4f} ± {meta['base_ppl_err']:.5f} "
          f"({meta['seconds']:.0f}s)", flush=True)
    return meta


_PM = r"(-?[0-9.]+(?:e[+-]?\d+)?)\s*±\s*(-?[0-9.]+(?:e[+-]?\d+)?)"


def parse_kld_log(text: str) -> dict:
    """Pull bartowski's columns (+ the KLD percentile ladder) out of llama-perplexity output."""
    def pm(label):
        m = re.search(re.escape(label) + r"\s*:\s*" + _PM, text)
        return (float(m.group(1)), float(m.group(2))) if m else None

    r = {
        "ppl_q": pm("Mean PPL(Q)"),
        "ppl_base": pm("Mean PPL(base)"),
        "ppl_ratio": pm("Mean PPL(Q)/PPL(base)"),
        "ppl_diff": pm("Mean PPL(Q)-PPL(base)"),
        "kld_mean": pm("Mean    KLD"),
    }
    m = re.search(r"Mean\s+Δp:\s*" + _PM + r"\s*%", text)
    r["dp_mean_pct"] = (float(m.group(1)), float(m.group(2))) if m else None
    m = re.search(r"RMS Δp\s*:\s*" + _PM + r"\s*%", text)
    r["dp_rms_pct"] = (float(m.group(1)), float(m.group(2))) if m else None
    m = re.search(r"Same top p:\s*" + _PM + r"\s*%", text)
    r["same_top_p_pct"] = (float(m.group(1)), float(m.group(2))) if m else None
    ladder = {}
    for lab in LADDER:
        pat = (rf"^\s*{re.escape(lab)}\s+KLD:\s*(-?[0-9.]+(?:e[+-]?\d+)?)" if lab.endswith("%")
               else rf"^{lab}\s+KLD:\s*(-?[0-9.]+(?:e[+-]?\d+)?)")
        m = re.search(pat, text, re.M)
        if m:
            ladder[lab] = float(m.group(1))
    r["kld_ladder"] = ladder
    missing = [k for k in ("ppl_q", "kld_mean", "same_top_p_pct") if r[k] is None]
    if missing:
        raise RuntimeError(f"llama-perplexity output lacks {missing}")
    return r


def measure(ppl_bin: str, gguf: Path, quant: str, basis: dict, data: Path,
            threads: int | None = None, timeout: int = 3 * 3600) -> dict:
    """Score one tier against the saved reference logits. Writes <gguf>.kld.log."""
    log = Path(str(gguf) + ".kld.log")
    t0 = time.time()
    text = _run([ppl_bin, "-m", str(gguf), "-f", str(data),
                 *_common_args(basis["ctx"], basis["chunks"], threads),
                 "--kl-divergence-base", basis["base_file"], "--kl-divergence"], log, timeout)
    row = {"quant": quant, "file": gguf.name, "bytes": gguf.stat().st_size,
           **parse_kld_log(text), "seconds": round(time.time() - t0, 1)}
    print(f"  {quant}: KLD {row['kld_mean'][0]:.6f} · same top-p "
          f"{row['same_top_p_pct'][0]:.3f}% · PPL {row['ppl_q'][0]:.4f} "
          f"({row['seconds']:.0f}s)", flush=True)
    return row


def append_row(out_dir: Path, row: dict) -> None:
    with open(out_dir / "kld_results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def load_rows(out_dir: Path) -> list[dict]:
    p = out_dir / "kld_results.jsonl"
    if not p.exists():
        return []
    rows = {}
    for ln in p.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[r["quant"]] = r          # last measurement of a tier wins
    return list(rows.values())


def _f(v, nd=6):
    return "—" if v is None else f"{v[0]:.{nd}g} ± {v[1]:.2g}"


def write_reports(out_dir: Path, basis: dict, rows: list[dict], model: str = "") -> Path:
    """perplexity.md (bartowski layout) + kld_results.json. Rows sorted by size, largest first."""
    rows = sorted(rows, key=lambda r: -r["bytes"])
    b = basis
    L = [f"## Perplexity{(' — ' + model) if model else ''}", "",
         f"Perplexity and KL divergence measured against the {b['ref']} model on "
         f"`wiki.test.raw` over {b['chunks']} chunks.", "",
         "<details>", "<summary>Click to view full KL-divergence statistics</summary>", "",
         f"- Reference model: {b['ref']}", "- Dataset: wiki.test.raw "
         f"(sha256 `{b['dataset_sha256'][:12]}…`)",
         f"- Chunks: {b['chunks']} × ctx {b['ctx']}",
         f"- Base PPL: {b['base_ppl']:.4f} ± {b['base_ppl_err']:.5f}",
         f"- llama.cpp build: {b.get('llama_build', '?')}", "", "### Summary", "",
         "| Quant | Size (GB) | PPL(Q) | PPL(Q)/PPL(base) | PPL(Q)-PPL(base) | Mean KLD | "
         "Mean Δp | RMS Δp | Same top-p |",
         "| ----- | ----: | ----- | ----- | ----- | ----- | ----- | ----- | ----- |"]
    for r in rows:
        dp, rms, st = r["dp_mean_pct"], r["dp_rms_pct"], r["same_top_p_pct"]
        L.append(f"| {r['quant']} | {r['bytes'] / 1e9:.2f} | {_f(r['ppl_q'])} | "
                 f"{_f(r['ppl_ratio'])} | {_f(r['ppl_diff'])} | {_f(r['kld_mean'])} | "
                 f"{'—' if dp is None else f'{dp[0]:.3f} ± {dp[1]:.3f}%'} | "
                 f"{'—' if rms is None else f'{rms[0]:.3f} ± {rms[1]:.3f}%'} | "
                 f"{st[0]:.3f} ± {st[1]:.3f}% |")
    L += ["", "### KLD percentile ladder", "",
          "| Quant | " + " | ".join(LADDER) + " |",
          "| ----- |" + " ----- |" * len(LADDER)]
    for r in rows:
        lad = r.get("kld_ladder", {})
        L.append(f"| {r['quant']} | " + " | ".join(
            f"{lad[k]:.6g}" if k in lad else "—" for k in LADDER) + " |")
    L += ["", "</details>", ""]
    md = out_dir / "perplexity.md"
    md.write_text("\n".join(L))
    (out_dir / "kld_results.json").write_text(json.dumps(
        {"basis": {k: v for k, v in b.items() if k != "base_file"}, "rows": rows}, indent=2))
    return md


def upload_reports(repo_id: str, out_dir: Path) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    for name in ("perplexity.md", "kld_results.json"):
        api.upload_file(path_or_fileobj=str(out_dir / name), path_in_repo=name,
                        repo_id=repo_id, repo_type="model",
                        commit_message=f"Add {name} (KLD vs unquantized reference)")


# ── CLI: measure an already-published repo ─────────────────────────────────
def _hub_ggufs(repo: str) -> list[tuple[str, int, str]]:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    out = []
    for s in info.siblings:
        n = s.rfilename
        if not n.endswith(".gguf") or n.startswith("mmproj") or "mmproj" in n:
            continue
        if re.search(r"-(F16|F32|BF16)\.gguf$", n):
            continue
        sha = s.lfs.sha256 if s.lfs else None
        out.append((n, s.size or 0, sha))
    return sorted(out, key=lambda t: -t[1])


def _quant_of(fname: str, stem: str) -> str:
    q = fname[:-len(".gguf")]
    return q[len(stem) + 1:] if q.startswith(stem + "-") else q


def _download(repo: str, fname: str, dest: Path) -> Path:
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    out = dest / fname
    if subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0:
        from huggingface_hub import get_token
        hdr = []
        tok = get_token()          # $HF_TOKEN or the `hf auth login` cache
        inp = None
        if tok:   # token goes through a 0600 input file, never onto the argv
            import tempfile
            fd, inp = tempfile.mkstemp(dir=str(dest), prefix=".aria_", suffix=".in")
            with os.fdopen(fd, "w") as f:
                f.write(f"{url}\n  header=Authorization: Bearer {tok}\n  out={fname}\n")
            hdr = ["-i", inp]
        else:
            hdr = [url, "-o", fname]
        try:
            subprocess.run(["aria2c", "-x", "16", "-s", "16", "-k", "16M", "-d", str(dest),
                            "--allow-overwrite=true", "--auto-file-renaming=false",
                            "--console-log-level=warn", "--summary-interval=0", *hdr],
                           check=True)
        finally:
            if inp:
                os.unlink(inp)
    else:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, fname, local_dir=str(dest))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="published GGUF repo, e.g. ManniX-ITA/X-GGUF")
    ap.add_argument("--ref", required=True, type=Path, help="unquantized reference GGUF (bf16/F16)")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--tiers", nargs="*", help="only these quant names (default: every tier)")
    ap.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--keep", action="store_true", help="keep downloaded tiers")
    ap.add_argument("--upload", action="store_true", help="push perplexity.md + kld_results.json")
    ap.add_argument("--allow-gemma4-a4b", action="store_true",
                    help="measure a Gemma-4 A4B anyway (its PPL/KLD is known to be meaningless)")
    a = ap.parse_args()
    if is_gemma4_a4b(a.ref) and not a.allow_gemma4_a4b:
        print(f"  kld: REFUSED for {a.ref.name} -- {A4B_SKIP_REASON}. "
              "Pass --allow-gemma4-a4b to measure it anyway.", flush=True)
        return 3

    a.work.mkdir(parents=True, exist_ok=True)
    ppl = find_perplexity()
    data = fetch_wikitext(a.work)
    basis = ref_pass(ppl, a.ref, data, a.work, a.ctx, a.chunks, a.threads)
    stem = a.repo.split("/")[-1].removesuffix("-GGUF")
    done = {r["quant"] for r in load_rows(a.work)}
    todo = [t for t in _hub_ggufs(a.repo)
            if (not a.tiers or _quant_of(t[0], stem) in a.tiers)]
    print(f"  kld: {len(todo)} tiers on {a.repo}, {len(done)} already measured", flush=True)
    failures = []
    for fname, size, sha in todo:
        q = _quant_of(fname, stem)
        if q in done:
            continue
        local = a.work / fname
        try:
            if not (local.exists() and local.stat().st_size == size):
                print(f"  {q}: download {size / 1e9:.2f} GB", flush=True)
                _download(a.repo, fname, a.work)
            if sha and _sha256(local) != sha:
                raise RuntimeError("sha256 differs from the Hub LFS oid")
            row = measure(ppl, local, q, basis, data, a.threads)
            append_row(a.work, row)
        except Exception as e:  # noqa: BLE001 -- record and continue; reported at the end
            print(f"  {q}: KLD FAILED — {e}", flush=True)
            failures.append((q, str(e)))
        finally:
            if not a.keep and local.exists():
                local.unlink()
    rows = load_rows(a.work)
    if rows:
        md = write_reports(a.work, basis, rows, stem)
        print(f"  kld: wrote {md} ({len(rows)} tiers)", flush=True)
        if a.upload:
            upload_reports(a.repo, a.work)
            print(f"  kld: uploaded perplexity.md + kld_results.json to {a.repo}", flush=True)
    for q, e in failures:
        print(f"  FAILED {q}: {e}", flush=True)
    print("KLD_DONE" if not failures else f"KLD_DONE_WITH_FAILURES n={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
