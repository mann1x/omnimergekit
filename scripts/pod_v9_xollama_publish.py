#!/usr/bin/env python3
"""pod_v9_xollama_publish.py — ON THE POD: publish gemma-4-20B-v9-agentic to ollama.com with xollama.

For every tier, as soon as its GGUF is on the HF GGUF repo, four tags are created and pushed:

    <T>              text
    vision-<T>       text + the Gemma 4 vision projector (second FROM)
    mtp-<T>          text + the Gemma 4 26B-A4B assistant drafter (DRAFT, draft_num_predict 2)
    mtp-vision-<T>   both

plus :latest = the text Q4_K_M. Every tag carries RENDERER gemma4 / PARSER gemma4 (xollama's
renderer re-injects prior reasoning exactly like Google's official template) and the
deployment sampler + think budget in params/ollama_params.json. No num_ctx.

A push is never trusted from its exit code: `ollama push` exits 0 on an unregistered key. Each
tag is read back from the registry (manifest layers, params blob, config renderer/parser)
before it counts. A tag already on the registry with the right content is skipped, so a
re-run resumes.

Two gates stand between a GGUF and the registry (bug-784: CD-IQ2_XS_L failed its sanity check
0/3 and was still pushed, because the publisher only waited for the file to appear on HF, and
quantize_gguf.py uploads BEFORE it checks and leaves failures in place):
  1. the tier's own `<T>: sanity PASS` verdict in quantize_gguf's log; a FAIL skips the tier;
  2. every created tag must answer the same three questions through xollama itself, as
     deployed (renderer, params, drafter, projector), before it is pushed.
The second gate runs on THIS host's GPU and engine: it cannot see a fault that only shows on
another GPU architecture or KV-cache type. The HF token is read from ~/.cache/huggingface/token and never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "127.0.0.1:22434"
ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
DRAFTER_REPO = "ManniX-ITA/gemma-4-26B-A4B-it-assistant-GGUF"
DRAFTER_FILE = "gemma-4-26B-A4B-it-assistant.Q8_0.gguf"
DRAFT_N = 2          # the length xollama's --drafter=auto sets; measured best for these heads
# The three questions of quantize_gguf.py SANITY_QUESTIONS, pass at >=2/3, same as there.
SANITY_QUESTIONS = [
    ("What is the capital of France? Reply ONLY with JSON: {\"capital\": \"...\"}", "Paris"),
    ("What is the capital of Japan? Reply ONLY with JSON: {\"capital\": \"...\"}", "Tokyo"),
    ("What is the capital of Germany? Reply ONLY with JSON: {\"capital\": \"...\"}", "Berlin"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def xo(*args: str, check: bool = True, timeout: int = 7200) -> subprocess.CompletedProcess:
    env = dict(os.environ, OLLAMA_HOST=HOST)
    r = subprocess.run(["xollama", *args], env=env, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"xollama {' '.join(args)} rc={r.returncode}: {(r.stderr or r.stdout)[-600:]}")
    return r


def api(path: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(f"http://{HOST}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def registry_get(url: str, accept: str | None = None) -> bytes | None:
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def registry_tag(ns_name: str, tag: str) -> dict | None:
    """Return {'layers': {mediaType-suffix: layer}, 'params': {...}, 'config': {...}} or None."""
    base = f"https://registry.ollama.ai/v2/{ns_name}"
    raw = registry_get(f"{base}/manifests/{tag}", ACCEPT)
    if raw is None:
        return None
    man = json.loads(raw)
    layers = {}
    for lay in man.get("layers", []):
        layers.setdefault(lay["mediaType"].rsplit(".", 1)[-1], []).append(lay)
    params = {}
    if "params" in layers:
        params = json.loads(registry_get(f"{base}/blobs/{layers['params'][0]['digest']}") or b"{}")
    config = json.loads(registry_get(f"{base}/blobs/{man['config']['digest']}") or b"{}")
    return {"layers": layers, "params": params, "config": config}


STORE = Path(os.environ.get("OLLAMA_MODELS", "/workspace/xollama_models"))


def manifest_path(ns_name: str, tag: str) -> Path:
    """The tag's manifest in the LOCAL store. xollama 0.40 writes a newly created model under
    manifests-v2/ollama.com/, older entries sit under manifests/registry.ollama.ai/; when both
    exist the newest file is the one the daemon just wrote (create, or the read-back pull)."""
    cands = [p for p in (STORE / "manifests-v2/ollama.com" / ns_name / tag,
                         STORE / "manifests/registry.ollama.ai" / ns_name / tag) if p.exists()]
    if not cands:
        raise RuntimeError(f"{ns_name}:{tag}: no manifest in the local store ({STORE})")
    return max(cands, key=lambda p: p.stat().st_mtime_ns)


def registry_readback(ns_name: str, tag: str) -> dict | None:
    """Authenticated read-back. The model is PRIVATE on ollama.com, so an anonymous registry GET
    is a 404 even after a good push. /api/pull fetches the registry's manifest with the push key and
    writes it over the local one (the blobs are already local, so nothing large moves); the manifest
    and blobs are then read from the store, which now holds exactly what the registry serves."""
    req = urllib.request.Request(f"http://{HOST}/api/pull", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": f"{ns_name}:{tag}"}).encode())
    last = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.strip():
                last = json.loads(line)
                if "error" in last:
                    return None
    if last.get("status") != "success":
        return None
    man = json.loads(manifest_path(ns_name, tag).read_text())

    def blob(d: str) -> bytes:
        return (STORE / "blobs" / d.replace(":", "-")).read_bytes()

    layers = {}
    for lay in man.get("layers", []):
        layers.setdefault(lay["mediaType"].rsplit(".", 1)[-1], []).append(lay)
    params = json.loads(blob(layers["params"][0]["digest"])) if "params" in layers else {}
    return {"layers": layers, "params": params, "config": json.loads(blob(man["config"]["digest"]))}


def model_digest(ns_name: str, tag: str) -> str:
    """Digest of the model layer of a tag as it sits in the LOCAL store right now."""
    man = json.loads(manifest_path(ns_name, tag).read_text())
    models = [lay["digest"] for lay in man.get("layers", []) if lay["mediaType"].endswith(".model")]
    if len(models) != 1:
        raise RuntimeError(f"{ns_name}:{tag}: {len(models)} model layers in the local manifest")
    return models[0]


def expected_layers(vision: bool, mtp: bool) -> dict:
    exp = {"model": 1}
    if vision:
        exp["projector"] = 1
    if mtp:
        exp["draft"] = 1
    return exp


def tag_ok(info: dict | None, want_params: dict, vision: bool, mtp: bool) -> tuple[bool, str]:
    if info is None:
        return False, "absent"
    have = {k: len(v) for k, v in info["layers"].items() if k in ("model", "projector", "draft")}
    if have != expected_layers(vision, mtp):
        return False, f"layers {have}"
    for k, v in want_params.items():
        got = info["params"].get(k)
        if isinstance(v, float):
            if not isinstance(got, (int, float)) or abs(got - v) > 1e-9:
                return False, f"param {k}={got!r}"
        elif got != v:
            return False, f"param {k}={got!r}"
    extra = set(info["params"]) - set(want_params)
    if extra:
        return False, f"unexpected params {sorted(extra)}"
    if info["config"].get("renderer") != "gemma4" or info["config"].get("parser") != "gemma4":
        return False, f"renderer/parser {info['config'].get('renderer')}/{info['config'].get('parser')}"
    return True, "ok"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def hf_lfs(api_, repo: str) -> dict:
    return {s.rfilename: s.lfs.sha256 for s in api_.model_info(repo, files_metadata=True).siblings if s.lfs}


def download(repo: str, fname: str, sha: str, dest: Path, token: str) -> Path:
    out = dest / fname
    if out.exists() and sha256(out) == sha:
        return out
    dest.mkdir(parents=True, exist_ok=True)
    inp = dest / ".aria_in"
    fd = os.open(inp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"https://huggingface.co/{repo}/resolve/main/{fname}\n  out={fname}\n"
                f"  header=Authorization: Bearer {token}\n")
    r = subprocess.run(["aria2c", "-x", "16", "-s", "16", "-k", "64M", "-c", "--file-allocation=none",
                        "--console-log-level=warn", "--summary-interval=0", "-d", str(dest), "-i", str(inp)],
                       capture_output=True, text=True)
    inp.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"aria2c {fname} rc={r.returncode}: {r.stdout[-400:]}")
    got = sha256(out)
    if got != sha:
        raise RuntimeError(f"{fname} sha {got[:16]} != LFS {sha[:16]}")
    return out


def modelfile(gguf: Path, mmproj: Path | None, drafter: Path | None, params: dict) -> str:
    lines = [f"FROM {gguf}"]
    if mmproj:
        lines.append(f"FROM {mmproj}")
    if drafter:
        lines.append(f"DRAFT {drafter}")
    lines += ["RENDERER gemma4", "PARSER gemma4"]
    for k, v in params.items():
        if isinstance(v, str) and k == "think_budget_message":
            assert '"' not in v, "message must not contain a double quote"
            lines.append(f'PARAMETER {k} "{v}"')
        else:
            lines.append(f"PARAMETER {k} {v}")
    return "\n".join(lines) + "\n"


def local_check(ref: str, want: dict, vision: bool, mtp: bool) -> None:
    show = api("/api/show", {"model": ref})
    mf = show.get("modelfile", "")
    if "RENDERER gemma4" not in mf or "PARSER gemma4" not in mf:
        raise RuntimeError(f"{ref}: renderer/parser missing in the created model")
    # /api/show has no top-level think_budget: it is a line of the `parameters` text,
    # e.g. `think_budget                   "medium"`.
    tb = next((ln.split(None, 1)[1].strip().strip('"') for ln in show.get("parameters", "").splitlines()
               if ln.split(None, 1)[:1] == ["think_budget"]), None)
    if tb != want["think_budget"]:
        raise RuntimeError(f"{ref}: think_budget {tb!r}")
    if want.get("think_budget_message") and want["think_budget_message"] not in show.get("parameters", "") \
            and json.dumps(want["think_budget_message"])[1:-1] not in show.get("parameters", ""):
        raise RuntimeError(f"{ref}: think_budget_message not carried verbatim")
    if mtp and not show.get("drafter"):
        raise RuntimeError(f"{ref}: no drafter attached")
    if vision and "vision" not in (show.get("capabilities") or []):
        raise RuntimeError(f"{ref}: no vision capability ({show.get('capabilities')})")


def sanity_verdict(log_path: Path, tier: str) -> str | None:
    """quantize_gguf's verdict for one tier: 'PASS', 'FAIL', or None while it is not written yet."""
    if not log_path.exists():
        return None
    pat = re.compile(rf"^\s+{re.escape(tier)}: sanity (PASS|FAIL) \(")
    verdict = None
    with open(log_path, errors="replace") as f:
        for line in f:
            m = pat.match(line)
            if m:
                verdict = m.group(1)    # the last verdict wins (a resumed run re-checks)
    return verdict


def generation_check(ref: str) -> None:
    """Ask the created tag the three sanity questions through xollama; raise below 2/3."""
    passed, seen = 0, []
    for prompt, expected in SANITY_QUESTIONS:
        d = api("/api/chat", {"model": ref, "stream": False, "keep_alive": 0,
                              "messages": [{"role": "user", "content": prompt}],
                              "options": {"num_ctx": 4096, "num_predict": 2048}}, timeout=900)
        m = d.get("message") or {}
        # The VISIBLE reply must carry the answer. quantize_gguf's check also accepts the
        # thinking block, so a tag that names the city while reasoning and then answers with
        # nothing (or garbage) would pass it; this gate must not.
        content = m.get("content") or ""
        passed += expected.lower() in content.lower()
        seen.append(f"{expected}->{content.strip()[:60]!r} (thinking {len(m.get('thinking') or '')} ch)")
    if passed < 2:
        raise RuntimeError(f"{ref}: generation check {passed}/3: {'; '.join(seen)}")
    log(f"  {ref} generation check {passed}/3: {'; '.join(seen)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="ManniX-ITA/gemma-4-20B-v9-agentic-GGUF")
    ap.add_argument("--ollama", default="mannix/gemma-4-20B-v9-agentic")
    ap.add_argument("--stem", default="gemma-4-20B-v9-agentic")
    ap.add_argument("--tiers", required=True, help="comma list, in publish order")
    ap.add_argument("--params", required=True, help="JSON of PARAMETER values (no num_ctx)")
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--scratch", default="/workspace/ollama_scratch")
    ap.add_argument("--keep", default="Q6_K", help="tiers whose GGUF is kept after publishing (for evals)")
    ap.add_argument("--keep-dir", default="/workspace/eval_gguf")
    ap.add_argument("--latest", default="Q4_K_M")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--sanity-log", default="/workspace/logs/quants.out",
                    help="quantize_gguf.py output; a tier is published only after its 'sanity PASS'")
    ap.add_argument("--report", default="/workspace/logs/xollama_publish.tsv")
    # REPLACE mode (2026-10-16, v9-agentic re-published from the v9-ml ml_k8 build): the HF repo and
    # the registry already hold the PREVIOUS build under the same file and tag names, so neither
    # "the file is on HF" nor "the tag reads back with the right params" says anything about which
    # weights are there. With --gguf-dir a tier is ready only when quantize_gguf's own sidecar
    # <gguf-dir>/<file>.sha256 exists AND the HF LFS sha equals it (= this run's upload); every tag
    # is re-created and pushed (only tags this run's --report already lists as ok are skipped, so a
    # restart resumes); and the registry's model-layer digest after the push must equal the one
    # `create` produced locally (ollama re-serialises a GGUF, so the HF sha cannot be compared).
    ap.add_argument("--gguf-dir", default=None, help="REPLACE mode: quantize_gguf output dir (sidecars)")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    token = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    assert token, "HF token file is empty"
    hf = HfApi(token=token)
    base_params = json.load(open(args.params))
    assert "num_ctx" not in base_params
    tiers = [t for t in args.tiers.split(",") if t]
    keep = set(args.keep.split(","))
    scratch = Path(args.scratch)
    mmproj = Path(args.mmproj)
    while not mmproj.exists():
        log(f"waiting for {mmproj}")
        time.sleep(args.poll)

    dl = hf_lfs(hf, DRAFTER_REPO)
    drafter = download(DRAFTER_REPO, DRAFTER_FILE, dl[DRAFTER_FILE], scratch / "drafter", token)
    log(f"drafter {drafter.name} sha {dl[DRAFTER_FILE][:16]}")

    report = Path(args.report)
    replace = args.gguf_dir is not None
    done_ok = set()
    if replace and report.exists():
        done_ok = {(ln.split("\t")[1], ln.split("\t")[2]) for ln in report.read_text().splitlines()
                   if ln.count("\t") >= 3 and ln.split("\t")[3] == "ok"}
    for t in tiers:
        fname = f"{args.stem}-{t}.gguf"
        while True:
            lfs = hf_lfs(hf, args.hf_repo)
            if replace:
                side = Path(args.gguf_dir) / f"{fname}.sha256"
                want_sha = side.read_text().split()[0] if side.exists() else None
                if want_sha and lfs.get(fname) == want_sha:
                    break
                log(f"{t}: waiting for THIS run's {fname} on {args.hf_repo} "
                    f"(sidecar {'absent' if not want_sha else want_sha[:16]}, hub {str(lfs.get(fname))[:16]})")
            elif fname in lfs:
                break
            else:
                log(f"{t}: waiting for {fname} on {args.hf_repo}")
            time.sleep(args.poll)
        while (verdict := sanity_verdict(Path(args.sanity_log), t)) is None:
            log(f"{t}: waiting for its sanity verdict in {args.sanity_log}")
            time.sleep(args.poll)
        if verdict != "PASS":
            log(f"{t}: sanity {verdict} in {args.sanity_log} -- NOT PUBLISHED")
            with open(report, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{args.ollama}:{t}\t-\tskipped_sanity_{verdict}\n")
            continue
        variants = [(t, False, False), (f"vision-{t}", True, False),
                    (f"mtp-{t}", False, True), (f"mtp-vision-{t}", True, True)]
        if t == args.latest:
            variants.append(("latest", False, False))
        want = {}
        for tag, vis, mtp in variants:
            p = dict(base_params)
            if mtp:
                p["draft_num_predict"] = DRAFT_N
            want[tag] = (p, vis, mtp)
        if replace:
            todo = [tag for tag in want if (f"{args.ollama}:{tag}", lfs[fname]) not in done_ok]
        else:
            todo = [tag for tag, (p, vis, mtp) in want.items()
                    if not tag_ok(registry_readback(args.ollama, tag), p, vis, mtp)[0]]
        if not todo:
            log(f"{t}: all {len(want)} tags already published, skipping")
            continue
        gguf = download(args.hf_repo, fname, lfs[fname], scratch, token)
        log(f"{t}: {fname} {gguf.stat().st_size / 1e9:.2f} GB sha {lfs[fname][:16]}; tags {todo}")
        for tag in todo:
            p, vis, mtp = want[tag]
            ref = f"{args.ollama}:{tag}"
            mf = scratch / f"Modelfile.{tag}"
            mf.write_text(modelfile(gguf, mmproj if vis else None, drafter if mtp else None, p))
            for attempt in range(1, 4):
                try:
                    xo("rm", ref, check=False)    # no stale local manifest can be read back
                    xo("create", ref, "-f", str(mf))
                    local_check(ref, p, vis, mtp)
                    made = model_digest(args.ollama, tag)
                    generation_check(ref)
                    xo("push", ref)
                    info = registry_readback(args.ollama, tag)
                    ok, why = tag_ok(info, p, vis, mtp)
                    if not ok or info is None:
                        raise RuntimeError(f"registry read-back: {why}")
                    served = info["layers"]["model"][0]["digest"]
                    if served != made:
                        raise RuntimeError(f"registry serves model {served[:19]}, created {made[:19]}")
                    log(f"  {ref} PUBLISHED (verified on the registry)")
                    with open(report, "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{ref}\t{lfs[fname]}\tok\n")
                    break
                except Exception as e:  # noqa: BLE001 — logged, retried, then fatal
                    log(f"  {ref} attempt {attempt} FAILED: {e}")
                    if attempt == 3:
                        log(f"FATAL {ref}")
                        return 2
                    time.sleep(60 * attempt)
            xo("rm", ref, check=False)
        if t in keep:
            Path(args.keep_dir).mkdir(parents=True, exist_ok=True)
            gguf.rename(Path(args.keep_dir) / fname)
            log(f"{t}: kept at {args.keep_dir}/{fname}")
        else:
            gguf.unlink()
    log("###### XOLLAMA_PUBLISH_DONE ######")
    return 0


if __name__ == "__main__":
    sys.exit(main())
