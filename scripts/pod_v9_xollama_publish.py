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
another GPU architecture or KV-cache type. The HF token is read from $HF_TOKEN, else
~/.cache/huggingface/token, and never printed.

OTHER MODELS (2026-10-25, JackOD-9B-Coder v3): every gemma-4 value above is a default, not a fork.
--renderer/--parser/--requires set the tag config; --drafter-repo/--drafter-file/--drafter-rev/
--drafter-sha pick the external drafter (a DFlash drafter is attached exactly like the gemma
assistant: one DRAFT layer, xollama infers draft-dflash from its arch); --draft-n is its
draft_num_predict; --mmproj none publishes no vision tags; --variants picks the tag kinds and
--mtp-prefix their names (an empty prefix puts the drafter tags at plain <T>, for a model name
that is itself the drafter variant, e.g. mannix/JackOD-9B-Coder-dflash). A built-in MTP head
needs no DRAFT layer: its draft_num_predict goes in --params like any other PARAMETER.
--sanity-log takes a comma list (the tiers can come from more than one quantize run).

--no-gate publishes without either GPU gate (no verdict wait, no generation check; the
registry read-back still runs), for a host whose GPU is busy. --sanity-round is the separate
action that certifies what is ALREADY published: every --tiers x --variants tag is pulled from
the registry, asked the three questions, and removed again; the failing tags are printed as
SANITY_ROUND_FAIL lines for manual removal (ollama.com has no delete API). No HF token is
needed for public repos.
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


TAG_CFG = {"renderer": "gemma4", "parser": "gemma4", "requires": None}   # set from argv in main()


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
    if info["config"].get("renderer") != TAG_CFG["renderer"] or info["config"].get("parser") != TAG_CFG["parser"]:
        return False, f"renderer/parser {info['config'].get('renderer')}/{info['config'].get('parser')}"
    if TAG_CFG["requires"] and info["config"].get("requires") != TAG_CFG["requires"]:
        return False, f"requires {info['config'].get('requires')!r}"
    return True, "ok"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def hf_lfs(api_, repo: str, revision: str | None = None) -> dict:
    info = api_.model_info(repo, revision=revision, files_metadata=True)
    return {s.rfilename: s.lfs.sha256 for s in info.siblings or [] if s.lfs}


def download(repo: str, fname: str, sha: str, dest: Path, token: str | None, rev: str = "main") -> Path:
    out = dest / fname
    if out.exists() and sha256(out) == sha:
        return out
    dest.mkdir(parents=True, exist_ok=True)
    inp = dest / ".aria_in"
    fd = os.open(inp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"https://huggingface.co/{repo}/resolve/{rev}/{fname}\n  out={fname}\n"
                + (f"  header=Authorization: Bearer {token}\n" if token else ""))
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
    lines += [f"RENDERER {TAG_CFG['renderer']}", f"PARSER {TAG_CFG['parser']}"]
    if TAG_CFG["requires"]:
        lines.append(f"REQUIRES {TAG_CFG['requires']}")
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
    if f"RENDERER {TAG_CFG['renderer']}" not in mf or f"PARSER {TAG_CFG['parser']}" not in mf:
        raise RuntimeError(f"{ref}: renderer/parser missing in the created model")
    # /api/show has no top-level think_budget: it is a line of the `parameters` text,
    # e.g. `think_budget                   "medium"`.
    tb = next((ln.split(None, 1)[1].strip().strip('"') for ln in show.get("parameters", "").splitlines()
               if ln.split(None, 1)[:1] == ["think_budget"]), None)
    if "think_budget" in want and tb != want["think_budget"]:
        raise RuntimeError(f"{ref}: think_budget {tb!r}")
    if want.get("think_budget_message") and want["think_budget_message"] not in show.get("parameters", "") \
            and json.dumps(want["think_budget_message"])[1:-1] not in show.get("parameters", ""):
        raise RuntimeError(f"{ref}: think_budget_message not carried verbatim")
    if mtp and not show.get("drafter"):
        raise RuntimeError(f"{ref}: no drafter attached")
    if vision and "vision" not in (show.get("capabilities") or []):
        raise RuntimeError(f"{ref}: no vision capability ({show.get('capabilities')})")


def sanity_verdict(log_paths: list[Path], tier: str) -> str | None:
    """quantize_gguf's verdict for one tier: 'PASS', 'FAIL', or None while it is not written yet.
    Several logs (several quantize runs) may be given; any FAIL wins over a PASS."""
    pat = re.compile(rf"^\s+{re.escape(tier)}: sanity (PASS|FAIL) \(")
    verdicts = []
    for log_path in log_paths:
        if not log_path.exists():
            continue
        verdict = None
        with open(log_path, errors="replace") as f:
            for line in f:
                m = pat.match(line)
                if m:
                    verdict = m.group(1)    # the last verdict wins (a resumed run re-checks)
        if verdict:
            verdicts.append(verdict)
    return "FAIL" if "FAIL" in verdicts else ("PASS" if verdicts else None)


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


def sanity_round(ns_name: str, tags: list[str], report: Path) -> int:
    """--sanity-round: certify tags that are ALREADY on the registry, as a user gets them.
    Each tag is pulled from ollama.com (authenticated, so a private model works too), asked the
    three questions of generation_check through xollama, then removed from the local store so
    one tag's blobs are resident at a time. Nothing is pushed or deleted on the registry:
    ollama.com has no delete API, so the failing tags are listed for the owner to remove by hand.
    Sentinel: SANITY_ROUND_DONE pass=<n> fail=<n>, then one SANITY_ROUND_FAIL line per failing tag."""
    failed = []
    for tag in tags:
        ref = f"{ns_name}:{tag}"
        xo("rm", ref, check=False)    # the read-back must be the registry's, not a stale local copy
        try:
            if registry_readback(ns_name, tag) is None:
                raise RuntimeError("pull failed (tag absent or key not registered)")
            generation_check(ref)
            verdict = "PASS"
        except Exception as e:  # noqa: BLE001 — a failing tag is the finding, not a crash
            log(f"  {ref} SANITY FAIL: {e}")
            verdict, _ = "FAIL", failed.append(tag)
        with open(report, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{ref}\tsanity_{verdict}\n")
        xo("rm", ref, check=False)
    log(f"###### SANITY_ROUND_DONE pass={len(tags) - len(failed)} fail={len(failed)} ######")
    for tag in failed:
        print(f"SANITY_ROUND_FAIL {ns_name}:{tag}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="ManniX-ITA/gemma-4-20B-v9-agentic-GGUF")
    ap.add_argument("--ollama", default="mannix/gemma-4-20B-v9-agentic")
    ap.add_argument("--stem", default="gemma-4-20B-v9-agentic")
    ap.add_argument("--tiers", required=True, help="comma list, in publish order")
    ap.add_argument("--params", help="JSON of PARAMETER values (no num_ctx); required to publish")
    ap.add_argument("--mmproj", help="vision projector, or 'none' for a text-only model; required to publish")
    ap.add_argument("--renderer", default="gemma4")
    ap.add_argument("--parser", default="gemma4")
    ap.add_argument("--requires", default=None, help="Modelfile REQUIRES (minimum ollama version)")
    ap.add_argument("--drafter-repo", default=DRAFTER_REPO)
    ap.add_argument("--drafter-file", default=DRAFTER_FILE)
    ap.add_argument("--drafter-rev", default="main", help="HF revision of the drafter (pin it)")
    ap.add_argument("--drafter-sha", default=None, help="expected sha256 of the drafter GGUF")
    ap.add_argument("--draft-n", type=int, default=DRAFT_N, help="draft_num_predict of the drafter tags")
    ap.add_argument("--variants", default="text,vision,mtp,mtp-vision",
                    help="tag kinds per tier, comma list of text|vision|mtp|mtp-vision; "
                         ":latest is the first kind at --latest")
    ap.add_argument("--mtp-prefix", default="mtp-", help="name prefix of the drafter tags ('' = plain <T>)")
    ap.add_argument("--scratch", default="/workspace/ollama_scratch")
    ap.add_argument("--keep", default="Q6_K", help="tiers whose GGUF is kept after publishing (for evals)")
    ap.add_argument("--keep-dir", default="/workspace/eval_gguf")
    ap.add_argument("--latest", default="Q4_K_M")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--sanity-log", default="/workspace/logs/quants.out",
                    help="quantize_gguf.py output(s), comma list; a tier is published only after its 'sanity PASS'")
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
    ap.add_argument("--no-gate", action="store_true",
                    help="publish WITHOUT the GPU gates (sanity-log verdict wait + generation check); "
                         "run --sanity-round on the published tags afterwards")
    ap.add_argument("--sanity-round", action="store_true",
                    help="separate action: pull every --tiers x --variants tag from the registry and run the "
                         "generation check on it; publishes nothing")
    args = ap.parse_args()

    tiers = [t for t in args.tiers.split(",") if t]
    kinds = [k for k in args.variants.split(",") if k]
    bad_kinds = set(kinds) - {"text", "vision", "mtp", "mtp-vision"}
    assert kinds and not bad_kinds, f"--variants: unknown kinds {sorted(bad_kinds)}"
    if args.sanity_round:
        pfx = args.mtp_prefix
        names = {"text": "{t}", "vision": "vision-{t}", "mtp": pfx + "{t}", "mtp-vision": pfx + "vision-{t}"}
        return sanity_round(args.ollama, [names[k].format(t=t) for t in tiers for k in kinds], Path(args.report))
    assert args.params and args.mmproj, "publishing needs --params and --mmproj"

    from huggingface_hub import HfApi
    tok_file = Path.home().joinpath(".cache/huggingface/token")
    token = os.environ.get("HF_TOKEN") or (tok_file.read_text().strip() if tok_file.exists() else None)
    if not token:
        log("no HF token: anonymous Hub access (public repos only)")
    hf = HfApi(token=token)
    TAG_CFG.update(renderer=args.renderer, parser=args.parser, requires=args.requires)
    base_params = json.load(open(args.params))
    assert "num_ctx" not in base_params
    keep = set(args.keep.split(","))
    scratch = Path(args.scratch)
    mmproj = None if args.mmproj == "none" else Path(args.mmproj)
    assert mmproj or not {"vision", "mtp-vision"} & set(kinds), "--mmproj none with a vision variant"
    while mmproj and not mmproj.exists():
        log(f"waiting for {mmproj}")
        time.sleep(args.poll)
    sanity_logs = [Path(p) for p in args.sanity_log.split(",") if p]

    drafter = None
    if {"mtp", "mtp-vision"} & set(kinds):
        dl = hf_lfs(hf, args.drafter_repo, args.drafter_rev)
        if args.drafter_sha and dl[args.drafter_file] != args.drafter_sha:
            raise RuntimeError(f"drafter {args.drafter_file} @ {args.drafter_rev}: LFS sha "
                               f"{dl[args.drafter_file][:16]} != --drafter-sha {args.drafter_sha[:16]}")
        drafter = download(args.drafter_repo, args.drafter_file, dl[args.drafter_file], scratch / "drafter",
                           token, rev=args.drafter_rev)
        log(f"drafter {drafter.name} sha {dl[args.drafter_file][:16]} ({args.drafter_repo}@{args.drafter_rev})")

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
        verdict = "PASS" if args.no_gate else None
        while verdict is None and (verdict := sanity_verdict(sanity_logs, t)) is None:
            log(f"{t}: waiting for its sanity verdict in {args.sanity_log}")
            time.sleep(args.poll)
        if verdict != "PASS":
            log(f"{t}: sanity {verdict} in {args.sanity_log} -- NOT PUBLISHED")
            with open(report, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{args.ollama}:{t}\t-\tskipped_sanity_{verdict}\n")
            continue
        pfx = args.mtp_prefix
        kind_tag = {"text": (t, False, False), "vision": (f"vision-{t}", True, False),
                    "mtp": (f"{pfx}{t}", False, True), "mtp-vision": (f"{pfx}vision-{t}", True, True)}
        variants = [kind_tag[k] for k in kinds]
        if t == args.latest:
            variants.append(("latest", *kind_tag[kinds[0]][1:]))
        want = {}
        for tag, vis, mtp in variants:
            p = dict(base_params)
            if mtp:
                p["draft_num_predict"] = args.draft_n
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
                    if not args.no_gate:
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
