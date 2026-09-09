#!/usr/bin/env python3
"""Re-publish every ollama tag of a model with a Go RENDERER/PARSER attached.

WHY THIS EXISTS
---------------
ollama only uses its built-in Go renderer/parser when the manifest CONFIG says so:

    server/images.go:361 ->  if m.Config.Renderer != "" || m.Config.Parser != ""

Those two fields are populated ONLY from the Modelfile's `RENDERER` / `PARSER`
directives. A model created from a bare GGUF leaves them EMPTY, so ollama silently
falls back to `gguf_chat_template` -- the jinja template baked into the file. That is
how `mannix/qwen3.6-27b-a3b-coder` shipped: 35 tags, every one of them with

    {"model_family":"qwen35moe", ...}                    <- no renderer, no parser

while the upstream reference `library/qwen3.6:35b`, same architecture, ships

    {"model_family":"qwen35moe", "renderer":"qwen3.5", "parser":"qwen3.5", ...}

The defect is invisible from `ollama list`, invisible from the model page, and
invisible from the GGUF -- it lives in a 462-byte config blob. So the check here is
NOT "did the Modelfile say RENDERER": it is "does the blob the REGISTRY serves say
renderer". Ask the service, never the flag.

WHAT IT DOES, PER TAG
---------------------
    pull -> create(FROM <tag> + RENDERER + PARSER) -> verify -> push -> registry
    verify -> rm + blob GC (bounded disk) -> .done marker (resumable)

`FROM <existing tag>` re-uses the source manifest's layers by digest, so the weights,
projector, params and license blobs are carried over BYTE-IDENTICAL -- nothing is
re-imported and nothing is re-quantized. That is asserted, not assumed: the model
layer digests are compared before/after and a mismatch is fatal for that tag.

USAGE
    python3 ollama_set_renderer_parser.py --model qwen3.6-27b-a3b-coder \
        --renderer qwen3.5 --parser qwen3.5 [--dry-run] [--only Q4_K_M,latest]

Launch detached:
    setsid nohup python3 ollama_set_renderer_parser.py --model ... \
        >/srv/ml/ollama_renderer_fix/launch.log 2>&1 </dev/null &
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

NS = "mannix"
OL = "/usr/local/bin/ollama"
def _resolve_store():
    """Where the DAEMON actually keeps models -- ask it, never assume.

    A hardcoded path silently breaks the whole sweep: every pull succeeds, then
    every manifest read comes back empty and each tag dies FAILPULL-nomodel-layer
    (bs2, 2026-09-08: daemon runs OLLAMA_MODELS=/mnt/sdc/ollama/models while this
    constant pointed at /usr/share/ollama/.ollama/models).

    Order: our own env -> the running daemon's env -> daemon HOME -> legacy default.
    """
    env = os.environ.get("OLLAMA_MODELS")
    if env and os.path.isdir(env):
        return env
    try:
        pid = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True,
                             text=True, timeout=15).stdout.split()
        if pid:
            with open("/proc/%s/environ" % pid[0], "rb") as f:
                envd = dict(
                    kv.split("=", 1) for kv in
                    f.read().decode("utf-8", "replace").split("\0") if "=" in kv)
            m = envd.get("OLLAMA_MODELS")
            if m and os.path.isdir(m):
                return m
            h = envd.get("HOME")
            if h and os.path.isdir(os.path.join(h, ".ollama", "models")):
                return os.path.join(h, ".ollama", "models")
    except Exception:
        pass
    return "/usr/share/ollama/.ollama/models"


STORE = _resolve_store()
BLOBS = os.path.join(STORE, "blobs")
REGISTRY = "https://registry.ollama.ai/v2"

# bs2 root fs must never drop under this (standing host rule). Checked BEFORE every
# pull, because the largest tier here is ~28 GB and a mid-run ENOSPC leaves a partial
# blob plus a half-written manifest.
FREE_FLOOR_GB = 220

# Smallest-first. A recipe that is going to fail (unknown renderer name, create
# rejected, push auth) fails on an 8 GB tier in minutes instead of a 28 GB one.
TIER_ORDER = [
    "IQ2_XS", "Q2_K_L", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K_XL",
    "IQ4_XS", "IQ4_NL", "Q4_K_S", "Q4_K_M", "Q4_K_L",
    "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q6_K", "Q6_K_L", "Q8_0",
]
TAG_RE = re.compile(r"^(CD-|mtp-)?(qat-)?(F16|Q[0-9][A-Za-z0-9_]*|IQ[0-9][A-Za-z0-9_]*)$")


def log(msg):
    print("[rp %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def run(cmd, to=7200):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=to)


def free_gb(path="/"):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


# ---------------------------------------------------------------- tag discovery
def real_tag(tag):
    t = tag[7:] if tag.startswith("vision-") else tag
    return t in ("qat", "latest") or bool(TAG_RE.match(t))


def tags_from_web(model):
    """The model page is the only public enumeration -- there is no /tags/list for a
    user namespace (it 404s). Noise from the page's CSS/JS is filtered by real_tag."""
    url = "https://ollama.com/%s/%s/tags" % (NS, model)
    try:
        html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
    except Exception as e:
        log("WARN tag page fetch failed (%s) -- falling back to TIER_ORDER" % e)
        return []
    raw = set(re.findall(r"%s:([A-Za-z0-9_.-]+)" % re.escape(model), html))
    return sorted(t for t in raw if real_tag(t))


def order_tags(tags):
    """text tier then its vision sibling, so both share the freshly-pulled model blob."""
    rank = {t: i for i, t in enumerate(TIER_ORDER)}
    out = []
    for t in TIER_ORDER:
        for v in (t, "vision-" + t):
            if v in tags:
                out.append(v)
    for t in sorted(tags):                      # anything TIER_ORDER did not cover
        base = t[7:] if t.startswith("vision-") else t
        if t not in out and base not in rank:
            out.append(t)
    return out


# ---------------------------------------------------------------- manifests
def local_manifest(model, tag):
    p = os.path.join(STORE, "manifests", "registry.ollama.ai", NS, model, tag)
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def layer_digests(man, media_suffix):
    if not man:
        return []
    return sorted(l["digest"] for l in man.get("layers", [])
                  if l["mediaType"].endswith(media_suffix))


def registry_config(model, tag):
    """The authoritative answer: what the REGISTRY will hand a puller."""
    base = "%s/%s/%s" % (REGISTRY, NS, model)
    try:
        with urllib.request.urlopen("%s/manifests/%s" % (base, tag), timeout=60) as r:
            man = json.load(r)
        with urllib.request.urlopen("%s/blobs/%s" % (base, man["config"]["digest"]),
                                    timeout=60) as r:
            return json.load(r)
    except Exception as e:
        return {"_error": str(e)}


def _blob_json(digest):
    """Read a local blob by digest. Returns {} when absent or unparseable."""
    b = os.path.join(BLOBS, digest.replace("sha256:", "sha256-"))
    try:
        with open(b) as f:
            return json.load(f)
    except Exception:
        return {}


def local_params(model, tag):
    """The params dict ollama would apply, read from the local store."""
    man = local_manifest(model, tag)
    for d in layer_digests(man, ".image.params"):
        return _blob_json(d)
    return {}


def registry_params(model, tag):
    """The params dict the REGISTRY serves. Ask the service, never the flag."""
    base = "%s/%s/%s" % (REGISTRY, NS, model)
    try:
        with urllib.request.urlopen("%s/manifests/%s" % (base, tag), timeout=60) as r:
            man = json.load(r)
        for lay in man.get("layers", []):
            if lay["mediaType"].endswith(".image.params"):
                with urllib.request.urlopen("%s/blobs/%s" % (base, lay["digest"]),
                                            timeout=60) as r:
                    return json.load(r)
    except Exception:
        pass
    return {}


def _params_match(have, want):
    """Every requested key present with the requested value (string-compared)."""
    return all(str(have.get(k)) == str(v) for k, v in want.items())


def gc_blobs(protect):
    """Purge blobs no local manifest references. Builds the reference set from ALL
    manifests, so unrelated local models (v7test, gemma31b-q6-128k, ...) are safe."""
    refs = set(protect)
    root = os.path.join(STORE, "manifests")
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            try:
                with open(os.path.join(dirpath, fn)) as f:
                    man = json.load(f)
            except Exception:
                continue
            # Both fields can be absent OR explicitly null in the store (ollama writes
            # `"layers": null` for some entries); `or {}` / `or []` covers both, while
            # a plain .get(k, default) does NOT -- that was a real crash, mid-sweep.
            for d in [(man.get("config") or {}).get("digest")] + \
                     [l.get("digest") for l in (man.get("layers") or [])]:
                if d:
                    refs.add(d.replace("sha256:", "sha256-"))
    n = 0
    for b in os.listdir(BLOBS) if os.path.isdir(BLOBS) else []:
        if not b.startswith("sha256-"):
            continue
        if b.endswith("-partial") or "-partial-" in b:
            os.remove(os.path.join(BLOBS, b)); n += 1; continue
        if b not in refs:
            os.remove(os.path.join(BLOBS, b)); n += 1
    log("  gc: purged %d orphan blob(s); free %.0f GB" % (n, free_gb()))


# ---------------------------------------------------------------- per-tag work
def process(model, tag, renderer, parser, workdir, done_dir, dry, params=None):
    full = "%s/%s:%s" % (NS, model, tag)
    params = params or {}
    suffix = ""
    if params:
        suffix = "__" + "_".join("%s%s" % (k, v) for k, v in sorted(params.items()))
    marker = os.path.join(done_dir,
                          "%s__%s%s.done" % (model, tag.replace("/", "_"), suffix))
    if os.path.exists(marker):
        return "DONE-CACHED"

    if free_gb() < FREE_FLOOR_GB:
        return "ABORT-DISK (%.0f GB free < %d GB floor)" % (free_gb(), FREE_FLOOR_GB)

    pre = registry_config(model, tag)
    if "_error" in pre:
        open(marker, "w").write("skip-missing %s\n" % pre["_error"])
        return "SKIP-MISSING"
    pre_p = registry_params(model, tag) if params else {}
    if (pre.get("renderer") == renderer and pre.get("parser") == parser
            and (not params or _params_match(pre_p, params))):
        open(marker, "w").write("already-correct\n")
        return "ALREADY-CORRECT"
    if dry:
        return "DRY (registry renderer=%r parser=%r params=%r)" % (
            pre.get("renderer"), pre.get("parser"),
            {k: pre_p.get(k) for k in params} if params else None)

    r = run([OL, "pull", full])
    if r.returncode != 0:
        return "FAILPULL: " + (r.stderr or r.stdout).strip()[:160]

    src = local_manifest(model, tag)
    src_model = layer_digests(src, ".image.model")
    src_proj = layer_digests(src, ".image.projector")
    src_params = layer_digests(src, ".image.params")
    if not src_model:
        return "FAILPULL-nomodel-layer"

    mfp = os.path.join(workdir, "Modelfile.%s.%s" % (model, tag.replace("/", "_")))
    with open(mfp, "w") as f:
        f.write("FROM %s\n" % full)
        f.write("RENDERER %s\n" % renderer)
        f.write("PARSER %s\n" % parser)
        for k, v in params.items():
            f.write("PARAMETER %s %s\n" % (k, v))

    r = run([OL, "create", full, "-f", mfp], to=3600)
    if r.returncode != 0:
        return "FAILCREATE: " + (r.stderr or r.stdout).strip()[:200]

    # --- verify LOCALLY: directives present, weights untouched ------------------
    after = run([OL, "show", full, "--modelfile"], to=180).stdout
    if ("RENDERER %s" % renderer) not in after or ("PARSER %s" % parser) not in after:
        return "FAILVERIFY-modelfile"
    new = local_manifest(model, tag)
    if layer_digests(new, ".image.model") != src_model:
        return "FAILVERIFY-model-layer-changed"       # weights were re-imported: STOP
    if layer_digests(new, ".image.projector") != src_proj:
        return "FAILVERIFY-projector-lost"            # vision capability dropped
    if not params:
        if layer_digests(new, ".image.params") != src_params:
            return "FAILVERIFY-params-changed"        # sampler defaults drifted
    else:
        # Params are being changed ON PURPOSE, so a digest match would mean the
        # edit did NOT take. Assert the SHAPE instead: every requested key set to
        # the requested value, and every pre-existing key carried over untouched.
        src_p = _blob_json(src_params[0]) if src_params else {}
        new_p = local_params(model, tag)
        if not _params_match(new_p, params):
            return "FAILVERIFY-param-not-applied %r" % (
                {k: new_p.get(k) for k in params},)
        dropped = {k: v for k, v in src_p.items()
                   if k not in params and str(new_p.get(k)) != str(v)}
        if dropped:
            return "FAILVERIFY-params-dropped %r" % (dropped,)

    r = run([OL, "push", full])
    if r.returncode != 0:
        return "FAILPUSH: " + (r.stderr or r.stdout).strip()[:160]

    # --- verify at the REGISTRY: this is the only check that proves the fix ------
    post = {}
    post_p = {}
    for _ in range(4):
        post = registry_config(model, tag)
        post_p = registry_params(model, tag) if params else {}
        if (post.get("renderer") == renderer and post.get("parser") == parser
                and (not params or _params_match(post_p, params))):
            break
        time.sleep(6)
    if post.get("renderer") != renderer or post.get("parser") != parser:
        return "FAILREG: registry config renderer=%r parser=%r" % (
            post.get("renderer"), post.get("parser"))
    if params and not _params_match(post_p, params):
        return "FAILREG-params: registry serves %r, wanted %r" % (
            {k: post_p.get(k) for k in params}, params)

    open(marker, "w").write("ok renderer=%s parser=%s params=%s file_type=%s\n"
                            % (renderer, parser,
                               {k: post_p.get(k) for k in params} if params else "-",
                               post.get("file_type")))
    return "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="repo name inside the mannix namespace")
    ap.add_argument("--renderer", required=True)
    ap.add_argument("--parser", required=True)
    ap.add_argument("--work", default="/srv/ml/ollama_renderer_fix")
    ap.add_argument("--only", default="", help="comma-separated tag subset (pilot runs)")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                    help="Modelfile PARAMETER to set on every tag, repeatable "
                         "(e.g. --param draft_num_predict=3). Without it the "
                         "params blob must stay byte-identical, as before.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    workdir = os.path.join(a.work, a.model)
    done_dir = os.path.join(workdir, "done")
    os.makedirs(done_dir, exist_ok=True)

    web = tags_from_web(a.model)
    known = [t for tier in TIER_ORDER for t in (tier, "vision-" + tier)] + ["latest"]
    tags = order_tags(sorted(set(web) | set(known)))
    if a.only:
        want = {t.strip() for t in a.only.split(",") if t.strip()}
        tags = [t for t in tags if t in want]

    log("=" * 78)
    params = {}
    for kv in a.param:
        if "=" not in kv:
            sys.exit("--param must be KEY=VALUE, got %r" % kv)
        k, v = kv.split("=", 1)
        params[k.strip()] = v.strip()
    log("model=%s/%s  renderer=%s  parser=%s  params=%s  dry=%s"
        % (NS, a.model, a.renderer, a.parser, params or "-", a.dry_run))
    log("store=%s" % STORE)
    log("tags from web: %d  |  candidate set: %d  |  free: %.0f GB"
        % (len(web), len(tags), free_gb()))
    log("order: %s" % ", ".join(tags))
    log("=" * 78)

    results = {}
    for i, tag in enumerate(tags, 1):
        log("---------- [%d/%d] %s ----------" % (i, len(tags), tag))
        try:
            res = process(a.model, tag, a.renderer, a.parser, workdir, done_dir,
                          a.dry_run, params=params)
        except subprocess.TimeoutExpired as e:
            res = "TIMEOUT: %s" % e
        except Exception as e:                          # noqa: BLE001 - keep the sweep alive
            res = "EXC: %r" % e
        results[tag] = res
        log("  -> %s" % res)
        if not a.dry_run:
            run([OL, "rm", "%s/%s:%s" % (NS, a.model, tag)], to=600)
            gc_blobs(protect=set())
        if res.startswith("ABORT"):
            log("STOPPING: %s" % res)
            break

    log("=" * 78)
    ok = [t for t, r in results.items() if r in ("OK", "DONE-CACHED", "ALREADY-CORRECT")]
    bad = {t: r for t, r in results.items() if t not in ok}
    log("OK/cached : %d/%d" % (len(ok), len(results)))
    for t, r in bad.items():
        log("  FAIL %-22s %s" % (t, r))
    log("OLLAMA_RENDERER_FIX_FIN model=%s failures=%d" % (a.model, len(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
