#!/usr/bin/env python3
"""Rebuild EVERY v6-coder ollama tag from the eos-fixed GGUFs, onto the v7-coder tag shape.

WHY THIS CANNOT BE A PARAM EDIT
-------------------------------
`ollama_strip_stop_v7.py` rebuilds a tag `FROM` the blob already in the local store, so it can
change parameters while keeping the weights. That is useless here: the v6-coder **weights
changed**. Every published v6 GGUF was rewritten (`eos 1 -> 106`, `eot -> 1`, chat template
`2e4c13d9 -> d9f21aac`), so each ollama model blob is stale and every tag has to be created
from the new file and re-pushed. There is no cheaper path -- the blob IS the artifact.

THE TARGET IS THE v7-coder TAG SHAPE, NOT "v6 MINUS ITS BUGS"
-------------------------------------------------------------
A registry sweep of the 62 published v6 tags found THREE different parameter blocks -- an
accretion of separate publish batches, not a measurement:

    n=12  CD-*            min_p .05 num_ctx 8192 repeat_last_n 256 repeat_penalty 1.1
                          stop ["<turn|>", "<|tool_response>"] temperature 0.6 top_p 0.95
    n=24  K-quants        min_p .05 num_ctx 8192 repeat_penalty 1.1 stop ["<turn|>"] temp 0.6
    n=26  I-quants, Q4_0… min_p .05 num_ctx 8192 repeat_penalty 1.1 temperature 0.6

Contrast v7-coder, where the ONLY axis that varies is `temperature` (0.9 / 0.8), and that one
varies because a per-tier loop gate measured it -- see
`memory/feedback_a_spread_is_a_measurement_not_a_tidyup_target.md`. v6's spread encodes no
measurement, so it is normalised onto v7's block with temperature pinned at 0.9 (user decision,
2026-07-31). If a per-tier v6 loop gate is ever run, THAT result overrides this file.

`stop` must go for the reason in `feedback_ollama_create_mines_jinja_for_stop.md`: token 106
`<turn|>` is already the GGUF's EOG, so the stop string cannot fire in the normal case -- the
only case it CAN fire is a coder model writing the literal characters `<turn|>`, truncating the
answer mid-sentence. Declaring RENDERER + PARSER stops `ollama create` re-mining the jinja.

TWO SENSES OF "TEMPLATE", BOTH PINNED TO v7
-------------------------------------------
1. the GGUF's embedded jinja  -> sha256 d9f21aac…, asserted on the DOWNLOADED file before use.
2. the ollama tag's template layer -> all 27 v7 tags carry the 13-byte passthrough stub
   `{{ .Prompt }}` (sha256 b507b9c2f6ca…), which is what `ollama create` writes when RENDERER
   owns turn termination. Asserted on the CREATED manifest. v6's `IQ4_K_M` today carries a
   different template layer -- a real template layer OVERRIDES the renderer, so that is a bug
   this run removes.

WHAT IS ASSERTED PER TAG BEFORE ANYTHING IS PUSHED
--------------------------------------------------
    params blob == TARGET_PARAMS exactly (not a superset, no stop)
    template layer digest == sha256("{{ .Prompt }}")
    config renderer == parser == "gemma4"
    model blob digest != the OLD published digest      <- proves the weights actually moved
    projector layer present iff the tag is vision-*
    draft layer present iff the tag is mtp-*
A tag failing any of these is not pushed and gets no marker, so a re-run retries it.

Disk-bounded: one source tier at a time (download -> create every tag off it -> push -> rm ->
orphan-blob gc -> delete), behind a free-space floor on the ROOT fs, which holds the ollama
blob store and must never drop below 200 GB free on bs2.

Usage:
  ollama_republish_v6.py --old-digests v6_old_digests.json \\
      [--only TAG[,TAG]] [--dry-run] [--limit N] [--temperature 0.9]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

NS = "mannix"
MODEL = "gemma4-98e-v6-coder"
HF_REPO = "ManniX-ITA/gemma-4-A4B-98e-v6-coder-it-GGUF"
HF_PREFIX = "gemma-4-A4B-98e-v6-coder-it-"
MMPROJ = "mmproj-gemma4.gguf"

WORK = os.environ.get("V6PUB_WORK", "/mnt/sdc/ml/v6_ollama")
FLOOR_PATH = os.environ.get("V6PUB_FLOOR_PATH", "/")
FLOOR_GB = float(os.environ.get("V6PUB_FLOOR_GB", "230"))

# The fixed GGUF fingerprint, from the already-published v7-coder Q6_K (NOT reconstructed).
WANT_EOS, WANT_EOT = 106, 1
WANT_TPL_SHA = "d9f21aac4764de694c7d3fb73e664e53b37eaad3b4ed14d7f1e4fbafeebc1ef6"
# What `ollama create` writes as the template layer when RENDERER owns turn termination.
TEMPLATE_STUB = b"{{ .Prompt }}"
TEMPLATE_STUB_SHA = hashlib.sha256(TEMPLATE_STUB).hexdigest()

# HF has an F16 tier but no F16 ollama tag: nothing to rebuild, and skipping it saves a 40 GB
# download. Skipping is a decision, not an oversight.
NO_OLLAMA_TAG = {"F16"}
# Published as an ollama tag but ABSENT from the HF GGUF repo and from every local disk. It is
# recovered by pulling the old tag and rewriting that blob's KV in place -- see recover_orphan.
ORPHAN_TIERS = {"IQ4_K_M"}


def log(msg):
    print("[v6pub %s] %s" % (time.strftime("%T %Z"), msg), flush=True)


def run(cmd, to=3600, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=to, env=e)


def rm(*paths):
    """Best-effort delete. Every failure path must leave zero large files behind: this runs
    against a disk floor and a leaked 20 GB tier stalls the whole batch."""
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def floor_ok():
    try:
        free = shutil.disk_usage(FLOOR_PATH).free / 1e9
    except Exception:
        return True, -1.0
    return free >= FLOOR_GB, free


def models_dir():
    """The dir the ollama *server* owns -- not the client's. bs2 runs `ollama serve` as user
    `ollama` with HOME=/usr/share/ollama, and a stale root-owned /root/.ollama also exists;
    verifying against the wrong one silently checks nothing."""
    env = os.environ.get("OLLAMA_MODELS")
    if env and os.path.isdir(os.path.join(env, "blobs")):
        return env
    for d in ("/usr/share/ollama/.ollama/models", "/root/.ollama/models",
              os.path.expanduser("~/.ollama/models")):
        if os.path.isdir(os.path.join(d, "blobs")):
            return d
    return None


def manifest_path(tag):
    md = models_dir()
    return os.path.join(md, "manifests", "registry.ollama.ai", NS, MODEL, tag) if md else None


def read_blob(digest):
    md = models_dir()
    p = os.path.join(md, "blobs", "sha256-" + digest.split(":")[-1])
    with open(p, "rb") as f:
        return f.read()


def inspect_created(tag):
    """Read the LOCAL manifest of a freshly created tag. Parsing `ollama show` cannot tell you
    a layer digest, and the digest is what proves the weights moved."""
    p = manifest_path(tag)
    if not p or not os.path.exists(p):
        raise RuntimeError("no local manifest at %s" % p)
    m = json.load(open(p))
    out = {"model": None, "projector": None, "params": {}, "template": None,
           "draft": None, "renderer": None, "parser": None}
    for lay in m.get("layers", []):
        kind = lay["mediaType"].rsplit(".", 1)[-1]
        if kind == "model":
            out["model"] = lay["digest"].split(":")[-1]
        elif kind == "projector":
            out["projector"] = lay["digest"].split(":")[-1]
        elif kind == "draft":
            out["draft"] = lay["digest"].split(":")[-1]
        elif kind == "template":
            out["template"] = lay["digest"].split(":")[-1]
        elif kind == "params":
            out["params"] = json.loads(read_blob(lay["digest"]) or b"{}")
    cfg = json.loads(read_blob(m["config"]["digest"]))
    out["renderer"] = cfg.get("renderer")
    out["parser"] = cfg.get("parser")
    return out


def gguf_meta(path):
    """(eos, eot, chat_template sha) of a local GGUF. Raises on an unparseable file."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    eos = eot = None
    tpl = None
    for fl in r.fields.values():
        if fl.name == "tokenizer.ggml.eos_token_id":
            eos = int(fl.parts[fl.data[0]][0])
        elif fl.name == "tokenizer.ggml.eot_token_id":
            eot = int(fl.parts[fl.data[0]][0])
        elif fl.name == "tokenizer.chat_template":
            tpl = b"".join(bytes(fl.parts[p]) for p in fl.data)
    return eos, eot, (hashlib.sha256(tpl).hexdigest() if tpl is not None else None)


def gc_blobs():
    """Delete blob files no manifest references. `ollama rm` drops only the manifest; the
    content-addressed blob survives, and an unchecked multi-tier cycle ballooned the store to
    128 GB on 2026-05-18. Unreferenced-by-any-manifest is safe by construction."""
    md = models_dir()
    if not md:
        return 0, 0
    ref = set()
    for root, _d, files in os.walk(os.path.join(md, "manifests")):
        for fn in files:
            try:
                txt = open(os.path.join(root, fn)).read()
            except Exception:
                continue
            ref.update(re.findall(r"sha256[:-]([0-9a-f]{64})", txt))
    n = freed = 0
    bd = os.path.join(md, "blobs")
    for b in os.listdir(bd):
        if not b.startswith("sha256-") or b[7:] in ref:
            continue
        fp = os.path.join(bd, b)
        try:
            sz = os.path.getsize(fp)
            os.remove(fp)
            n += 1
            freed += sz
        except Exception:
            pass
    return n, freed


def build_modelfile(gguf, params, mmproj=None, draft=None):
    """RENDERER + PARSER are mandatory: without them `ollama create` mines the GGUF's jinja for
    stop strings and silently re-adds `stop "<turn|>"`.

    TEMPLATE is the identity passthrough `{{ .Prompt }}`, matching all 27 published v7-coder
    tags. Omitting it is NOT equivalent: ollama then writes no template layer at all, and the
    tag stops being byte-comparable to v7 (the first gate run caught exactly that). With
    RENDERER declared the Go renderer owns chat turn assembly, so the stub is inert there and
    only serves the raw /api/generate path -- where identity passthrough is the correct
    behaviour. What must never appear is a REAL chat template, which would override the
    renderer; v6's old `IQ4_K_M` tag carries one, and this run removes it."""
    lines = ["FROM %s" % os.path.abspath(gguf)]
    if mmproj:
        lines.append("FROM %s" % os.path.abspath(mmproj))
    if draft:
        lines.append("DRAFT %s" % os.path.abspath(draft))
    lines += ['TEMPLATE """%s"""' % TEMPLATE_STUB.decode(), "RENDERER gemma4", "PARSER gemma4"]
    for k in sorted(params):
        lines.append("PARAMETER %s %s" % (k, params[k]))
    return "\n".join(lines) + "\n"


def publish(tag, gguf, params, old_digests, mmproj=None, draft=None, dry=False):
    full = "%s/%s:%s" % (NS, MODEL, tag)
    marker = os.path.join(WORK, "done", tag + ".done")
    if os.path.exists(marker):
        return "DONE-CACHED"
    if dry:
        return "DRY(from %s%s%s)" % (os.path.basename(gguf), " +mmproj" if mmproj else "",
                                     " +draft" if draft else "")

    mf = os.path.join(WORK, "Modelfile.%s" % tag)
    open(mf, "w").write(build_modelfile(gguf, params, mmproj, draft))
    r = run(["ollama", "create", full, "-f", mf], to=7200)
    if r.returncode != 0:
        return "FAILCREATE: " + (r.stderr or r.stdout).strip()[-160:]

    def reject(why):
        """A rejected tag must not leave its blob behind. `ollama create` has already copied
        the full tier into the store; without this the first gate failure alone cost 34 GB of
        root fs, and 30 of them would breach the 200 GB floor."""
        run(["ollama", "rm", full], to=300)
        return why

    try:
        got = inspect_created(tag)
    except Exception as e:
        return reject("FAILINSPECT: %s" % str(e)[:120])

    if got["params"] != params:
        return reject("FAILVERIFY-params(%s != %s)"
                      % (json.dumps(got["params"], sort_keys=True),
                         json.dumps(params, sort_keys=True)))
    if got["template"] != TEMPLATE_STUB_SHA:
        return reject("FAILVERIFY-template(%s != v7 stub %s)"
                      % ((got["template"] or "none")[:12], TEMPLATE_STUB_SHA[:12]))
    if got["renderer"] != "gemma4" or got["parser"] != "gemma4":
        return reject("FAILVERIFY-renderer(renderer=%s parser=%s)"
                      % (got["renderer"], got["parser"]))
    old = old_digests.get(tag)
    if old and got["model"].startswith(old):
        return reject("FAILVERIFY-blob-unchanged(%s -- old weights republished?)" % old)
    want_proj = tag.startswith("vision-")
    if want_proj != bool(got["projector"]):
        return reject("FAILVERIFY-projector(want=%s got=%s)" % (want_proj, bool(got["projector"])))
    want_draft = tag.startswith("mtp-")
    if want_draft != bool(got["draft"]):
        return reject("FAILVERIFY-draft(want=%s got=%s)" % (want_draft, bool(got["draft"])))
    if want_proj:
        show = run(["ollama", "show", full], to=300).stdout
        if "vision" not in show.lower():
            return reject("FAILVERIFY-vision-capability-lost")

    r = run(["ollama", "push", full], to=14400)
    if r.returncode != 0:
        run(["ollama", "rm", full], to=300)
        return "FAILPUSH: " + (r.stderr or r.stdout).strip()[-160:]

    run(["ollama", "rm", full], to=300)
    rm(mf)
    open(marker, "w").write("ok %s\n" % got["model"])
    return "OK(%s)" % got["model"][:12]


def fetch_tier(fn):
    from huggingface_hub import hf_hub_download
    dl = os.path.join(WORK, "dl")
    os.makedirs(dl, exist_ok=True)
    return hf_hub_download(repo_id=HF_REPO, filename=fn, local_dir=dl)


def recover_orphan(tier):
    """`IQ4_K_M` exists as an ollama tag but not on HF and not on any local disk. The published
    ollama model blob IS a GGUF, so pull the old tag, rewrite that blob's KV with
    gguf_new_metadata, and build from the result. This preserves an artifact that would
    otherwise be unfixable -- do NOT substitute a different tier's file for it."""
    full = "%s/%s:%s" % (NS, MODEL, tier)
    r = run(["ollama", "pull", full], to=14400)
    if r.returncode != 0:
        raise RuntimeError("pull failed: " + (r.stderr or r.stdout)[-160:])
    got = inspect_created(tier)
    src = os.path.join(models_dir(), "blobs", "sha256-" + got["model"])
    out = os.path.join(WORK, "dl", "%s%s.recovered.gguf" % (HF_PREFIX, tier))
    rm(out)
    cmd = [sys.executable, "-m", "gguf.scripts.gguf_new_metadata",
           "--chat-template-file", os.path.join(WORK, "tpl.jinja"),
           "--special-token-by-id", "eos", str(WANT_EOS),
           "--special-token-by-id", "eot", str(WANT_EOT),
           "--force", src, out]
    r = run(cmd, to=14400)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("retag failed: " + (r.stderr or r.stdout)[-160:])
    run(["ollama", "rm", full], to=300)
    gc_blobs()
    return out


def recover_draft():
    """The mtp-Q4_K_M drafter (446 MB) exists in no local checkout. Pull the old tag and reuse
    its draft blob verbatim, so the rebuilt tag ships a byte-identical drafter."""
    full = "%s/%s:mtp-Q4_K_M" % (NS, MODEL)
    r = run(["ollama", "pull", full], to=14400)
    if r.returncode != 0:
        raise RuntimeError("pull failed: " + (r.stderr or r.stdout)[-160:])
    got = inspect_created("mtp-Q4_K_M")
    if not got["draft"]:
        raise RuntimeError("published mtp tag has no draft layer")
    src = os.path.join(models_dir(), "blobs", "sha256-" + got["draft"])
    dst = os.path.join(WORK, "drafter-mtp-Q4_K_M.gguf")
    shutil.copyfile(src, dst)
    run(["ollama", "rm", full], to=300)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-digests", required=True,
                    help="JSON {tag: old_model_blob_digest_prefix} from the registry sweep")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    params = {"min_p": 0.05, "num_ctx": 32768, "repeat_penalty": 1.1,
              "temperature": a.temperature, "top_k": 64, "top_p": 0.95}
    old = json.load(open(a.old_digests))
    os.makedirs(os.path.join(WORK, "done"), exist_ok=True)

    from huggingface_hub import HfApi
    info = HfApi().model_info(HF_REPO)
    tiers = sorted(s.rfilename[len(HF_PREFIX):-5] for s in info.siblings
                   if s.rfilename.startswith(HF_PREFIX) and s.rfilename.endswith(".gguf"))
    tiers = [t for t in tiers if t not in NO_OLLAMA_TAG] + sorted(ORPHAN_TIERS)

    units = []
    for t in tiers:
        tags = [t, "vision-" + t]
        if t == "Q4_K_M":
            tags += ["latest", "mtp-Q4_K_M"]
        units.append((t, tags))
    if a.only:
        want = set(a.only.split(","))
        units = [(t, [g for g in tg if g in want]) for t, tg in units]
        units = [u for u in units if u[1]]
    if a.limit:
        units = units[:a.limit]

    log("model=%s/%s tiers=%d tags=%d params=%s dry=%s"
        % (NS, MODEL, len(units), sum(len(u[1]) for u in units),
           json.dumps(params, sort_keys=True), a.dry_run))
    log("store=%s floor=%s>=%.0fGB" % (models_dir(), FLOOR_PATH, FLOOR_GB))

    mmproj = drafter = None
    t0, summary = time.time(), {}
    for i, (tier, tags) in enumerate(units, 1):
        pending = [g for g in tags if not os.path.exists(os.path.join(WORK, "done", g + ".done"))]
        if not pending:
            log("[%2d/%d] %-14s -> DONE-CACHED (%d tags)" % (i, len(units), tier, len(tags)))
            summary["DONE-CACHED"] = summary.get("DONE-CACHED", 0) + len(tags)
            continue
        ok, free = floor_ok()
        if not ok:
            log("ABORT-DISKFLOOR(%s free=%.0fGB < %.0fGB)" % (FLOOR_PATH, free, FLOOR_GB))
            return 2

        src = None
        try:
            if tier in ORPHAN_TIERS:
                if not a.dry_run:
                    src = recover_orphan(tier)
            else:
                src = fetch_tier("%s%s.gguf" % (HF_PREFIX, tier)) if not a.dry_run else "DRY"
            if not a.dry_run:
                eos, eot, tsha = gguf_meta(src)
                if (eos, eot, tsha) != (WANT_EOS, WANT_EOT, WANT_TPL_SHA):
                    log("[%2d/%d] %-14s -> ABORT-STALE-SOURCE(eos=%s eot=%s tpl=%s)"
                        % (i, len(units), tier, eos, eot, (tsha or "none")[:8]))
                    rm(src)
                    summary["ABORT-STALE-SOURCE"] = summary.get("ABORT-STALE-SOURCE", 0) + 1
                    continue
                if any(g.startswith("vision-") for g in pending) and mmproj is None:
                    mmproj = fetch_tier(MMPROJ)
                if "mtp-Q4_K_M" in pending and drafter is None:
                    drafter = recover_draft()
        except Exception as e:
            log("[%2d/%d] %-14s -> FAILSOURCE: %s" % (i, len(units), tier, str(e)[:140]))
            summary["FAILSOURCE"] = summary.get("FAILSOURCE", 0) + 1
            rm(src)
            continue

        for tag in pending:
            st = publish(tag, src, params, old,
                         mmproj=mmproj if tag.startswith("vision-") else None,
                         draft=drafter if tag.startswith("mtp-") else None,
                         dry=a.dry_run)
            key = st.split("(")[0].split(":")[0]
            summary[key] = summary.get(key, 0) + 1
            log("[%2d/%d] %-22s -> %s" % (i, len(units), tag, st))
        if not a.dry_run:
            rm(src)
            n, freed = gc_blobs()
            if n:
                log("       gc: %d orphan blobs, %.1f GB" % (n, freed / 1e9))

    rm(mmproj)
    dt = int(time.time() - t0)
    log("=== V6 OLLAMA REPUBLISH SUMMARY (%dm%02ds) ===" % (dt // 60, dt % 60))
    for k in sorted(summary):
        log("   %-24s %d" % (k, summary[k]))
    bad = sum(v for k, v in summary.items() if k.startswith(("FAIL", "ABORT")))
    log("V6PUB-COMPLETE failures=%d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
