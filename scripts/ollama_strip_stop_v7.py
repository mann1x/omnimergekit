#!/usr/bin/env python3
"""Republish every v7-coder / v7-coderx ollama tag WITHOUT the phantom `stop "<turn|>"`.

WHY
---
Every published v7 tag carries `PARAMETER stop <turn|>` that appears in no Modelfile we ever
wrote. `ollama create` MINES the GGUF's embedded jinja `chat_template` for stop strings when
the Modelfile does not declare RENDERER/PARSER; for Gemma-4 that yields `<turn|>`. Google's own
`library/gemma4:26b` params blob is 42 bytes and has no stop at all.

The stop is not merely redundant, it is harmful. Token 106 `<turn|>` is already the GGUF's
`eos_token_id`, so it terminates as EOG and never reaches the text stream -- the stop CANNOT
fire in the normal case. The only case it can fire is a *coder* model writing the literal
characters `<turn|>` (discussing chat templates, or inside a fence), which silently truncates
the answer mid-sentence. Human validation on the stop-free `:test` tag: working tetris;
`:test2` (Google's bare 3 params, temp 1.0): non-working. So we keep our sampler and drop the
stop, changing exactly ONE thing.

HOW (and why not the obvious way)
---------------------------------
`ollama_bake_antiloop.py` does `FROM <tag>` + PARAMETER lines. That CANNOT be reused here:
`FROM <existing model>` INHERITS AND MERGES the parent's params, so a PARAMETER line can only
override or add -- never subtract. A stop-free variant must be built `FROM` the GGUF blob.

`ollama show <tag> --modelfile` emits exactly that: `FROM <blobpath>` plus TEMPLATE / RENDERER /
PARSER / PARAMETER lines, and for vision tags a SECOND `FROM <mmproj blob>`. So we derive it,
delete the one `PARAMETER stop` line, and recreate. Because RENDERER/PARSER are present in the
derived Modelfile, ollama does not re-mine the jinja and the stop does not come back.

INVARIANT ASSERTED PER TAG (this is the whole point -- do not weaken it):
    params_after == params_before - {stop}
plus renderer/parser preserved, FROM-count preserved (projector!), and for vision-* tags the
`vision` capability still reported after create. Anything else = refuse to push that tag.

PER-QUANT TEMPERATURE IS DELIBERATE -- DO NOT NORMALIZE IT
----------------------------------------------------------
A registry sweep after the run shows 38 tags at `temperature 0.9` and 16 at `0.8`. That is NOT
drift: the temperature is chosen PER QUANT TIER by the per-tier loop gate, because low-bit
tiers loop at a different threshold than high-bit ones. The `params_after == params_before -
stop` invariant is what preserves it. Never "fix" the spread by writing one canonical param
block across all tags -- that would silently undo a measured, per-tier decision.

Usage:  ollama_strip_stop_v7.py [--dry-run] [--only TAG[,TAG...]] [--models M[,M]]
Resumable via .done markers. Disk-bounded: pull -> create -> push -> rm -> orphan-blob gc.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

NS = "mannix"
DEFAULT_MODELS = ["gemma4-98e-v7-coder", "gemma4-98e-v7-coderx"]
WORK = os.environ.get("STRIPSTOP_WORK", "/mnt/sdc/ml/ollama_stripstop")

# bs2's ollama daemon keeps blobs in /root/.ollama/models -- on the ROOT fs, which must never
# drop below 200 GB free. A 20 GB tag is transient (pull -> push -> rm -> gc) but the gc has
# failed before (orphan blobs ballooned that dir to 128 GB on 2026-05-18), so do not trust it:
# check the real floor before every pull and stop the batch rather than eat the margin.
FLOOR_PATH = os.environ.get("STRIPSTOP_FLOOR_PATH", "/root")
FLOOR_GB = float(os.environ.get("STRIPSTOP_FLOOR_GB", "220"))


def floor_ok():
    try:
        free = shutil.disk_usage(FLOOR_PATH).free / 1e9
    except Exception:
        return True, -1.0
    return free >= FLOOR_GB, free
TAG_RE = re.compile(r"^(CD-|mtp-)?(qat-)?(F16|Q[0-9][A-Za-z0-9_]*|IQ[0-9][A-Za-z0-9_]*)$")


def real_tag(tag):
    t = tag[7:] if tag.startswith("vision-") else tag
    return t in ("qat", "latest") or bool(TAG_RE.match(t))


def get_tags(model):
    """HTML scrape. The registry's /v2/<repo>/tags/list returns 404 for ollama.com, so the
    scrape is the only enumeration available -- verified 2026-07-30."""
    url = "https://ollama.com/%s/%s/tags" % (NS, model)
    html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
    raw = set(re.findall(r"%s:([A-Za-z0-9_.-]+)" % re.escape(model), html))
    return sorted(t for t in raw if real_tag(t))


def run(cmd, to=7200):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=to)


def params_of(mf):
    """set of 'key value' for every PARAMETER line."""
    out = set()
    for line in mf.splitlines():
        if line.startswith("PARAMETER "):
            out.add(line[len("PARAMETER "):].strip())
    return out


def rp_of(mf):
    return sorted(l.strip() for l in mf.splitlines() if l.startswith(("RENDERER", "PARSER")))


def from_count(mf):
    return sum(1 for l in mf.splitlines() if l.startswith("FROM "))


def models_dir():
    for c in (os.environ.get("OLLAMA_MODELS"),
              "/srv/dev-disk-by-uuid-92295e2c-12bd-4d15-a50c-1d80e1a33ee8/spool/ollama_models",
              "/usr/share/ollama/.ollama/models", "/root/.ollama/models"):
        if c and os.path.isdir(os.path.join(c, "blobs")):
            return c
    return None


def gc_blobs():
    """Delete blob files no manifest references. `ollama rm` drops only the manifest; the
    content-addressed blob survives and a multi-tier cycle ballooned /root/.ollama to 128 GB
    on 2026-05-18. Unreferenced-by-any-manifest is safe by construction."""
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
            for m in re.findall(r"sha256[:-]([0-9a-f]{64})", txt):
                ref.add(m)
    n = freed = 0
    bd = os.path.join(md, "blobs")
    for b in os.listdir(bd):
        if not b.startswith("sha256-"):
            continue
        if b[7:] not in ref:
            p = os.path.join(bd, b)
            try:
                sz = os.path.getsize(p)
                os.remove(p)
                n += 1
                freed += sz
            except Exception:
                pass
    return n, freed


def process(model, tag, dry):
    full = "%s/%s:%s" % (NS, model, tag)
    done = os.path.join(WORK, "done")
    marker = os.path.join(done, "%s__%s.done" % (model, tag.replace("/", "_")))
    if os.path.exists(marker):
        return "DONE-CACHED"

    ok, free = floor_ok()
    if not ok:
        return "ABORT-DISKFLOOR(%s free=%.0fGB < %.0fGB)" % (FLOOR_PATH, free, FLOOR_GB)

    r = run(["ollama", "pull", full])
    if r.returncode != 0:
        blob = (r.stderr + r.stdout).lower()
        if "not found" in blob or "404" in blob or "file does not exist" in blob:
            open(marker, "w").write("skip404\n")
            return "SKIP404"
        return "FAILPULL: " + (r.stderr or r.stdout).strip()[:140]

    before = run(["ollama", "show", full, "--modelfile"], to=300).stdout
    if not before.strip():
        return "FAILSHOW-empty"
    p_before, rp_before, nfrom = params_of(before), rp_of(before), from_count(before)

    # RENDERER/PARSER are LOAD-BEARING: without them `ollama create` re-mines the GGUF jinja
    # and puts the stop straight back. Never "fix" this by adding them silently -- if a tag
    # lacks them, that tag needs a human look.
    if not any(l.startswith("RENDERER") for l in rp_before) or \
       not any(l.startswith("PARSER") for l in rp_before):
        return "FAILNO-RENDERER/PARSER %s" % rp_before

    stops = {p for p in p_before if p.startswith("stop")}
    if not stops:
        open(marker, "w").write("already-clean\n")
        return "ALREADY-CLEAN"

    kept = [l for l in before.splitlines()
            if not (l.startswith("PARAMETER ") and l[len("PARAMETER "):].lstrip().startswith("stop"))]
    mf = os.path.join(WORK, "mf_%s_%s.txt" % (model, tag.replace("/", "_")))
    open(mf, "w").write("\n".join(kept) + "\n")

    if dry:
        return "DRY(stop=%s from=%d params=%d)" % (sorted(stops), nfrom, len(p_before))

    r = run(["ollama", "create", full, "-f", mf], to=3600)
    if r.returncode != 0:
        return "FAILCREATE: " + (r.stderr or r.stdout).strip()[:140]

    after = run(["ollama", "show", full, "--modelfile"], to=300).stdout
    p_after = params_of(after)
    # THE invariant: exactly one thing changed.
    if p_after != p_before - stops:
        return "FAILVERIFY-params(+%s -%s)" % (sorted(p_after - (p_before - stops)),
                                               sorted((p_before - stops) - p_after))
    if rp_of(after) != rp_before:
        return "FAILVERIFY-renderer(%s->%s)" % (rp_before, rp_of(after))
    if from_count(after) != nfrom:
        return "FAILVERIFY-from(%d->%d projector lost?)" % (nfrom, from_count(after))
    if tag.startswith("vision-"):
        show = run(["ollama", "show", full], to=300).stdout
        if "vision" not in show.lower():
            return "FAILVERIFY-vision-capability-lost"

    r = run(["ollama", "push", full])
    if r.returncode != 0:
        return "FAILPUSH: " + (r.stderr or r.stdout).strip()[:140]

    run(["ollama", "rm", full], to=600)
    n, freed = gc_blobs()
    try:
        os.remove(mf)
    except Exception:
        pass
    open(marker, "w").write("ok\n")
    return "OK(gc %d blobs, %.1f GB)" % (n, freed / 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated tags to restrict to")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    a = ap.parse_args()
    os.makedirs(os.path.join(WORK, "done"), exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    t0 = time.time()
    summary = {}
    print("models_dir=%s work=%s dry=%s" % (models_dir(), WORK, a.dry_run), flush=True)
    for model in a.models.split(","):
        try:
            tags = get_tags(model)
        except Exception as e:
            print("[%s] FAILED to list tags: %s" % (model, e), flush=True)
            continue
        if only:
            tags = [t for t in tags if t in only]
        print("=== %s : %d tags ===" % (model, len(tags)), flush=True)
        for i, tag in enumerate(tags, 1):
            st = process(model, tag, a.dry_run)
            summary[st.split("(")[0].split(":")[0]] = \
                summary.get(st.split("(")[0].split(":")[0], 0) + 1
            print("[%s %2d/%d] %-22s -> %s" % (model.replace("gemma4-98e-", ""), i, len(tags),
                                               tag, st), flush=True)
            if st.startswith("ABORT-DISKFLOOR"):
                print("STRIPSTOP-ABORTED: disk floor hit; resume with the same command "
                      "(markers make it idempotent).", flush=True)
                return 2
    dt = int(time.time() - t0)
    print("=== STRIPSTOP SUMMARY (%dm%02ds) ===" % (dt // 60, dt % 60), flush=True)
    for k in sorted(summary):
        print("   %-28s %d" % (k, summary[k]), flush=True)
    bad = sum(v for k, v in summary.items() if k.startswith("FAIL"))
    print("STRIPSTOP-COMPLETE failures=%d" % bad, flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
