#!/usr/bin/env python3
"""Rewrite EOG tokens + chat template in EVERY published GGUF tier of an HF repo, in place.

WHY
---
Published `gemma-4-A4B-98e-v6-coder-it-GGUF` tiers carry `eos_token_id = 1 (<eos>)`. For
Gemma 4 the real turn terminator is token **106 `<turn|>`**; with eos=1 the model never
emits a recognised EOG and control tokens leak into the answer text. The v7-coder cohort was
already corrected to `eos=106 / eot=1` plus the google-18683-based loop-fixed chat template,
so v6 is the odd one out.

Nothing here re-quantizes. `gguf_new_metadata` copies the file and rewrites the KV block;
tensors are untouched, so this is CPU-only and needs no GPU and no imatrix.

TARGET IS TAKEN FROM A SHIPPED ARTIFACT, NOT RECONSTRUCTED
----------------------------------------------------------
`--template-file` should be the template DUMPED FROM AN ALREADY-FIXED PUBLISHED TIER
(gguf_probe_header.py --out-template against the v7-coder Q6_K URL). A local
`chat_template.jinja` lying around on a work host is NOT the same thing -- 17466 / 18051 /
19177-byte variants all coexist on bs2, and picking the wrong one silently ships a third
template version. Pass --template-sha to pin it; the run refuses to start if it disagrees.

INVARIANT ASSERTED PER TIER, BEFORE ANYTHING IS UPLOADED
--------------------------------------------------------
    eos == --eos   AND   eot == --eot   AND   chat_template sha == target sha
    AND tensor count unchanged   AND file is a parseable GGUF
A tier that fails any of these is NOT uploaded and does NOT get a marker, so a re-run retries
it. The `.sha256` sidecar is regenerated from the NEW bytes -- the published sidecar goes
stale the instant the KV block changes, and a stale checksum is worse than none.

Disk-bounded: download -> rewrite -> verify -> upload -> delete, one tier at a time, with a
free-space floor checked before each download (the F16 tier alone is ~40 GB).

Usage:
  gguf_retag_republish.py --repo ManniX-ITA/gemma-4-A4B-98e-v6-coder-it-GGUF \\
      --template-file tpl/chat_template.v7fixed.jinja --template-sha <sha256> \\
      [--eos 106] [--eot 1] [--only TIER[,TIER]] [--dry-run] [--limit N]
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

WORK = os.environ.get("RETAG_WORK", "/mnt/sdc/ml/gguf_retag")
FLOOR_PATH = os.environ.get("RETAG_FLOOR_PATH", "/mnt/sdc")
FLOOR_GB = float(os.environ.get("RETAG_FLOOR_GB", "150"))
PY = sys.executable

# mmproj carries vision-projector tensors and NO tokenizer/template KV, so there is nothing
# here to fix. Skipping it is a decision, not an oversight -- do not silently include it.
SKIP = re.compile(r"^mmproj", re.I)


def log(msg):
    print("[retag %s] %s" % (time.strftime("%T %Z"), msg), flush=True)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rm(*paths):
    """Best-effort delete. Every failure path must leave zero large files behind: the
    republisher runs against a disk floor and a leaked 40 GB F16 stalls the whole batch."""
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


def read_meta(path):
    """(eos, eot, template_text, n_tensors) from a local GGUF. Raises on unparseable file."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    out = {"eos": None, "eot": None, "tpl": None, "ntensor": len(r.tensors)}
    for fl in r.fields.values():
        if fl.name == "tokenizer.ggml.eos_token_id":
            out["eos"] = int(fl.parts[fl.data[0]][0])
        elif fl.name == "tokenizer.ggml.eot_token_id":
            out["eot"] = int(fl.parts[fl.data[0]][0])
        elif fl.name == "tokenizer.chat_template":
            out["tpl"] = b"".join(bytes(fl.parts[p]) for p in fl.data)
    return out


def tpl_sha(b):
    return hashlib.sha256(b).hexdigest() if b is not None else None


def process(api, repo, fn, tpl_path, target_tpl_sha, eos, eot, dry):
    from huggingface_hub import hf_hub_download
    marker = os.path.join(WORK, "done", fn.replace("/", "_") + ".done")
    if os.path.exists(marker):
        return "DONE-CACHED"
    ok, free = floor_ok()
    if not ok:
        return "ABORT-DISKFLOOR(%s free=%.0fGB < %.0fGB)" % (FLOOR_PATH, free, FLOOR_GB)

    dl = os.path.join(WORK, "dl")
    os.makedirs(dl, exist_ok=True)
    try:
        src = hf_hub_download(repo_id=repo, filename=fn, local_dir=dl)
    except Exception as e:
        return "FAILDL: %s" % str(e)[:160]

    try:
        before = read_meta(src)
    except Exception as e:
        rm(src)
        return "FAILPARSE-before: %s" % str(e)[:120]

    if before["eos"] == eos and before["eot"] == eot and tpl_sha(before["tpl"]) == target_tpl_sha:
        rm(src)
        open(marker, "w").write("already-correct\n")
        return "ALREADY-OK"

    if dry:
        rm(src)
        return "DRY(eos %s->%s eot %s->%s tpl %s->%s)" % (
            before["eos"], eos, before["eot"], eot,
            (tpl_sha(before["tpl"]) or "none")[:8], target_tpl_sha[:8])

    tmp = src + ".retag.tmp"
    rm(tmp)
    cmd = [PY, "-m", "gguf.scripts.gguf_new_metadata",
           "--chat-template-file", tpl_path,
           "--special-token-by-id", "eos", str(eos),
           "--special-token-by-id", "eot", str(eot),
           "--force", src, tmp]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    if r.returncode != 0 or not os.path.exists(tmp):
        rm(src, tmp)
        return "FAILREWRITE: " + (r.stderr or r.stdout).strip()[-160:]

    # Verify the REWRITTEN file before it can reach the hub. A rewrite that lost tensors or
    # produced an unparseable header must never be uploaded over a good artifact.
    try:
        after = read_meta(tmp)
    except Exception as e:
        rm(src, tmp)
        return "FAILPARSE-after: %s" % str(e)[:120]
    if after["eos"] != eos or after["eot"] != eot:
        rm(src, tmp)
        return "FAILVERIFY-eog(eos=%s eot=%s)" % (after["eos"], after["eot"])
    if tpl_sha(after["tpl"]) != target_tpl_sha:
        rm(src, tmp)
        return "FAILVERIFY-tpl(%s != %s)" % ((tpl_sha(after["tpl"]) or "none")[:12],
                                             target_tpl_sha[:12])
    if after["ntensor"] != before["ntensor"]:
        rm(src, tmp)
        return "FAILVERIFY-tensors(%d->%d)" % (before["ntensor"], after["ntensor"])

    # The published .sha256 describes the OLD bytes; regenerate from the new ones.
    digest = sha256_file(tmp)
    side = os.path.join(WORK, "dl", fn + ".sha256")
    open(side, "w").write("%s  %s\n" % (digest, fn))

    try:
        api.upload_file(path_or_fileobj=tmp, path_in_repo=fn, repo_id=repo,
                        commit_message="retag: eos=%d eot=%d + loop-fixed chat template (%s)"
                                       % (eos, eot, fn))
        api.upload_file(path_or_fileobj=side, path_in_repo=fn + ".sha256", repo_id=repo,
                        commit_message="retag: refresh sha256 sidecar (%s)" % fn)
    except Exception as e:
        rm(src, tmp, side)
        return "FAILUPLOAD: %s" % str(e)[:160]

    rm(src, tmp, side)
    open(marker, "w").write("ok %s\n" % digest)
    return "OK(eos %s->%s, tpl %s->%s)" % (before["eos"], eos,
                                           (tpl_sha(before["tpl"]) or "none")[:8],
                                           target_tpl_sha[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--template-file", required=True)
    ap.add_argument("--template-sha", default=None,
                    help="sha256 the template file MUST have. Refuse to run otherwise.")
    ap.add_argument("--eos", type=int, default=106)
    ap.add_argument("--eot", type=int, default=1)
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    os.makedirs(os.path.join(WORK, "done"), exist_ok=True)

    tsha = sha256_file(a.template_file)
    if a.template_sha and tsha != a.template_sha:
        log("FATAL template sha mismatch: file=%s expected=%s" % (tsha, a.template_sha))
        return 2
    log("template %s sha=%s (%d bytes)"
        % (a.template_file, tsha[:16], os.path.getsize(a.template_file)))

    api = HfApi()
    info = api.model_info(a.repo)
    tiers = sorted(s.rfilename for s in info.siblings
                   if s.rfilename.endswith(".gguf") and not SKIP.match(s.rfilename))
    if a.only:
        want = set(a.only.split(","))
        tiers = [t for t in tiers if t in want or any(t.endswith("-%s.gguf" % w) for w in want)]
    if a.limit:
        tiers = tiers[:a.limit]
    log("repo=%s tiers=%d eos=%d eot=%d dry=%s" % (a.repo, len(tiers), a.eos, a.eot, a.dry_run))

    t0, summary = time.time(), {}
    for i, fn in enumerate(tiers, 1):
        st = process(api, a.repo, fn, a.template_file, tsha, a.eos, a.eot, a.dry_run)
        key = st.split("(")[0].split(":")[0]
        summary[key] = summary.get(key, 0) + 1
        log("[%2d/%d] %-52s -> %s" % (i, len(tiers), fn, st))
        if st.startswith("ABORT-DISKFLOOR"):
            log("RETAG-ABORTED: disk floor hit; resume with the same command (idempotent).")
            return 2
    dt = int(time.time() - t0)
    log("=== RETAG SUMMARY (%dm%02ds) ===" % (dt // 60, dt % 60))
    for k in sorted(summary):
        log("   %-24s %d" % (k, summary[k]))
    bad = sum(v for k, v in summary.items() if k.startswith("FAIL"))
    log("RETAG-COMPLETE failures=%d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
