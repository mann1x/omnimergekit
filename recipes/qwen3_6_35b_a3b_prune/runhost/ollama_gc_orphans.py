#!/usr/bin/env python3
"""Free ollama blobs that no manifest references.

WHY THIS IS NEEDED. `ollama rm <tag>` deletes the MANIFEST but leaves the blobs; the daemon
only prunes unreferenced blobs at startup. A publish loop that creates → pushes → removes one
tier at a time therefore accumulates the full weight of every tier it has finished. Measured
2026-08-21 on bs2: the store grew 113 G → 207 G over 10 CoderX tiers while holding 11
manifests and zero coderx tags, and the loop correctly refused to continue at the /mnt/sdc
floor. 161.9 GB was orphaned.

SAFETY. The keep-set is built from every manifest's `layers[].digest` plus `config.digest`.
**An empty reference set is a bug signal, not a licence to delete** — a wrong or missing
manifests dir would make every blob look orphaned and one pass would wipe the whole library.
Both `manifests` and `keep` are asserted non-empty before anything is removed (2026-08-05
rule, [[feedback_ollama_store_is_the_daemon_users_home]]).

Resolves the store the way the DAEMON sees it, never from a literal path: OLLAMA_MODELS →
`systemctl show ollama -p Environment` → the daemon user's ~/.ollama/models.

Usage:  ollama_gc_orphans.py [--apply]     (default is a dry run)
"""
import glob
import json
import os
import re
import subprocess
import sys


def resolve_store() -> str:
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return env
    try:  # the unit's own Environment= wins over our shell
        out = subprocess.run(["systemctl", "show", "ollama", "-p", "Environment"],
                             capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"OLLAMA_MODELS=(\S+)", out)
        if m:
            return m.group(1)
        user = subprocess.run(["systemctl", "show", "ollama", "-p", "User", "--value"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        if user:
            import pwd
            return os.path.join(pwd.getpwnam(user).pw_dir, ".ollama", "models")
    except Exception:
        pass
    return "/usr/share/ollama/.ollama/models"


def main() -> int:
    apply = "--apply" in sys.argv
    root = resolve_store()
    mdir, bdir = os.path.join(root, "manifests"), os.path.join(root, "blobs")
    if not os.path.isdir(mdir) or not os.path.isdir(bdir):
        print(f"REFUSE: {root} does not look like an ollama store")
        return 2

    mans = [m for m in glob.glob(mdir + "/**/*", recursive=True) if os.path.isfile(m)]
    keep, unreadable = set(), 0
    for m in mans:
        try:
            j = json.load(open(m))
        except Exception:
            unreadable += 1
            continue
        for layer in (j.get("layers") or []) + ([j["config"]] if j.get("config") else []):
            d = layer.get("digest")
            if d:
                keep.add(d.replace(":", "-"))

    # An unreadable manifest is a hole in the keep-set -- its blobs would look orphaned.
    if unreadable:
        print(f"REFUSE: {unreadable} manifest(s) unreadable; keep-set would be incomplete")
        return 3
    if not mans:
        print(f"REFUSE: zero manifests under {mdir} -- that is a bug signal, not an empty library")
        return 3
    if not keep:
        print("REFUSE: reference set is EMPTY -- refusing to treat every blob as orphaned")
        return 3

    blobs = glob.glob(bdir + "/*")
    orph = [b for b in blobs if os.path.basename(b) not in keep]
    freed = sum(os.path.getsize(b) for b in orph)
    kept = sum(os.path.getsize(b) for b in blobs if os.path.basename(b) in keep)
    print(f"store={root}")
    print(f"manifests={len(mans)} referenced={len(keep)} blobs={len(blobs)} "
          f"orphans={len(orph)}")
    print(f"reclaimable={freed/1e9:.1f} GB   keeping={kept/1e9:.1f} GB")

    if not apply:
        print("DRY RUN -- pass --apply to delete")
        return 0
    n = 0
    for b in orph:
        try:
            os.remove(b)
            n += 1
        except OSError as e:
            print(f"  could not remove {os.path.basename(b)}: {e}")
    print(f"OLLAMA_GC_OK reclaimed {freed/1e9:.1f} GB from {n} orphan blobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
