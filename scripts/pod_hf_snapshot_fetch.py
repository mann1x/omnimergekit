"""pod_hf_snapshot_fetch.py — ON THE POD: download a whole HF model repo at a pinned revision with aria2c.

Every file is listed from the repo at `<revision>`; LFS files are checked against their sha256, small files
against their size. The token goes into a 0600 aria2c input file that is deleted after the transfer (never in
argv). A `FETCH_OK` marker file is written into `<dest>` only when every file passed.

usage: pod_hf_snapshot_fetch.py <repo_id> <revision> <dest_dir>
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    repo, rev, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "FETCH_OK").unlink(missing_ok=True)
    tok_file = Path.home() / ".cache/huggingface/token"
    token = tok_file.read_text().strip() if tok_file.exists() else ""
    sib = HfApi(token=token or None).model_info(repo, revision=rev, files_metadata=True).siblings or []
    assert sib, f"{repo}@{rev}: no files listed"
    todo = []
    for s in sib:
        out = dest / s.rfilename
        ok = out.exists() and (sha256(out) == s.lfs.sha256 if s.lfs else out.stat().st_size == s.size)
        if not ok:
            todo.append(s)
    print(f"{repo}@{rev[:10]}: {len(sib)} files, {len(todo)} to fetch", flush=True)
    if todo:
        inp = dest / ".aria_in"
        fd = os.open(inp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            for s in todo:
                f.write(f"https://huggingface.co/{repo}/resolve/{rev}/{s.rfilename}\n  out={s.rfilename}\n")
                if token:
                    f.write(f"  header=Authorization: Bearer {token}\n")
        r = subprocess.run(["aria2c", "-x", "16", "-s", "16", "-j", "4", "-k", "64M", "-c",
                            "--file-allocation=none", "--console-log-level=warn", "--summary-interval=0",
                            "-d", str(dest), "-i", str(inp)], capture_output=True, text=True)
        inp.unlink(missing_ok=True)
        if r.returncode != 0:
            print(f"FETCH_FAIL aria2c rc={r.returncode}: {r.stdout[-400:]}", flush=True)
            sys.exit(1)
    bad = []
    for s in sib:
        out = dest / s.rfilename
        if not out.exists():
            bad.append(f"{s.rfilename}: missing")
        elif s.lfs and sha256(out) != s.lfs.sha256:
            bad.append(f"{s.rfilename}: sha256 != LFS")
        elif not s.lfs and out.stat().st_size != s.size:
            bad.append(f"{s.rfilename}: size {out.stat().st_size} != {s.size}")
    if bad:
        print("FETCH_FAIL " + "; ".join(bad), flush=True)
        sys.exit(1)
    (dest / "FETCH_OK").write_text(f"{repo} {rev}\n")
    print(f"FETCH_OK {repo}@{rev[:10]} {sum(s.size or 0 for s in sib) / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
