"""pod_v9_quant_fetch.py — ON THE POD: download published v9-agentic GGUF tiers for a quant comparison.

Each tier comes from the HF GGUF repo with aria2c (token in a 0600 input file that is deleted after
the transfer, never in argv), is checked against the repo's LFS sha256, and gets a `<file>.ok`
marker that the eval queues wait for. Tiers are fetched in the order given, so the first queue cell
can start while the rest download.

usage: pod_v9_quant_fetch.py <dest_dir> <tier> [<tier> ...]
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "ManniX-ITA/gemma-4-20B-v9-agentic-GGUF"
STEM = "gemma-4-20B-v9-agentic"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch(fname: str, sha: str, dest: Path, token: str) -> Path:
    out = dest / fname
    if not (out.exists() and sha256(out) == sha):
        inp = dest / ".aria_in"
        fd = os.open(inp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"https://huggingface.co/{REPO}/resolve/main/{fname}\n  out={fname}\n"
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
    Path(f"{out}.ok").write_text(sha + "\n")
    return out


def main() -> None:
    dest = Path(sys.argv[1])
    tiers = sys.argv[2:]
    dest.mkdir(parents=True, exist_ok=True)
    token = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    assert token, "HF token file is empty"
    lfs = {s.rfilename: s.lfs.sha256 for s in HfApi(token=token).model_info(REPO, files_metadata=True).siblings
           if s.lfs}
    for t in tiers:
        fname = f"{STEM}-{t}.gguf"
        if fname not in lfs:
            print(f"FETCH_FAIL {t}: {fname} not in {REPO}", flush=True)
            sys.exit(1)
        try:
            p = fetch(fname, lfs[fname], dest, token)
        except Exception as e:  # noqa: BLE001 — report and stop; queues waiting on this tier time out
            print(f"FETCH_FAIL {t}: {e}", flush=True)
            sys.exit(1)
        print(f"FETCH_OK {t} {p.stat().st_size / 1e9:.2f} GB sha {lfs[fname][:16]}", flush=True)
    print("FETCH_ALL_OK", flush=True)


if __name__ == "__main__":
    main()
