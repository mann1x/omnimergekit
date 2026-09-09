#!/usr/bin/env python3
"""Publish the vision projector to both Ornith GGUF repos.

The projector is mmproj-Qwen3.6-27B-A3B-Coder-F16.gguf. It is NOT a re-derivation --
it is the Qwen3.6-A3B-Coder tower, which works on Ornith because Ornith is the same
architecture (Qwen3_5MoeForConditionalGeneration) with the same hidden_size (2048).
VERIFIED by loading it in llama-server against Ornith and reading a 5-band
non-guessable test image 5/5 correct, with a no-image control arm returning nothing.

Filename follows the REPO NAME (see feedback_gguf_filename_follows_the_repo_name).
Provenance is recorded in the card, not smuggled into the filename.
"""
import os
import sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi

SRC = "/mnt/sdc/ream-work/mmproj/mmproj-Qwen3.6-27B-A3B-Coder-F16.gguf"
TARGETS = [
    ("ManniX-ITA/Ornith-1.5-27B-A3B-Coder-MTP-GGUF",
     "mmproj-Ornith-1.5-27B-A3B-Coder-F16.gguf"),
    ("ManniX-ITA/Ornith-1.5-27B-A3B-CoderX-MTP-GGUF",
     "mmproj-Ornith-1.5-27B-A3B-CoderX-F16.gguf"),
]


def main():
    if not os.path.exists(SRC):
        print("MISSING SRC", SRC)
        return 2
    api = HfApi()
    print("src %s (%d bytes)" % (os.path.basename(SRC), os.path.getsize(SRC)), flush=True)
    for repo, name in TARGETS:
        api.upload_file(path_or_fileobj=SRC, path_in_repo=name, repo_id=repo,
                        repo_type="model",
                        commit_message="add vision projector (verified on this arm)")
        files = api.list_repo_files(repo)
        ok = name in files
        print("  %-50s -> %s  present=%s" % (repo.split("/")[-1], name, ok), flush=True)
        if not ok:
            return 3
    print("MMPROJ_UPLOAD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
