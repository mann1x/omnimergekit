#!/usr/bin/env bash
# MultiPL-E phase-2 evaluation — runs the nuprl language-sandbox image against a
# directory of per-problem *.json generations produced by multipl_e_generate.py.
#
# Usage:
#   multipl_e_evaluate.sh <generations_dir> <output_dir>
#
# The image `ghcr.io/nuprl/multipl-e-evaluation` is loaded from a local tar if
# present (MPE_IMAGE_TAR, default the solidpc backup_models path), else pulled.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <generations_dir> <output_dir>" >&2
    exit 1
fi

GEN_DIR="$1"
OUT_DIR="$2"
IMG="${MPE_IMAGE:-ghcr.io/nuprl/multipl-e-evaluation:latest}"
IMAGE_TAR="${MPE_IMAGE_TAR:-/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/multipl_e/docker_images/multipl-e-evaluation.tar}"

[[ -d "$GEN_DIR" ]] || { echo "ERROR: $GEN_DIR not found" >&2; exit 1; }
mkdir -p "$OUT_DIR"

N=$(ls "$GEN_DIR"/*.json 2>/dev/null | wc -l)
echo "[eval] generations: $N files in $GEN_DIR"
[[ "$N" -gt 0 ]] || { echo "ERROR: no generations to evaluate"; exit 1; }

# Ensure image is loaded (tar first to avoid a network pull when cached).
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    if [[ -f "$IMAGE_TAR" ]]; then
        echo "[eval] loading image from $IMAGE_TAR …"
        docker load -i "$IMAGE_TAR"
    else
        echo "[eval] pulling image (no tar at $IMAGE_TAR) …"
        docker pull "$IMG"
    fi
fi

echo "[eval] running $IMG against $GEN_DIR …"
docker run --rm \
    --network none \
    -v "$GEN_DIR":/inputs:ro \
    -v "$OUT_DIR":/outputs \
    "$IMG" \
    --dir /inputs --output-dir /outputs

echo "[eval] computing pass@1 …"
python3 - "$OUT_DIR" <<'PY'
import json, sys, glob, os
out_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(out_dir, "*.results.json")))
if not files:
    print(f"ERROR: no *.results.json in {out_dir}", file=sys.stderr)
    sys.exit(2)
n_total = len(files)
n_pass = 0
for f in files:
    d = json.load(open(f))
    res = d.get("results") or []
    ok = any(r.get("status") == "OK" for r in res[:1])  # pass@1 = first completion
    if ok:
        n_pass += 1
pct = 100.0 * n_pass / n_total
print(f"[eval] {os.path.basename(out_dir)}: pass@1 = {n_pass}/{n_total} = {pct:.2f}%")
with open(os.path.join(out_dir, "_summary.json"), "w") as f:
    json.dump({"n_total": n_total, "n_pass": n_pass, "pass_at_1": pct/100.0}, f, indent=2)
PY
