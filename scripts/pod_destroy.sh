#!/usr/bin/env bash
# pod_destroy.sh — safe wrapper around prov_destroy with archive guard.
#
# Refuses to destroy a Linode/vast pod until the user confirms eval_results/
# imatrix.dat have been rsynced off the pod (per project rule:
# "Eval results are SACRED — NEVER purge a pod without rsync").
#
# Usage:
#   pod_destroy.sh --label gemma4-eval-stack2
#   pod_destroy.sh --provider linode --id 12345
#   pod_destroy.sh --label scratch --force        # skip archive guard
#
# Looks up the label in ~/.cache/omk-pods/<label>.env when --id is omitted.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pod_provider.sh"

LABEL=""; ID=""; FORCE=0; PROVIDER_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --label)    LABEL="$2"; shift 2 ;;
        --id)       ID="$2"; shift 2 ;;
        --provider) PROVIDER_ARG="$2"; shift 2 ;;
        --force)    FORCE=1; shift ;;
        -h|--help)  sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done

CACHE="$HOME/.cache/omk-pods"
if [ -z "$ID" ]; then
    [ -z "$LABEL" ] && { echo "need --id or --label" >&2; exit 2; }
    ENV_FILE="$CACHE/${LABEL}.env"
    [ ! -f "$ENV_FILE" ] && { echo "no cached env for label $LABEL" >&2; exit 2; }
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    [ -n "$PROVIDER" ] && export PROVIDER
fi
[ -n "$PROVIDER_ARG" ] && export PROVIDER="$PROVIDER_ARG"
prov_resolve

# Archive guard
if [ "$FORCE" != "1" ]; then
    cat <<EOF >&2
WARN: about to destroy $PROVIDER instance $ID (label=${LABEL:-?}).

Before destroy, you should have rsynced from the pod:
  - eval_results/             (samples_*.jsonl, results_*.json, summary.json)
  - imatrix.dat               (if you ran any quant chain)
  - any GGUFs you didn't already push to HF / ollama

Pass --force to bypass this prompt.
Type YES to proceed:
EOF
    read -r ANSWER
    [ "$ANSWER" = "YES" ] || { echo "aborted" >&2; exit 1; }
fi

prov_destroy "$ID"
echo "destroyed: $PROVIDER $ID"

# Untrack persistent
LIST="$CACHE/persistent.json"
[ -f "$LIST" ] && /root/anaconda3/envs/omnimergekit/bin/python - "$LIST" "$ID" <<'PYEOF'
import json, sys
path, iid = sys.argv[1:3]
try:
    data = json.load(open(path))
except Exception:
    data = []
data = [d for d in data if str(d.get("id")) != str(iid)]
json.dump(data, open(path,"w"), indent=2)
PYEOF

# Move env file to .destroyed/ for audit trail
if [ -n "$LABEL" ]; then
    mkdir -p "$CACHE/.destroyed"
    mv "$CACHE/${LABEL}.env" "$CACHE/.destroyed/${LABEL}.$(date +%s).env" 2>/dev/null || true
fi
