#!/usr/bin/env bash
# pod_provision.sh — provider-agnostic instance creator.
#
# Wraps pod_provider.sh's prov_create + prov_wait_ready and emits a
# connection file (provider, instance_id, host, port, ssh_cmd) that
# downstream bootstrap scripts can source.
#
# Usage:
#   pod_provision.sh --provider linode --label "gemma4-eval-stack2" \
#                    --type g2-gpu-rtx6000-1 --region eu-central
#   pod_provision.sh --provider vast --type RTX_4090
#   # auto-detect provider from env tokens:
#   pod_provision.sh --label "scratch"
#
# Writes connection info to:
#   ~/.cache/omk-pods/<label>.env   (sourceable; key=value)
# Prints the same on stdout.
#
# Long-running mode (--persistent) skips the destroy-on-exit hint and adds
# the instance to ~/.cache/omk-pods/persistent.json for tracking.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pod_provider.sh
source "$SCRIPT_DIR/pod_provider.sh"

LABEL=""; PROVIDER_ARG=""; TYPE=""; REGION=""; IMAGE=""; DISK=""; PERSISTENT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --provider)   PROVIDER_ARG="$2"; shift 2 ;;
        --label)      LABEL="$2"; shift 2 ;;
        --type)       TYPE="$2"; shift 2 ;;
        --region)     REGION="$2"; shift 2 ;;
        --image)      IMAGE="$2"; shift 2 ;;
        --disk)       DISK="$2"; shift 2 ;;
        --persistent) PERSISTENT=1; shift ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done

[ -n "$PROVIDER_ARG" ] && export PROVIDER="$PROVIDER_ARG"
prov_resolve

LABEL="${LABEL:-omk-$(date +%Y%m%d-%H%M%S)}"
TYPE="${TYPE:-$(prov_default_type)}"
REGION="${REGION:-$(prov_default_region)}"
IMAGE="${IMAGE:-$(prov_default_image)}"
DISK="${DISK:-200}"

echo "[provision] provider=$PROVIDER type=$TYPE region=$REGION image=$IMAGE label=$LABEL persistent=$PERSISTENT" >&2

# Create
ID=$(prov_create --type "$TYPE" --region "$REGION" --image "$IMAGE" --label "$LABEL" --disk "$DISK")
[ -z "$ID" ] && { echo "[provision] create returned empty id" >&2; exit 1; }
echo "[provision] instance id: $ID" >&2

# Wait for ready
echo "[provision] waiting for SSH..." >&2
read -r HOST PORT < <(prov_wait_ready "$ID")
SSH_CMD=$(prov_ssh_cmd "$ID")

# Write connection file
CACHE="$HOME/.cache/omk-pods"
mkdir -p "$CACHE"
ENV_FILE="$CACHE/${LABEL}.env"
cat > "$ENV_FILE" <<EOF
PROVIDER=$PROVIDER
INSTANCE_ID=$ID
LABEL=$LABEL
HOST=$HOST
PORT=$PORT
SSH_CMD="$SSH_CMD"
CREATED=$(date -Iseconds)
PERSISTENT=$PERSISTENT
EOF
echo "[provision] env file: $ENV_FILE" >&2

# Track persistents for budget visibility
if [ "$PERSISTENT" = "1" ]; then
    LIST="$CACHE/persistent.json"
    /root/anaconda3/envs/omnimergekit/bin/python - "$LIST" "$LABEL" "$PROVIDER" "$ID" "$HOST" "$PORT" <<'PYEOF'
import json, sys, os
path, label, prov, iid, host, port = sys.argv[1:7]
data = []
if os.path.exists(path):
    try: data = json.load(open(path))
    except Exception: data = []
data = [d for d in data if d.get("label") != label]
data.append({"label": label, "provider": prov, "id": iid, "host": host, "port": port})
json.dump(data, open(path, "w"), indent=2)
PYEOF
    echo "[provision] tracked as persistent in $LIST" >&2
fi

# Stdout output (parsable)
cat "$ENV_FILE"
