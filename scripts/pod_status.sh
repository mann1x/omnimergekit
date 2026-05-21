#!/usr/bin/env bash
# pod_status.sh — at-a-glance status across all providers + persistent set.
#
# Usage:
#   pod_status.sh                    # all providers we have a token for
#   pod_status.sh --provider linode  # one provider
#   pod_status.sh --persistents      # only the tracked-persistent set
#
# Output: one line per active instance:
#   <provider>  <id>  <status>  <type>  <region>  <hourly>  <uptime>  <label>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pod_provider.sh"

ONLY=""; PERSIST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --provider)    ONLY="$2"; shift 2 ;;
        --persistents) PERSIST=1; shift ;;
        *) shift ;;
    esac
done

list_linode() {
    [ -z "${LINODE_TOKEN:-}" ] && return 0
    PROVIDER=linode prov_list | /root/anaconda3/envs/omnimergekit/bin/python -c '
import json, sys, datetime
data = json.load(sys.stdin)
PRICE = {
    "g2-gpu-rtx6000-1": 1.50, "g2-gpu-rtx6000-2": 3.00, "g2-gpu-rtx6000-4": 6.00,
    "g1-gpu-rtx6000-1": 1.50, "g6-standard-1": 0.012,
}
for v in data:
    iid = v["id"]; status = v.get("status","?"); typ = v.get("type","?")
    region = v.get("region","?"); label = v.get("label","")
    try:
        dt = datetime.datetime.fromisoformat(v.get("created","").replace("Z","+00:00"))
        up = f"{(datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds()/3600:.1f}h"
    except Exception:
        up = "?"
    print(f"linode  {iid}  {status:14}  {typ:24}  {region:14}  ${PRICE.get(typ,0):.2f}/h  {up:>6}  {label}")
'
}

list_vast() {
    [ -z "${VAST_AI_API_KEY:-}" ] && return 0
    PROVIDER=vast prov_list 2>/dev/null | /root/anaconda3/envs/omnimergekit/bin/python -c '
import json, sys
try: data = json.load(sys.stdin)
except Exception: data = []
for v in data:
    iid = v.get("id","?"); status = v.get("actual_status","?"); typ = v.get("gpu_name","?")
    geo = v.get("geolocation","?"); h = v.get("dph_total",0); label = v.get("label","") or ""
    up_h = (v.get("duration",0) or 0)/3600.0
    print(f"vast    {iid}  {status:14}  {typ:24}  {geo:14}  ${h:.2f}/h  {up_h:>5.1f}h  {label}")
' || true
}

if [ "$PERSIST" = "1" ]; then
    LIST="$HOME/.cache/omk-pods/persistent.json"
    if [ -f "$LIST" ]; then
        /root/anaconda3/envs/omnimergekit/bin/python - "$LIST" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
for e in data:
    print(f"{e['provider']}  {e['id']}  persistent      host={e['host']}:{e['port']}  label={e['label']}")
PYEOF
    else
        echo "(no persistent pods tracked)"
    fi
    exit 0
fi

printf "%-7s %-12s %-14s %-24s %-14s %-9s %-6s %s\n" PROVIDER ID STATUS TYPE REGION HOURLY UPTIME LABEL
case "$ONLY" in
    linode) list_linode ;;
    vast)   list_vast   ;;
    "")     list_vast; list_linode ;;
    *)      echo "unknown provider $ONLY" >&2; exit 2 ;;
esac
