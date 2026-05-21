# pod_provider.sh — provider-agnostic pod control shim.
#
# Source this in any pod_* script that wants to drive vast.ai, RunPod, or
# Linode through one set of calls. The provider is selected via:
#   1. --provider {vast,runpod,linode} flag parsed by the caller
#   2. PROVIDER env var
#   3. auto-detect by which API token is exported in the shell
#
# Exports a small set of functions; everything else stays provider-specific
# in the calling script.
#
# Functions:
#   prov_resolve              # set PROVIDER if unset; export PROVIDER
#   prov_create  --type ID --region R [--disk N] [--label L] [--image IMG]
#                             # echoes the new instance ID on success
#   prov_show    <id>         # echoes JSON
#   prov_status  <id>         # echoes simple status: provisioning|running|stopped|error
#   prov_wait_ready <id>      # blocks until SSH is reachable, then echoes "host port"
#   prov_ssh_cmd  <id>        # echoes the ssh command string (no exec)
#   prov_destroy <id>         # destroys the instance
#   prov_list                 # list active instances
#   prov_balance              # current account balance / credit remaining
#
# Conventions:
#   - All functions emit machine-parseable output on stdout.
#   - Errors go to stderr with a "PROV-ERR:" prefix; exit code nonzero.
#   - Existing scripts that already call `vastai` directly keep working —
#     you only adopt this shim for new --provider-aware code paths.

# Intentionally NOT setting `set -eu` at file scope — sourcing this would
# leak strict mode into the caller's shell, which is surprising. Each
# wrapper script (pod_provision.sh, pod_destroy.sh, pod_status.sh) sets
# its own shell options.

# ----- provider resolution ----------------------------------------------------
prov_resolve() {
    if [ -n "${PROVIDER:-}" ]; then
        case "$PROVIDER" in
            vast|runpod|linode) ;;
            *) echo "PROV-ERR: unknown PROVIDER=$PROVIDER" >&2; return 2;;
        esac
        return 0
    fi
    # Auto-detect: prefer linode > vast > runpod (for new code; legacy scripts
    # ignore this and continue calling vastai directly)
    if [ -n "${LINODE_TOKEN:-}" ];      then export PROVIDER=linode
    elif [ -n "${VAST_AI_API_KEY:-}" ]; then export PROVIDER=vast
    elif [ -n "${RUNPOD_API_KEY:-}" ];  then export PROVIDER=runpod
    else
        echo "PROV-ERR: no provider token in environment (LINODE_TOKEN, VAST_AI_API_KEY, RUNPOD_API_KEY)" >&2
        return 2
    fi
}

# ----- defaults per provider ---------------------------------------------------
# Caller can override via --type / --region / etc.

prov_default_type() {
    case "${PROVIDER:-}" in
        linode)   echo "g2-gpu-rtx6000-1" ;;       # RTX 6000 Ada, ~$1.50/h
        vast)     echo "RTX_4090" ;;
        runpod)   echo "NVIDIA RTX 4090" ;;
    esac
}

prov_default_region() {
    case "${PROVIDER:-}" in
        linode)   echo "eu-central" ;;             # Frankfurt (low-latency from solidpc)
        vast)     echo "" ;;                       # vast picks any
        runpod)   echo "EU-NL-1" ;;
    esac
}

prov_default_image() {
    case "${PROVIDER:-}" in
        linode)   echo "linode/ubuntu24.04" ;;
        vast)     echo "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04" ;;
        runpod)   echo "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04" ;;
    esac
}

# ----- linode backend ----------------------------------------------------------

_linode_cli() {
    # linode-cli reads ~/.config/linode-cli/* OR LINODE_CLI_TOKEN env. Use the
    # env path so a missing config file (fresh host) still works.
    LINODE_CLI_TOKEN="${LINODE_TOKEN:?LINODE_TOKEN unset}" /root/anaconda3/envs/omnimergekit/bin/linode-cli --json --suppress-warnings "$@"
}

_prov_create_linode() {
    local type="$1" region="$2" image="$3" label="$4" disk="$5"
    # disk arg is in GiB. linode-cli takes it implicitly from the type's storage;
    # for larger disks we attach a Block Storage volume separately.
    local out
    out=$(_linode_cli linodes create \
        --type "$type" \
        --region "$region" \
        --image "$image" \
        --label "$label" \
        --booted true \
        --authorized_keys "$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub)" \
        2>&1) || { echo "PROV-ERR: linode create failed: $out" >&2; return 1; }
    # Extract id
    echo "$out" | /root/anaconda3/envs/omnimergekit/bin/python -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id'] if isinstance(d, list) else d['id'])
"
}

_prov_show_linode()    { _linode_cli linodes view "$1"; }
_prov_destroy_linode() { _linode_cli linodes delete "$1"; }
_prov_list_linode()    { _linode_cli linodes list; }
_prov_balance_linode() { _linode_cli account view; }

_prov_status_linode() {
    local s
    s=$(_prov_show_linode "$1" | /root/anaconda3/envs/omnimergekit/bin/python -c "
import json, sys
d = json.load(sys.stdin)
print((d[0] if isinstance(d, list) else d).get('status',''))
")
    case "$s" in
        running)      echo running ;;
        provisioning|booting|rebooting|migrating|cloning|restoring|resizing)  echo provisioning ;;
        offline|stopped|shutting_down) echo stopped ;;
        *)            echo error ;;
    esac
}

_prov_wait_ready_linode() {
    local id="$1"
    local host port=22
    # poll for status=running, then resolve public IPv4
    while true; do
        local s; s=$(_prov_status_linode "$id" || echo error)
        [ "$s" = "running" ] && break
        [ "$s" = "error" ]   && { echo "PROV-ERR: linode $id entered error state" >&2; return 1; }
        sleep 10
    done
    host=$(_prov_show_linode "$id" | /root/anaconda3/envs/omnimergekit/bin/python -c "
import json, sys
d = json.load(sys.stdin)
v = d[0] if isinstance(d, list) else d
print(v['ipv4'][0])
")
    # Probe SSH (StackScript may still be running at first contact)
    for i in $(seq 1 60); do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p $port "root@$host" "echo SSH_OK" 2>/dev/null | grep -q SSH_OK; then
            echo "$host $port"
            return 0
        fi
        sleep 5
    done
    echo "PROV-ERR: linode $id never accepted SSH" >&2
    return 1
}

_prov_ssh_cmd_linode() {
    local host; host=$(_prov_show_linode "$1" | /root/anaconda3/envs/omnimergekit/bin/python -c "
import json, sys
d = json.load(sys.stdin)
v = d[0] if isinstance(d, list) else d
print(v['ipv4'][0])
")
    echo "ssh -o StrictHostKeyChecking=no -p 22 root@$host"
}

# ----- vast backend (thin wrapper over existing vastai CLI) -------------------

_prov_create_vast() {
    local type="$1" region="$2" image="$3" label="$4" disk="$5"
    # find a cheap offer matching type
    local offer_id
    offer_id=$(vastai search offers "gpu_name=$type num_gpus=1 cuda_max_good>=13 verified=True" \
        --raw 2>/dev/null | /root/anaconda3/envs/omnimergekit/bin/python -c "
import json, sys
d = json.load(sys.stdin)
if not d: sys.exit('PROV-ERR: no vast offers')
print(sorted(d, key=lambda x: x.get('dph_total',1e9))[0]['id'])
" )
    vastai create instance "$offer_id" \
        --image "$image" --disk "$disk" --label "$label" \
        --env "-e HF_HUB_ENABLE_HF_TRANSFER=1" \
        --ssh true --direct 2>&1 | grep -oE 'new_contract.: [0-9]+' | awk '{print $2}'
}
_prov_show_vast()    { vastai show instance "$1" --raw; }
_prov_destroy_vast() { vastai destroy instance "$1"; }
_prov_list_vast()    { vastai show instances --raw; }
_prov_balance_vast() { vastai show user --raw; }
_prov_status_vast() {
    local s; s=$(_prov_show_vast "$1" | /root/anaconda3/envs/omnimergekit/bin/python -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('actual_status',''))
")
    case "$s" in
        running) echo running ;;
        loading|created) echo provisioning ;;
        stopped|exited) echo stopped ;;
        *) echo error ;;
    esac
}
_prov_wait_ready_vast() {
    local id="$1" info host port
    while true; do
        local s; s=$(_prov_status_vast "$id" || echo error)
        [ "$s" = "running" ] && break
        [ "$s" = "error" ]   && { echo "PROV-ERR: vast $id error state" >&2; return 1; }
        sleep 10
    done
    info=$(_prov_show_vast "$id")
    host=$(echo "$info" | /root/anaconda3/envs/omnimergekit/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['ssh_host'])")
    port=$(echo "$info" | /root/anaconda3/envs/omnimergekit/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['ssh_port'])")
    for i in $(seq 1 60); do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$port" "root@$host" "echo SSH_OK" 2>/dev/null | grep -q SSH_OK; then
            echo "$host $port"
            return 0
        fi
        sleep 5
    done
    return 1
}
_prov_ssh_cmd_vast() {
    local info; info=$(_prov_show_vast "$1")
    local host; host=$(echo "$info" | /root/anaconda3/envs/omnimergekit/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['ssh_host'])")
    local port; port=$(echo "$info" | /root/anaconda3/envs/omnimergekit/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['ssh_port'])")
    echo "ssh -o StrictHostKeyChecking=no -p $port root@$host"
}

# ----- runpod backend (stub — pod scripts that need it should fill in) -------
_prov_create_runpod()      { echo "PROV-ERR: runpod backend not yet wired" >&2; return 1; }
_prov_show_runpod()        { echo "PROV-ERR: runpod backend not yet wired" >&2; return 1; }
_prov_destroy_runpod()     { echo "PROV-ERR: runpod backend not yet wired" >&2; return 1; }
_prov_list_runpod()        { echo "PROV-ERR: runpod backend not yet wired" >&2; return 1; }
_prov_balance_runpod()     { echo "PROV-ERR: runpod backend not yet wired" >&2; return 1; }
_prov_status_runpod()      { echo "PROV-ERR: runpod backend not yet wired" >&2; return 1; }
_prov_wait_ready_runpod()  { echo "PROV-ERR: runpod backend not yet wired" >&2; return 1; }
_prov_ssh_cmd_runpod()     { echo "PROV-ERR: runpod backend not yet wired" >&2; return 1; }

# ----- dispatch ---------------------------------------------------------------

_dispatch() {
    local op="$1"; shift
    prov_resolve
    case "$PROVIDER" in
        linode) "_prov_${op}_linode" "$@" ;;
        vast)   "_prov_${op}_vast" "$@" ;;
        runpod) "_prov_${op}_runpod" "$@" ;;
    esac
}

prov_create()     {
    local type="$(prov_default_type)" region="$(prov_default_region)"
    local image="$(prov_default_image)" label="omk-$(date +%Y%m%d-%H%M)" disk=200
    while [ $# -gt 0 ]; do
        case "$1" in
            --type)   type="$2";   shift 2 ;;
            --region) region="$2"; shift 2 ;;
            --image)  image="$2";  shift 2 ;;
            --label)  label="$2";  shift 2 ;;
            --disk)   disk="$2";   shift 2 ;;
            *) echo "PROV-ERR: unknown arg $1" >&2; return 2 ;;
        esac
    done
    _dispatch create "$type" "$region" "$image" "$label" "$disk"
}
prov_show()       { _dispatch show "$@"; }
prov_destroy()    { _dispatch destroy "$@"; }
prov_list()       { _dispatch list "$@"; }
prov_balance()    { _dispatch balance "$@"; }
prov_status()     { _dispatch status "$@"; }
prov_wait_ready() { _dispatch wait_ready "$@"; }
prov_ssh_cmd()    { _dispatch ssh_cmd "$@"; }
