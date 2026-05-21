# Linode / Akamai Cloud — pod fleet setup

We use Linode for **long-running** persistent pods (Ollama mirror, stable
serving endpoints, evaluation pods we don't want reclaimed). For ephemeral
1-shot eval jobs vast.ai is still 5–10× cheaper and remains the default.

The provider abstraction lives in `scripts/pod_provider.sh` — sourcing it
gives any script `prov_create`, `prov_status`, `prov_wait_ready`,
`prov_destroy`, `prov_balance`, `prov_list`. Top-level convenience wrappers:

| Script | Purpose |
|---|---|
| `scripts/pod_provision.sh --provider linode --label X` | provision + wait-for-SSH + write `~/.cache/omk-pods/X.env` |
| `scripts/pod_status.sh` | one-line state per active pod across all providers we hold tokens for |
| `scripts/pod_destroy.sh --label X` | safe destroy with eval_results/imatrix archive guard (per `feedback_eval_results_are_sacred.md`) |

## One-time setup

```bash
# 1. Get an API token
#    cloud.linode.com → My Profile → API Tokens → Create a Personal Access Token
#    Scopes: Linodes RW + Account RO + StackScripts RW + Volumes RW + Domains R
#    Save somewhere safe — the token is shown once.

# 2. Stash it next to the existing pod tokens (~/.bashrc on solidpc):
echo 'export LINODE_TOKEN="linode_pat_..."' >> ~/.bashrc
source ~/.bashrc

# 3. Sanity-check
linode-cli account view                      # account + balance
linode-cli linodes list                       # zero rows initially is fine
scripts/pod_status.sh                         # cross-provider view
```

## Default GPU instance — `g2-gpu-rtx6000-1`

| Region | Frankfurt (`eu-central`) — closest to solidpc |
| Type   | `g2-gpu-rtx6000-1` (1× RTX 6000 Ada, 48 GB VRAM, 24 vCPU, 232 GB RAM) |
| Image  | `linode/ubuntu24.04` (kernel 6.8, CUDA 13 drivers via runfile install) |
| Disk   | 200 GiB primary; attach a Block Storage Volume for HF cache (≥ 500 GiB) |
| Hourly | ~$1.50/h on-demand → ~$1080/month if left on |

To browse other GPU classes:
```bash
linode-cli linodes types --json | jq '.[] | select(.class=="gpu") | {id, label, vcpus, memory, disk, price:.price.hourly}'
```

## Common ops

```bash
# Provision a long-running eval pod
scripts/pod_provision.sh \
    --provider linode \
    --label  gemma4-eval-stack2 \
    --type   g2-gpu-rtx6000-1 \
    --region eu-central \
    --disk   200 \
    --persistent

# SSH into it (uses the cached env file)
source ~/.cache/omk-pods/gemma4-eval-stack2.env
$SSH_CMD

# Push the canonical bootstrap on the pod
scp -P "$PORT" scripts/pod_bootstrap_reeval.sh "root@$HOST:/root/"
$SSH_CMD "bash /root/pod_bootstrap_reeval.sh"

# Check what's running everywhere (vast + linode)
scripts/pod_status.sh

# When done — eval_results archived FIRST, then:
scripts/pod_destroy.sh --label gemma4-eval-stack2
```

## Budget tracking

The team budget is €2-3K/month with our share ~€1-1.5K/month. At
`g2-gpu-rtx6000-1` rate ($1.50/h ≈ €1.40/h):

| Usage pattern | Monthly cost |
|---|---|
| 1 pod 24/7        | ~€1000 |
| 1 pod 12 h/day    | ~€500  |
| 2 pods 24/7       | ~€2000 (over budget — stop one before adding) |

The Linode console shows live billing, and `linode-cli account view` returns
`balance` (negative = credit). When the new account drops, the 2-week
unrestricted period is a great time to sanity-check H100 pricing for 31B
NVFP4A16 evals and decide if upgrading the default type is worth it.

## Block Storage for HF cache (recommended)

Persistent pods benefit from a detachable Block Storage Volume for the HF
cache + model weights — survives instance rebuilds, can be moved between
pods in the same region:

```bash
# Create a 1 TB volume in eu-central
linode-cli volumes create --label hf-cache --region eu-central --size 1000

# Attach to a running instance
linode-cli volumes attach <volume_id> --linode_id <instance_id>

# On the pod, format + mount once:
mkfs.ext4 /dev/disk/by-id/scsi-0Linode_Volume_hf-cache
mkdir /workspace/hf-cache
mount /dev/disk/by-id/scsi-0Linode_Volume_hf-cache /workspace/hf-cache
export HF_HOME=/workspace/hf-cache/.huggingface
```

## Provider hand-off

Existing pod scripts (`pod_setup_eval_envs.sh`, `pod_bootstrap_reeval.sh`)
are provider-agnostic by design — they run on the SSH host without caring
how it was provisioned. The `--provider` flag only matters at provision and
destroy time. Vast.ai-specific scripts under `backup_models/scripts/pod_*`
remain untouched; new work that needs Linode goes through the new
`pod_provision.sh` wrapper.
