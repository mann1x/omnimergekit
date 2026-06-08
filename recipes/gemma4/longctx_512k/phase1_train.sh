#!/bin/bash
# phase1_train.sh — blackswan-2 training launcher (T87).
#
# Runs on the POD. Dual-GPU:
#   GPU 0 = LoRA continued pretrain on YaRN-patched config
#   GPU 1 = probe watcher (every new ckpt → quick NIAH-256k probe; abort if 3× <80%)
#
# Both processes launched in nohup + disown; PIDs recorded in
# /srv/ml/runs/longctx_512k/<T>/{train,probe}.pid for the orchestrator to
# poll. tmux session 'longctx_<T>' wraps both for attach-and-watch.
#
# ### COUNCIL — read brief §2-3 + plan §"Phase 1". Key knobs:
#   --tokens     250M (default; council may want 100M/200M/300M)
#   --pack-len   256k (default; council may want 128k or 512k pack-up frac)
#   --pack-512-frac 0.5  (only 0.0 forces no 512k chunks; relevant for 31B Option A vs B)
#   --lr         1e-4 (LoRA scale)
#   --rank       16
#   --alpha      32

set -uo pipefail

BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TRAINER=$BM/scripts/phase1_train_yarn_lora.py
PROBE=$BM/scripts/phase1_probe_watcher.py
DATA_PACK=$BM/scripts/pack_pg19_math_rpv2.py
PATCH_YARN=$BM/scripts/patch_yarn_config.py

TARGET=""
DO_RUN=0
TOKENS=250_000_000
PACK_LEN=262144
PACK_512_FRAC=0.5
LR=1e-4
RANK=16
ALPHA=32
while [ $# -gt 0 ]; do
  case "$1" in
    --target)         TARGET=$2; shift 2;;
    --run)            DO_RUN=1; shift;;
    --tokens)         TOKENS=$2; shift 2;;
    --pack-len)       PACK_LEN=$2; shift 2;;
    --pack-512-frac)  PACK_512_FRAC=$2; shift 2;;
    --lr)             LR=$2; shift 2;;
    --rank)           RANK=$2; shift 2;;
    --alpha)          ALPHA=$2; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

case "$TARGET" in
  31b) BASE_DIR=$BM/google/gemma-4-31B-it
       OUT_PREFIX=$BM/runs/longctx_512k/31b
       # 31B can't fit 512k packed chunks on a single 96GB Blackwell w/o activation ckpt.
       # Plan Option A: force 512k frac to 0 for 31B unless council says otherwise.
       PACK_512_FRAC=0.0  # OVERRIDE — see plan §Hardware
       ;;
  v6c) BASE_DIR=$BM/google/gemma-4-A4B-98e-v6-coder-it
       OUT_PREFIX=$BM/runs/longctx_512k/v6c
       ;;
  *) echo "FATAL: --target must be 31b|v6c"; exit 2;;
esac

YARN_CFG_DIR=$OUT_PREFIX/yarn_patched_base
DATA_DIR=$OUT_PREFIX/packed_data
CKPT_DIR=$OUT_PREFIX/ckpts
LOG=$OUT_PREFIX/train.log
PROBE_LOG=$OUT_PREFIX/probe.log

mkdir -p "$OUT_PREFIX" "$CKPT_DIR" "$DATA_DIR" "$YARN_CFG_DIR"

echo "=== Phase 1 train: $TARGET ==="
echo "  base_dir     : $BASE_DIR"
echo "  yarn_cfg_dir : $YARN_CFG_DIR"
echo "  data_dir     : $DATA_DIR"
echo "  ckpt_dir     : $CKPT_DIR"
echo "  tokens       : $TOKENS"
echo "  pack_len     : $PACK_LEN ($((PACK_LEN/1024))k)"
echo "  pack_512_frac: $PACK_512_FRAC"
echo "  LR / r / α   : $LR / $RANK / $ALPHA"
echo

if [ "$DO_RUN" -ne 1 ]; then
  echo "[dry-run] would:"
  echo "  1. patch_yarn_config.py --src $BASE_DIR --dst $YARN_CFG_DIR --factor 2.0"
  echo "  2. pack_pg19_math_rpv2.py --out $DATA_DIR --tokens $TOKENS --pack-len $PACK_LEN --pack-512-frac $PACK_512_FRAC"
  echo "  3. CUDA_VISIBLE_DEVICES=0 phase1_train_yarn_lora.py … (GPU 0, ~6-14h)"
  echo "  4. CUDA_VISIBLE_DEVICES=1 phase1_probe_watcher.py … (GPU 1, polls ckpts)"
  exit 0
fi

# ---------- Step 1: patch config ----------
if [ ! -f "$YARN_CFG_DIR/config.json" ]; then
  echo "[step 1/4] patching YaRN config"
  "$PY" "$PATCH_YARN" --src "$BASE_DIR" --dst "$YARN_CFG_DIR" --factor 2.0 --native 262144 || exit 1
fi

# ---------- Step 2: pack data ----------
if [ ! -f "$DATA_DIR/.done" ]; then
  echo "[step 2/4] packing data ~$TOKENS tokens"
  "$PY" "$DATA_PACK" \
    --out "$DATA_DIR" \
    --tokens "$TOKENS" \
    --pack-len "$PACK_LEN" \
    --pack-512-frac "$PACK_512_FRAC" \
    --tokenizer "$YARN_CFG_DIR" || exit 1
  touch "$DATA_DIR/.done"
fi

# ---------- Step 3+4: launch trainer + probe in tmux ----------
echo "[step 3+4/4] launching trainer on GPU 0 + probe watcher on GPU 1"
tmux kill-session -t "longctx_$TARGET" 2>/dev/null || true
tmux new-session -d -s "longctx_$TARGET" -x 220 -y 50 "bash -c '
  echo \"=== longctx_$TARGET train+probe session ===\";
  echo;
  # Trainer
  nohup env CUDA_VISIBLE_DEVICES=0 \\
      \"$PY\" \"$TRAINER\" \\
        --yarn-cfg-dir \"$YARN_CFG_DIR\" \\
        --data-dir \"$DATA_DIR\" \\
        --ckpt-dir \"$CKPT_DIR\" \\
        --tokens \"$TOKENS\" \\
        --lr \"$LR\" --rank \"$RANK\" --alpha \"$ALPHA\" \\
    > \"$LOG\" 2>&1 &
  TRAIN_PID=\$!; disown
  echo \$TRAIN_PID > \"$OUT_PREFIX/train.pid\"
  echo \"  trainer PID \$TRAIN_PID — log $LOG\";
  # Probe
  nohup env CUDA_VISIBLE_DEVICES=1 \\
      \"$PY\" \"$PROBE\" \\
        --ckpt-dir \"$CKPT_DIR\" \\
        --base-dir \"$YARN_CFG_DIR\" \\
        --abort-on 3 --threshold 0.80 \\
        --kill-pid \$TRAIN_PID \\
    > \"$PROBE_LOG\" 2>&1 &
  PROBE_PID=\$!; disown
  echo \$PROBE_PID > \"$OUT_PREFIX/probe.pid\"
  echo \"  probe   PID \$PROBE_PID — log $PROBE_LOG\";
  bash
'"

sleep 3
tmux list-sessions | grep "longctx_$TARGET"
echo
echo "Attach with:  ssh linode-blackswan-2 -t 'tmux attach -t longctx_$TARGET'"
echo "PIDs in:      $OUT_PREFIX/{train,probe}.pid"
echo "Logs in:      $LOG  +  $PROBE_LOG"
