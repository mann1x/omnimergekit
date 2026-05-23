#!/usr/bin/env bash
# install_mpe_toolchains_native.sh — provision a vast.ai pod for NATIVE (no-docker)
# MultiPL-E evaluation. See eval/EVAL_PROTOCOL.md §1.4 "MultiPL-E" for the stack
# lock this reproduces. Validated pod 37268930 2026-05-23: MPE-10 128e = 100%
# rs/java/js after running this.
#
# SECURITY: native MPE runs model-generated code UNSANDBOXED. Throwaway pods only.
#
# Idempotent. Pins:
#   MultiPL-E harness  commit 3025a531af74   (the eval logic — language scripts)
#   rustc/cargo        apt (1.75.0 on ubuntu 22.04) — rust needs no external deps
#   javac              default-jdk (OpenJDK 11)
#   javatuples 1.2     /usr/multiple/javatuples-1.2.jar  (eval_java.py HARDCODES this)
#   node               v20 (NodeSource) — apt's node 12 breaks `node:` imports
#   python deps        datasets, sqlitedict, tqdm
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

MPE_HARNESS="${MPE_HARNESS:-/workspace/MultiPL-E}"
MPE_COMMIT="${MPE_COMMIT:-3025a531af74}"
JAVATUPLES_VER="1.2"
JAVATUPLES_PATH="/usr/multiple/javatuples-${JAVATUPLES_VER}.jar"   # path is hardcoded in eval_java.py

echo "[mpe-setup $(date -Iseconds)] START"

echo "[mpe-setup] base toolchains (rustc, cargo, default-jdk, git, curl)"
apt-get update -y >/tmp/mpe_apt_update.log 2>&1
apt-get install -y rustc cargo default-jdk git curl ca-certificates \
    >/tmp/mpe_apt_base.log 2>&1 || { echo "FATAL apt base"; tail -20 /tmp/mpe_apt_base.log; exit 1; }

echo "[mpe-setup] javatuples ${JAVATUPLES_VER} -> ${JAVATUPLES_PATH}"
mkdir -p "$(dirname "$JAVATUPLES_PATH")"
if [ ! -f "$JAVATUPLES_PATH" ]; then
    curl -fsSL -o "$JAVATUPLES_PATH" \
        "https://repo1.maven.org/maven2/org/javatuples/javatuples/${JAVATUPLES_VER}/javatuples-${JAVATUPLES_VER}.jar" \
        || { echo "FATAL javatuples download"; exit 1; }
fi
echo "[mpe-setup] javatuples sha256: $(sha256sum "$JAVATUPLES_PATH" | cut -c1-16)…  (expect 2eda5b19…)"

echo "[mpe-setup] node 20 (NodeSource); purge apt node 12 first to avoid file conflict"
if ! node --version 2>/dev/null | grep -qE "^v(1[6-9]|[2-9][0-9])"; then
    apt-get purge -y libnode-dev libnode72 nodejs nodejs-doc >/tmp/mpe_node_purge.log 2>&1 || true
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/tmp/mpe_nodesource.log 2>&1
    apt-get install -y nodejs >/tmp/mpe_node_install.log 2>&1 \
        || { echo "FATAL node install"; tail -20 /tmp/mpe_node_install.log; exit 1; }
fi
echo "[mpe-setup] node $(node --version)  npm $(npm --version 2>/dev/null)"
node -e "require('node:assert').equal(1,1)" \
    && echo "[mpe-setup] node: import scheme OK" \
    || { echo "FATAL node too old for node: imports"; exit 1; }

echo "[mpe-setup] python deps (datasets, sqlitedict, tqdm)"
"${OMK_PYTHON:-/usr/bin/python3}" -m pip install -q datasets sqlitedict tqdm \
    || { echo "FATAL pip deps"; exit 1; }

echo "[mpe-setup] MultiPL-E harness @ ${MPE_COMMIT} -> ${MPE_HARNESS}"
if [ ! -d "$MPE_HARNESS/.git" ]; then
    git clone https://github.com/nuprl/MultiPL-E "$MPE_HARNESS" >/tmp/mpe_clone.log 2>&1 \
        || { echo "FATAL clone"; exit 1; }
fi
git -C "$MPE_HARNESS" fetch --depth 1 origin "$MPE_COMMIT" >/dev/null 2>&1 || true
git -C "$MPE_HARNESS" checkout -q "$MPE_COMMIT" 2>/dev/null \
    || echo "[mpe-setup] WARN: could not checkout ${MPE_COMMIT} (shallow clone?); using HEAD $(git -C "$MPE_HARNESS" rev-parse --short HEAD)"
[ -f "$MPE_HARNESS/evaluation/src/main.py" ] \
    || { echo "FATAL harness main.py missing"; exit 1; }

echo "[mpe-setup] verify import chain (matlab/etc must be lazy)"
( cd "$MPE_HARNESS/evaluation/src" && "${OMK_PYTHON:-/usr/bin/python3}" -c "import containerized_eval; print('[mpe-setup] containerized_eval import OK')" ) \
    || { echo "FATAL harness import"; exit 1; }

echo "[mpe-setup $(date -Iseconds)] DONE — run MPE with MPE_MODE=native MPE_HARNESS=$MPE_HARNESS, template mode:chat"
