#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
PROBE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBE_CANN="${ASCEND_HOME_PATH:-}"
PROBE_JOBS=2
PROBE_CMAKE_ARGS=()
for PROBE_ARG in "$@"; do
    case "$PROBE_ARG" in
        --cann-root=*) PROBE_CANN="${PROBE_ARG#*=}" ;;
        --jobs=*) PROBE_JOBS="${PROBE_ARG#*=}" ;;
        *) PROBE_CMAKE_ARGS+=("$PROBE_ARG") ;;
    esac
done
if [[ -z "$PROBE_CANN" ]]; then
    echo "请先加载服务器 CANN 环境，或传入 --cann-root=目录（含 include/acl 和 lib64）" >&2
    exit 2
fi
if [[ ! -f "$PROBE_CANN/include/acl/acl_rt.h" ]]; then
    PROBE_ARCH="$(uname -m)-linux"
    if [[ -f "$PROBE_CANN/$PROBE_ARCH/include/acl/acl_rt.h" ]]; then
        PROBE_CANN="$PROBE_CANN/$PROBE_ARCH"
    fi
fi
cmake -S "$PROBE_DIR" -B "$PROBE_DIR/build" -DCANN_ROOT="$PROBE_CANN" -DCMAKE_BUILD_TYPE=Release "${PROBE_CMAKE_ARGS[@]}"
cmake --build "$PROBE_DIR/build" --parallel "$PROBE_JOBS"
