#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PT_HPU_LAZY_MODE=0
export GC_KERNEL_PATH="${GC_KERNEL_PATH:-/usr/lib/habanalabs/libtpc_kernels.so}"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv/bin/python in $PWD. Run ./init-hpu.sh first." >&2
    exit 1
fi

qwen35_tpc_library="$PWD/build/hpu/libcomfy_qwen35_tpc.so"
qwen35_tpc_bridge="$PWD/build/hpu/comfy_qwen35_tpc_bridge.so"
if [[ "${COMFY_HPU_DISABLE_QWEN35_TPC:-0}" != "1" ]]; then
    if [[ ! -f "$qwen35_tpc_library" || ! -f "$qwen35_tpc_bridge" ]]; then
        echo "Missing Gaudi2 Qwen3.5 TPC kernels in $PWD/build/hpu. Run ./init-hpu.sh first (or set COMFY_HPU_DISABLE_QWEN35_TPC=1 to disable)." >&2
        exit 1
    fi
    if [[ -f scripts/gaudi/build_qwen35_tpc.py ]] && ! .venv/bin/python scripts/gaudi/build_qwen35_tpc.py --check >/dev/null 2>&1; then
        echo "Gaudi2 Qwen3.5 TPC kernels in $PWD/build/hpu are out of date. Run ./init-hpu.sh first (or set COMFY_HPU_DISABLE_QWEN35_TPC=1 to disable)." >&2
        exit 1
    fi
    if [[ ":$GC_KERNEL_PATH:" != *":$qwen35_tpc_library:"* ]]; then
        export GC_KERNEL_PATH="$GC_KERNEL_PATH:$qwen35_tpc_library"
    fi
fi

exec .venv/bin/python main.py --disable-dynamic-vram --disable-pinned-memory \
    --disable-async-offload --disable-xformers --use-pytorch-cross-attention \
    --disable-all-custom-nodes --listen 127.0.0.1 --port 8189 "$@"
