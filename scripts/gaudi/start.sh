#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PT_HPU_LAZY_MODE=0
export GC_KERNEL_PATH="${GC_KERNEL_PATH:-/usr/lib/habanalabs/libtpc_kernels.so}"
exec .venv/bin/python main.py --disable-dynamic-vram --disable-pinned-memory \
    --disable-async-offload --disable-xformers --use-pytorch-cross-attention \
    --disable-all-custom-nodes --listen 127.0.0.1 --port 8189 "$@"
