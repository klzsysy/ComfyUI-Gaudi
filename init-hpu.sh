#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(pwd -P)

usage() {
    cat <<'EOF'
Usage: ./init-hpu.sh [options]

Initialize the local ComfyUI Gaudi2 environment:
  1. Discover the Habana base venv under /opt/gaudi-install/venv-* (or --base-venv / GAUDI_BASE_VENV)
  2. Create .venv linked to the Habana base site-packages via gaudi-base.pth
  3. Install ComfyUI dependencies without shadowing vendor Habana PyTorch
  4. Build the optional Gaudi2 Qwen3.5 TPC-C kernel library in build/hpu/

Options:
  --base-venv PATH   Use an explicit Habana base venv directory
  --recreate-venv    Remove and recreate .venv from scratch
  --skip-deps        Skip pip dependency installation
  --rebuild-tpc      Force rebuild of build/hpu Qwen3.5 TPC kernels
  --skip-tpc         Skip building optional Qwen3.5 TPC kernels
  -h, --help         Show this help message
EOF
}

base_venv="${GAUDI_BASE_VENV:-}"
install_root="${GAUDI_INSTALL_ROOT:-/opt/gaudi-install}"
recreate_venv=0
skip_deps=0
rebuild_tpc=0
skip_tpc=0

while (( $# > 0 )); do
    case "$1" in
        --base-venv)
            if (( $# < 2 )); then
                echo "Missing value for --base-venv" >&2
                exit 1
            fi
            base_venv="$2"
            shift 2
            ;;
        --recreate-venv)
            recreate_venv=1
            shift
            ;;
        --skip-deps)
            skip_deps=1
            shift
            ;;
        --rebuild-tpc|--force-tpc)
            rebuild_tpc=1
            shift
            ;;
        --skip-tpc)
            skip_tpc=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

discover_base_venv() {
    local candidate
    local matches=()
    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] || continue
        if [[ -x "$candidate/bin/python" ]] && compgen -G "$candidate/lib/python*/site-packages/habana_frameworks" > /dev/null; then
            matches+=("$candidate")
        fi
    done < <({ compgen -G "$install_root/venv-*" || true; } | sort -V -r)
    if (( ${#matches[@]} > 0 )); then
        printf '%s\n' "${matches[0]}"
        return 0
    fi
    return 1
}

if [[ -z "$base_venv" ]]; then
    if ! base_venv=$(discover_base_venv); then
        echo "No Habana Gaudi base venv found matching $install_root/venv-*." >&2
        echo "Pass --base-venv <path> or set GAUDI_BASE_VENV to your Gaudi Python venv." >&2
        exit 1
    fi
fi

if [[ ! -x "$base_venv/bin/python" ]]; then
    echo "Invalid Habana base venv (missing bin/python): $base_venv" >&2
    exit 1
fi
if ! compgen -G "$base_venv/lib/python*/site-packages/habana_frameworks" > /dev/null; then
    echo "Invalid Habana base venv (missing habana_frameworks): $base_venv" >&2
    exit 1
fi

base_venv=$(cd "$base_venv" && pwd -P)
echo "Using Habana base venv: $base_venv"

if (( recreate_venv )) && [[ -d .venv ]]; then
    echo "Removing existing .venv (--recreate-venv)..."
    rm -rf .venv
fi

base_py_ver=$("$base_venv/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if [[ ! -x .venv/bin/python ]]; then
    echo "Creating .venv from $base_venv/bin/python (Python $base_py_ver)..."
    "$base_venv/bin/python" -m venv .venv
else
    venv_py_ver=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    if [[ "$venv_py_ver" != "$base_py_ver" ]]; then
        echo "Existing .venv uses Python $venv_py_ver, but base venv uses Python $base_py_ver." >&2
        echo "Re-run with ./init-hpu.sh --recreate-venv." >&2
        exit 1
    fi
    echo "Reusing existing .venv (Python $venv_py_ver)."
fi

if [[ -d "$base_venv/lib/python$base_py_ver/site-packages" ]]; then
    base_site="$base_venv/lib/python$base_py_ver/site-packages"
elif [[ -d "$base_venv/lib64/python$base_py_ver/site-packages" ]]; then
    base_site="$base_venv/lib64/python$base_py_ver/site-packages"
else
    base_site=$("$base_venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
fi
venv_site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$base_site" > "$venv_site/gaudi-base.pth"
echo "Linked vendor packages via $venv_site/gaudi-base.pth -> $base_site"

tmp_dir=""
trap '[[ -n "${tmp_dir:-}" ]] && rm -rf "$tmp_dir"' EXIT

if (( ! skip_deps )); then
    tmp_dir=$(mktemp -d)

    filtered_reqs="$tmp_dir/requirements-hpu.txt"
    templates_req="$tmp_dir/templates-req.txt"
    .venv/bin/python - "$root/requirements.txt" "$filtered_reqs" "$templates_req" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text().splitlines()
skip_vendor = {"torch", "torchvision", "torchaudio"}
filtered = []
templates = []
for raw in src:
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    pkg = re.split(r"[<>=!~;\[\s@]", line, maxsplit=1)[0].strip().lower().replace("_", "-")
    if pkg in skip_vendor:
        continue
    if pkg == "comfyui-workflow-templates":
        templates.append(line)
        continue
    filtered.append(line)
Path(sys.argv[2]).write_text("\n".join(filtered) + ("\n" if filtered else ""))
Path(sys.argv[3]).write_text("\n".join(templates) + ("\n" if templates else ""))
PY

    echo "Installing ComfyUI Python dependencies into .venv (preserving vendor Habana PyTorch)..."
    if [[ -s "$filtered_reqs" ]]; then
        .venv/bin/python -m pip install --no-deps -r "$filtered_reqs"
    fi
    .venv/bin/python -m pip install \
        multidict propcache trampoline greenlet python-dotenv \
        kornia-rs mako aiohappyeyeballs aiosignal async-timeout attrs frozenlist \
        'setuptools>=77.0.3' pytest
    if [[ -s "$templates_req" ]]; then
        .venv/bin/python -m pip install -r "$templates_req"
    fi
    if [[ -f "$root/manager_requirements.txt" ]]; then
        .venv/bin/python -m pip install -r "$root/manager_requirements.txt"
    fi

    # Verify vendor Habana PyTorch is NOT shadowed by a local CUDA torch inside .venv.
    .venv/bin/python - "$base_site" <<'PY'
import os
import sys
import torch
import habana_frameworks.torch  # noqa: F401

base_site = os.path.realpath(sys.argv[1])
torch_path = os.path.realpath(torch.__file__)
if not torch_path.startswith(base_site + os.sep):
    raise SystemExit(
        f"ERROR: torch ({torch_path}) is not loaded from Habana base site-packages ({base_site})."
    )
print(f"Verified Habana PyTorch: {torch.__version__} ({torch_path})")
PY
fi

if (( ! skip_tpc )) && [[ "${COMFY_HPU_DISABLE_QWEN35_TPC:-0}" != "1" ]]; then
    echo "Building Gaudi2 Qwen3.5 TPC-C kernels in $root/build/hpu..."
    tpc_flags=(--ensure)
    if (( rebuild_tpc )); then
        tpc_flags=(--force)
    fi
    PT_HPU_LAZY_MODE=0 .venv/bin/python scripts/gaudi/build_qwen35_tpc.py "${tpc_flags[@]}"
fi

echo "ComfyUI HPU initialization complete. Start the server with ./start-hpu.sh or bash scripts/gaudi/start.sh."
