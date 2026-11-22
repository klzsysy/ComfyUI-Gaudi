"""Build the optional Gaudi2 Qwen3.5 TPC-C kernels with the active Habana SDK.

Run during environment setup via ``./init-hpu.sh`` (or directly with ``.venv/bin/python``).
``scripts/gaudi/start.sh`` verifies the built artifacts via ``--check`` and adds
``build/hpu/libcomfy_qwen35_tpc.so`` to ``GC_KERNEL_PATH``.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


KERNELS = (
    ("topk_s1", "topk_s1_bmax.c"),
    ("topk_s2", "topk_s2_blocks.c"),
    ("topk_s3", "topk_s3_extract.c"),
    ("gdn_rec", "gdn_recurrence.c"),
    ("gdn_conv", "gdn_conv_silu.c"),
    ("rms_q", "rmsnorm_quant.c"),
    ("gated_rms_q", "gated_rmsnorm_quant.c"),
    ("sig_mul_q", "sig_mul_quant.c"),
    ("silu_mul_s1", "silu_mul_quant_s1.c"),
    ("silu_mul_s2", "silu_mul_quant_s2.c"),
    ("qk_norm_rope", "qk_norm_rope.c"),
    ("sample_accept", "sample_tree_accept.c"),
    ("dual_rms_cat", "dual_rmsnorm_cat_quant.c"),
    ("moss_situ_q", "moss_situ_quant.c"),
    ("moss_qkv_prep", "moss_qkv_prep.c"),
)

EXTRA_SOURCES = ("params.h", "glue.cpp", "bridge.cpp")
FINGERPRINT_NAME = "qwen35_tpc.fingerprint"
RUNTIME_KEYS = ("schema", "sources_sha256", "python", "habana_frameworks", "habana_plugin", "torch", "torch_lib")


def _resolve_tpc_clang():
    override = os.environ.get("TPC_CLANG")
    if override:
        p = Path(override)
        return str(p) if p.is_file() and os.access(p, os.X_OK) else None
    found = shutil.which("tpc-clang")
    if found:
        return found
    for candidate in ("/opt/habanalabs/bin/tpc-clang", "/usr/bin/tpc-clang"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _resolve_habana_include():
    override = os.environ.get("HABANA_INCLUDE")
    candidates = [override] if override else ["/usr/include/habanalabs", "/opt/habanalabs/include"]
    for directory in candidates:
        if directory and (Path(directory) / "tpc_kernel_lib_interface.h").is_file():
            return directory
    return None


def _stat_sig(path):
    if not path:
        return None
    try:
        st = os.stat(path)
        return [os.path.realpath(path), st.st_size, st.st_mtime_ns]
    except OSError:
        return None


def _spec_paths(mod_name):
    spec = importlib.util.find_spec(mod_name)
    if spec is None:
        return None
    paths = []
    if spec.origin:
        paths.append(os.path.realpath(spec.origin))
    if spec.submodule_search_locations:
        paths.extend(os.path.realpath(p) for p in spec.submodule_search_locations)
    return paths


def compute_fingerprint(root: Path, compiler=None, gxx=None, habana_inc=None):
    source = root / "comfy" / "text_encoders" / "hpu_qwen35_tpc"
    h = hashlib.sha256()
    input_files = [Path(__file__).resolve()]
    input_files.extend(source / fn for _, fn in KERNELS)
    input_files.extend(source / fn for fn in EXTRA_SOURCES)
    for path in input_files:
        h.update(str(path.name).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")

    habana_locs = _spec_paths("habana_frameworks") or []
    torch_locs = _spec_paths("torch") or []
    habana_lib_sig = None
    for loc in habana_locs:
        cand = Path(loc) / "torch" / "lib" / "libhabana_pytorch2_plugin.so"
        if cand.is_file():
            habana_lib_sig = _stat_sig(cand)
            break
    torch_lib_sig = None
    for loc in torch_locs:
        p = Path(loc)
        base = p.parent if p.is_file() else p
        cand = base / "lib" / "libtorch_cpu.so"
        if cand.is_file():
            torch_lib_sig = _stat_sig(cand)
            break

    return {
        "schema": 1,
        "sources_sha256": h.hexdigest(),
        "python": [list(sys.version_info[:3]), os.path.realpath(sys.executable)],
        "tpc_clang": _stat_sig(compiler),
        "gxx": _stat_sig(gxx),
        "habana_header": _stat_sig(Path(habana_inc) / "tpc_kernel_lib_interface.h") if habana_inc else None,
        "habana_frameworks": habana_locs,
        "habana_plugin": habana_lib_sig,
        "torch": torch_locs,
        "torch_lib": torch_lib_sig,
    }


def _outputs_ready(build: Path, fingerprint: dict, runtime_only: bool = False):
    kernel_lib = build / "libcomfy_qwen35_tpc.so"
    bridge_lib = build / "comfy_qwen35_tpc_bridge.so"
    fp_file = build / FINGERPRINT_NAME
    if not (kernel_lib.is_file() and bridge_lib.is_file() and fp_file.is_file()):
        return False
    try:
        saved = json.loads(fp_file.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(saved, dict):
        return False
    if runtime_only:
        return all(saved.get(k) == fingerprint.get(k) for k in RUNTIME_KEYS)
    return saved == fingerprint


def _check_prereqs():
    compiler = _resolve_tpc_clang()
    if not compiler:
        return None, "tpc-clang not found (checked TPC_CLANG, PATH, /opt/habanalabs/bin/tpc-clang)"
    gxx = shutil.which("g++")
    if not gxx:
        return None, "g++ compiler not found in PATH"
    habana_inc = _resolve_habana_include()
    if not habana_inc:
        return None, "tpc_kernel_lib_interface.h not found (checked HABANA_INCLUDE, /usr/include/habanalabs)"
    if importlib.util.find_spec("habana_frameworks") is None:
        return None, "habana_frameworks Python package not found in active environment"
    if importlib.util.find_spec("torch") is None:
        return None, "torch Python package not found in active environment"
    return (compiler, gxx, habana_inc), None


def _build_locked(root: Path, build: Path, compiler: str, gxx: str, habana_inc: str, fingerprint: dict):
    source = root / "comfy" / "text_encoders" / "hpu_qwen35_tpc"
    pid = os.getpid()

    def compile_kernel(item):
        sym, filename = item
        obj = build / f"qwen35_{sym}.o"
        tmp_obj = build / f"qwen35_{sym}.{pid}.o"
        subprocess.run(
            [compiler, "-march=gaudi2", "-O2", "-c", str(source / filename), "-o", str(tmp_obj)],
            check=True,
        )
        os.replace(tmp_obj, obj)
        return sym, obj

    workers = min(4, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        compiled = list(pool.map(compile_kernel, KERNELS))

    asm_lines = [".section .rodata"]
    for sym, obj in compiled:
        asm_lines.append(f".global {sym}_start, {sym}_end")
        asm_lines.append(f"{sym}_start:")
        asm_lines.append(f'.incbin "{obj}"')
        asm_lines.append(f"{sym}_end:")
    asm_lines.append('.section .note.GNU-stack,"",@progbits\n')

    blobs_s = build / "qwen35_blobs.S"
    blobs_o = build / "qwen35_blobs.o"
    blobs_s.write_text("\n".join(asm_lines))
    subprocess.run([gxx, "-fPIC", "-c", str(blobs_s), "-o", str(blobs_o)], check=True)

    kernel_lib = build / "libcomfy_qwen35_tpc.so"
    tmp_kernel = build / f"libcomfy_qwen35_tpc.so.tmp.{pid}"
    subprocess.run(
        [
            gxx, "-shared", "-fPIC", "-O2",
            f"-I{habana_inc}",
            f"-I{source}",
            str(source / "glue.cpp"),
            str(blobs_o),
            "-o", str(tmp_kernel),
        ],
        check=True,
    )
    os.replace(tmp_kernel, kernel_lib)

    # Clear stale bridge build artifacts so ABI/header upgrades always trigger a clean compile.
    for stale in ("bridge.o", "bridge.o.d", "build.ninja", "comfy_qwen35_tpc_bridge.so"):
        stale_path = build / stale
        if stale_path.exists():
            stale_path.unlink()

    import habana_frameworks.torch  # noqa: F401
    from torch.utils.cpp_extension import load

    package = Path(habana_frameworks.torch.__file__).parent
    os.environ.setdefault("MAX_JOBS", "2")
    load(
        name="comfy_qwen35_tpc_bridge",
        sources=[str(source / "bridge.cpp")],
        build_directory=str(build),
        extra_include_paths=[str(package / "include"), str(source)],
        extra_ldflags=[
            f'-L{package / "lib"}',
            "-lhabana_pytorch2_plugin",
            f'-Wl,-rpath,{package / "lib"}',
        ],
        is_python_module=False,
        verbose=True,
    )

    fp_tmp = build / f"{FINGERPRINT_NAME}.tmp.{pid}"
    fp_tmp.write_text(json.dumps(fingerprint, sort_keys=True, indent=2) + "\n")
    os.replace(fp_tmp, build / FINGERPRINT_NAME)
    print(f"Built Gaudi2 Qwen3.5 TPC kernels in {build}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build optional Gaudi2 Qwen3.5 TPC-C kernels.")
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Skip building when existing libraries match current sources and active SDK.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 if libraries are present and up to date, non-zero otherwise without building.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a full rebuild even when --ensure is passed.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    build = root / "build" / "hpu"

    prereqs, reason = _check_prereqs()
    if prereqs is None:
        # Allow slim runtime environments without compilers to reuse verified prebuilt libraries.
        if (args.ensure or args.check) and not args.force:
            rt_fp = compute_fingerprint(root)
            if _outputs_ready(build, rt_fp, runtime_only=True):
                return 0
        print(f"Skipping Gaudi2 Qwen3.5 TPC kernel build: {reason}", file=sys.stderr)
        return 1

    compiler, gxx, habana_inc = prereqs
    fingerprint = compute_fingerprint(root, compiler, gxx, habana_inc)

    if args.check:
        return 0 if _outputs_ready(build, fingerprint) else 1

    if args.ensure and not args.force and _outputs_ready(build, fingerprint):
        return 0

    try:
        build.mkdir(parents=True, exist_ok=True)
        lock_path = build / ".build.lock"
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if args.ensure and not args.force and _outputs_ready(build, fingerprint):
                return 0
            _build_locked(root, build, compiler, gxx, habana_inc, fingerprint)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"Gaudi2 Qwen3.5 TPC kernel build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
