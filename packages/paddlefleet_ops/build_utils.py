# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as get_pkg_version
from pathlib import Path

import backends
from packaging.version import Version

logger = logging.getLogger(__name__)

# packages/paddlefleet_ops/
PKG_ROOT = Path(__file__).parent.resolve()
# workspace root (packages/paddlefleet_ops/ → packages/ → workspace root)
ROOT_DIR = PKG_ROOT.parent.parent.resolve()

OPS_DIR = PKG_ROOT / "src" / "paddlefleet_ops"
THIRD_PARTY_INSTALL_TEMP = PKG_ROOT / "src" / "_third_party_install_temp"


def remove_path(path: Path) -> None:
    """Removes a path (file, directory, or symlink) if it exists."""
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def create_symlink(src: Path, dst: Path) -> None:
    """Creates a symlink from src to dst, overwriting dst if it exists."""
    remove_path(dst)
    logger.info(f"Symlinking {src} -> {dst}")
    dst.symlink_to(src, target_is_directory=src.is_dir())


@dataclass
class Artifact:
    """
    Defines a mapping from a path in installation directory to a target name in ops directory.

    source_rel_path: Relative path from the library's installation directory (e.g., 'deep_gemm').
    target_name: Name of the symlink/directory to create in 'src/paddlefleet/ops' (e.g., 'deep_gemm').
    """

    source_rel_path: str
    target_name: str


class EcosystemLibrary:
    """
    Represents an external ecosystem operator library.
    Encapsulates logic for building and installing library.
    """

    def __init__(
        self,
        name: str,
        source_rel_path: str,
        artifacts: list[Artifact],
        extra_env: dict[str, str] | None = None,
        include_dirs: list[str] | None = None,
    ):
        self.name = name
        # source_rel_path is relative to PKG_ROOT (where third_party/ lives)
        self.source_dir = PKG_ROOT / source_rel_path
        # Install into a subdirectory named after the library
        self.install_dir = THIRD_PARTY_INSTALL_TEMP / name
        self.artifacts = artifacts
        self._extra_env = extra_env or {}
        self._include_dirs = include_dirs or []

    def build(self) -> None:
        """Builds the library unconditionally."""
        logger.info(f"Building ecosystem library: {self.name}")
        # Clean any stale artifacts from a previous (possibly partial) build
        # so that pip install does not warn about existing directories and
        # leave inconsistent state.
        if self.install_dir.exists():
            logger.info(f"Removing stale install dir: {self.install_dir}")
            shutil.rmtree(self.install_dir)
        self.install_dir.mkdir(parents=True, exist_ok=True)

        # Also clean the source-tree build directory to avoid
        # "[Errno 17] File exists: ...dist-info" from setuptools/bdist_wheel.
        source_build_dir = self.source_dir / "build"
        if source_build_dir.exists():
            logger.info(f"Removing stale source build dir: {source_build_dir}")
            shutil.rmtree(source_build_dir)

        # Special pre-build step for DeepGEMM: link CUTLASS headers into deep_gemm/include
        if self.name.lower() == "deepgemm":
            cutlass_root = (
                self.source_dir / "third-party" / "cutlass" / "include"
            )
            target_include_dir = self.source_dir / "deep_gemm" / "include"
            target_include_dir.mkdir(parents=True, exist_ok=True)

            links = {
                cutlass_root / "cutlass": target_include_dir / "cutlass",
                cutlass_root / "cute": target_include_dir / "cute",
            }

            for src, dst in links.items():
                create_symlink(src, dst)

        # pip install . --target  <install_dir> --no-deps --no-build-isolation --no-compile
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            ".",
            "--target",
            str(self.install_dir),
            "--no-deps",
            "--no-build-isolation",
            "--no-compile",
            "-v",
        ]

        try:
            _env = os.environ.copy()
            _env.update(self._extra_env)
            if self._include_dirs:
                abs_dirs = [
                    str(self.source_dir / d) for d in self._include_dirs
                ]
                extra = os.pathsep.join(abs_dirs)
                for var in ("C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
                    existing = _env.get(var, "")
                    _env[var] = (
                        f"{extra}{os.pathsep}{existing}" if existing else extra
                    )
            subprocess.check_call(cmd, cwd=self.source_dir, env=_env)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build {self.name}: {e}")
            raise

    def install(self, use_symlinks: bool = False) -> None:
        """Installs artifacts to ops directory via symlink or copy."""
        for artifact in self.artifacts:
            # Artifact source path is relative to the installation directory
            src = self.install_dir / artifact.source_rel_path
            dst = OPS_DIR / artifact.target_name

            if use_symlinks:
                create_symlink(src, dst)
            else:
                remove_path(dst)
                logger.info(f"Copying {src} -> {dst}")
                if src.is_dir():
                    shutil.copytree(
                        src, dst, symlinks=False, dirs_exist_ok=True
                    )
                else:
                    shutil.copy(src, dst)

            if artifact.target_name == "deep_ep_cpp.so":
                cmd = [
                    "patchelf",
                    "--add-rpath",
                    "$ORIGIN/../nvidia/nvshmem/lib",
                    dst,
                ]
                try:
                    subprocess.check_call(cmd)
                except subprocess.CalledProcessError:
                    cmd_str = " ".join(cmd)
                    logger.error(f"Failed to run {cmd_str}.")
                    raise


def check_submodule_updated():
    if backends.IS_NVIDIA:
        required_paths = [
            "third_party/DeepGEMM/setup.py",
            "third_party/DeepEP/setup.py",
            "third_party/HybridEP/setup.py",
            "third_party/HybridEP/third-party/nccl/Makefile",
            "third_party/quack/pyproject.toml",
            "third_party/sonic-moe/pyproject.toml",
            "third_party/flash-attention/setup.py",
            "third_party/FlashMLA/setup.py",
        ]
        missing_paths = [
            path for path in required_paths if not (PKG_ROOT / path).exists()
        ]
        if missing_paths:
            logger.error(
                "\033[91m Found uninitialized submodules. Please use "
                "'git submodule update --init --recursive' from the PaddleFormers "
                f"workspace root to fix. Missing: {', '.join(missing_paths)}\033[0m"
            )
            sys.exit(1)
    elif backends.IS_XPU:
        pass


def check_patchelf_exists():
    """Checks if patchelf is installed."""
    if shutil.which("patchelf") is None:
        logger.error(
            "\033[31m Error: 'patchelf' not found in PATH.\033[0m\n"
            "\033[31m Please install 'patchelf' using your package manager (apt, yum, conda, uv, etc.) before proceeding.\033[0m"
        )
        sys.exit(1)


_SUPPORTED_CUDA_ARCHS = {"8.0", "9.0", "10.0", "10.3"}


def check_cuda_arch_list():
    """Validate PADDLE_CUDA_ARCH_LIST early, before any compilation starts.

    Called at the top of build_wheel so a bad value is caught immediately
    rather than after DeepGEMM/DeepEP have already spent minutes compiling.
    """
    raw = os.environ.get("PADDLE_CUDA_ARCH_LIST", "").strip()
    if not raw:
        return

    tokens = [t.strip() for t in re.split(r"[;,]", raw) if t.strip()]
    arch_pattern = re.compile(r"^\d+\.\d+$")
    bad_format = [t for t in tokens if not arch_pattern.match(t)]
    if bad_format:
        raise ValueError(
            f"\n\nInvalid PADDLE_CUDA_ARCH_LIST value: {raw!r}\n"
            f"  Bad token(s): {bad_format}\n"
            f"  Expected format: dot-separated X.Y values, semicolon- or comma-delimited.\n"
            f'  Example: PADDLE_CUDA_ARCH_LIST="9.0" or "9.0;10.0" or "8.0,9.0"\n'
            f"  Note: spaces are NOT valid separators.\n"
            f"  Supported archs: {sorted(_SUPPORTED_CUDA_ARCHS)}"
        )

    unsupported = [t for t in tokens if t not in _SUPPORTED_CUDA_ARCHS]
    if unsupported:
        raise ValueError(
            f"\n\nUnsupported arch(es) in PADDLE_CUDA_ARCH_LIST: {unsupported}\n"
            f"  Full value: {raw!r}\n"
            f"  Supported archs: {sorted(_SUPPORTED_CUDA_ARCHS)}\n"
            f'  Example: PADDLE_CUDA_ARCH_LIST="9.0" or "9.0;10.0;10.3"'
        )


def _detect_local_gpu_arch():
    """Auto-detect the compute capability of the first visible GPU via nvidia-smi.

    Returns a string like '9.0', or None if detection fails.
    DeepEP requires GPU compute capability >= 9.0 (SM90+).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        caps = {
            line.strip() for line in out.decode().splitlines() if line.strip()
        }
        # Return the first available architecture (usually all GPUs are same)
        return caps[0] if caps else None
    except Exception:
        return None


def get_cuda_version():
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise FileNotFoundError(
            "nvcc command not found. Please make sure CUDA toolkit is installed and nvcc is in PATH."
        )

    result = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    version_output = result.stdout

    match = re.search(r"release (\d+)\.(\d+)", version_output)
    if not match:
        raise ValueError(
            f"Cannot parse CUDA version from nvcc output:\n{version_output}"
        )
    cuda_major = int(match.group(1))
    cuda_minor = int(match.group(2))

    if cuda_major < 12:
        raise ValueError(
            f"CUDA version must be >= 12. Detected version: {cuda_major}.{cuda_minor}"
        )
    return cuda_major, cuda_minor


def get_special_build_deps():
    if backends.IS_NVIDIA:
        cuda_major, cuda_minor = get_cuda_version()
        deps = []
        # for deep_ep build
        if platform.machine() == "aarch64":
            deps.append("nvidia-nvshmem-cu13>=3.3.9,<3.5")
            return deps
        if cuda_major == 12:
            if cuda_minor > 6:
                deps.append("paddle-nvidia-nvshmem-cu12>=3.3.9,<3.5")
            else:
                deps.append("nvidia-nvshmem-cu12>=3.3.9,<3.5")
        elif cuda_major == 13:
            deps.append("paddle-nvidia-nvshmem-cu13>=3.3.9,<3.5")
        else:
            raise ValueError(
                f"Unsupported CUDA version: {cuda_major}.{cuda_minor}."
            )
        return deps
    elif backends.IS_XPU:
        try:
            xpu_version = get_pkg_version("paddlepaddle-xpu")
        except PackageNotFoundError:
            xpu_version = None

        if xpu_version is not None:
            if Version(xpu_version) < Version("3.3.0"):
                raise ValueError(
                    f"paddlepaddle-xpu {xpu_version} is too old, >=3.3.0 required."
                )
            deps = [f"paddlepaddle-xpu=={xpu_version}"]
        else:
            deps = ["paddlepaddle-xpu>=3.3.0"]
        return deps
    else:
        return []


def get_libs():
    cuda_major, cuda_minor = get_cuda_version()

    # Allow CI or users to pin the arch list via PADDLE_CUDA_ARCH_LIST.
    # Falls back to sensible defaults derived from the detected CUDA version:
    #   < 12.8  → SM90 only
    #   ≥ 12.8 / 13.x → SM90 + SM100 + SM103
    _default_arch = (
        "9.0" if (cuda_major == 12 and cuda_minor < 8) else "9.0;10.0;10.3"
    )
    _raw = (
        os.environ.get("PADDLE_CUDA_ARCH_LIST")
        or _detect_local_gpu_arch()
        or _default_arch
    )
    # Normalize: some callers use comma-separated (e.g. "8.0,9.0,10.0,10.3").
    # Paddle's _get_cuda_arch_flags only accepts semicolon-separated values,
    # and DeepEP only supports SM90/SM100/SM103 — drop anything outside that set.
    _supported = {"9.0", "10.0", "10.3"}
    _deep_ep_arch = (
        ";".join(a for a in re.split(r"[;,]", _raw) if a.strip() in _supported)
        or _default_arch
    )

    LIBRARIES: list[EcosystemLibrary] = [
        EcosystemLibrary(
            name="DeepGEMM",
            source_rel_path="third_party/DeepGEMM",
            artifacts=[
                # Updated paths to point to installation directory
                Artifact("deep_gemm", "deep_gemm"),
                Artifact("deep_gemm_cpp", "deep_gemm_cpp"),
            ],
            include_dirs=[
                "deep_gemm/include",
                "third-party/cutlass/include",
                "third-party/fmt/include",
            ],
        ),
        EcosystemLibrary(
            name="DeepEP",
            source_rel_path="third_party/DeepEP",
            artifacts=[
                Artifact("deep_ep", "deep_ep"),
                Artifact("deep_ep_cpp.so", "deep_ep_cpp.so"),
            ],
            extra_env={"PADDLE_CUDA_ARCH_LIST": _deep_ep_arch},
            include_dirs=["csrc/"],
        ),
        EcosystemLibrary(
            name="flash-attention",
            source_rel_path="third_party/flash-attention/flashmask",
            artifacts=[
                Artifact("flash_mask", "flash_mask"),
            ],
            extra_env={"FLASHMASK_BUILD": "fa4"},
            include_dirs=[
                "flash_mask/flashmask_attention_v3/csrc",
                "flash_mask/flashmask_attention_v3",
                "flash_mask/flashmask_attention_v3/cutlass/include",
            ],
        ),
        EcosystemLibrary(
            name="FlashMLA",
            source_rel_path="third_party/FlashMLA",
            artifacts=[
                Artifact("flash_mla", "flash_mla"),
            ],
            extra_env={
                "PADDLE_CUDA_ARCH_LIST": "",
                "FLASH_MLA_DISABLE_SM90": str("9.0" not in _deep_ep_arch),
                "FLASH_MLA_DISABLE_SM100": str(
                    (cuda_major, cuda_minor) <= (12, 8)
                ),
            },
        ),
    ]
    if (cuda_major, cuda_minor) >= (12, 9):
        LIBRARIES.append(
            EcosystemLibrary(
                name="HybridEP",
                source_rel_path="third_party/HybridEP",
                artifacts=[
                    Artifact("deep_ep", "hybrid_ep"),
                    Artifact("hybrid_ep_cpp.so", "hybrid_ep_cpp.so"),
                ],
                extra_env={
                    "HYBRID_EP_MULTINODE": "1",
                    "HYBRID_EP_SKIP_DEEP_EP": "1",
                    "PADDLE_CUDA_ARCH_LIST": _deep_ep_arch,
                },
            ),
        )
    if sys.version_info >= (3, 12):
        LIBRARIES.append(
            EcosystemLibrary(
                name="quack",
                source_rel_path="third_party/quack",
                artifacts=[
                    Artifact("quack", "quack"),
                ],
            )
        )
        LIBRARIES.append(
            EcosystemLibrary(
                name="sonic-moe",
                source_rel_path="third_party/sonic-moe",
                artifacts=[
                    Artifact("sonicmoe", "sonicmoe"),
                ],
            )
        )
        LIBRARIES.append(
            EcosystemLibrary(
                name="cudnn",
                source_rel_path="third_party/cudnn-frontend",
                artifacts=[
                    Artifact("cudnn", "cudnn"),
                ],
                extra_env={
                    "FETCHCONTENT_SOURCE_DIR_DLPACK": str(
                        PKG_ROOT / "third_party" / "dlpack"
                    ),
                },
            )
        )
    return LIBRARIES
