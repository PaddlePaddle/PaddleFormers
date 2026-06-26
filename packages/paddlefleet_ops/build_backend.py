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

import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import backends
from build_utils import (
    check_cuda_arch_list,
    check_patchelf_exists,
    check_submodule_updated,
    get_libs,
    get_special_build_deps,
)
from setuptools import build_meta as orig

backends.init_backend_type()

logger = logging.getLogger(__name__)

# Root of this sub-package (packages/paddlefleet_ops/)
_pkg_root = Path(__file__).parent.resolve()
# Workspace root (two levels up: packages/paddlefleet_ops/ → packages/ → workspace root)
_workspace_root = _pkg_root.parent.parent.resolve()
_pkg_rel_path = _pkg_root.relative_to(_workspace_root).as_posix()
_version_txt = _pkg_root / "version.txt"


def is_git_repo():
    return (_workspace_root / ".git").exists()


def get_git_commit_hash(cwd: Path | None) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .strip()
        .decode("utf-8")
    )


def _get_current_branch(cwd: Path | None) -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd
        )
        .strip()
        .decode("utf-8")
    )


def _find_base_branch(cwd: Path) -> str:
    """Determine the base branch (develop or release/*).

    If the current branch IS develop or release/*, use it directly.
    Otherwise, default to 'develop'.
    Returns the branch name without remote prefix (e.g. "develop" or "release/0.2").
    """
    current = _get_current_branch(cwd)
    if current == "develop" or current.startswith("release/"):
        return current

    # Default to develop for non-base branches
    return "develop"


def _get_last_packages_commit(cwd: Path, base_branch: str) -> str:
    """Get the last commit that modified the packages/ directory based on current state.

    Always searches from the current branch or HEAD (detached), never uses other branches.
    base_branch is only used to determine version suffix (dev/post), not for search.
    """
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "packages/"],
        cwd=cwd,
        capture_output=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().decode()

    return get_git_commit_hash(cwd)


def _generate_version_info():
    """Generate version info file from version.txt + git commit hash.

    CI release builds may override via PADDLEFLEET_VERSION env var.
    Otherwise the version is derived as:
      - develop branch: <base_version>.dev<YYYYMMDD>+<commit_hash_8>
      - release/* branch: <base_version>.post<YYYYMMDD>+<commit_hash_8>

    Both paddleformers and paddlefleet_ops live in the same repo, so the
    commit and date always reflect HEAD (same as paddleformers build_backend).
    """
    version_py = _pkg_root / "src" / "paddlefleet_ops" / "version.py"

    if version_py.exists() and not is_git_repo():
        logger.info(
            "The version.py file already exists (not in git repo), keeping it"
        )
        return

    if os.environ.get("PADDLEFLEET_VERSION") is not None:
        final_version = os.environ["PADDLEFLEET_VERSION"]
    else:
        base_version = _version_txt.read_text().strip()
        base_branch = _find_base_branch(_workspace_root)
        commit_short = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=11", "HEAD"],
                cwd=_workspace_root,
            )
            .strip()
            .decode("utf-8")
        )
        date_str = (
            subprocess.check_output(
                [
                    "git",
                    "log",
                    "-1",
                    "--format=%cd",
                    "--date=format:%Y%m%d",
                    "HEAD",
                ],
                cwd=_workspace_root,
            )
            .strip()
            .decode("utf-8")
        )
        if base_branch.startswith("release/"):
            final_version = f"{base_version}.post{date_str}+{commit_short}"
        else:
            final_version = f"{base_version}.dev{date_str}+{commit_short}"

    git_commit_hash = get_git_commit_hash(_workspace_root)

    with open(version_py, "w") as f:
        f.write(
            "# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.\n"
        )
        f.write("#\n")
        f.write(
            '# Licensed under the Apache License, Version 2.0 (the "License");\n'
        )
        f.write(
            "# you may not use this file except in compliance with the License.\n"
        )
        f.write("# You may obtain a copy of the License at\n")
        f.write("#\n")
        f.write("#     http://www.apache.org/licenses/LICENSE-2.0\n")
        f.write("#\n")
        f.write(
            "# Unless required by applicable law or agreed to in writing, software\n"
        )
        f.write(
            '# distributed under the License is distributed on an "AS IS" BASIS,\n'
        )
        f.write(
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        )
        f.write(
            "# See the License for the specific language governing permissions and\n"
        )
        f.write("# limitations under the License.\n")
        f.write("\n")
        f.write('"""Generate version info file with git metadata."""\n')
        f.write("\n")
        f.write(f'__version__ = "{final_version}"\n')
        f.write(f'commit = "{git_commit_hash}"\n')
    logger.info(f"Created version.py with version {final_version}")
    return final_version


# Generate version info as soon as this module is imported
_generate_version_info()


def _prepare_ecosystem(use_symlinks: bool):
    """Iterates over all registered libraries and prepares them."""
    if backends.IS_NVIDIA:
        libs = get_libs()
        # Build all libraries in parallel to exploit multi-core machines;
        # install sequentially afterwards because install() has ordered
        # side-effects (e.g. patchelf on deep_ep_cpp.so).
        with ThreadPoolExecutor(max_workers=len(libs)) as executor:
            futures = {executor.submit(lib.build): lib for lib in libs}
            for future in as_completed(futures):
                future.result()  # re-raise any build exception immediately
        for lib in libs:
            lib.install(use_symlinks=use_symlinks)
    elif backends.IS_XPU:
        # xpu specific preparations
        pass
    elif backends.IS_ILUVATAR_GPU:
        pass
    elif backends.IS_METAX_GPU:
        pass


def _clean_egg_info():
    """Remove stale *.egg-info directories under _pkg_root.

    setuptools' egg_info command reads the SOURCES.txt it finds inside an
    existing egg-info directory.  If a previous (possibly failed) build left
    one behind that contains absolute paths, every subsequent build will fail
    with "path is not relative".  Deleting it forces setuptools to regenerate
    SOURCES.txt from scratch with correct relative paths.

    egg-info directories can appear at the top level *or* inside src/
    (e.g. src/paddlefleet_ops.egg-info/), so we use rglob to cover both.
    """
    for egg_info_dir in _pkg_root.rglob("*.egg-info"):
        if egg_info_dir.is_dir():
            logger.info(f"Removing stale egg-info: {egg_info_dir}")
            shutil.rmtree(egg_info_dir, ignore_errors=True)


def get_requires_for_build_wheel(config_settings=None):
    return get_special_build_deps()


def get_requires_for_build_sdist(config_settings=None):
    return get_special_build_deps()


def get_requires_for_build_editable(config_settings=None):
    return get_special_build_deps()


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return orig.prepare_metadata_for_build_wheel(
        metadata_directory, config_settings
    )


def prepare_metadata_for_build_editable(
    metadata_directory, config_settings=None
):
    return orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    check_cuda_arch_list()
    check_patchelf_exists()
    check_submodule_updated()
    _prepare_ecosystem(use_symlinks=False)
    return orig.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_editable(
    wheel_directory, config_settings=None, metadata_directory=None
):
    check_cuda_arch_list()
    check_patchelf_exists()
    check_submodule_updated()
    _prepare_ecosystem(use_symlinks=True)
    return orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    raise RuntimeError(
        "Currently, we don't support building sdist. "
        "Please re-build paddlefleet_ops with `--wheel` option. "
        "For example, run `uv build --package paddlefleet_ops --wheel`."
    )
    # TODO(dev): Enable source distribution build when it's ready.
    # check_submodule_updated()
    # return orig.build_sdist(sdist_directory, config_settings)
