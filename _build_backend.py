# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, TypeVar

from setuptools import build_meta as orig

_workspace_root = Path(__file__).parent.resolve()
_build_version_py = _workspace_root / "_paddleformers_build_version.py"
_fleet_version_py = _workspace_root / "paddleformers" / "fleet" / "_version.py"
T = TypeVar("T")


def _is_git_repo() -> bool:
    return (_workspace_root / ".git").exists()


def _get_current_branch() -> str:
    if os.environ.get("BRANCH"):
        return os.environ["BRANCH"]
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_workspace_root
    ).strip().decode("utf-8")


def _find_base_branch() -> str:
    current = _get_current_branch()
    if current == "develop" or current.startswith("release/"):
        return current
    return "develop"


def _get_last_commit() -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H"],
        cwd=_workspace_root,
        capture_output=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().decode("utf-8")
    raise RuntimeError("Cannot find any commit for PaddleFormers version")


def _get_commit_date(commit: str) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%cd", "--date=format:%Y%m%d", commit],
        cwd=_workspace_root,
    ).strip().decode("utf-8")


def _generate_version_info() -> tuple[str, str]:
    if os.environ.get("PADDLEFORMERS_VERSION") is not None:
        commit = _get_last_commit() if _is_git_repo() else "unknown"
        return os.environ["PADDLEFORMERS_VERSION"], commit

    base_version = "1.2.0"
    if not _is_git_repo():
        return base_version, "unknown"

    base_branch = _find_base_branch()
    commit = _get_last_commit()
    commit_short = commit[:8]
    date_str = _get_commit_date(commit)
    if base_branch.startswith("release/"):
        return f"{base_version}.post{date_str}+{commit_short}", commit
    return f"{base_version}.dev{date_str}+{commit_short}", commit


def _render_fleet_version_py(version: str, commit: str) -> str:
    return (
        "# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.\n"
        "#\n"
        "# Licensed under the Apache License, Version 2.0 (the \"License\");\n"
        "# you may not use this file except in compliance with the License.\n"
        "# You may obtain a copy of the License at\n"
        "#\n"
        "#     http://www.apache.org/licenses/LICENSE-2.0\n"
        "#\n"
        "# Unless required by applicable law or agreed to in writing, software\n"
        "# distributed under the License is distributed on an \"AS IS\" BASIS,\n"
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        "# See the License for the specific language governing permissions and\n"
        "# limitations under the License.\n"
        "\n"
        '"""Generated PaddleFormers Fleet version metadata."""\n'
        "\n"
        f'__version__ = "{version}"\n'
        f'commit = "{commit}"\n'
    )


@contextmanager
def _temporary_build_version():
    build_original = _build_version_py.read_text() if _build_version_py.exists() else None
    fleet_original = _fleet_version_py.read_text() if _fleet_version_py.exists() else None
    version, commit = _generate_version_info()
    try:
        _build_version_py.write_text(f'__version__ = "{version}"\n')
        _fleet_version_py.write_text(_render_fleet_version_py(version, commit))
        yield
    finally:
        if build_original is None:
            _build_version_py.unlink(missing_ok=True)
        else:
            _build_version_py.write_text(build_original)
        if fleet_original is None:
            _fleet_version_py.unlink(missing_ok=True)
        else:
            _fleet_version_py.write_text(fleet_original)


def _with_temporary_build_version(func: Callable[..., T], *args, **kwargs) -> T:
    with _temporary_build_version():
        return func(*args, **kwargs)


def get_requires_for_build_sdist(config_settings=None):
    return _with_temporary_build_version(orig.get_requires_for_build_sdist, config_settings)


def get_requires_for_build_wheel(config_settings=None):
    return _with_temporary_build_version(orig.get_requires_for_build_wheel, config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _with_temporary_build_version(
        orig.prepare_metadata_for_build_wheel,
        metadata_directory,
        config_settings,
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _with_temporary_build_version(
        orig.build_wheel,
        wheel_directory,
        config_settings,
        metadata_directory,
    )


def build_sdist(sdist_directory, config_settings=None):
    return _with_temporary_build_version(
        orig.build_sdist,
        sdist_directory,
        config_settings,
    )
