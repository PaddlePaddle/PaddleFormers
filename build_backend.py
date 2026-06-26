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

"""Lightweight PEP 517 build backend for paddleformers.

Generates ``paddleformers/_version.py`` at build time and delegates every
actual build hook to ``setuptools.build_meta``.
"""

import logging
import os
import subprocess
from pathlib import Path

from setuptools import build_meta as orig

logger = logging.getLogger(__name__)

_pkg_root = Path(__file__).parent.resolve()


def is_git_repo() -> bool:
    return (_pkg_root / ".git").exists()


def get_git_commit_hash(cwd: Path) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .strip()
        .decode("utf-8")
    )


def _generate_version_info() -> str:
    """Generate ``paddleformers/_version.py`` with git metadata."""
    version_file = _pkg_root / "version.txt"
    base_version = version_file.read_text().strip()

    git_commit_hash = get_git_commit_hash(_pkg_root)

    version_py = _pkg_root / "paddleformers" / "_version.py"

    if version_py.exists() and not is_git_repo():
        logger.info("_version.py already exists (not in git repo), keeping it")
        return base_version

    if os.environ.get("PADDLEFORMERS_VERSION") is not None:
        final_version = os.environ["PADDLEFORMERS_VERSION"]
    else:
        commit_short = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=11", "HEAD"],
                cwd=_pkg_root,
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
                cwd=_pkg_root,
            )
            .strip()
            .decode("utf-8")
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=_pkg_root,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        if branch.startswith("release/"):
            final_version = f"{base_version}.post{date_str}+{commit_short}"
        else:
            final_version = f"{base_version}.dev{date_str}+{commit_short}"

    with open(version_py, "w") as f:
        f.write(
            "# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.\n"
            "#\n"
            '# Licensed under the Apache License, Version 2.0 (the "License");\n'
            "# you may not use this file except in compliance with the License.\n"
            "# You may obtain a copy of the License at\n"
            "#\n"
            "#     http://www.apache.org/licenses/LICENSE-2.0\n"
            "#\n"
            "# Unless required by applicable law or agreed to in writing, software\n"
            '# distributed under the License is distributed on an "AS IS" BASIS,\n'
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
            "# See the License for the specific language governing permissions and\n"
            "# limitations under the License.\n"
            "\n"
            '"""Auto-generated version info — do not edit."""\n'
            "\n"
            f'__version__ = "{final_version}"\n'
            f'commit = "{git_commit_hash}"\n'
        )
    logger.info(f"Created _version.py with version {final_version}")
    return final_version


_generate_version_info()


def get_requires_for_build_wheel(config_settings=None):
    return orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return orig.get_requires_for_build_editable(config_settings)


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
    return orig.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_editable(
    wheel_directory, config_settings=None, metadata_directory=None
):
    return orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    return orig.build_sdist(sdist_directory, config_settings)
