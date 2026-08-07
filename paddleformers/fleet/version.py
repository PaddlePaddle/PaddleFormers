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

try:
    from ._version import (
        __version__ as __version__,
        commit as commit,
    )
except:

    def _get_version_from_source():
        from datetime import datetime
        from pathlib import Path

        workspace_root = Path(__file__).parent.parent.parent.resolve()
        version_file = workspace_root / "version.txt"
        if not version_file.exists():
            raise RuntimeError("version.txt not found in workspace root")
        base_version = version_file.read_text().strip()
        if not base_version:
            raise RuntimeError("version.txt is empty")
        date_str = datetime.now().strftime("%Y%m%d")
        return f"{base_version}.dev{date_str}"

    def _get_commit_from_source():
        import subprocess
        from pathlib import Path

        workspace_root = Path(__file__).parent.parent.parent.resolve()
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace_root,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )

    __version__ = _get_version_from_source()
    commit = _get_commit_from_source()
