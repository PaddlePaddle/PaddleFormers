# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.version import git

commit = "unknown"

paddleformers_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if commit.endswith("unknown") and git.is_git_repo(paddleformers_dir) and git.have_git():
    commit = git.git_revision(paddleformers_dir).decode("utf-8")
    if git.is_dirty(paddleformers_dir):
        commit += ".dirty"
del paddleformers_dir


__all__ = ["show"]


def show():
    """Get the corresponding commit id of paddleformers.

    Returns:
        The commit-id of paddleformers will be output.

        full_version: version of paddleformers


    Examples:
        .. code-block:: python

            import paddleformers

            paddleformers.version.show()
            # commit: 1ef5b94a18773bb0b1bba1651526e5f5fc5b16fa

    """
    print("commit:", commit)
