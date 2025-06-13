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

"""Git utilities."""
# https://github.com/python/mypy/blob/2c2d126cc742f2467045d36780c33bb8fb77a614/mypy/git.py#L1-L34
# Used also from setup.py, so don't pull in anything additional here (like mypy or typing):
from __future__ import annotations

import os
import subprocess


def is_git_repo(dir: str) -> bool:
    """Is the given directory version-controlled with git?"""
    return os.path.exists(os.path.join(dir, ".git"))


def have_git() -> bool:
    """Can we run the git executable?"""
    try:
        subprocess.check_output(["git", "--help"])
        return True
    except subprocess.CalledProcessError:
        return False
    except OSError:
        return False


def git_revision(dir: str) -> bytes:
    """Get the SHA-1 of the HEAD of a git repository."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dir).strip()


def is_dirty(dir: str) -> bool:
    """Check whether a git repository has uncommitted changes."""
    output = subprocess.check_output(["git", "status", "-uno", "--porcelain"], cwd=dir)
    return output.strip() != b""
