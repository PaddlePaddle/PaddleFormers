# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import subprocess
import sys

from pkg_resources import parse_version


def read_version_of_remote_package(name: str) -> str:
    """get version of remote package,

    adapted from: https://stackoverflow.com/a/58649262/6894382

    Args:
        name (str): the name of package

    Returns:
        str: the version of package
    """
    latest_version = str(
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "{}==random".format(name)], capture_output=True, text=True
        )
    )
    latest_version = latest_version[latest_version.find("(from versions:") + 15 :]
    latest_version = latest_version[: latest_version.find(")")]
    latest_version = latest_version.replace(" ", "").split(",")[-1]
    return latest_version


def read_version_of_local_package(version_file_path: str) -> str:
    """get version of local package

    Args:
        version_file_path (str): the path of `VERSION` file

    Returns:
        str: the version of local package
    """
    with open(version_file_path, "r", encoding="utf-8") as f:
        version = f.read().strip()
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)

    args = parser.parse_args()

    version_file_map = {
        "ppdiffusers": "ppdiffusers/VERSION",
        "paddle-pipelines": "pipelines/VERSION",
    }
    remote_version = read_version_of_remote_package(args.name)

    if args.name == "paddlenlp":
        local_version = str(subprocess.check_output(["python", "setup.py", "--version"], text=True))
    elif args.name in version_file_map:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_version_file = os.path.join(PROJECT_ROOT, version_file_map[args.name])
        local_version = read_version_of_local_package(local_version_file)
    else:
        raise ValueError(f"package<{args.name}> not supported")

    should_deploy = str(parse_version(remote_version) < parse_version(local_version)).lower()
    print(f"should_deploy={should_deploy}")
