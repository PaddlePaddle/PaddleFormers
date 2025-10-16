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

import sys

from paddleformers.cli.export.export import run_export
from paddleformers.cli.train.tuner import run_tuner


def launch():
    """
    Distributed launch
    """

    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        raise ValueError("len(sys.argv) mush be larger than 1")

    if command == "train":
        run_tuner()
    elif command == "export":
        run_export()
    else:
        raise ValueError(f"Unknown command : {command}")


if __name__ == "__main__":
    launch()
