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

import re
import subprocess
import sys

fork_point_sha = subprocess.check_output("git merge-base develop HEAD".split()).decode("utf-8")
modified_files = (
    subprocess.check_output(f"git diff --diff-filter=d --name-only {fork_point_sha}".split()).decode("utf-8").split()
)

valid_dirs = "|".join(sys.argv[1:])
regex = re.compile(rf"^({valid_dirs}).*?\.(py|md)$")

relevant_modified_files = [x for x in modified_files if regex.match(x)]
print(" ".join(relevant_modified_files), end="")
