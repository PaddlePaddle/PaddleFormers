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

"""
ci.integration_test.check_log_for_exitcode
"""

import sys


def check_tests(log_path: str, check_string="Training completed") -> bool:
    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            if check_string in line:
                print(f"Found '{check_string}' string in log file.'")
                print("Test passed.")
                return True
    print(f"Did not find '{check_string}' string in log file.'")
    print("Test failed.")
    return False


if __name__ == "__main__":
    log_path = sys.argv[1]
    check_str = sys.argv[2] if len(sys.argv) > 2 else "Training completed"
    result = check_tests(log_path, check_string=check_str)
    if result:
        sys.exit(0)
    else:
        sys.exit(1)
