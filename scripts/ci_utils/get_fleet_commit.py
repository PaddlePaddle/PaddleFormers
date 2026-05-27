# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 读取setup.py文件 拿到fleet的commit id
def get_fleet_commit_id(setup_file_path):
    with open(setup_file_path, "r") as f:
        for line in f:
            if "paddlefleet==" in line:
                commit_id = line.split("paddlefleet==")[1].strip().strip('"').strip("'")
                commit_id = commit_id.split("+")[1].strip()  # 如果有版本号，取最后的commit id部分
                commit_id = commit_id.split('"')[0].strip()  # 如果有版本号，取最后的commit id部分
                return commit_id
    raise ValueError("FLEET_COMMIT_ID not found in setup.py")


if __name__ == "__main__":
    fleet_commit_id = get_fleet_commit_id("setup.py")
    print(fleet_commit_id)
