#!/bin/bash

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

file_path="/tmp/pdc/init_step.state"
# 使用 grep 和 awk 提取最外层的 step 值
step_value=$(grep -o '"step":[0-9]*' "$file_path" | head -n 1 | awk -F ':' '{print $2}') 
echo "$step_value" 
