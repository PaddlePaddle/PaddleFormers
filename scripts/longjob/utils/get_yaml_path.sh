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

# 设定文件名
file="/tmp/pdc/yaml.state"

# 检查文件是否存在
if [ -f "$file" ]; then
    # 使用grep查找包含"path"的行，然后使用cut切分字符串提取路径
    path=$(grep -o '"path": *"[^"]*"' "$file" | cut -d '"' -f 4)
else
    # 文件不存在，path为空
    if [ -n "$PDC_YAML_PATH" ] && [ "$PDC_YAML_PATH" != "None" ]; then
        path="$PDC_YAML_PATH"
    else
        path="$CONF_FILE"
    fi
fi

# 输出path变量的值，以便其他脚本可以捕获
echo "$path"
