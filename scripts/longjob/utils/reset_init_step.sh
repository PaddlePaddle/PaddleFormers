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

STEP=$1
CURRENT_TIME=$(date +%s)
FILE="/tmp/pdc/init_step.state"
mkdir -p /tmp/pdc

# 判断文件是否存在
if [ ! -f "$FILE" ]; then
    # 文件不存在，写入新的JSON内容
    echo "{\"step\": $STEP, \"update_time\": $CURRENT_TIME}" > "$FILE"
else
    # 文件存在，进行替换
    sed -i '0,/"step":[0-9]\+/{s/"step":[0-9]\+/"step":'$STEP'/; s/"update_time":[0-9]\+/"update_time":'$CURRENT_TIME'/}' "$FILE"
fi
