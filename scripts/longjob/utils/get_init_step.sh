#!/bin/bash

file_path="/tmp/pdc/init_step.state"
# 使用 grep 和 awk 提取最外层的 step 值
step_value=$(grep -o '"step":[0-9]*' "$file_path" | head -n 1 | awk -F ':' '{print $2}') 
echo "$step_value" 
