#!/bin/bash

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
