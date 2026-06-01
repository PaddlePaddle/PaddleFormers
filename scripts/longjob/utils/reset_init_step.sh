#!/bin/bash

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
