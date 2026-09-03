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

Flash_Checkpoint_DIR="/shared/dev/shm/flash"

# 检查目录是否存在
if [ ! -d "$Flash_Checkpoint_DIR" ]; then
  echo "FlashCheckpoint目录 $Flash_Checkpoint_DIR 不存在。"
  return
fi

# 查找并删除以 "checkpoint-" 开头的目录
for dir in "$Flash_Checkpoint_DIR"/checkpoint-*; do
  if [ -d "$dir" ]; then
    echo "正在删除FlashCheckpoint目录: $dir"
    mpirun rm -rf "$dir"
  fi
done

echo "删除FlashCheckpoint目录完成。"
