#!/bin/bash

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
