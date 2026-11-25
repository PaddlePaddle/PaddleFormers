# -*- coding: utf-8 -*-
# !/usr/bin/env python3

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

"""
@version: 1.0
@file: deperacate_checkpoints.py
@time: 2025/04/19 22:58:07
@Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved

Deperacate checkpoints if they may be saved after launching the training process.

"""
import os
import re
import shutil
import sys
from datetime import datetime

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
DEPERECATED_SUFFIX = f"-old-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"


def main(local_checkpoint_dir, start_step):
    """
    deperacate checkpoints after 'start_step'
    """
    available_ckpt_steps = []
    for ckpt in os.listdir(local_checkpoint_dir):
        ckpt_step = _re_checkpoint.search(ckpt)
        if ckpt_step is not None:
            available_ckpt_steps.append(ckpt_step.groups()[0])
    for ckpt_step in available_ckpt_steps:
        if int(ckpt_step) > int(start_step):
            original_ckpt_path = os.path.join(local_checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{ckpt_step}")
            deperacated_ckpt_path = original_ckpt_path + DEPERECATED_SUFFIX
            print(f"moving {original_ckpt_path} to {deperacated_ckpt_path} ...")
            shutil.move(original_ckpt_path, deperacated_ckpt_path)


if __name__ == "__main__":
    assert len(sys.argv) == 3
    local_checkpoint_dir = sys.argv[1]
    start_step = sys.argv[2]
    main(local_checkpoint_dir, start_step)
