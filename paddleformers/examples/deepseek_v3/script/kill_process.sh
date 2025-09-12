#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

set -x
skip_kill_time=${1:-"False"}
function kill_impl() {
    skip_kill_time=$1
    # kill aadiff test finally.
    pids=`ps -ef | grep pretrain.py | grep -v grep | awk '{print $2}'`
    if [[ "$pids" != "" ]] ; then
        echo $pids
        echo $pids | xargs kill -9
    fi

    echo "Killing processes on gpu"
    lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill -9 {}
}

kill_impl $skip_kill_time || true