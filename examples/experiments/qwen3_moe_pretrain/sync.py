# -*- coding: utf-8 -*-
# !/usr/bin/env python3

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

"""
@version: 1.0
@file: sync.py
@time: 2023/07/03 17:22:19
@Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved



"""
import distutils.util as util
import multiprocessing
import os
import sys
from shlex import quote


def get_self_ip():
    """
    get_self_ip
    """
    # ip = subprocess.check_output(['hostname', '-i']).decode().strip()
    ip = os.getenv("POD_IP")
    return ip


def get_ips():
    """
    get_ips
    """
    path = os.path.join(os.path.dirname(__file__), "iplist.txt")
    if not os.path.exists(path):
        path = "/root/paddlejob/workspace/hostfile"
        assert os.path.exists(path), "iplist file not exists!"

    with open(path, "r") as f:
        return [ip.strip().split(" ")[0] for ip in f.readlines() if ip.strip() and not ip.strip().startswith("#")]


def concat_cmd(cmd):
    """
    concat_cmd
    """
    return " ".join([quote(c) for c in cmd])


def sync_file(path):
    """
    sync_file
    """
    if isinstance(path, (list, tuple)):
        for p in path:
            sync_file(p)
        return

    path = os.path.abspath(path)
    # dirname = os.path.dirname(path)

    self_ip = get_self_ip()
    ips = get_ips()
    all_cmds = []
    for ip in ips:
        if ip != self_ip:
            # cmd = "mkdir -p {0}; scp -r {2} {1}:{2}".format(dirname, ip, path)
            # ssh_cmd = concat_cmd(['ssh', ip, cmd])
            ssh_cmd = "scp -r {1} {0}:{1}".format(ip, path)
            all_cmds.append(ssh_cmd)
    if len(all_cmds) > 0:
        with multiprocessing.Pool(min(len(all_cmds), 20)) as pool:
            pool.map(execute_cmd, all_cmds)


def execute_cmd(cmd):
    """
    execute_cmd
    """
    print(cmd)
    # assert os.system(cmd) == 0
    return os.system(cmd)


def sync_cmd(cmd):
    """
    sync_cmd
    """
    self_ip = get_self_ip()
    ips = get_ips()
    joined_cmd = "cd {0}; {1}".format(os.path.abspath("."), concat_cmd(cmd))

    all_cmds = []
    for ip in ips:
        is_self = ip == self_ip
        if not is_self:
            exe_cmd = concat_cmd(["ssh", ip, joined_cmd])
        else:
            exe_cmd = joined_cmd

        if is_self and not util.strtobool(os.getenv("HAS_SELF", "1")):
            continue

        all_cmds.append(exe_cmd)

    with multiprocessing.Pool(len(all_cmds)) as pool:
        pool.map(execute_cmd, all_cmds)


if __name__ == "__main__":
    mode = sys.argv[1]
    assert mode in ["file", "cmd"]
    if mode == "file":
        sync_file(sys.argv[2:])
    elif mode == "cmd":
        sync_cmd(sys.argv[2:])
    else:
        raise ValueError("Invalid mode: {}".format(mode))
