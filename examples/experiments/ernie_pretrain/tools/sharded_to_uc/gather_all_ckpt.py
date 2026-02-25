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

import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--org_path",
        type=str,
        required=True,
        help="The path of checkpoint to be gathered.",
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        required=True,
        help="The output path to gather all checkpoints.",
    )
    parser.add_argument("--hostfile_path", type=str, required=True, help="The file for hostfile.")
    args = parser.parse_args()
    return args


def parse_path(args):
    pwd = subprocess.run(["pwd"], capture_output=True, text=True).stdout.strip()
    org_path = args.org_path
    tgt_path = args.tgt_path
    assert not org_path.startswith("/"), "Please path relative path instead of absolute path to org_path."
    assert not tgt_path.startswith("/"), "Please path relative path instead of absolute path to tgt_path."
    assert not os.path.exists(tgt_path), f"{tgt_path} exist, please check"
    os.makedirs(f"{tgt_path}")
    return f"{pwd}/{org_path}", f"{pwd}/{tgt_path}"


def get_ip_list(args):
    hostfile_path = args.hostfile_path
    assert os.path.exists(hostfile_path), f"{hostfile_path} not exist, please check"
    hostnames = subprocess.run(["cat", hostfile_path], capture_output=True, text=True).stdout.split("\n")
    for i in range(len(hostnames)):
        hostname = hostnames[i].split(" ")[0]
        hostnames[i] = hostname
    if hostnames[-1] == "":
        hostnames.pop(-1)
    local_host = subprocess.run(["hostname", "-i"], capture_output=True, text=True).stdout.split("\n")[0]
    return hostnames, local_host


def gather_ckpt(org_path, tgt_path, hostnames, local_host):
    for i, hostname in enumerate(hostnames):
        # moving ckpt
        print(f"ssh moving {hostname}:{org_path} to {hostname}:{tgt_path}_{i}")
        subprocess.run(["ssh", hostname, f"rm -rf {tgt_path}_{i}"], capture_output=True, text=True)
        rst = subprocess.run(
            ["ssh", hostname, f"cp -r {org_path} {tgt_path}_{i}"],
            capture_output=True,
            text=True,
        )
        assert rst.stderr == "", f"error happened when moving ckpt at {hostname} with stderr {rst.stderr}"

        # non-local, should scp from remote
        if hostname != local_host:
            # compressing
            print(f"ssh compressing {hostname}:{tgt_path}_{i} to {hostname}:{tgt_path}_{i}.tar")
            rst = subprocess.run(
                ["ssh", hostname, f"tar -cPf {tgt_path}_{i}.tar {tgt_path}_{i}"],
                capture_output=True,
                text=True,
            )
            assert rst.stderr == "", f"error happend when compressing ckpt at {hostname} with stderr {rst.stderr}"

            # scp
            print(f"scp {hostname}:{tgt_path}_{i} to local")
            subprocess.run(["rm", "-rf", f"{tgt_path}_{i}.tar"], capture_output=True, text=True)
            rst = subprocess.run(
                ["scp", f"{hostname}:{tgt_path}_{i}.tar", "."],
                capture_output=True,
                text=True,
            )
            assert rst.stderr == "", f"error happend when scp ckpt for {hostname} with stderr {rst.stderr}"

            # clear remote
            subprocess.run(
                ["ssh", hostname, f"rm -rf {tgt_path}_{i}"],
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["ssh", hostname, f"rm -rf {tgt_path}_{i}.tar"],
                capture_output=True,
                text=True,
            )

            # decompressing
            print(f"decompressing {tgt_path}_{i}.tar")
            rst = subprocess.run(["tar", "-xPf", f"{tgt_path}_{i}.tar"], capture_output=True, text=True)
            assert rst.stderr == "", f"error happend when decompressing {tgt_path}_{i}.tar with stderr {rst.stderr}"

            # clear local tar
            subprocess.run(["rm", "-rf", f"{tgt_path}_{i}.tar"], capture_output=True, text=True)

        # remove useless file
        subprocess.run(
            f"rm -rf {tgt_path}_{i}/saved_signal_*",
            capture_output=True,
            text=True,
            shell=True,
        )
        if i != 0:
            subprocess.run(
                f"rm -rf {tgt_path}_{i}/scheduler.pdparams",
                capture_output=True,
                text=True,
                shell=True,
            )

        # moving ckpt
        print(f"moving ckpt from {tgt_path}_{i} to {tgt_path}")
        subprocess.run(
            f"mv {tgt_path}_{i}/* {tgt_path}",
            capture_output=True,
            text=True,
            shell=True,
        )

        # clear local path
        subprocess.run(["rm", "-rf", f"{tgt_path}_{i}"], capture_output=True, text=True)


if __name__ == "__main__":
    args = parse_args()
    org_path, tgt_path = parse_path(args)
    print(f"gather ckpt from {org_path} to {tgt_path}")
    hostnames, local_host = get_ip_list(args)
    print(f"gather ips list {hostnames}, and local_host is {local_host}")
    gather_ckpt(org_path, tgt_path, hostnames, local_host)
    print("done gathered all ckpt")
