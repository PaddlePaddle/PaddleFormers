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
"""cli
"""
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

import paddle

from .utils.process import detect_device, set_ascend_environment, terminate_process_tree

script_dir = Path(__file__).parent.resolve()
parent_dir = script_dir.parent.parent

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

os.environ["PYTHONPATH"] = f"{parent_dir!s}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"


USAGE = (
    "-" * 60
    + "\n"
    + "| Usage:                                                              |\n"
    + "|   paddleformers-cli train -h: model finetuning                      |\n"
    + "|   paddleformers-cli export -h: model export                         |\n"
    + "|   paddleformers-cli version: show version info                      |\n"
    + "|   paddleformers-cli help: show helping info                         |\n"
    + "-" * 60
)


WELCOME = "-" * 60 + "\n" + "Welcome to PaddleFormers Cli" + "\n" + "-" * 60


def main():
    """cli main process"""
    from . import launcher
    from .export.export import run_export
    from .train.tuner import run_tuner

    COMMAND_MAP = {
        "train": run_tuner,
        "export": run_export,
        "version": partial(print, WELCOME),
        "help": partial(print, USAGE),
    }

    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    distributed_funcs = ["train", "export"]
    paddleformers_dist_log = os.getenv("PADDLEFORMERS_DIST_LOG", "paddleformers_dist_log")
    nnodes = os.getenv("NNODES", "1")
    master_ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "8080")
    current_device = detect_device()
    if current_device == "xpu":
        num_xpus = paddle.device.xpu.device_count()
        default_xpus = ",".join(map(str, range(0, num_xpus)))
        visible_cards = os.getenv("XPU_VISIBLE_DEVICES", default_xpus)
    elif current_device == "npu":
        num_npus = len(paddle.device.get_available_custom_device())
        default_npus = ",".join(map(str, range(0, num_npus)))
        visible_cards = os.getenv("ASCEND_RT_VISIBLE_DEVICES", default_npus)
    elif current_device == "iluvatar_gpu":
        num_iluvatar_gpus = len(paddle.device.get_available_custom_device())
        default_iluvatar_gpus = ",".join(map(str, range(0, num_iluvatar_gpus)))
        visible_cards = os.getenv("CUDA_VISIBLE_DEVICES", default_iluvatar_gpus)
    else:
        import GPUtil

        num_gpus = len(GPUtil.getGPUs())
        # Create a default GPU list string (e.g., "0,1,2" for 3 GPUs)
        default_gpus = ",".join(map(str, range(0, num_gpus)))
        # Get the CUDA_VISIBLE_DEVICES environment variable value,
        # use the default GPU list if the environment variable is not set
        visible_cards = os.getenv("CUDA_VISIBLE_DEVICES", default_gpus)

    for key in [
        "PADDLE_TRAINERS_NUM",
        "PADDLE_TRAINER_ID",
        "PADDLE_WORKERS_IP_PORT_LIST",
        "PADDLE_TRAINERS",
        "PADDLE_NUM_GRADIENT_SERVERS",
        "PADDLE_ELASTIC_JOB_ID",
        "PADDLE_TRAINER_ENDPOINTS",
        "DISTRIBUTED_TRAINER_ENDPOINTS",
        "FLAGS_START_PORT",
        "PADDLE_ELASTIC_TIMEOUT",
    ]:
        if key in os.environ:
            del os.environ[key]

    os.environ["FLAGS_set_to_1d"] = "False"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["FLAGS_dataloader_use_file_descriptor"] = "False"

    if current_device == "xpu":
        os.environ["FLAGS_use_stride_kernel"] = "0"
        os.environ["XPU_PADDLE_L3_SIZE"] = "0"
        os.environ["XPUAPI_DEFAULT_SIZE"] = "2205258752"

        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"

        os.environ["BKCL_TREE_THRESHOLD"] = "0"
        os.environ["BKCL_ENABLE_XDR"] = "1"
        os.environ["BKCL_RDMA_FORCE_TREE"] = "1"
        os.environ["BKCL_RDMA_NICS"] = "eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"
        os.environ["BKCL_SOCKET_IFNAME"] = "eth0"
        os.environ["BKCL_FORCE_L3_RDMA"] = "0"
        os.environ["BKCL_USE_AR"] = "1"
        os.environ["BKCL_RING_OPT"] = "1"
        os.environ["BKCL_RING_HOSTID_USE_RANK"] = "1"

        os.environ["XPU_PADDLE_FC_LOCAL_INT16"] = "1"
        os.environ["XPU_AUTO_BF16_TF32_RADIO"] = "10"
        os.environ["XPU_AUTO_BF16_TF32"] = "1"
    elif current_device == "npu":
        os.environ["FLAGS_allocator_strategy_kernel"] = "auto_growth"
        os.environ["FLAGS_npu_jit_compile"] = "0"
        try:
            set_ascend_environment()
        except Exception as e:
            print("Unexpected error setting Ascend environment: %s", e)
    elif current_device == "iluvatar_gpu":
        os.environ["PADDLE_XCCL_BACKEND"] = "iluvatar_gpu"
        os.environ["LD_PRELOAD"] = "/usr/local/corex/lib64/libcuda.so.1"
        os.environ["FLAGS_embedding_deterministic"] = "1"

    if command in distributed_funcs:

        # if os.path.exists(paddleformers_dist_log):
        #     try:
        #         shutil.rmtree(paddleformers_dist_log)
        #         print(f"Succeed to delete {paddleformers_dist_log}.")
        #     except Exception as e:
        #         print(f"Error occurs while deleting {paddleformers_dist_log}: {e}")

        # launch distributed training
        env = deepcopy(os.environ)
        args_to_pass = " ".join(shlex.quote(arg) for arg in sys.argv[1:])
        if current_device == "iluvatar_gpu":
            current_device = "gpu"
        command = (
            f"python -m paddle.distributed.launch --log_dir {paddleformers_dist_log} "
            f"--{current_device}s {visible_cards} --master {master_ip}:{master_port} "
            f"--nnodes {nnodes} {launcher.__file__} {args_to_pass}"
        )
        command = shlex.split(command)
        process = subprocess.Popen(
            command,
            env=env,
        )

        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nReceived interrupt, terminating server...")
            terminate_process_tree(process.pid)
            sys.exit(1)
        except Exception as e:
            print(f"Server process failed: {e}")
            terminate_process_tree(process.pid)
            sys.exit(1)
        finally:
            sys.exit(process.returncode)

    elif command in COMMAND_MAP:
        COMMAND_MAP[command]()
    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
