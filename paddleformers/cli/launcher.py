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

import os
import shutil
import sys

BIND_TRAINER_NUMA_EXECED = "BIND_TRAINER_NUMA_EXECED"


def _reexec_with_numactl(local_rank, numa_node):
    if os.getenv(BIND_TRAINER_NUMA_EXECED) == "1":
        return

    numactl = shutil.which("numactl")
    if not numactl:
        raise RuntimeError("BIND_TRAINER_NUMA=1 requires numactl on PATH")

    env = os.environ.copy()
    env[BIND_TRAINER_NUMA_EXECED] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    args = [
        numactl,
        f"--cpunodebind={numa_node}",
        f"--membind={numa_node}",
        sys.executable,
        "-u",
        *sys.argv,
    ]
    print(
        f"[PaddleFormers] Re-exec trainer with numactl: local_rank={local_rank}, numa_node={numa_node}",
        flush=True,
    )
    os.execvpe(numactl, args, env)


def _maybe_bind_trainer_numa():
    if os.getenv("BIND_TRAINER_NUMA", "0").lower() not in ("1", "true", "on", "yes"):
        return

    rank_env = os.getenv("PADDLE_LOCAL_RANK") or os.getenv("FLAGS_selected_gpus")
    if not rank_env:
        raise RuntimeError("BIND_TRAINER_NUMA=1 requires PADDLE_LOCAL_RANK or FLAGS_selected_gpus")

    local_rank = int(rank_env.split(",", 1)[0])

    if local_rank in (0, 1):
        numa_node = 0
    elif local_rank in (2, 3):
        numa_node = 1
    else:
        raise RuntimeError(f"BIND_TRAINER_NUMA=1 only supports local rank 0-3, got {local_rank}")

    _reexec_with_numactl(local_rank, numa_node)
    print(
        f"[PaddleFormers] Trainer NUMA affinity enabled: local_rank={local_rank}, "
        f"numa_node={numa_node}, policy=numactl_cpunodebind_membind",
        flush=True,
    )


def launch():
    """
    Distributed launch
    """

    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        raise ValueError("len(sys.argv) mush be larger than 1")

    if command == "train":
        _maybe_bind_trainer_numa()
        from paddleformers.cli.train.tuner import run_tuner

        run_tuner()
    elif command == "export":
        from paddleformers.cli.export.export import run_export

        run_export()
    else:
        raise ValueError(f"Unknown command : {command}")


if __name__ == "__main__":
    launch()
