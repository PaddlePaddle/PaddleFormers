# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


import copy
import logging
import os
import subprocess
import sys
import time
import unittest

import paddle
from paddle.distributed.utils.launch_utils import (
    TrainerProc,
    find_free_ports,
    get_cluster,
    terminate_local_procs,
    watch_local_trainers,
)

from paddleformers.utils.downloader import get_path_from_url_with_filelock

logger = logging.getLogger("root")


def get_cluster_from_args(selected_gpus, num_nodes=1):
    cluster_node_ips = "127.0.0.1"
    node_ip = "127.0.0.1"

    node_ips = [x.strip() for x in cluster_node_ips.split(",")]

    node_ips.index(node_ip)

    free_ports = None

    free_ports = find_free_ports(len(selected_gpus))
    if free_ports is not None:
        free_ports = list(free_ports)

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])

    return get_cluster(node_ips, node_ip, trainer_endpoints, selected_gpus)


def get_gpus(selected_gpus):
    selected_gpus = [x.strip() for x in selected_gpus.split(",")]
    return selected_gpus


def start_local_trainers_cpu(trainer_endpoints, training_script, training_script_args, log_dir=None):
    current_env = copy.copy(os.environ.copy())
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    n_rank = len(trainer_endpoints)
    print(trainer_endpoints)
    for rank_id, endpoint in enumerate(trainer_endpoints):
        proc_env = {
            "PADDLE_DISTRI_BACKEND": "gloo",
            "PADDLE_TRAINER_ID": "%d" % rank_id,
            "PADDLE_CURRENT_ENDPOINT": "%s" % endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % n_rank,
            "PADDLE_TRAINER_ENDPOINTS": ",".join(trainer_endpoints),
        }

        current_env.update(proc_env)

        print("trainer proc env:{}".format(current_env))

        assert os.getenv("WITH_COVERAGE", "OFF") == "OFF", "Gloo don't support WITH_COVERAGE."
        cmd = "python -u " + training_script

        print("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None

        proc = subprocess.Popen(cmd.split(" "), env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = rank_id
        tp.log_fn = fn
        tp.cmd = cmd

        procs.append(tp)

    return procs


def start_local_trainers(
    cluster, pod, training_script, training_script_args, log_dir=None, num_nodes=1, hack_output_dir=True
):
    current_env = copy.copy(os.environ.copy())
    # paddle broadcast ncclUniqueId use socket, and
    # proxy maybe make trainers unreachable, so delete them.
    # if we set them to "", grpc will log error message "bad uri"
    # so just delete them.
    # current_env.pop("http_proxy", None)
    # current_env.pop("https_proxy", None)

    procs = []
    for idx, t in enumerate(pod.trainers):
        local_rank = idx % (len(pod.trainers) // num_nodes)
        node_rank = idx // (len(pod.trainers) // num_nodes)
        proc_env = {
            "FLAGS_selected_gpus": "%s" % ",".join([str(g) for g in t.gpus]),
            "PADDLE_GLOBAL_SIZE": f"{len(pod.trainers)}",
            "PADDLE_LOCAL_SIZE": f"{len(pod.trainers)//num_nodes}",
            "PADDLE_GLOBAL_RANK": f"{idx}",
            "PADDLE_LOCAL_RANK": f"{local_rank}",
            "PADDLE_NNODES": f"{num_nodes}",
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints()),
            # compatible env
            "PADDLE_CURRENT_ENDPOINT": "%s" % t.endpoint,
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_RANK_IN_NODE": f"{local_rank}",
        }

        current_env.update(proc_env)

        logger.debug(f"trainer proc env:{current_env}")

        if hack_output_dir and num_nodes > 1:
            dir_idx = training_script_args.index("--output_dir") + 1
            script_args = copy.deepcopy(training_script_args)
            script_args[dir_idx] = f"{script_args[dir_idx]}/node_{node_rank}"
        else:
            script_args = copy.deepcopy(training_script_args)

        cmd = [sys.executable, "-u", training_script] + script_args

        logger.info(f"start trainer proc:{cmd} env:{proc_env}")

        fn = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            fn = open("%s/workerlog.n%d.c%d" % (log_dir, node_rank, local_rank), "a")
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.local_rank = idx
        tp.log_fn = fn
        tp.log_offset = fn.tell() if fn else None
        tp.cmd = cmd

        procs.append(tp)

    return procs


class TestMultipleGpus(unittest.TestCase):
    def setUp(self):
        self.selected_gpus = get_gpus("0,1")
        self.num_nodes = 1

    def run_1gpu(self, *args, **kwargs):
        self.selected_gpus = get_gpus("0")
        self.run_n_gpu(*args, **kwargs)

    def run_2gpu(self, *args, **kwargs):
        self.selected_gpus = get_gpus("0,1")
        self.run_n_gpu(*args, **kwargs)

    def run_4gpu(self, *args, **kwargs):
        self.selected_gpus = get_gpus("0,1,2,3")
        self.run_n_gpu(*args, **kwargs)

    def run_8gpu(self, *args, **kwargs):
        self.selected_gpus = get_gpus("0,1,2,3,4,5,6,7")
        self.run_n_gpu(*args, **kwargs)

    def run_n1c2(self, *args, **kwargs):
        self.selected_gpus = get_gpus("0,1")
        self.num_nodes = 1
        self.run_n_gpu(*args, **kwargs)

    def run_n1c8(self, *args, **kwargs):
        self.selected_gpus = get_gpus("0,1,2,3,4,5,6,7")
        self.num_nodes = 1
        self.run_n_gpu(*args, **kwargs)

    def run_n2c4(self, *args, **kwargs):
        self.selected_gpus = get_gpus("0,1,2,3,4,5,6,7")
        self.num_nodes = 2
        self.run_n_gpu(*args, **kwargs)

    def run_n4c2(self, *args, **kwargs):
        self.num_nodes = 4
        self.selected_gpus = get_gpus("0,1,2,3,4,5,6,7")
        self.run_n_gpu(*args, **kwargs)

    def run_n8c1(self, *args, **kwargs):
        self.num_nodes = 8
        self.selected_gpus = get_gpus("0,1,2,3,4,5,6,7")
        self.run_n_gpu(*args, **kwargs)

    def run_n_gpu(
        self,
        target_file_name,
        log_dir="./log",
        **kwargs,
    ):
        if not paddle.framework.core.is_compiled_with_cuda() or paddle.framework.core.get_cuda_device_count() == 0:
            return

        # selected_gpus = get_gpus("0,1")
        cluster = None
        pod = None

        cluster, pod = get_cluster_from_args(self.selected_gpus)
        script_args = []
        for k, v in kwargs.items():
            script_args.append("--" + str(k))
            script_args.append(str(v))

        procs = start_local_trainers(
            cluster,
            pod,
            # allocator_strategy=allocator_strategy,
            log_dir=log_dir,
            training_script=target_file_name,
            training_script_args=script_args,
            num_nodes=self.num_nodes,
        )

        try:
            while True:
                alive = watch_local_trainers(procs, cluster.trainers_endpoints())

                if not alive:
                    print("Local procs complete, POD info:{}".format(pod))
                    break
                time.sleep(0.5)
        finally:
            terminate_local_procs(procs)

    def prepare_inputs_data(self, input_dir, files):
        os.makedirs(input_dir, exist_ok=True)
        for file in files:
            file_name = file.split("/")[-1]
            file_path = os.path.join(input_dir, file_name)
            if not os.path.exists(file_path):
                get_path_from_url_with_filelock(file, root_dir=input_dir)


class TestMultipleWithGloo(unittest.TestCase):
    def run_2cpu(self, target_file_name):

        cluster, pod = get_cluster_from_args([0, 1])  # tmp use. for getting trainer_nranks()

        procs = start_local_trainers_cpu(
            cluster.trainers_endpoints(),
            training_script=target_file_name,
            training_script_args=[],
        )

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_nranks())

            if not alive:
                print("Local procs complete, POD info:{}".format(pod))
                break
            time.sleep(3)
