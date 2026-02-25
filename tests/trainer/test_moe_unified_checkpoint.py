# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

environment_variables = {
    "NCCL_ALGO": "Tree",
    "NVIDIA_TF32_OVERRIDE": "0",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_DEBUG": "INFO",
    "FLAGS_embedding_deterministic": "1",
    "FLAGS_cudnn_deterministic": "1",
    "Flags_mp_aysnc_allreduce": "1",
    "Flags_skip_mp_c_identity": "1",
    "FLAGS_shard_norm_align_dp": "0",
    "FLAGS_shard_use_reduce": "1",
    "test_ci_no_save_model": "1",
}

moe_arguments = {
    "model_name_or_path": "__internal_testing__/unified-ckpt-qwen2moe",
    "dataset_name_or_path": "./unified_checkpoint/peft_input/data/",
    "output_dir": "./unified_checkpoint/checkpoints/qwen2moe_sft_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "per_device_eval_batch_size": 8,
    "eval_accumulation_steps": 16,
    "learning_rate": 3e-04,
    "max_steps": 10,
    "save_steps": 6,
    "warmup_steps": 30,
    "logging_steps": 1,
    "evaluation_strategy": "no",
    "save_strategy": "steps",
    "src_length": 1024,
    "max_length": 2048,
    "bf16": "true",
    "fp16_opt_level": "O2",
    "do_train": "true",
    "do_eval": "false",
    "disable_tqdm": "true",
    "eval_with_do_generation": "false",
    "recompute": "true",
    "recompute_granularity": "full",
    "save_total_limit": 1,
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "sharding": "",
    "lora": "false",
    "unified_checkpoint": 1,
    "continue_training": 0,
    "sequence_parallel": 0,
}


def check_acc(log_dir="log"):
    file_path = os.path.join(log_dir, "workerlog.n0.c0")
    cmd = "grep -a 'global_step: 10' " + file_path + " | awk -F ','  '{print $2}' | awk  '{print $6}'"
    import subprocess

    res = subprocess.check_output(cmd, shell=True, text=True)
    res = [float(x) for x in res.split()]

    return res


seed = 2024

rng = np.random.default_rng(seed=seed)


# @pytest.mark.xdist_group(name="UC")
# class TestUnifiedCheckpointBase(TestMultipleGpus):
#     @classmethod
#     @property
#     def __test__(cls):
#         return cls != TestUnifiedCheckpointBase

#     def setUp(self):
#         """
#         1. update runfirst and rerun to run defined different config
#         2. update need_allclose to True if you want to check the result
#         3. update rtol to the relative value you want to check
#         """

#         self.configs = get_pretrain_arguments(moe_arguments)
#         os.environ.update(environment_variables)

#         file_ = "https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz"
#         input_dir = "unified_checkpoint/peft_input/"
#         os.makedirs(input_dir, exist_ok=True)
#         file_path = os.path.join(input_dir, "AdvertiseGen.tar.gz")
#         if not os.path.exists(file_path):
#             get_path_from_url_with_filelock(file_, root_dir=input_dir)

#         self.need_allclose = True
#         self.rtol = 1e-7

#         self.run_file = "llm/run_finetune.py"

#     def runfirst(self, train_args):
#         self.run_n1c8(self.run_file, **train_args)

#     def rerun(self, train_args):
#         self.run_n1c8(self.run_file, **train_args)

#     @require_paddle_at_least_8_gpu
#     def testTP4DP2(self):
#         remove_logs()
#         remove_ckpt(moe_arguments["output_dir"])

#         train_args = self.configs["TP4DP2"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testTP2Sharding4(self):
#         remove_logs()
#         remove_ckpt(moe_arguments["output_dir"])

#         train_args = self.configs["TP2Sharding4"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)


# @pytest.mark.xdist_group(name="UC")
# class TestUnifiedCheckpointFull(TestUnifiedCheckpointBase):
#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testTP2Sharding4V2(self):
#         remove_logs()
#         remove_ckpt(moe_arguments["output_dir"])

#         train_args = self.configs["TP2Sharding4"]
#         train_args.update({"split_param": True})
#         train_args.update({"amp_master_grad": True})
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)
