# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# import os
# import shutil

# import numpy as np
# import pytest

# from paddleformers.utils.downloader import get_path_from_url_with_filelock
# from tests.parallel_launch import TestMultipleGpus
# from tests.testing_utils import require_paddle_at_least_8_gpu, skip_for_none_ce_case
# from tests.trainer.trainer_utils import get_pretrain_arguments

# environment_variables = {
#     "NCCL_ALGO": "Tree",
#     "NVIDIA_TF32_OVERRIDE": "0",
#     "NCCL_IB_TIMEOUT": "22",
#     "NCCL_DEBUG": "INFO",
#     "FLAGS_embedding_deterministic": "1",
#     "FLAGS_cudnn_deterministic": "1",
#     "Flags_mp_aysnc_allreduce": "1",
#     "Flags_skip_mp_c_identity": "1",
#     "FLAGS_shard_norm_align_dp": "0",
#     "FLAGS_shard_use_reduce": "1",
#     "test_ci_no_save_model": "1",
# }

# lora_arguments = {
#     "model_name_or_path": "__internal_testing__/unified-ckpt-llama-170m-for-peft",
#     "dataset_name_or_path": "./unified_checkpoint/peft_input/data/",
#     "output_dir": "./unified_checkpoint/checkpoints/llama_lora_ckpts",
#     "per_device_train_batch_size": 4,
#     "gradient_accumulation_steps": 8,
#     "per_device_eval_batch_size": 8,
#     "eval_accumulation_steps": 16,
#     "learning_rate": 3e-04,
#     "max_steps": 15,
#     "save_steps": 10,
#     "warmup_steps": 30,
#     "logging_steps": 1,
#     "evaluation_strategy": "no",
#     "save_strategy": "steps",
#     "src_length": 1024,
#     "max_length": 2048,
#     "fp16": "true",
#     "fp16_opt_level": "O2",
#     "do_train": "true",
#     "do_eval": "false",
#     "disable_tqdm": "true",
#     "eval_with_do_generation": "false",
#     "recompute": "true",
#     "save_total_limit": 1,
#     "tensor_model_parallel_size": 1,
#     "pipeline_model_parallel_size": 1,
#     "lora": "true",
#     "zero_padding": "false",
#     "unified_checkpoint": 1,
# }


# # convert from N1C8 to N2C4 or N2C4 to N1C8
# MAX_CONVERT_CONFIGS = 1  # max: 16, min: 1

# seed = 2024

# rng = np.random.default_rng(seed=seed)


# def random_sample(keys, k):
#     return rng.permutation(list(keys))[0:k].tolist()


# def check_acc(log_dir="log"):
#     file_path = os.path.join(log_dir, "workerlog.n0.c0")
#     cmd = "grep -a 'global_step: 15' " + file_path + " | awk -F ','  '{print $2}' | awk  '{print $6}'"
#     import subprocess

#     res = subprocess.check_output(cmd, shell=True, text=True)
#     res = [float(x) for x in res.split()]

#     return res


# def remove_logs(log_dir="log"):
#     if os.path.exists(log_dir):
#         shutil.rmtree(log_dir)


# def remove_ckpt(ckpt_dir):
#     if os.path.exists(ckpt_dir):
#         shutil.rmtree(ckpt_dir)


# @pytest.mark.xdist_group(name="UC")
# class TestUnifiedCheckpointSingle(TestMultipleGpus):
#     def setUp(self):
#         self.config = lora_arguments
#         os.environ.update(environment_variables)

#         file_ = "https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz"
#         input_dir = "unified_checkpoint/peft_input/"
#         os.makedirs(input_dir, exist_ok=True)
#         file_path = os.path.join(input_dir, "AdvertiseGen.tar.gz")
#         if not os.path.exists(file_path):
#             get_path_from_url_with_filelock(file_, root_dir=input_dir)

#         self.need_allclose = True
#         self.rtol = 1e-7
#         self.run_lora_file = "llm/run_finetune.py"
#         self.num_nodes = 1

#     def runfirst(self, train_args):
#         self.run_1gpu(self.run_lora_file, **train_args)

#     def rerun(self, train_args):
#         self.run_1gpu(self.run_lora_file, **train_args)

#     @skip_for_none_ce_case
#     def testDP1(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         self.runfirst(self.config)
#         self.rerun(self.config)
#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)


# # Test Unified Checkpoint Hybrid Parallel Strategy on N1C8 and N2C4
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

#         self.configs = get_pretrain_arguments(lora_arguments)
#         os.environ.update(environment_variables)

#         file_ = "https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz"
#         input_dir = "unified_checkpoint/peft_input/"
#         os.makedirs(input_dir, exist_ok=True)
#         file_path = os.path.join(input_dir, "AdvertiseGen.tar.gz")
#         if not os.path.exists(file_path):
#             get_path_from_url_with_filelock(file_, root_dir=input_dir)

#         self.need_allclose = True
#         self.rtol = 1e-7

#         self.run_lora_file = "llm/run_finetune.py"

#     def runfirst(self, train_args):
#         self.run_n1c8(self.run_lora_file, **train_args)

#     def rerun(self, train_args):
#         self.run_n1c8(self.run_lora_file, **train_args)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testTP4PP2(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["TP4PP2"]
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
#         remove_ckpt(lora_arguments["output_dir"])

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
#     def testTP8(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["TP8"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testTP4DP2(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["TP4DP2"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testTP4Sharding2(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["TP4Sharding2"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testTP2PP4(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["TP2PP4"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testPP8(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["PP8"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testPP4DP2(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["PP4DP2"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testPP4Sharding2(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["PP4Sharding2"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testSharding8S1(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["Sharding8S1"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testSharding8S2(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["Sharding8S2"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testSharding4S1DP2(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["Sharding4S1DP2"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testSharding4S2DP2(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["Sharding4S2DP2"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testSharding2S1DP4(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["Sharding2S1DP4"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testSharding2S2DP4(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["Sharding2S2DP4"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)

#     @skip_for_none_ce_case
#     @require_paddle_at_least_8_gpu
#     def testDP8(self):
#         remove_logs()
#         remove_ckpt(lora_arguments["output_dir"])

#         train_args = self.configs["DP8"]
#         self.runfirst(train_args)
#         self.rerun(train_args)

#         if self.need_allclose:
#             res = check_acc()
#             assert len(res) == 2
#             np.testing.assert_allclose(res[0], res[1], self.rtol)


# @pytest.mark.skipif(True, reason="Skip for None CE")
# class TestUnifiedCheckpointOnN2C4(TestUnifiedCheckpointBase):
#     def setUp(self):
#         super().setUp()
#         self.need_allclose = True
#         self.rtol = 1e-7

#     def runfirst(self, train_args):
#         self.run_n2c4(self.run_lora_file, **train_args)

#     def rerun(self, train_args):
#         self.run_n2c4(self.run_lora_file, **train_args)


# @pytest.mark.skipif(True, reason="Skip for None CE")
# class TestUnifiedCheckpointOnN1C8CheckpointCompatible(TestUnifiedCheckpointBase):
#     def setUp(self):
#         super().setUp()

#         self.need_allclose = True
#         self.rtol = 1e-7

#     def runfirst(self, train_args):
#         train_args["unified_checkpoint"] = 0
#         self.run_n1c8(self.run_lora_file, **train_args)

#     def rerun(self, train_args):
#         train_args["unified_checkpoint"] = 1
#         self.run_n1c8(self.run_lora_file, **train_args)


# @pytest.mark.skipif(True, reason="Skip for None CE")
# class TestPaddleCheckpointOnN1C8Reset(TestUnifiedCheckpointBase):
#     def setUp(self):
#         super().setUp()

#         self.need_allclose = True
#         self.rtol = 1e-7

#     def runfirst(self, train_args):
#         train_args["unified_checkpoint"] = 0
#         self.run_n1c8(self.run_lora_file, **train_args)

#     def rerun(self, train_args):
#         train_args["unified_checkpoint"] = 0
#         self.run_n1c8(self.run_lora_file, **train_args)


# @pytest.mark.skipif(True, reason="Skip for None CE")
# class TestUnifiedCheckpointOnN2C4CheckpointCompatible(TestUnifiedCheckpointBase):
#     def setUp(self):
#         super().setUp()

#         self.need_allclose = True
#         self.rtol = 1e-7

#     def runfirst(self, train_args):
#         train_args["unified_checkpoint"] = 0
#         self.run_n2c4(self.run_lora_file, **train_args)

#     def rerun(self, train_args):
#         train_args["unified_checkpoint"] = 1
#         self.run_n2c4(self.run_lora_file, **train_args)
