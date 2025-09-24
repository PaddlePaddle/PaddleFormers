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

import os
import shutil

import numpy as np
import pytest

from paddleformers.trainer.unified_checkpoint.utils import UnifiedCheckpointOption
from tests.parallel_launch import TestMultipleGpus
from tests.testing_utils import (
    require_paddle_at_least_2_gpu,
    require_paddle_at_least_8_gpu,
    skip_for_none_ce_case,
)
from tests.trainer.trainer_utils import get_pretrain_arguments

# export NVIDIA_TF32_OVERRIDE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=xgbe0
# export NCCL_IB_TIMEOUT=22
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_IB_GDR_LEVEL=4
# export NCCL_SOCKET_IFNAME=eth2


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
    "FLAGS_eager_communication_connection": "1",  # no lazy init comm group
    "test_ci_no_save_model": "1",
}

pretrain_arguments = {
    "model_name_or_path": "./tests/trainer/unified-ckpt-llama-170m",
    "tokenizer_name_or_path": "facebook/llama-7b",
    "input_dir": "./unified_checkpoint/data/llama",
    "output_dir": "./unified_checkpoint/checkpoints/llama_pretrain_ckpts",
    "split": "1,0,0",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "per_device_eval_batch_size": 8,
    "tensor_parallel_degree": 2,
    "pipeline_parallel_degree": 4,
    "sharding": "",
    "virtual_pp_degree": 1,
    "sequence_parallel": 0,
    "use_flash_attention": "false",
    "use_fused_rms_norm": "false",
    "max_seq_len": 1024,
    "learning_rate": 3e-04,
    "min_learning_rate": 1e-05,
    "warmup_steps": 100,
    "logging_steps": 1,
    "max_steps": 15,
    "save_steps": 10,
    "eval_steps": 1000,
    "weight_decay": 0.01,
    "fp16": "true",
    "fp16_opt_level": "O2",
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 0,
    "continue_training": 0,
    "do_train": "true",
    "do_eval": "false",
    "do_predict": "false",
    "disable_tqdm": "true",
    "recompute": 0,
    "unified_checkpoint": 1,
    "distributed_dataloader": 0,
    "recompute_granularity": "full",
    "save_total_limit": 2,
}

# GBS: 16 MAX_steps: 30

# convert from N1C8 to N2C4 or N2C4 to N1C8
MAX_CONVERT_CONFIGS = 1  # max: 16, min: 1

seed = 2024

rng = np.random.default_rng(seed=seed)


def random_sample(keys, k):
    return rng.permutation(list(keys))[0:k].tolist()


def check_acc(log_dir="log"):
    file_path = os.path.join(log_dir, "workerlog.n0.c0")
    cmd = "grep -a 'global_step: 15' " + file_path + " | awk -F ','  '{print $2}' | awk  '{print $6}'"
    import subprocess

    res = subprocess.check_output(cmd, shell=True, text=True)
    res = [float(x) for x in res.split()]

    return res


def remove_logs(log_dir="log"):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)


def remove_ckpt(ckpt_dir):
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)


def move_checkpoint_N1C8_to_N2C4():
    save_steps = pretrain_arguments["save_steps"]
    mode = rng.choice([1, 2, 3])
    base_ckpt_path = os.path.join(pretrain_arguments["output_dir"], "checkpoint-%d" % save_steps)
    node0_ckpt_path = os.path.join(pretrain_arguments["output_dir"], "node_0", "checkpoint-%d" % save_steps)
    node1_ckpt_path = os.path.join(pretrain_arguments["output_dir"], "node_1", "checkpoint-%d" % save_steps)
    os.system("mkdir -p %s" % node0_ckpt_path)
    os.system("mkdir -p %s" % node1_ckpt_path)

    # 1. only machine-0 holds the checkpoint.
    # 2. only machin-1 holds the checkpoint.
    # 3. randomly split one-machine checkpoint into two machines.
    if mode == 1:
        os.system("mv %s/* %s" % (base_ckpt_path, node0_ckpt_path))
    elif mode == 2:
        os.system("mv %s/* %s" % (base_ckpt_path, node1_ckpt_path))
    else:
        # randomly split checkpoint.
        os.system("mv %s/* %s" % (base_ckpt_path, node0_ckpt_path))
        for filename in os.listdir(node0_ckpt_path):
            move_flag = rng.integers(0, 2)
            file_path = os.path.join(node0_ckpt_path, filename)
            if move_flag:
                os.system("mv %s %s" % (file_path, node1_ckpt_path))


def move_checkpoint_N2C4_to_N1C8():
    save_steps = pretrain_arguments["save_steps"]
    base_ckpt_path = os.path.join(pretrain_arguments["output_dir"], "checkpoint-%d" % save_steps)
    node0_ckpt_path = os.path.join(pretrain_arguments["output_dir"], "node_0", "checkpoint-%d" % save_steps)
    os.system("mv %s %s" % (node0_ckpt_path, os.path.join(pretrain_arguments["output_dir"])))

    node1_ckpt_path = os.path.join(pretrain_arguments["output_dir"], "node_1", "checkpoint-%d" % save_steps)
    if os.path.exists(node1_ckpt_path):
        # Force coverage
        os.system("mv -f %s/* %s" % (node1_ckpt_path, base_ckpt_path))


# https://pytest-xdist.readthedocs.io/en/latest/distribution.html
# Test Unified Checkpoint Hybrid Parallel Strategy on N1C8 and N2C4
@pytest.mark.xdist_group(name="UC")
class TestUnifiedCheckpointBase(TestMultipleGpus):
    @classmethod
    @property
    def __test__(cls):
        return cls != TestUnifiedCheckpointBase

    def setUp(self):
        """
        1. update runfirst and rerun to run defined different config
        2. update need_allclose to True if you want to check the result
        3. update rtol to the relative value you want to check
        """

        self.configs = get_pretrain_arguments(pretrain_arguments)
        os.environ.update(environment_variables)

        files = [
            "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin",
            "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx",
        ]
        self.prepare_inputs_data(pretrain_arguments["input_dir"], files)

        self.need_allclose = True
        self.rtol = 1e-7

        self.run_pretrain_file = "llm/run_pretrain.py"

    def runfirst(self, train_args):
        self.run_n1c8(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        self.run_n1c8(self.run_pretrain_file, **train_args)

    @skip_for_none_ce_case
    @require_paddle_at_least_8_gpu
    def testTP4PP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP4PP2"]
        self.runfirst(train_args)
        self.rerun(train_args)

        if self.need_allclose:
            res = check_acc()
            assert len(res) == 2
            np.testing.assert_allclose(res[0], res[1], self.rtol)

    @skip_for_none_ce_case
    @require_paddle_at_least_8_gpu
    def testTP2Sharding4(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP2Sharding4"]
        self.runfirst(train_args)
        self.rerun(train_args)

        if self.need_allclose:
            res = check_acc()
            assert len(res) == 2
            np.testing.assert_allclose(res[0], res[1], self.rtol)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN2C4(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        self.need_allclose = True
        self.rtol = 1e-7

    def runfirst(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)


# Test Unified Checkpoint Hybrid Parallel Strategy Convert on N2C4
@pytest.mark.skipif(True, reason="Skip for failed")
class TestUnifiedCheckpointOnN2C4Dynamic(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            self.run_n2c4(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


# Test Unified Checkpoint Hybrid Parallel Strategy and Devices Convert Between N1C8 and N2C4
@pytest.mark.skipif(True, reason="Skip for failed")
class TestUnifiedCheckpointOnN1C8ToN2C4(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        self.run_n1c8(self.run_pretrain_file, **train_args)
        move_checkpoint_N1C8_to_N2C4()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            self.run_n2c4(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


@pytest.mark.skipif(True, reason="Skip for failed")
class TestUnifiedCheckpointOnN2C4ToN1C8(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)
        move_checkpoint_N2C4_to_N1C8()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            self.run_n1c8(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


# Test Unified Checkpoint Config on N1C8
@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN1C8SkipSaveModelWeight(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value

        self.need_allclose = True
        self.rtol = 1e-7

    def runfirst(self, train_args):
        self.run_n1c8(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        self.run_n1c8(self.run_pretrain_file, **train_args)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN1C8MasterWeightCompatibleO1ToO2(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value

        self.need_allclose = False

    def runfirst(self, train_args):
        train_args["fp16_opt_level"] = "O1"
        self.run_n1c8(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        train_args["fp16_opt_level"] = "O2"
        self.run_n1c8(self.run_pretrain_file, **train_args)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN1C8MasterWeightCompatibleO2ToO1(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value

        self.need_allclose = False

    def runfirst(self, train_args):
        train_args["fp16_opt_level"] = "O2"
        self.run_n1c8(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        train_args["fp16_opt_level"] = "O1"
        self.run_n1c8(self.run_pretrain_file, **train_args)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN1C8CheckpointCompatible(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()

        self.need_allclose = True
        self.rtol = 1e-7

    def runfirst(self, train_args):
        train_args["unified_checkpoint"] = 0
        self.run_n1c8(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        train_args["unified_checkpoint"] = 1
        self.run_n1c8(self.run_pretrain_file, **train_args)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestPaddleCheckpointOnN1C8Reset(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()

        self.need_allclose = True
        self.rtol = 1e-7

    def runfirst(self, train_args):
        train_args["unified_checkpoint"] = 0
        self.run_n1c8(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        train_args["unified_checkpoint"] = 0
        self.run_n1c8(self.run_pretrain_file, **train_args)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestPaddleCheckpointOnN1C2Reset(TestMultipleGpus):
    def setUp(self):
        self.configs = get_pretrain_arguments(pretrain_arguments)
        os.environ.update(environment_variables)

        files = [
            "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin",
            "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx",
        ]
        self.prepare_inputs_data(pretrain_arguments["input_dir"], files)

        self.need_allclose = True
        self.rtol = 1e-7

        self.run_pretrain_file = "llm/run_pretrain.py"

    def runfirst(self, train_args):
        train_args["unified_checkpoint"] = 0
        self.run_n1c2(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        train_args["unified_checkpoint"] = 0
        self.run_n1c2(self.run_pretrain_file, **train_args)

    @skip_for_none_ce_case
    @require_paddle_at_least_2_gpu
    def testTP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP2"]

        self.runfirst(train_args)
        self.rerun(train_args)

        if self.need_allclose:
            res = check_acc()
            assert len(res) == 2
            np.testing.assert_allclose(res[0], res[1], self.rtol)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN1C2Reset(TestMultipleGpus):
    def setUp(self):
        self.configs = get_pretrain_arguments(pretrain_arguments)
        os.environ.update(environment_variables)

        files = [
            "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin",
            "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx",
        ]
        self.prepare_inputs_data(pretrain_arguments["input_dir"], files)

        self.need_allclose = True
        self.rtol = 1e-7

        self.run_pretrain_file = "llm/run_pretrain.py"
        self.filelists = [
            "config.json",
            "master_weights-00001-of-00002.safetensors",
            "master_weights-00002-of-00002.safetensors",
            "master_weights.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "optimizer-00001-of-00002.safetensors",
            "optimizer-00002-of-00002.safetensors",
            "optimizer.safetensors.index.json",
            "rng_state_2.pth",
            "scaler.pdparams",
            "scheduler.pdparams",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "trainer_state.json",
            "training_args.bin",
        ]

    def runfirst(self, train_args):
        train_args["unified_checkpoint"] = 1
        self.run_n1c2(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        train_args["unified_checkpoint"] = 1
        self.run_n1c2(self.run_pretrain_file, **train_args)

    @skip_for_none_ce_case
    @require_paddle_at_least_2_gpu
    def testTP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP2"]

        self.runfirst(train_args)
        self.rerun(train_args)

        if self.need_allclose:
            res = check_acc()
            assert len(res) == 2
            np.testing.assert_allclose(res[0], res[1], self.rtol)

    @skip_for_none_ce_case
    @require_paddle_at_least_2_gpu
    def testFileLists(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        save_steps = pretrain_arguments["save_steps"]
        base_ckpt_path = os.path.join(pretrain_arguments["output_dir"], "checkpoint-%d" % save_steps)

        train_args = self.configs["TP2"]
        self.runfirst(train_args)
        assert sorted(self.filelists) == sorted(os.listdir(base_ckpt_path))
        self.rerun(train_args)

        if self.need_allclose:
            res = check_acc()
            assert len(res) == 2
            np.testing.assert_allclose(res[0], res[1], self.rtol)

        # Test skip_save_model_weight
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args["unified_checkpoint_config"] = "skip_save_model_weight"
        self.runfirst(train_args)
        unsave_filelists = [
            "master_weights-00001-of-00002.safetensors",
            "master_weights-00002-of-00002.safetensors",
            "master_weights.safetensors.index.json",
        ]
        cur_filelists = [file for file in self.filelists if file not in unsave_filelists]
        assert sorted(cur_filelists) == sorted(os.listdir(base_ckpt_path))
        self.rerun(train_args)

        if self.need_allclose:
            res = check_acc()
            assert len(res) == 2
            np.testing.assert_allclose(res[0], res[1], self.rtol)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN1C8AsyncSaveToDisk(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key]["unified_checkpoint_config"] = UnifiedCheckpointOption.ASYNC_SAVE.value

        self.need_allclose = True
        self.rtol = 1e-7

    def runfirst(self, train_args):
        self.run_n1c8(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        self.run_n1c8(self.run_pretrain_file, **train_args)


# Test Unified Checkpoint Config on N2C4
@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN2C4SkipSaveModelWeight(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value

        self.need_allclose = True
        self.rtol = 1e-7

    def runfirst(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN2C4MasterWeightCompatibleO1ToO2(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value

        self.need_allclose = False

    def runfirst(self, train_args):
        train_args["fp16_opt_level"] = "O1"
        self.run_n2c4(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        train_args["fp16_opt_level"] = "O2"
        self.run_n2c4(self.run_pretrain_file, **train_args)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN2C4MasterWeightCompatibleO2ToO1(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value

        self.need_allclose = False

    def runfirst(self, train_args):
        train_args["fp16_opt_level"] = "O2"
        self.run_n2c4(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        train_args["fp16_opt_level"] = "O1"
        self.run_n2c4(self.run_pretrain_file, **train_args)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN2C4CheckpointCompatible(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()

        self.need_allclose = True
        self.rtol = 1e-7

    def runfirst(self, train_args):
        train_args["unified_checkpoint"] = 0
        self.run_n2c4(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        train_args["unified_checkpoint"] = 1
        self.run_n2c4(self.run_pretrain_file, **train_args)


@pytest.mark.skipif(True, reason="Skip for None CE")
class TestUnifiedCheckpointOnN2C4AsyncSaveToDisk(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key]["unified_checkpoint_config"] = UnifiedCheckpointOption.ASYNC_SAVE.value

        self.need_allclose = True
        self.rtol = 1e-7

    def runfirst(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)

    def rerun(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)


# Test Unified Checkpoint Hybrid Parallel Strategy and Devices Convert Between N1C8 and N2C4
# With Unified Checkpoint Config
@pytest.mark.skipif(True, reason="Skip for failed, hang")
class TestUnifiedCheckpointOnN1C8ToN2C4SkipSaveModelWeight(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value

        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        self.run_n1c8(self.run_pretrain_file, **train_args)
        move_checkpoint_N1C8_to_N2C4()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            self.run_n2c4(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


@pytest.mark.skipif(True, reason="Skip for failed, hang")
class TestUnifiedCheckpointOnN1C8ToN2C4MasterWeightCompatibleO1ToO2(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value

        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        train_args["fp16_opt_level"] = "O1"
        self.run_n1c8(self.run_pretrain_file, **train_args)
        move_checkpoint_N1C8_to_N2C4()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            config["fp16_opt_level"] = "O2"
            self.run_n2c4(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


@pytest.mark.skipif(True, reason="Skip for failed, hang")
class TestUnifiedCheckpointOnN1C8ToN2C4MasterWeightCompatibleO2ToO1(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value

        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        train_args["fp16_opt_level"] = "O2"
        self.run_n1c8(self.run_pretrain_file, **train_args)
        move_checkpoint_N1C8_to_N2C4()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            config["fp16_opt_level"] = "O1"
            self.run_n2c4(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


@pytest.mark.skipif(True, reason="Skip for failed, hang")
class TestUnifiedCheckpointOnN1C8ToN2C4AsyncSaveToDisk(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key]["unified_checkpoint_config"] = UnifiedCheckpointOption.ASYNC_SAVE.value

        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        self.run_n1c8(self.run_pretrain_file, **train_args)
        move_checkpoint_N1C8_to_N2C4()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            self.run_n2c4(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


@pytest.mark.skipif(True, reason="Skip for failed, hang")
class TestUnifiedCheckpointOnN2C4ToN1C8SkipSaveModelWeight(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value

        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)
        move_checkpoint_N2C4_to_N1C8()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            self.run_n1c8(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


@pytest.mark.skipif(True, reason="Skip for failed, hang")
class TestUnifiedCheckpointOnN2C4ToN1C8MasterWeightCompatibleO1ToO2(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value

        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        train_args["fp16_opt_level"] = "O1"
        self.run_n2c4(self.run_pretrain_file, **train_args)
        move_checkpoint_N2C4_to_N1C8()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            config["fp16_opt_level"] = "O2"
            self.run_n1c8(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


@pytest.mark.skipif(True, reason="Skip for failed, hang")
class TestUnifiedCheckpointOnN2C4ToN1C8MasterWeightCompatibleO2ToO1(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key][
                "unified_checkpoint_config"
            ] = UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value

        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        train_args["fp16_opt_level"] = "O2"
        self.run_n2c4(self.run_pretrain_file, **train_args)
        move_checkpoint_N2C4_to_N1C8()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            config["fp16_opt_level"] = "O1"
            self.run_n1c8(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)


@pytest.mark.skipif(True, reason="Skip for failed, hang")
class TestUnifiedCheckpointOnN2C4ToN1C8AsyncSaveToDisk(TestUnifiedCheckpointBase):
    def setUp(self):
        super().setUp()
        for config_key in self.configs:
            self.configs[config_key]["unified_checkpoint"] = 1
            self.configs[config_key]["unified_checkpoint_config"] = UnifiedCheckpointOption.ASYNC_SAVE.value

        self.need_allclose = False
        self.rtol = 1e-4
        self.k = MAX_CONVERT_CONFIGS  # max: 16, min: 1

    def runfirst(self, train_args):
        self.run_n2c4(self.run_pretrain_file, **train_args)
        move_checkpoint_N2C4_to_N1C8()

    def rerun(self, train_args):
        configs = random_sample(self.configs.keys(), k=self.k)
        for config_name in configs:
            print(f"Rerun using {config_name}")
            config = self.configs[config_name]
            self.run_n1c8(self.run_pretrain_file, **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=self.rtol)
