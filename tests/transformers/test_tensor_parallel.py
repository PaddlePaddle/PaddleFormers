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


import glob
import os
import sys
import tempfile
from pathlib import Path

import paddle
from paddle.distributed import fleet

sys.path.append(str(Path(__file__).parent.parent.parent))
from tests.parallel_launch import TestMultipleGpus
from tests.testing_utils import require_paddle_at_least_2_gpu

tp_size = paddle.distributed.get_world_size()
tp_rank = 0
if tp_size > 1:
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tp_size,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    tp_rank = hcg.get_model_parallel_rank()
    mp_group = hcg.get_model_parallel_group()


def prepare_config(config):
    config.hidden_size = 512
    config.num_layers = 2
    config.num_hidden_layers = 2
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    config.intermediate_size = config.hidden_size * 3
    config.tensor_parallel_degree = tp_size
    config.tensor_parallel_rank = tp_rank
    return config


def common_test_load(model_class, tempdir):
    paddle.distributed.barrier()
    if model_class is not None:
        model_class.from_pretrained(tempdir)
        paddle.distributed.barrier()
        if paddle.distributed.get_rank() == 0:
            files = glob.glob(tempdir + "/*")
            for f in files:
                os.remove(f)
        paddle.distributed.barrier()


def common_test_merge(model, model_class=None):
    rank = paddle.distributed.get_rank()
    is_main_process = rank == 0
    object_list = []
    with tempfile.TemporaryDirectory() as tempdir:
        paddle.distributed.all_gather_object(object_list, tempdir, group=mp_group)
        tempdir = object_list[0]
        # test merge one
        model.save_pretrained(save_dir=tempdir, merge_tensor_parallel=True, is_main_process=is_main_process)
        common_test_load(model_class, tempdir)
        # test merge shard
        model.save_pretrained(
            tempdir,
            merge_tensor_parallel=True,
            variant=f"tp{rank:0>2d}",
            max_shard_size="5MB",
            is_main_process=is_main_process,
        )
        common_test_load(model_class, tempdir)
        # test save tp
        model.save_pretrained(tempdir, max_shard_size="5MB", is_main_process=is_main_process)
        common_test_load(model_class, tempdir)
        # test save shard safe
        model.save_pretrained(tempdir, max_shard_size="5MB", safe_serialization=True, is_main_process=is_main_process)
        common_test_load(model_class, tempdir)
        # test save safe tensor
        model.save_pretrained(tempdir, safe_serialization=True, is_main_process=is_main_process)
        common_test_load(model_class, tempdir)
        paddle.distributed.barrier()


def _test_llama():
    from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig()
    config = prepare_config(config)
    model = LlamaForCausalLM.from_config(config)
    common_test_merge(model, LlamaForCausalLM)


def _test_chatglm():
    from paddlenlp.transformers import ChatGLMConfig, ChatGLMForCausalLM

    config = ChatGLMConfig()
    config = prepare_config(config)
    model = ChatGLMForCausalLM.from_config(config)
    common_test_merge(model, ChatGLMForCausalLM)


def _test_bloom():
    from paddlenlp.transformers import BloomConfig, BloomForCausalLM

    config = BloomConfig()
    config = prepare_config(config)
    model = BloomForCausalLM.from_config(config)
    common_test_merge(model, BloomForCausalLM)


def _test_qwen2():
    from paddlenlp.transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config()
    config = prepare_config(config)
    model = Qwen2ForCausalLM.from_config(config)
    common_test_merge(model, Qwen2ForCausalLM)


def _test_gemma():
    from paddlenlp.transformers import GemmaConfig, GemmaForCausalLM

    config = GemmaConfig()
    config = prepare_config(config)
    model = GemmaForCausalLM.from_config(config)
    common_test_merge(model, GemmaForCausalLM)


@require_paddle_at_least_2_gpu
class TestTensorParallel(TestMultipleGpus):
    def test_model_load_merge(self):
        self.run_2gpu(__file__)


if __name__ == "__main__":
    _test_llama()
    _test_chatglm()
    _test_bloom()
    _test_gemma()
    _test_qwen2()
