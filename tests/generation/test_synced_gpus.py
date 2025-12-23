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
import sys
import tempfile
from pathlib import Path

import paddle

from paddleformers.generation import GenerationConfig
from paddleformers.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddleformers.transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent.parent))
from tests.parallel_launch import TestMultipleGpus
from tests.transformers.test_modeling_common import ids_tensor


class ShardingStage3Tester(TestMultipleGpus):
    def test_synced_gpus_greedy(self):
        # test this file
        self.run_2gpu(__file__)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Paddleformers/tiny-random-llama")
    model = AutoModelForCausalLM.from_pretrained("Paddleformers/tiny-random-llama")
    model.config.eos_token_id = -1
    world_size = paddle.distributed.get_world_size()

    with tempfile.TemporaryDirectory() as tempdir:
        args_dict = {
            "sharding": "stage3",
            "sharding_parallel_size": world_size,
            "fp16": True,
            "fp16_opt_level": "O2",
            "output_dir": os.path.join(tempdir, "output"),
        }
        parser = PdArgumentParser((TrainingArguments,))
        args = parser.parse_dict(args_dict)[0]
        trainer = Trainer(model, args=args, tokenizer=tokenizer)
        trainer.create_optimizer_and_scheduler(num_training_steps=10)
        trainer._wrap_model(trainer.model_wrapped)
    model = trainer.model
    model.eval()
    input_ids = ids_tensor([1, 5], vocab_size=model.config.vocab_size, dtype="int64")
    attention_mask = paddle.ones_like(input_ids, dtype="bool")
    input_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "synced_gpus": True,
    }
    generation_config = GenerationConfig(max_length=10 + paddle.distributed.get_rank(), trunc_input=False)

    def test_synced_gpus_greedy():
        with paddle.no_grad():
            generation_config.decode_strategy = "greedy_search"
            model.generate(**input_kwargs, generation_config=generation_config)

    def test_synced_gpus_sample():
        with paddle.no_grad():
            generation_config.decode_strategy = "sampling"
            generation_config.top_k = 8
            model.generate(**input_kwargs, generation_config=generation_config)

    def test_synced_gpus_beam_search():
        with paddle.no_grad():
            generation_config.decode_strategy = "beam_search"
            generation_config.num_beams = 4
            model.generate(**input_kwargs, generation_config=generation_config)

    def test_synced_gpus_group_beam_search():
        with paddle.no_grad():
            generation_config.decode_strategy = "beam_search"
            generation_config.num_beams = 4
            generation_config.num_beam_groups = 2
            model.generate(**input_kwargs, generation_config=generation_config)

    test_synced_gpus_greedy()
    test_synced_gpus_sample()
    test_synced_gpus_beam_search()
    test_synced_gpus_group_beam_search()
