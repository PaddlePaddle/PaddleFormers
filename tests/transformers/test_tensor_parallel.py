# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import glob
import os
import sys
import tempfile
from pathlib import Path

import paddle
from paddle.distributed import fleet

sys.path.append(str(Path(__file__).parent.parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline-parallel-size", type=int, default=1)
parallel_args, _ = parser.parse_known_args()
world_size = paddle.distributed.get_world_size()
pp_size = parallel_args.pipeline_parallel_size
if pp_size <= 0 or world_size % pp_size:
    raise ValueError("pipeline-parallel-size must be a positive divisor of world size.")
tp_size = world_size // pp_size
tp_rank = 0
if world_size > 1:
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tp_size,
        "pp_degree": pp_size,
        "sharding_degree": 1,
    }
    strategy.pipeline_configs = {"micro_batch_size": 2, "accumulate_steps": 1}
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
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.tensor_model_parallel_size = tp_size
    config.tensor_parallel_rank = tp_rank
    config.head_dim = config.hidden_size // config.num_attention_heads
    return config


def common_test_load(model_class, tempdir):
    paddle.distributed.barrier()
    if model_class is not None:
        model_class.from_pretrained(tempdir, convert_from_hf=False, load_checkpoint_format="")
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
        model.save_pretrained(
            save_dir=tempdir,
            merge_tensor_parallel=True,
            is_main_process=is_main_process,
            save_safetensors=False,
            save_checkpoint_format="",
        )
        common_test_load(model_class, tempdir)
        # test merge shard
        model.save_pretrained(
            tempdir,
            merge_tensor_parallel=True,
            variant=f"tp{rank:0>2d}",
            max_shard_size="5MB",
            is_main_process=is_main_process,
            save_safetensors=False,
            save_checkpoint_format="",
        )
        common_test_load(model_class, tempdir)
        # test save tp
        model.save_pretrained(
            tempdir,
            max_shard_size="5MB",
            is_main_process=is_main_process,
            save_safetensors=False,
            save_checkpoint_format="",
        )
        common_test_load(model_class, tempdir)
        # test save shard safe
        model.save_pretrained(
            tempdir,
            max_shard_size="5MB",
            safe_serialization=True,
            is_main_process=is_main_process,
            save_safetensors=False,
            save_checkpoint_format="",
        )
        common_test_load(model_class, tempdir)
        # test save safe tensor
        model.save_pretrained(
            tempdir,
            safe_serialization=True,
            is_main_process=is_main_process,
            save_safetensors=False,
            save_checkpoint_format="",
        )
        common_test_load(model_class, tempdir)
        paddle.distributed.barrier()


def _test_llama():
    from paddleformers.transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig()
    config = prepare_config(config)
    LlamaForCausalLM.from_config(config)
    # common_test_merge(model, LlamaForCausalLM)


def _test_qwen2():
    from paddleformers.transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config()
    config = prepare_config(config)
    Qwen2ForCausalLM.from_config(config)
    # common_test_merge(model, Qwen2ForCausalLM)


def _test_olmo3():
    import numpy as np
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
    )

    from paddleformers.transformers import Olmo3Config
    from paddleformers.transformers.olmo3.modeling import Olmo3Attention

    if tp_size < 2:
        raise ValueError("Olmo3 tensor parallel regression requires at least two ranks.")
    for kv_heads in (4, 2):
        for sequence_parallel in (False, True):
            paddle.seed(2026)
            config = Olmo3Config(
                hidden_size=32,
                num_attention_heads=4,
                num_key_value_heads=kv_heads,
                num_hidden_layers=1,
                layer_types=["full_attention"],
                attention_dropout=0.0,
                fuse_rms_norm=False,
                _attn_implementation="eager",
            )
            reference = Olmo3Attention(config, 0)
            tp_config = Olmo3Config.from_dict(config.to_dict())
            tp_config.tensor_model_parallel_size = tp_size
            tp_config.tensor_parallel_rank = tp_rank
            tp_config.sequence_parallel = sequence_parallel
            parallel = Olmo3Attention(tp_config, 0)
            state = reference.state_dict()
            for name in ("q_norm.weight", "k_norm.weight"):
                state[name] = paddle.linspace(0.5, 1.5, state[name].shape[0])
            reference.set_state_dict(state)
            sharded_state = {}
            for name, value in state.items():
                if name.startswith(("q_proj.", "k_proj.", "v_proj.")):
                    value = paddle.split(value, tp_size, axis=-1)[tp_rank]
                elif name == "o_proj.weight":
                    value = paddle.split(value, tp_size, axis=0)[tp_rank]
                sharded_state[name] = value
            parallel.set_state_dict(sharded_state)
            values = paddle.arange(2 * 6 * 32, dtype="float32").reshape([2, 6, 32]) / 384
            full_input = values.clone()
            full_input.stop_gradient = False
            local_input = values.clone()
            local_input.stop_gradient = False
            local_hidden = ScatterOp.apply(local_input.reshape([-1, 32])) if sequence_parallel else local_input
            position_embeddings = (paddle.ones([2, 6, 8]), paddle.zeros([2, 6, 8]))
            mask = paddle.triu(paddle.full([2, 1, 6, 6], -1e4), diagonal=1)
            expected = reference(full_input, attention_mask=mask, position_embeddings=position_embeddings)[0]
            actual = parallel(local_hidden, attention_mask=mask, position_embeddings=position_embeddings)[0]
            if sequence_parallel:
                actual = GatherOp.apply(actual).reshape([2, 6, 32])
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
            expected.square().mean().backward()
            actual.square().mean().backward()
            np.testing.assert_allclose(local_input.grad.numpy(), full_input.grad.numpy(), atol=1e-5, rtol=1e-5)
            for name, parameter in parallel.named_parameters():
                gradient = dict(reference.named_parameters())[name].grad
                if name.startswith(("q_proj.", "k_proj.", "v_proj.")):
                    gradient = paddle.split(gradient, tp_size, axis=-1)[tp_rank]
                elif name == "o_proj.weight":
                    gradient = paddle.split(gradient, tp_size, axis=0)[tp_rank]
                assert parameter.grad is not None, name
                np.testing.assert_allclose(parameter.grad.numpy(), gradient.numpy(), atol=1e-5, rtol=1e-5)


def _test_pipeline_parallel():
    import numpy as np

    from paddleformers.transformers import (
        Olmo3Config,
        Olmo3ForCausalLM,
        Olmo3ForCausalLMPipe,
        Phi3Config,
        Phi3ForCausalLM,
        Phi3ForCausalLMPipe,
    )

    for config_class, model_class, pipe_class in (
        (Phi3Config, Phi3ForCausalLM, Phi3ForCausalLMPipe),
        (Olmo3Config, Olmo3ForCausalLM, Olmo3ForCausalLMPipe),
    ):
        for tied in (False, True):
            for recompute_enabled in (False, True):
                paddle.seed(2026)
                config = config_class(
                    vocab_size=64,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=pp_size,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    pad_token_id=1,
                    use_cache=False,
                    tie_word_embeddings=tied,
                    fuse_rms_norm=False,
                    _attn_implementation="eager",
                )
                if config_class is Olmo3Config:
                    config.layer_types = ["sliding_attention"] * (pp_size - 1) + ["full_attention"]
                    config.sliding_window = 3
                reference = model_class(config)
                reference.eval()
                pipe_config = config_class.from_dict(config.to_dict())
                pipe_config.tensor_parallel_output = False
                if recompute_enabled:
                    pipe_config.recompute_granularity = "full"
                    pipe_config.recompute_method = "uniform"
                    pipe_config.recompute_num_layers = 1
                    pipe_config.recompute_use_reentrant = False
                pipe = pipe_class(pipe_config)
                state = reference.state_dict()
                actions = {}
                if tp_size > 1:
                    for name, parameter in pipe.state_dict().items():
                        if getattr(parameter, "is_distributed", False):
                            actions[name] = lambda value, axis=parameter.split_axis: paddle.split(
                                value, tp_size, axis=axis
                            )[tp_rank]
                    state = {name: actions[name](value) if name in actions else value for name, value in state.items()}
                pipe.set_state_dict(dict(state))
                parallel = fleet.distributed_model(pipe)
                input_ids = paddle.to_tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]], dtype="int64")
                labels = paddle.to_tensor([[3, 4, 5, 6, 7, 2], [9, 10, 11, 12, 13, 2]], dtype="int64")
                with paddle.no_grad():
                    expected = reference(input_ids, return_dict=True).logits
                    outputs = parallel.eval_batch([input_ids, labels], compute_loss=False)
                if hcg.get_stage_id() == pp_size - 1:
                    actual = outputs[0]
                    if isinstance(actual, (tuple, list)):
                        actual = actual[0]
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
                reference.train()
                reference_loss = reference(input_ids, labels=labels, return_dict=True).loss
                reference_loss.backward()
                reference_optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=reference.parameters())
                reference_optimizer.step()
                optimizer = fleet.distributed_optimizer(paddle.optimizer.SGD(learning_rate=0.01, parameters=pipe.parameters()))
                actual_loss = parallel.train_batch([input_ids, labels], optimizer)
                np.testing.assert_allclose(actual_loss.numpy(), reference_loss.numpy(), atol=1e-5, rtol=1e-5)
                updated = reference.state_dict()
                if tp_size > 1:
                    updated = {name: actions[name](value) if name in actions else value for name, value in updated.items()}
                for name, value in pipe.state_dict().items():
                    np.testing.assert_allclose(value.numpy(), updated[name].numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    if pp_size > 1:
        _test_pipeline_parallel()
    else:
        _test_llama()
        _test_qwen2()
        if tp_size > 1:
            _test_olmo3()
