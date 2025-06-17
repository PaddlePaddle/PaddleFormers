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

environment_variables = {
    "NVIDIA_TF32_OVERRIDE": "0",
    "FLAGS_embedding_deterministic": "1",
    "FLAGS_cudnn_deterministic": "1",
}
for k, v in environment_variables.items():
    os.environ[k] = v
import unittest
from typing import Optional, Tuple

import paddle
import paddle.device
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.recompute import recompute as original_recompute

from paddleformers.trainer.training_args import TrainingArguments
from paddleformers.transformers.refined_recompute import no_recompute as rr_no_recompute
from paddleformers.transformers.refined_recompute import recompute as rr_recompute
from paddleformers.utils.import_utils import is_paddle_cuda_available

ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
}
dtype = paddle.float16


class PyLayerMatmul(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensor()
        if hasattr(a, "main_grad"):
            a.main_grad.add_(paddle.ones_like(a.main_grad))
        if hasattr(b, "main_grad"):
            b.main_grad.add_(paddle.ones_like(b.main_grad))
        grad_a = paddle.matmul(dy, b, transpose_y=True)
        grad_b = paddle.matmul(a, dy, transpose_x=True)
        return grad_a, grad_b


pylayer_matmul = PyLayerMatmul.apply


class BertConfig:
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        max_position_embeddings: int = 1024,
        type_vocab_size: int = 2,
        initializer_range: float = 0.2,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        layer_norm_eps: float = 1e-12,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        num_labels=2,
        recompute=False,
        use_rr_recompute=False,
        recompute_use_reentrant=False,
        **kwargs
    ):
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_labels = num_labels
        self.recompute = recompute
        self.use_rr_recompute = use_rr_recompute
        self.recompute_use_reentrant = recompute_use_reentrant


class BertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", paddle.arange(config.max_position_embeddings, dtype="int64").reshape((1, -1))
        )

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        input_shape = input_ids.shape
        seq_length = input_ids.shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:

        reshape_fn = lambda x: x.reshape([0, 0, -1, self.attention_head_size])
        # compute q,k,v
        query_layer = reshape_fn(self.query(hidden_states))
        key_layer = reshape_fn(self.key(hidden_states))
        value_layer = reshape_fn(self.value(hidden_states))

        context_layer = rr_no_recompute(
            F.scaled_dot_product_attention,
            query=query_layer,
            key=key_layer,
            value=value_layer,
            is_causal=True,
            enable=self.config.use_rr_recompute and self.config.recompute,
        )

        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size,
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, None) if output_attentions else (context_layer,)

        return outputs


class BertSelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor, input_tensor: paddle.Tensor) -> paddle.Tensor:
        hidden_states = rr_no_recompute(
            self.dense, hidden_states, enable=self.config.use_rr_recompute and self.config.recompute
        )
        hidden_states = self.dropout(hidden_states)

        hidden_states = rr_no_recompute(
            self.LayerNorm, hidden_states + input_tensor, enable=self.config.use_rr_recompute and self.config.recompute
        )
        return hidden_states


class BertAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False)
        self.dense.weight.main_grad = paddle.zeros_like(self.dense.weight).cast("float32")
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        def pylayer_dense(hidden_states):
            return pylayer_matmul(hidden_states, self.dense.weight)

        hidden_states = rr_no_recompute(
            pylayer_dense, hidden_states, enable=self.config.use_rr_recompute and self.config.recompute
        )
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor, input_tensor: paddle.Tensor) -> paddle.Tensor:
        def custom_dense(hidden_states, weight, bias=None):
            return F.linear(hidden_states, weight, bias)

        bias = self.dense.bias * 1.1
        hidden_states = rr_no_recompute(
            custom_dense,
            hidden_states,
            weight=self.dense.weight,
            bias=bias,
            enable=self.config.use_rr_recompute and self.config.recompute,
            keys_ignore_to_save=["bias"],
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        # self attn
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # ffn
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs

        return outputs


class BertEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer_module in self.layer:
            # add hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.training and self.config.recompute:
                recompute_function = rr_recompute if self.config.use_rr_recompute else original_recompute
                layer_outputs = recompute_function(
                    layer_module,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    use_reentrant=self.config.recompute_use_reentrant,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]

            # add self attn
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )


class BertPreTrainedModel(nn.Layer):
    def _init_weights(self, module):
        """Initialize the weights"""
        pass


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[paddle.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_ids.shape, dtype=paddle.int64)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return encoder_outputs


class BertRefinedRecomputeTest(unittest.TestCase):
    def no_pp_fwd_bwd(
        self,
        recompute=False,
        use_rr_recompute=False,
        recompute_use_reentrant=False,
        num_hidden_layers=4,
        shape=[2, 64],
    ):
        paddle.set_default_dtype(dtype)
        paddle.seed(42)
        config = BertConfig(
            num_hidden_layers=num_hidden_layers,
            recompute=recompute,
            use_rr_recompute=use_rr_recompute,
            recompute_use_reentrant=recompute_use_reentrant,
        )
        model = BertModel(config)
        model.train()
        input_ids = paddle.randint(10, config.vocab_size, shape=shape)
        gpu_mem_used_before = paddle.device.cuda.memory_allocated()
        outputs = model(input_ids=input_ids)[0]
        gpu_mem_used_after = paddle.device.cuda.memory_allocated()
        outputs.sum().backward()

        # div = 1024**3 # GB
        div = 1  # KB
        return (
            model,
            round((gpu_mem_used_after - gpu_mem_used_before) / div, 2),
            round(paddle.device.cuda.max_memory_allocated() / div, 2),
        )

    @unittest.skipIf(not is_paddle_cuda_available(), "refined-recompute only support on gpu")
    def test_refined_recompute(self):
        raw_dtype = paddle.get_default_dtype()

        model1, mem_usage_forward1, max_mem_usage_forward1 = self.no_pp_fwd_bwd(
            recompute=True, use_rr_recompute=False
        )  # with recompute
        model2, mem_usage_forward2, max_mem_usage_forward2 = self.no_pp_fwd_bwd(
            recompute=True, use_rr_recompute=True
        )  # with rr recompute
        model3, mem_usage_forward3, max_mem_usage_forward3 = self.no_pp_fwd_bwd(
            recompute=False, use_rr_recompute=False
        )  # without recompute

        name_list = [n for n, _ in model1.named_parameters()]

        for param1, param2, name in zip(model1.parameters(), model3.parameters(), name_list):
            # test main grad
            if "intermediate.dense.weight" in name:
                self.assertTrue(param1.main_grad.sum().item() > 0)
                self.assertTrue(param2.main_grad.sum().item() > 0)
            self.assertTrue(paddle.equal_all(param1.grad.cast("float32"), param2.grad.cast("float32")))

        for param1, param2, name in zip(model2.parameters(), model3.parameters(), name_list):
            # test main grad
            if "intermediate.dense.weight" in name:
                self.assertTrue(param1.main_grad.sum().item() > 0)
                self.assertTrue(param2.main_grad.sum().item() > 0)
            self.assertTrue(paddle.equal_all(param1.grad.cast("float32"), param2.grad.cast("float32")))

        # self.assertTrue(mem_usage_forward1 < mem_usage_forward2 < mem_usage_forward3)
        # self.assertTrue(max_mem_usage_forward1 < max_mem_usage_forward2 < max_mem_usage_forward3)

        del model1, model2, model3
        paddle.device.cuda.empty_cache()
        paddle.set_default_dtype(raw_dtype)

    def pp_fwd_bwd(
        self,
        recompute=False,
        use_rr_recompute=False,
        recompute_use_reentrant=False,
        num_iter=4,
        shape=[2, 64],
    ):
        paddle.set_default_dtype(dtype)
        paddle.seed(42)
        config = BertConfig(
            num_hidden_layers=1,
            recompute=recompute,
            use_rr_recompute=use_rr_recompute,
            recompute_use_reentrant=recompute_use_reentrant,
        )
        layer = BertLayer(config)
        layer.train()

        x = paddle.randn([*shape, config.hidden_size])
        x.stop_gradient = False
        x_copy = x

        if layer.training and config.recompute:
            recompute_function = rr_recompute if config.use_rr_recompute else original_recompute
            for _ in range(num_iter):
                x = recompute_function(layer, x, use_reentrant=config.recompute_use_reentrant)[0]
        else:
            for _ in range(num_iter):
                x = layer(x)[0]

        x.sum().backward()

        return x_copy.grad, layer

    @unittest.skipIf(not is_paddle_cuda_available(), "refined-recompute-pp only support on gpu")
    def test_refined_recompute_pp(self):
        paddle.set_device("gpu")
        raw_dtype = paddle.get_default_dtype()
        grad1, layer1 = self.pp_fwd_bwd(recompute=True, use_rr_recompute=False)
        grad2, layer2 = self.pp_fwd_bwd(recompute=True, use_rr_recompute=True)
        grad3, layer3 = self.pp_fwd_bwd(recompute=False, use_rr_recompute=False)

        name_list = [n for n, _ in layer1.named_parameters()]

        for param1, param2, name in zip(layer1.parameters(), layer3.parameters(), name_list):
            # test main grad
            if "intermediate.dense.weight" in name:
                self.assertTrue(param1.main_grad.sum().item() > 0)
                self.assertTrue(param2.main_grad.sum().item() > 0)
            self.assertTrue(paddle.equal_all(param1.grad.cast("float32"), param2.grad.cast("float32")))

        self.assertTrue(paddle.equal_all(grad1.cast("float32"), grad3.cast("float32")))
        for param1, param2, name in zip(layer2.parameters(), layer3.parameters(), name_list):
            # test main grad
            if "intermediate.dense.weight" in name:
                self.assertTrue(param1.main_grad.sum().item() > 0)
                self.assertTrue(param2.main_grad.sum().item() > 0)
            self.assertTrue(paddle.equal_all(param1.grad.cast("float32"), param2.grad.cast("float32")))

        self.assertTrue(paddle.equal_all(grad2.cast("float32"), grad3.cast("float32")))

        del grad1, grad2, grad3
        del layer1, layer2, layer3
        paddle.device.cuda.empty_cache()
        paddle.set_default_dtype(raw_dtype)


class TestRefinedRecomputeModel(unittest.TestCase):
    def setUp(self):
        self.args = TrainingArguments(
            output_dir="./",
            do_train=True,
            max_steps=100,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            refined_recompute="attention_column_ln:1,attention_row_ln:2,flash_attn:-1,mlp_column_ln:2,mlp_row_ln:-1",
        )

    @unittest.skipIf(not is_paddle_cuda_available(), "refined-recompute-pp only support on gpu")
    def test_llama_refined_recompute(self):
        paddle.set_device("gpu")
        from paddleformers.transformers.llama import LlamaConfig, LlamaModel

        llama_model = "__internal_testing__/tiny-random-llama"
        config = LlamaConfig.from_pretrained(llama_model)
        config.recompute = True
        config.recompute_granularity = "full"
        config.recompute_use_reentrant = False
        config.sequence_parallel = False
        config.use_flash_attention = True
        config.refined_recompute = self.args.refined_recompute
        model = LlamaModel.from_config(config=config, dtype="bfloat16")
        input_ids = paddle.randint(0, 100, shape=[1, 1024], dtype="int64")
        output = model(input_ids)
        output[0].mean().backward()

    @unittest.skipIf(not is_paddle_cuda_available(), "refined-recompute-pp only support on gpu")
    def test_qwen_refined_recompute(self):
        paddle.set_device("gpu")
        from paddleformers.transformers.qwen import QWenConfig, QWenModel

        llama_model = "__internal_testing__/tiny-random-qwen"
        config = QWenConfig.from_pretrained(llama_model)
        config.recompute = True
        config.recompute_granularity = "full"
        config.recompute_use_reentrant = False
        config.sequence_parallel = False
        config.use_flash_attention = True
        config.refined_recompute = self.args.refined_recompute
        config.seq_length = 1024
        model = QWenModel.from_config(config=config, dtype="bfloat16")
        input_ids = paddle.randint(0, 100, shape=[1, 1024], dtype="int64")
        output = model(input_ids)
        output[0].mean().backward()

    @unittest.skipIf(not is_paddle_cuda_available(), "refined-recompute-pp only support on gpu")
    def test_qwen2_refined_recompute(self):
        paddle.set_device("gpu")
        from paddleformers.transformers.qwen2 import Qwen2Config, Qwen2Model

        llama_model = "__internal_testing__/tiny-random-qwen2"
        config = Qwen2Config.from_pretrained(llama_model)
        config.recompute = True
        config.recompute_granularity = "full"
        config.recompute_use_reentrant = False
        config.sequence_parallel = False
        config.use_flash_attention = True
        config.refined_recompute = self.args.refined_recompute
        model = Qwen2Model.from_config(config=config, dtype="bfloat16")
        input_ids = paddle.randint(0, 100, shape=[1, 1024], dtype="int64")
        output = model(input_ids)
        output[0].mean().backward()
