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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleformers.generation.utils import GenerationMixin
from paddleformers.transformers.model_utils import PretrainedModel

from .configuration import DiffTransformerConfig


class RMSNorm(nn.Layer):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(
            shape=[dim],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        return x * paddle.rsqrt(paddle.mean(x**2, axis=-1, keepdim=True) + self.eps) * self.weight


class DiffAttn(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias_attr=False)
        self.k_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias_attr=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias_attr=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias_attr=False)

        self.lambda_init = config.lambda_init
        self.lambda_q1 = self.create_parameter(shape=[self.head_dim], dtype="float32")
        self.lambda_k1 = self.create_parameter(shape=[self.head_dim], dtype="float32")
        self.lambda_q2 = self.create_parameter(shape=[self.head_dim], dtype="float32")
        self.lambda_k2 = self.create_parameter(shape=[self.head_dim], dtype="float32")

        self.subln = RMSNorm(self.hidden_size)

    def _scaled_dot_product_attention(self, query, key, value, attention_mask):
        if query.dtype in (paddle.float16, paddle.bfloat16):
            return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False)

        query = query.transpose([0, 2, 1, 3])
        key = key.transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])
        scores = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        return paddle.matmul(F.softmax(scores, axis=-1), value).transpose([0, 2, 1, 3])

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(bsz, seq_len, 2, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(bsz, seq_len, 2, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(bsz, seq_len, self.num_heads, self.head_dim)

        q1, q2 = q[:, :, 0], q[:, :, 1]
        k1, k2 = k[:, :, 0], k[:, :, 1]

        attn1 = self._scaled_dot_product_attention(q1, k1, v, attention_mask)
        attn2 = self._scaled_dot_product_attention(q2, k2, v, attention_mask)

        lambda_1 = paddle.exp(paddle.sum(self.lambda_q1 * self.lambda_k1, axis=-1))
        lambda_2 = paddle.exp(paddle.sum(self.lambda_q2 * self.lambda_k2, axis=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn = attn1 - lambda_full * attn2
        attn = attn * (1 - self.lambda_init)

        attn = attn.reshape(bsz, seq_len, -1)
        attn = self.subln(attn)

        return self.o_proj(attn), None, None


class DiffTransformerBlock(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = DiffAttn(config, layer_idx)
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False),
            nn.Silu(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=False),
        )

    def forward(self, x, attention_mask=None, **kwargs):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DiffTransformerPreTrainedModel(PretrainedModel):
    config_class = DiffTransformerConfig
    base_model_prefix = "model"


class DiffTransformerModel(DiffTransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.LayerList([DiffTransformerBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed_tokens(input_ids)
        causal_mask = GenerationMixin._prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_ids.shape,
            past_key_values_length=0,
            dtype=x.dtype,
        )
        for layer in self.layers:
            x = layer(x, attention_mask=causal_mask)
        return self.norm(x)


class DiffTransformerForCausalLM(DiffTransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = DiffTransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape([-1, logits.shape[-1]])
            shift_labels = labels[:, 1:].reshape([-1])
            loss = F.cross_entropy(shift_logits, shift_labels)

        if loss is not None:
            return loss, logits
        return logits

    @classmethod
    def _gen_aoa_config(cls, config):
        """AOA checkpoint conversion config. Linear weights need transpose (PyTorch [out,in] -> Paddle [in,out])."""
        aoa_statements = [
            "model.embed_tokens.weight -> model.embed_tokens.weight",
            "model.norm.weight -> model.norm.weight",
            "lm_head.weight^T -> lm_head.weight",
            "model.layers.$LAYER_ID.norm1.weight -> model.layers.$LAYER_ID.norm1.weight",
            "model.layers.$LAYER_ID.norm2.weight -> model.layers.$LAYER_ID.norm2.weight",
            "model.layers.$LAYER_ID.attn.q_proj.weight^T -> model.layers.$LAYER_ID.attn.q_proj.weight",  # ^T
            "model.layers.$LAYER_ID.attn.k_proj.weight^T -> model.layers.$LAYER_ID.attn.k_proj.weight",  # ^T
            "model.layers.$LAYER_ID.attn.v_proj.weight^T -> model.layers.$LAYER_ID.attn.v_proj.weight",  # ^T
            "model.layers.$LAYER_ID.attn.o_proj.weight^T -> model.layers.$LAYER_ID.attn.o_proj.weight",  # ^T
            "model.layers.$LAYER_ID.attn.lambda_q1 -> model.layers.$LAYER_ID.attn.lambda_q1",
            "model.layers.$LAYER_ID.attn.lambda_k1 -> model.layers.$LAYER_ID.attn.lambda_k1",
            "model.layers.$LAYER_ID.attn.lambda_q2 -> model.layers.$LAYER_ID.attn.lambda_q2",
            "model.layers.$LAYER_ID.attn.lambda_k2 -> model.layers.$LAYER_ID.attn.lambda_k2",
            "model.layers.$LAYER_ID.attn.subln.weight -> model.layers.$LAYER_ID.attn.subln.weight",
            "model.layers.$LAYER_ID.mlp.0.weight^T -> model.layers.$LAYER_ID.mlp.0.weight",  # ^T
            "model.layers.$LAYER_ID.mlp.2.weight^T -> model.layers.$LAYER_ID.mlp.2.weight",  # ^T
        ]
        return {"aoa_statements": aoa_statements}
