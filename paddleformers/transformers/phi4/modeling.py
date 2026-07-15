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

import copy
import math
from typing import Any, Dict, Optional

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute

from ...generation.utils import _make_sliding_window_mask
from ...nn.criterion.interface import CriterionLayer
from ...nn.lm_head import LMHead as GeneralLMHead
from ...utils.log import logger
from ...utils.masking_utils import _expand_2d_mask, _make_causal_mask
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Phi4Config


def swiglu(x, y):
    return y * F.silu(x)


def selective_scan_paddle(x, dt, A, B, C, D, z=None, delta_bias=None, delta_softplus=True, return_last_state=False):
    batch, d_inner, seq_len = x.shape
    _, d_state, _ = B.shape
    orig_dtype = x.dtype

    x = x.astype("float32")
    dt = dt.astype("float32")
    A = A.astype("float32")
    B = B.astype("float32")
    C = C.astype("float32")
    D = D.astype("float32")

    if delta_bias is not None:
        dt = dt + delta_bias.reshape([1, -1, 1]).astype("float32")
    if delta_softplus:
        dt = F.softplus(dt)

    dA = paddle.exp(paddle.einsum("bdl,ds->bdsl", dt, A))
    dB = paddle.einsum("bdl,bsl->bdsl", dt, B)

    state = paddle.zeros([batch, d_inner, d_state], dtype="float32")
    outputs = []

    for i in range(seq_len):
        state = state * dA[:, :, :, i] + x[:, :, i : i + 1] * dB[:, :, :, i]
        y = paddle.einsum("bds,bs->bd", state, C[:, :, i])
        y = y + D * x[:, :, i]
        outputs.append(y)

    y = paddle.stack(outputs, axis=2)

    if z is not None:
        y = y * F.silu(z.astype("float32"))

    y = y.astype(orig_dtype)
    if return_last_state:
        return y, state.astype(orig_dtype)
    return y


class Phi4RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size], dtype=paddle.get_default_dtype(), default_initializer=nn.initializer.Constant(1.0)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight.astype(input_dtype) * hidden_states.astype(input_dtype)


PHI_NORM_CLASS = nn.LayerNorm


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class Phi4DiffAttention(nn.Layer):
    def __init__(self, head_dim, layer_idx):
        super().__init__()
        self.head_dim = head_dim
        self.lambda_init = lambda_init_fn(layer_idx if layer_idx is not None else 0)
        self.lambda_q1 = paddle.create_parameter(
            shape=[head_dim], dtype="float32", default_initializer=nn.initializer.Normal(mean=0.0, std=0.1)
        )
        self.lambda_k1 = paddle.create_parameter(
            shape=[head_dim], dtype="float32", default_initializer=nn.initializer.Normal(mean=0.0, std=0.1)
        )
        self.lambda_q2 = paddle.create_parameter(
            shape=[head_dim], dtype="float32", default_initializer=nn.initializer.Normal(mean=0.0, std=0.1)
        )
        self.lambda_k2 = paddle.create_parameter(
            shape=[head_dim], dtype="float32", default_initializer=nn.initializer.Normal(mean=0.0, std=0.1)
        )
        self.subln = Phi4RMSNorm(2 * head_dim, eps=1e-5)

    @staticmethod
    def _repeat_kv(x, n_rep):
        batch, n_heads, slen, head_dim = x.shape
        if n_rep == 1:
            return x
        x = x.unsqueeze(2).tile([1, 1, n_rep, 1, 1])
        return x.reshape([batch, n_heads * n_rep, slen, head_dim])

    def forward(self, q, k, v, attention_mask=None):
        bsz, n_heads, seqlen, head_dim = q.shape
        n_kv_heads = k.shape[1]
        n_rep = n_heads // n_kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        q1 = q[:, 0::2]
        q2 = q[:, 1::2]
        k1 = k[:, 0::2]
        k2 = k[:, 1::2]
        v1 = v[:, 0::2]
        v2 = v[:, 1::2]

        if n_rep > 1:
            k1 = self._repeat_kv(k1, n_rep)
            k2 = self._repeat_kv(k2, n_rep)
            v1 = self._repeat_kv(v1, n_rep)
            v2 = self._repeat_kv(v2, n_rep)

        attn_w1 = paddle.matmul(q1, k1.transpose([0, 1, 3, 2])) * scale
        attn_w2 = paddle.matmul(q2, k2.transpose([0, 1, 3, 2])) * scale

        if attention_mask is not None:
            attn_w1 = attn_w1 + attention_mask
            attn_w2 = attn_w2 + attention_mask

        attn_w1 = F.softmax(attn_w1, axis=-1, dtype="float32").cast(q.dtype)
        attn_w2 = F.softmax(attn_w2, axis=-1, dtype="float32").cast(q.dtype)

        attn11 = paddle.matmul(attn_w1, v1)
        attn12 = paddle.matmul(attn_w1, v2)
        attn21 = paddle.matmul(attn_w2, v1)
        attn22 = paddle.matmul(attn_w2, v2)

        attn1 = paddle.concat([attn11, attn12], axis=-1)
        attn2 = paddle.concat([attn21, attn22], axis=-1)

        lambda_1 = paddle.exp((self.lambda_q1.cast(q.dtype) * self.lambda_k1.cast(q.dtype)).sum())
        lambda_2 = paddle.exp((self.lambda_q2.cast(q.dtype) * self.lambda_k2.cast(q.dtype)).sum())
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn = attn1 - lambda_full * attn2
        attn = self.subln(attn) * (1 - self.lambda_init)

        attn = attn.reshape([bsz, n_heads // 2, seqlen, 2, head_dim])
        attn = attn.transpose([0, 1, 3, 2, 4]).reshape([bsz, n_heads, seqlen, head_dim])
        return attn


class Phi4MLP(nn.Layer):
    def __init__(self, config: Phi4Config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias_attr=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=False)

    def forward(self, hidden_states):
        y = self.fc1(hidden_states)
        gate, y = paddle.chunk(y, 2, axis=-1)
        return self.fc2(y * F.silu(gate))


class Phi4Attention(nn.Layer):
    def __init__(self, config: Phi4Config, layer_idx: Optional[int] = None, yoco_cross: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended"
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.yoco_cross = yoco_cross

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=True)
        if yoco_cross:
            self.Wqkv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=True)
        else:
            self.Wqkv = nn.Linear(self.hidden_size, op_size, bias_attr=True)
        self.inner_cross_attn = Phi4DiffAttention(self.head_dim, self.layer_idx)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        yoco_key_values=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        if self.yoco_cross:
            query_states = self.Wqkv(hidden_states)
            query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
            key_states, value_states = yoco_key_values
        else:
            qkv = self.Wqkv(hidden_states)
            query_pos = self.num_heads * self.head_dim
            query_states = qkv[..., :query_pos]
            key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
            value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

            query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
            key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
                [0, 2, 1, 3]
            )
            value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
                [0, 2, 1, 3]
            )

            if past_key_value is not None:
                cache_kwargs = {"cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

        yoco_key_values = key_states, value_states

        causal_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            else:
                causal_mask = attention_mask

        attn_weights = None
        attn_output = self.inner_cross_attn(query_states, key_states, value_states, attention_mask=causal_mask)
        attn_output = F.dropout(attn_output, p=self.attention_dropout, training=self.training)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([bsz, q_len, self.hidden_size])
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, yoco_key_values


class Phi4Mamba(nn.Layer):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        yoco_cross=False,
        yoco_kv=False,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.yoco_cross = yoco_cross
        self.yoco_kv = yoco_kv
        self.act = nn.Silu()

        if self.yoco_cross:
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias_attr=bias)
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias_attr=bias)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias_attr=bias)

            self.conv1d = nn.Conv1D(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                bias_attr=conv_bias,
            )

            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias_attr=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias_attr=True)

            A = paddle.arange(1, self.d_state + 1, dtype=paddle.float32)
            A = A.unsqueeze(0).tile([self.d_inner, 1])
            A_log = paddle.log(A)
            self.A_log = paddle.create_parameter(
                shape=A_log.shape, dtype="float32", default_initializer=nn.initializer.Assign(A_log)
            )

            self.D = paddle.create_parameter(
                shape=[self.d_inner], dtype="float32", default_initializer=nn.initializer.Constant(1.0)
            )

            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias_attr=bias)

    def forward(self, hidden_states, inference_params=None, mask=None, yoco_key_values=None, cache_position=None):
        if self.yoco_cross:
            out = self.in_proj(hidden_states)
            if yoco_key_values is not None:
                out = swiglu(out, yoco_key_values)
            else:
                out = out * self.act(out)
            out = self.out_proj(out)
            return out, yoco_key_values

        batch, seqlen, _ = hidden_states.shape
        conv_state, ssm_state = None, None

        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params)
            if cache_position is not None and cache_position[0] > 0:
                out, _, _, yoco_key_values = self.step(hidden_states, conv_state, ssm_state, yoco_key_values)
                return out, yoco_key_values

        xz = self.in_proj(hidden_states)
        xz = xz.transpose([0, 2, 1])

        A = -paddle.exp(self.A_log.astype("float32"))

        if (not self.yoco_kv) and self.use_fast_path and (inference_params is not None):
            raise NotImplementedError("Mamba fast path requires selective_scan_cuda kernel")
        else:
            x, z = paddle.chunk(xz, 2, axis=1)
            if self.yoco_kv:
                z = z.transpose([0, 2, 1])
            if mask is not None:
                x = x * mask.unsqueeze(1).astype(x.dtype)

            if conv_state is not None:
                conv_state_update = F.pad(x, [self.d_conv - x.shape[-1], 0])
                conv_state.set_value(conv_state_update.astype(conv_state.dtype))

            x_conv = x.astype(self.conv1d.weight.dtype)
            x = self.act(self.conv1d(x_conv)[..., :seqlen]).astype(x.dtype)

            if mask is not None:
                x = x * mask.unsqueeze(1).astype(x.dtype)

            x_dbl = self.x_proj(x.transpose([0, 2, 1]).reshape([batch * seqlen, self.d_inner]))
            dt = x_dbl[:, : self.dt_rank]
            B = x_dbl[:, self.dt_rank : self.dt_rank + self.d_state]
            C = x_dbl[:, self.dt_rank + self.d_state :]

            dt = paddle.matmul(dt, self.dt_proj.weight).reshape([batch, seqlen, self.d_inner]).transpose([0, 2, 1])
            B = B.reshape([batch, seqlen, self.d_state]).transpose([0, 2, 1])
            C = C.reshape([batch, seqlen, self.d_state]).transpose([0, 2, 1])

            y = selective_scan_paddle(
                x,
                dt,
                A,
                B,
                C,
                self.D.astype("float32"),
                z=None if self.yoco_kv else z,
                delta_bias=self.dt_proj.bias.astype("float32") if self.dt_proj.bias is not None else None,
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.set_value(last_state.astype(ssm_state.dtype))

            y = y.transpose([0, 2, 1])
            if self.yoco_kv:
                yoco_key_values = y
                y = swiglu(z, y)
            out = self.out_proj(y)

        return out, yoco_key_values

    def step(self, hidden_states, conv_state, ssm_state, yoco_key_values):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time"
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = paddle.chunk(xz, 2, axis=-1)

        conv_state_new = paddle.roll(conv_state, shifts=-1, axis=-1)
        conv_state_new[:, :, -1] = x
        conv_state.set_value(conv_state_new)

        x_conv = paddle.sum(conv_state * self.conv1d.weight.squeeze(1).astype(dtype), axis=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias.astype(dtype)
        x = self.act(x_conv)

        x_db = self.x_proj(x)
        dt, B, C = paddle.split(x_db, [self.dt_rank, self.d_state, self.d_state], axis=-1)
        dt = self.dt_proj(dt)

        A = -paddle.exp(self.A_log.astype("float32"))
        dt = F.softplus(dt.astype("float32")).astype(dtype)
        dA = paddle.exp(paddle.einsum("bd,dn->bdn", dt.astype("float32"), A))
        dB = paddle.einsum("bd,bn->bdn", dt.astype("float32"), B.astype("float32"))
        ssm_state_new = ssm_state.astype("float32") * dA + x.astype("float32").unsqueeze(2) * dB
        ssm_state.set_value(ssm_state_new.astype(ssm_state.dtype))

        y = paddle.einsum("bdn,bn->bd", ssm_state.astype(dtype), C)
        y = y + self.D.astype(dtype) * x

        if self.yoco_kv:
            yoco_key_values = y.unsqueeze(1)
            y = swiglu(z, y)
        else:
            y = y * self.act(z)

        out = self.out_proj(y.unsqueeze(1))
        return out, None, None, yoco_key_values

    def _get_states_from_cache(self, inference_params):
        conv_state = inference_params.key_cache[self.layer_idx]
        ssm_state = inference_params.value_cache[self.layer_idx]
        return conv_state, ssm_state


class Phi4Cache:
    def __init__(
        self,
        config: Phi4Config,
        batch_size: int = None,
        max_cache_len: int = None,
        dtype=None,
        max_batch_size: Optional[int] = None,
    ):
        self.dtype = dtype if dtype is not None else paddle.get_default_dtype()
        self._max_cache_len = max_cache_len
        self._max_batch_size = batch_size or max_batch_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.global_attn_idx = config.num_hidden_layers // 2 + 1
        self.num_layers = config.num_hidden_layers
        self.key_cache = []
        self.value_cache = []
        self.config = config

        intermediate_size = config.mamba_expand * config.hidden_size
        ssm_state_size = config.mamba_d_state
        conv_kernel_size = config.mamba_d_conv
        self.conv_kernel_size = conv_kernel_size

        for layer_idx in range(config.num_hidden_layers):
            use_mamba = config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
            if use_mamba:
                if self._max_batch_size is not None:
                    conv_state = paddle.zeros(
                        [self._max_batch_size, intermediate_size, conv_kernel_size], dtype=self.dtype
                    )
                    ssm_state = paddle.zeros(
                        [self._max_batch_size, intermediate_size, ssm_state_size], dtype=self.dtype
                    )
                else:
                    conv_state = None
                    ssm_state = None
                self.key_cache.append(conv_state)
                self.value_cache.append(ssm_state)
            else:
                self.key_cache.append(None)
                self.value_cache.append(None)

    @property
    def max_batch_size(self):
        return self._max_batch_size

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None):
        if layer_idx >= len(self.key_cache):
            raise ValueError(f"Layer index {layer_idx} out of range for cache with {len(self.key_cache)} layers")

        use_mamba = self.config.mb_per_layer > 0 and layer_idx % self.config.mb_per_layer == 0
        if use_mamba:
            return key_states, value_states

        sliding_window = None
        if (
            self.config.sliding_window is not None
            and layer_idx < len(self.config.sliding_window)
            and self.config.sliding_window[layer_idx] is not None
            and layer_idx != self.global_attn_idx
        ):
            sliding_window = self.config.sliding_window[layer_idx]

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = paddle.concat([self.key_cache[layer_idx], key_states], axis=2)
            self.value_cache[layer_idx] = paddle.concat([self.value_cache[layer_idx], value_states], axis=2)

            if sliding_window is not None and self.key_cache[layer_idx].shape[2] > sliding_window:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, -sliding_window:, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, -sliding_window:, :]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = None):
        if layer_idx is None:
            layer_idx = self.global_attn_idx

        if layer_idx >= len(self.key_cache):
            return 0

        if self.key_cache[layer_idx] is None:
            return 0

        return self.key_cache[layer_idx].shape[2]

    def get_max_cache_shape(self) -> Optional[int]:
        return self._max_cache_len

    def reset(self):
        for layer_idx in range(len(self.key_cache)):
            use_mamba = self.config.mb_per_layer > 0 and layer_idx % self.config.mb_per_layer == 0
            if use_mamba:
                if self.key_cache[layer_idx] is not None:
                    self.key_cache[layer_idx].zero_()
                    self.value_cache[layer_idx].zero_()
            else:
                self.key_cache[layer_idx] = None
                self.value_cache[layer_idx] = None

    def reorder_cache(self, beam_idx):
        if not isinstance(beam_idx, paddle.Tensor):
            beam_idx = paddle.to_tensor(beam_idx)
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                self.key_cache[layer_idx] = paddle.index_select(self.key_cache[layer_idx], beam_idx, axis=0)
            if self.value_cache[layer_idx] is not None:
                self.value_cache[layer_idx] = paddle.index_select(self.value_cache[layer_idx], beam_idx, axis=0)


class Phi4DecoderLayer(nn.Layer):
    def __init__(self, config: Phi4Config, layer_idx: int):
        super().__init__()
        self.mlp = Phi4MLP(config)
        self.input_layernorm = PHI_NORM_CLASS(config.hidden_size, epsilon=config.layer_norm_eps)

        self.yoco_kv = False
        self.yoco_cross = False
        self.yoco_mb = False
        self.layer_idx = layer_idx

        assert config.num_hidden_layers % 4 == 0, "n_layer should be divisible by 4 for Phi4"
        if layer_idx >= config.num_hidden_layers // 2:
            self.yoco_mb = True
            self.yoco_kv = layer_idx >= (config.num_hidden_layers // 2 + 1)
            self.yoco_cross = layer_idx >= (config.num_hidden_layers // 2 + 2)
            if layer_idx >= (config.num_hidden_layers // 2 + 1):
                config = copy.deepcopy(config)
                config.sliding_window = None
        self.config = config

        self.use_mamba = config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
        if self.use_mamba:
            factory_kwargs = {
                "d_conv": config.mamba_d_conv,
                "d_state": config.mamba_d_state,
                "expand": config.mamba_expand,
                "use_fast_path": False,
                "dtype": None,
            }
            self.attn = Phi4Mamba(
                config.hidden_size,
                layer_idx=layer_idx,
                yoco_cross=self.yoco_cross,
                yoco_kv=self.yoco_mb,
                **factory_kwargs,
            )
        else:
            self.attn = Phi4Attention(config, layer_idx=layer_idx, yoco_cross=self.yoco_cross)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = PHI_NORM_CLASS(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        ssm_output=None,
        yoco_key_values=None,
        causal_mask=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_dtype = hidden_states.dtype
        hidden_states = self.input_layernorm(hidden_states.astype(self.input_layernorm.weight.dtype)).astype(
            hidden_dtype
        )

        if self.use_mamba:
            mamba_mask = attention_mask if attention_mask is not None and attention_mask.ndim == 2 else None
            attn_outputs, ssm_output = self.attn(
                hidden_states=hidden_states,
                inference_params=past_key_value,
                mask=mamba_mask,
                yoco_key_values=ssm_output,
                cache_position=cache_position,
            )
            residual = residual.astype("float32")
            self_attn_weights = None
        else:
            layer_mask = causal_mask
            if (
                self.config.sliding_window is not None
                and self.config.sliding_window[self.layer_idx] is not None
                and layer_mask is not None
            ):
                if past_key_value is not None and cache_position[0] > 0:
                    layer_mask = layer_mask[:, :, :, -self.config.sliding_window[self.layer_idx] :]

            attn_outputs, self_attn_weights, yoco_key_values = self.attn(
                hidden_states=hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                yoco_key_values=yoco_key_values,
            )

        hidden_states = (residual + self.resid_attn_dropout(attn_outputs)).astype(hidden_dtype)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states.astype(self.post_attention_layernorm.weight.dtype)
        ).astype(hidden_dtype)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)
        outputs += (ssm_output,)
        outputs += (yoco_key_values,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Phi4PretrainedModel(PretrainedModel):
    config_class = Phi4Config
    base_model_prefix = "model"
    transpose_weight_keys = [
        "fc1",
        "fc2",
        "Wqkv",
        "out_proj",
        "in_proj",
        "x_proj",
        "dt_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Phi4Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        num_layers = config.num_hidden_layers
        yoco_cross_start = num_layers // 2 + 2

        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.final_layernorm.weight -> {model_prefix}final_layernorm.weight",
            f"model.final_layernorm.bias -> {model_prefix}final_layernorm.bias",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.layers.$LAYER_ID.input_layernorm.bias -> {model_prefix}layers.$LAYER_ID.input_layernorm.bias",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.bias -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.bias",
            f"model.layers.$LAYER_ID.mlp.fc1.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.fc1.weight",
            f"model.layers.$LAYER_ID.mlp.fc2.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.fc2.weight",
        ]

        for layer_id in range(num_layers):
            is_mamba = config.mb_per_layer > 0 and layer_id % config.mb_per_layer == 0
            is_yoco_cross = layer_id >= yoco_cross_start

            if is_mamba and not is_yoco_cross:
                aoa_statements.extend(
                    [
                        f"model.layers.{layer_id}.attn.A_log -> {model_prefix}layers.{layer_id}.attn.A_log",
                        f"model.layers.{layer_id}.attn.D -> {model_prefix}layers.{layer_id}.attn.D",
                        f"model.layers.{layer_id}.attn.conv1d.bias -> {model_prefix}layers.{layer_id}.attn.conv1d.bias",
                        f"model.layers.{layer_id}.attn.conv1d.weight -> {model_prefix}layers.{layer_id}.attn.conv1d.weight",
                        f"model.layers.{layer_id}.attn.dt_proj.bias -> {model_prefix}layers.{layer_id}.attn.dt_proj.bias",
                        f"model.layers.{layer_id}.attn.dt_proj.weight^T -> {model_prefix}layers.{layer_id}.attn.dt_proj.weight",
                        f"model.layers.{layer_id}.attn.in_proj.weight^T -> {model_prefix}layers.{layer_id}.attn.in_proj.weight",
                        f"model.layers.{layer_id}.attn.out_proj.weight^T -> {model_prefix}layers.{layer_id}.attn.out_proj.weight",
                        f"model.layers.{layer_id}.attn.x_proj.weight^T -> {model_prefix}layers.{layer_id}.attn.x_proj.weight",
                    ]
                )
            elif is_mamba and is_yoco_cross:
                aoa_statements.extend(
                    [
                        f"model.layers.{layer_id}.attn.in_proj.weight^T -> {model_prefix}layers.{layer_id}.attn.in_proj.weight",
                        f"model.layers.{layer_id}.attn.out_proj.weight^T -> {model_prefix}layers.{layer_id}.attn.out_proj.weight",
                    ]
                )
            else:
                aoa_statements.extend(
                    [
                        f"model.layers.{layer_id}.attn.Wqkv.weight^T -> {model_prefix}layers.{layer_id}.attn.Wqkv.weight",
                        f"model.layers.{layer_id}.attn.Wqkv.bias -> {model_prefix}layers.{layer_id}.attn.Wqkv.bias",
                        f"model.layers.{layer_id}.attn.out_proj.weight^T -> {model_prefix}layers.{layer_id}.attn.out_proj.weight",
                        f"model.layers.{layer_id}.attn.out_proj.bias -> {model_prefix}layers.{layer_id}.attn.out_proj.bias",
                        f"model.layers.{layer_id}.attn.inner_cross_attn.lambda_q1 -> {model_prefix}layers.{layer_id}.attn.inner_cross_attn.lambda_q1",
                        f"model.layers.{layer_id}.attn.inner_cross_attn.lambda_k1 -> {model_prefix}layers.{layer_id}.attn.inner_cross_attn.lambda_k1",
                        f"model.layers.{layer_id}.attn.inner_cross_attn.lambda_q2 -> {model_prefix}layers.{layer_id}.attn.inner_cross_attn.lambda_q2",
                        f"model.layers.{layer_id}.attn.inner_cross_attn.lambda_k2 -> {model_prefix}layers.{layer_id}.attn.inner_cross_attn.lambda_k2",
                        f"model.layers.{layer_id}.attn.inner_cross_attn.subln.weight -> {model_prefix}layers.{layer_id}.attn.inner_cross_attn.subln.weight",
                    ]
                )

        if cls != cls.base_model_class:
            if config.tie_word_embeddings:
                aoa_statements.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    def _init_weights(self, layer):
        std = self.config.initializer_range
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(paddle.normal(mean=0.0, std=std, shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros(shape=layer.bias.shape))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(paddle.normal(mean=0.0, std=std, shape=layer.weight.shape))
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(paddle.zeros(shape=[layer.weight.shape[-1]]))


@register_base_model
class Phi4Model(Phi4PretrainedModel):
    def __init__(self, config: Phi4Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.LayerList(
            [Phi4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = PHI_NORM_CLASS(config.hidden_size, epsilon=config.layer_norm_eps)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _build_causal_mask(attention_mask, input_shape, past_key_values_length, dtype, sliding_window_size=None):
        if input_shape[-1] <= 1 and past_key_values_length > 0:
            return None
        if attention_mask is not None and attention_mask.ndim == 2:
            expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
            causal = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
            if sliding_window_size is not None:
                window_mask = _make_sliding_window_mask(
                    input_shape, past_key_values_length=past_key_values_length, window_size=sliding_window_size
                )
                combined = causal & window_mask
            else:
                combined = causal
            expanded_attn_mask = expanded_attn_mask & combined
            return paddle.where(expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min).astype(dtype)
        elif attention_mask is not None and attention_mask.ndim >= 3:
            if attention_mask.dtype == paddle.bool:
                return paddle.where(attention_mask, 0.0, paddle.finfo(dtype).min).astype(dtype)
            return attention_mask
        else:
            causal = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
            if sliding_window_size is not None:
                window_mask = _make_sliding_window_mask(
                    input_shape, past_key_values_length=past_key_values_length, window_size=sliding_window_size
                )
                causal = causal & window_mask
            return paddle.where(causal.cast("bool"), 0.0, paddle.finfo(dtype).min).astype(dtype)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = Phi4Cache(
                config=self.config,
                max_batch_size=batch_size,
                max_cache_len=inputs_embeds.shape[1],
                dtype=str(inputs_embeds.dtype).replace("paddle.", ""),
            )

        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_seen_tokens = past_key_values.get_seq_length()
            cache_position = paddle.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], dtype=paddle.int64
            )

        if attention_mask is not None and use_cache and not self.training:
            seq_len = attention_mask.shape[1]
            if seq_len > 1:
                is_padding_right = attention_mask[:, -1].sum().item() != batch_size
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Phi4."
                    )

        hidden_states = inputs_embeds

        past_key_values_length = 0
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            past_key_values_length = past_key_values.get_seq_length()
        input_shape = (batch_size, seq_length)
        causal_mask = self._build_causal_mask(attention_mask, input_shape, past_key_values_length, inputs_embeds.dtype)
        sliding_window_sizes = []
        if self.config.sliding_window:
            for sw in self.config.sliding_window:
                if sw is not None and sw not in sliding_window_sizes:
                    sliding_window_sizes.append(sw)
        sliding_causal_masks = {}
        for sw in sliding_window_sizes:
            sliding_causal_masks[sw] = self._build_causal_mask(
                attention_mask, input_shape, past_key_values_length, inputs_embeds.dtype, sliding_window_size=sw
            )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        ssm_output = None
        yoco_key_values = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.layer_idx
            sw = (
                self.config.sliding_window[layer_idx]
                if self.config.sliding_window is not None and layer_idx < len(self.config.sliding_window)
                else None
            )
            layer_causal_mask = sliding_causal_masks.get(sw, causal_mask) if sw is not None else causal_mask

            if self.gradient_checkpointing and self.training:
                layer_outputs = recompute(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    ssm_output,
                    yoco_key_values,
                    layer_causal_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    ssm_output=ssm_output,
                    yoco_key_values=yoco_key_values,
                    causal_mask=layer_causal_mask,
                )

            hidden_states = layer_outputs[0]
            ssm_output = layer_outputs[1]
            yoco_key_values = layer_outputs[2]

            if output_attentions:
                all_self_attns += (layer_outputs[3],)

        hidden_dtype = hidden_states.dtype
        hidden_states = self.final_layernorm(hidden_states.astype(self.final_layernorm.weight.dtype)).astype(
            hidden_dtype
        )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Phi4ForCausalLM(Phi4PretrainedModel):
    def __init__(self, config: Phi4Config):
        super().__init__(config)
        self.model = Phi4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        loss_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels, loss_mask)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if cache_position is None:
            if past_key_values is None:
                cache_position = paddle.arange(0, input_ids.shape[1], dtype=paddle.int64)
            else:
                past_seen_tokens = 0
                if hasattr(past_key_values, "get_seq_length"):
                    past_seen_tokens = past_key_values.get_seq_length()
                cache_position = paddle.arange(
                    past_seen_tokens, past_seen_tokens + input_ids.shape[1], dtype=paddle.int64
                )

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs
