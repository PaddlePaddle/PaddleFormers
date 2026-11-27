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
"""MiMo model configuration."""

from ..qwen2.configuration import Qwen2Config


class MiMoConfig(Qwen2Config):
    r"""
    This is the configuration class for Xiaomi MiMo text models.

    MiMo-7B uses a Qwen2-compatible decoder backbone and adds one or more MTP
    layers for speculative decoding. The default values match
    `XiaomiMiMo/MiMo-7B-Base`.

    Args:
        num_mtp_layers (int, optional):
            Number of frozen auxiliary layers retained in checkpoints. Defaults
            to the HF `num_nextn_predict_layers` value.
        num_nextn_predict_layers (int, optional, defaults to 1):
            HF checkpoint alias for `num_mtp_layers`. Internally the generic
            training field is kept at zero: auxiliary MTP training is disabled.
    """

    model_type = "mimo"

    def __init__(
        self,
        vocab_size=151680,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=32768,
        max_window_layers=32,
        rms_norm_eps=1e-5,
        rope_theta=640000,
        sliding_window=32768,
        use_mrope=False,
        head_dim=128,
        attention_bias=True,
        num_nextn_predict_layers=1,
        num_mtp_layers=None,
        **kwargs,
    ):
        tokenizer_class = kwargs.get("tokenizer_class", None)
        if isinstance(tokenizer_class, (list, tuple)):
            kwargs["tokenizer_class"] = tokenizer_class[0] if len(tokenizer_class) > 0 else None
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=max_window_layers,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            **kwargs,
        )
        # HF uses this name for stored MTP tensors. PaddleFormers uses the
        # same name for the number of auxiliary objectives to train.
        self.num_mtp_layers = num_nextn_predict_layers if num_mtp_layers is None else num_mtp_layers
        if self.num_mtp_layers < 0:
            raise ValueError("num_mtp_layers must be non-negative")
        self.num_nextn_predict_layers = 0
        self.use_mrope = use_mrope
        self.head_dim = head_dim
        self.attention_bias = attention_bias

    def to_dict(self, saving_file=False):
        output = super().to_dict(saving_file=saving_file)
        if saving_file:
            output["num_nextn_predict_layers"] = output.pop("num_mtp_layers")
        return output


__all__ = ["MiMoConfig"]
