# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
""" Paddle Aixcoder model."""

from functools import partial

import paddle
import paddle.nn as nn

# Import components from Llama since Aixcoder is very similar to Llama
from ..llama.modeling import LlamaLMHead, LlamaModel, LlamaPretrainingCriterion
from ..model_outputs import CausalLMOutputWithCrossAttentions
from ..model_utils import PretrainedModel
from .configuration import AixcoderConfig

__all__ = [
    "AixcoderModel",
    "AixcoderForCausalLM",
    "AixcoderPretrainedModel",
]


class AixcoderPretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AixcoderConfig
    base_model_prefix = "aixcoder"
    _no_split_modules = ["AixcoderDecoderLayer"]
    _supports_sdp = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=std,
                    shape=module.weight.shape,
                ).astype(module.weight.dtype)
            )
            if module.bias is not None:
                module.bias.set_value(paddle.zeros_like(module.bias))
        elif isinstance(module, nn.Embedding):
            module.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=std,
                    shape=module.weight.shape,
                ).astype(module.weight.dtype)
            )
            if module._padding_idx is not None:
                with paddle.no_grad():
                    module.weight[module._padding_idx] = 0

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: AixcoderConfig, is_split=True):
        """Get tensor parallel mappings for splitting/merging model weights across devices."""
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                # Row Linear
                "aixcoder.embed_tokens.weight": partial(fn, is_column=False),
                "aixcoder.layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "aixcoder.layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            if config.tie_word_embeddings:
                base_actions["lm_head.weight"] = partial(fn, is_column=False)
            else:
                base_actions["lm_head.weight"] = partial(fn, is_column=True)

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("aixcoder.embed_tokens.weight")

            # Column Linear
            if hasattr(config, "fuse_attention_qkv") and config.fuse_attention_qkv:
                base_actions["aixcoder.layers.0.self_attn.qkv_proj.weight"] = partial(fn, is_column=True)
            else:
                base_actions["aixcoder.layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
                # if we have enough num_key_value_heads to split, then split it.
                if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                    base_actions["aixcoder.layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                    base_actions["aixcoder.layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

            if hasattr(config, "fuse_attention_ffn") and config.fuse_attention_ffn:
                base_actions["aixcoder.layers.0.mlp.gate_up_fused_proj.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True
                )
            else:
                base_actions["aixcoder.layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions["aixcoder.layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "aixcoder.layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("aixcoder.layers.0.", f"aixcoder.layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings


class AixcoderModel(LlamaModel):
    """
    Aixcoder Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).

    This model inherits from [`PretrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Paddle [`paddle.nn.Layer`](https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/nn/Layer_en.html) subclass. Use it as a regular Paddle Layer and refer to the Paddle
    documentation for all matter related to general usage and behavior.
    """

    config_class = AixcoderConfig

    def __init__(self, config: AixcoderConfig):
        # Call the parent LlamaModel with the config
        super().__init__(config)
        self.config = config


class AixcoderForCausalLM(AixcoderPretrainedModel):
    """
    Aixcoder Model transformer with a language modeling head on top.
    """

    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Use 'aixcoder' as attribute name for the model
        self.aixcoder = AixcoderModel(config)

        if config.tie_word_embeddings:
            self.lm_head = LlamaLMHead(config, embedding_weights=self.aixcoder.embed_tokens.weight, transpose_y=True)
            self.tie_weights()
        else:
            self.lm_head = LlamaLMHead(config)

        self.criterion = LlamaPretrainingCriterion(config)

    def get_input_embeddings(self):
        return self.aixcoder.embed_tokens

    def set_input_embeddings(self, value):
        self.aixcoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.aixcoder = decoder

    def get_decoder(self):
        return self.aixcoder

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        # Handle case where input_ids is None but inputs_embeds is provided
        if input_ids is None and inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
            attention_mask = kwargs.get("attention_mask", None)

            model_inputs = {"inputs_embeds": inputs_embeds}
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )
            return model_inputs

        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
        attention_mask = kwargs.get("attention_mask", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.cat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = paddle.cat(
                [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)], axis=-1
            )

        return model_kwargs

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: AixcoderConfig, is_fuse=False):
        """Get parameter fuse/split mappings for attention and FFN layers."""
        from ..conversion_utils import split_or_fuse_func

        fn = split_or_fuse_func(is_fuse=is_fuse)

        # last key is fused key, other keys are to be fused.
        fuse_qkv_keys = (
            "aixcoder.layers.0.self_attn.q_proj.weight",
            "aixcoder.layers.0.self_attn.k_proj.weight",
            "aixcoder.layers.0.self_attn.v_proj.weight",
            "aixcoder.layers.0.self_attn.qkv_proj.weight",
        )
        fuse_gate_up_keys = (
            "aixcoder.layers.0.mlp.gate_proj.weight",
            "aixcoder.layers.0.mlp.up_proj.weight",
            "aixcoder.layers.0.mlp.gate_up_fused_proj.weight",
        )
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)

        def get_tensor_parallel_fuse_mappings(num_layers, fuse_attention_qkv=False, fuse_attention_ffn=False):
            final_actions = {}
            base_actions = {}

            if fuse_attention_qkv:
                base_actions = {
                    fuse_qkv_keys[-1]: partial(fn, split_nums=[num_heads, num_key_value_heads, num_key_value_heads])
                }
            if fuse_attention_ffn:
                base_actions.update({fuse_gate_up_keys[-1]: fn})

            for key, action in base_actions.items():
                if "aixcoder.layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("aixcoder.layers.0.", f"aixcoder.layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_fuse_mappings(
            config.num_hidden_layers,
            fuse_attention_qkv=fuse_attention_qkv,
            fuse_attention_ffn=getattr(config, "fuse_attention_ffn", False),
        )

        return mappings

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        loss_mask=None,
        **kwargs,
    ):
        r"""
        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AixcoderTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

                Indices can be obtained using [`AixcoderTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
                `past_key_values`).

                If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
                and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
                information on the default strategy.

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.

                [What are position IDs?](../glossary#position-ids)
            past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
                have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
                of shape `(batch_size, sequence_length)`.
            inputs_embeds (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
                model's internal embedding lookup matrix.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Example:

        ```python
        >>> from paddleformers.transformers import AixcoderTokenizer, AixcoderForCausalLM

        >>> model = AixcoderForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AixcoderTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pd")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_new_tokens=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.aixcoder(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
