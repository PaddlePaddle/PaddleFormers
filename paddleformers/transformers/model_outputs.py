# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import functools
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Optional, Tuple

import numpy as np
import paddle
from paddle import Tensor
from paddle.distributed.fleet.utils import recompute
from paddle.nn import MultiHeadAttention
from paddle.nn.layer.transformer import _convert_attention_mask

from .utils import adapt_stale_fwd_patch


def tuple_output(outputs: Tuple[Tensor], loss: Optional[Tensor] = None):
    """re-construct the outputs with one method which contains the simple logic

    Args:
        outputs (Tuple[Tensor]): the source of the outputs
        loss (Optional[Tensor], optional): the loss of the model. Defaults to None.
    """
    if loss is not None:
        outputs = (loss,) + outputs
    if len(outputs) == 1:
        return outputs[0]
    return outputs


def convert_encoder_output(encoder_output):
    """
    Convert encoder_output from tuple to class:`~paddleformers.transformers.model_outputs.BaseModelOutput`.

    Args:
        encoder_output (tuple or ModelOutput):
            The output of the encoder, a tuple consists `last_hidden_state`, `hidden_states`(optional), `attentions`(optional).
            The data type of `last_hidden_state` is float32 and its shape is [batch_size, sequence_length, hidden_size].
    """
    return BaseModelOutput(
        last_hidden_state=encoder_output[0],
        hidden_states=encoder_output[1] if len(encoder_output) > 1 else None,
        attentions=encoder_output[2] if len(encoder_output) > 2 else None,
    )


def layer_init_wrapper(func):
    @functools.wraps(func)
    def _impl(self, *args, **kwargs):
        enable_recompute = kwargs.pop("enable_recompute", False)
        func(self, *args, **kwargs)
        if paddle.in_dynamic_mode():
            self.enable_recompute = enable_recompute
        else:
            self.enable_recompute = False

    return _impl


@paddle.jit.not_to_static
def _transformer_encoder_layer_fwd(self, src, src_mask=None, cache=None, output_attentions=False):
    self.self_attn.need_weights = output_attentions
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    residual = src
    if self.normalize_before:
        src = self.norm1(src)

    attn_outputs = self.self_attn(src, src, src, src_mask, cache)
    if isinstance(attn_outputs, tuple):
        src = attn_outputs[0]
        outputs = attn_outputs[1:]
    else:
        src = attn_outputs
        outputs = None

    src = residual + self.dropout1(src)
    if not self.normalize_before:
        src = self.norm1(src)

    residual = src
    if self.normalize_before:
        src = self.norm2(src)
    src = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = residual + self.dropout2(src)
    if not self.normalize_before:
        src = self.norm2(src)

    return src if outputs is None else ((src,) + outputs[::-1])  # hidden_states, cache, attentions


@paddle.jit.not_to_static
def _transformer_decoder_layer_fwd(
    self,
    tgt,
    memory,
    tgt_mask=None,
    memory_mask=None,
    cache=None,
    output_attentions=False,
):
    residual = tgt

    # self attention
    self.self_attn.need_weights = output_attentions
    tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)

    if self.normalize_before:
        tgt = self.norm1(tgt)

    self_attn_outputs = self.self_attn(tgt, tgt, tgt, tgt_mask, cache[0] if cache else None)
    # self_attn_outputs = (tgt, attn_weights, incremental_cache) or only tgt
    if isinstance(self_attn_outputs, type(tgt)):
        tgt = self_attn_outputs
    else:
        tgt = self_attn_outputs[0]
        if output_attentions:
            self_attn_weights = self_attn_outputs[1]
        if cache:
            incremental_cache = self_attn_outputs[-1]

    tgt = residual + self.dropout1(tgt)
    if not self.normalize_before:
        tgt = self.norm1(tgt)

    residual = tgt

    # cross attention
    if memory is not None:
        self.cross_attn.need_weights = output_attentions
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        if self.normalize_before:
            tgt = self.norm2(tgt)

        cross_attn_outputs = self.cross_attn(tgt, memory, memory, memory_mask, cache[1] if cache else None)
        if isinstance(cross_attn_outputs, type(tgt)):
            tgt = cross_attn_outputs
        else:
            tgt = cross_attn_outputs[0]
            if output_attentions:
                cross_attn_weights = cross_attn_outputs[1]
            if cache:
                static_cache = cross_attn_outputs[-1]

        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt

    if self.normalize_before:
        tgt = self.norm3(tgt)
    tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = residual + self.dropout3(tgt)
    if not self.normalize_before:
        tgt = self.norm3(tgt)

    if not output_attentions and cache is None:
        return tgt
    else:
        outputs = (tgt,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights if memory is not None else None)
        if cache:
            outputs += ((incremental_cache, static_cache if memory is not None else None),)
        return outputs


@paddle.jit.not_to_static
def _transformer_decoder_fwd(
    self,
    tgt,
    memory=None,
    tgt_mask=None,
    memory_mask=None,
    cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=False,
):
    tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
    if memory is not None:
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

    new_caches = [] if cache else None
    all_hidden_states = [tgt] if output_hidden_states else None
    all_self_attns = [] if output_attentions else None
    all_cross_attns = [] if output_attentions else None

    for i, mod in enumerate(self.layers):
        if cache is None:
            # if output has no gradient, recompute is unnecessary
            memory_stop_gradient = memory is not None and memory.stop_gradient
            has_gradient = (not tgt.stop_gradient) or (not memory_stop_gradient)
            if self.enable_recompute and has_gradient:
                outputs = recompute(mod, tgt, memory, tgt_mask, memory_mask, None, output_attentions)
            else:
                outputs = mod(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    cache=None,
                    output_attentions=output_attentions,
                )
        else:
            outputs = mod(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                cache=cache[i] if cache else None,
                output_attentions=output_attentions,
            )
        if isinstance(outputs, type(tgt)):
            tgt = outputs
        else:
            tgt = outputs[0]
        if cache:
            new_caches.append(outputs[-1])
        if output_attentions:
            all_self_attns.append(outputs[1])
            all_cross_attns.append(outputs[2])
        if output_hidden_states:
            all_hidden_states.append(tgt)

    if self.norm is not None:
        tgt = self.norm(tgt)
        if output_hidden_states:
            all_hidden_states[-1] = tgt

    if not return_dict:
        if isinstance(outputs, type(tgt)):
            return tgt

        temp_list = [
            tgt,
            new_caches if cache else None,
            all_hidden_states,
            all_self_attns,
            all_cross_attns,
        ]
        return tuple(v for v in temp_list if v is not None)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=tgt,
        past_key_values=new_caches,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        cross_attentions=all_cross_attns,
    )


@paddle.jit.not_to_static
def _transformer_encoder_fwd(
    self, src, src_mask=None, cache=None, output_attentions=False, output_hidden_states=False, return_dict=False
):
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    output = src
    # To get cache from None when use_cache is True, which is compatible with HF
    # while HF requires decoder. The implementation here uses cache update in the
    # MultiHeadAttention not so efficiently, and maybe optimize it later.
    if cache is None and getattr(self, "_use_cache", False):
        cache = [tuple(self.layers[0].gen_cache(src))] * len(self.layers)
    # To be compatible with `TransformerEncoder.forward`, `_use_cache` defaults
    # to True when cache is not None.
    new_caches = [] if cache is not None and getattr(self, "_use_cache", True) else None
    all_attentions = [] if output_attentions else None
    # NOTE: Also includes embedding output which is same as HF.
    all_hidden_states = [output] if output_hidden_states else None
    for i, mod in enumerate(self.layers):
        # if output has no gradient, recompute is unnecessary
        has_gradient = not output.stop_gradient
        if self.enable_recompute and has_gradient:
            # Note: recompute do not support pass as **kwargs yet.
            layer_outputs = recompute(
                mod,
                output,
                src_mask,
                None
                if cache is None
                else cache[i]
                if isinstance(cache[i], MultiHeadAttention.Cache)
                else MultiHeadAttention.Cache(*cache[i]),
                output_attentions,
            )
        else:
            layer_outputs = mod(
                output,
                src_mask=src_mask,
                cache=None
                if cache is None
                else cache[i]
                if isinstance(cache[i], MultiHeadAttention.Cache)
                else MultiHeadAttention.Cache(*cache[i]),
                output_attentions=output_attentions,
            )

        if isinstance(layer_outputs, tuple):
            output = layer_outputs[0]
            outputs = layer_outputs[1:]
        else:
            output = layer_outputs
            outputs = None

        if output_hidden_states:
            all_hidden_states.append(output)
        if output_attentions:
            all_attentions.append(outputs[-1])
        if new_caches is not None:
            new_caches.append(outputs[0] if isinstance(cache[i], MultiHeadAttention.Cache) else (tuple(outputs[0])))

    if self.norm is not None:
        output = self.norm(output)

        if output_hidden_states:
            all_hidden_states[-1] = output

    if not return_dict:
        outputs = tuple(
            tuple(v) if isinstance(v, list) else v
            for v in [
                output,
                new_caches,
                all_hidden_states,
                all_attentions,
            ]
            if v is not None
        )
        if len(outputs) == 1:
            return output
        else:
            return outputs

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=output,
        past_key_values=new_caches,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
    )


_transformer_encoder_fwd.__name__ = "forward"
_transformer_encoder_layer_fwd.__name__ = "forward"
# patches of paddle.nn.Transformer to get all hidden_states and attentions
paddle.nn.TransformerEncoderLayer.forward = _transformer_encoder_layer_fwd
paddle.nn.TransformerDecoderLayer.forward = _transformer_decoder_layer_fwd
paddle.nn.TransformerEncoder.forward = _transformer_encoder_fwd
paddle.nn.TransformerDecoder.forward = _transformer_decoder_fwd

_encoder_init = paddle.nn.TransformerEncoder.__init__
_decoder_init = paddle.nn.TransformerDecoder.__init__
paddle.nn.TransformerEncoder.__init__ = layer_init_wrapper(_encoder_init)
paddle.nn.TransformerDecoder.__init__ = layer_init_wrapper(_decoder_init)


def _get_wrap_setattr(cls):
    def _wrap_setattr(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(cls, self).__setattr__(name, value)

    return _wrap_setattr


paddle.nn.TransformerEncoderLayer.__setattr__ = functools.wraps(paddle.nn.TransformerEncoderLayer.__setattr__)(
    _get_wrap_setattr(paddle.nn.TransformerEncoderLayer)
)
paddle.nn.TransformerEncoder.__setattr__ = functools.wraps(paddle.nn.TransformerEncoder.__setattr__)(
    _get_wrap_setattr(paddle.nn.TransformerEncoder)
)
paddle.nn.TransformerDecoder.__setattr__ = functools.wraps(paddle.nn.TransformerDecoder.__setattr__)(
    _get_wrap_setattr(paddle.nn.TransformerDecoder)
)


def is_tensor(x):
    if isinstance(x, paddle.Tensor):
        return True

    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # note(guosheng): Convert list to tuple automatically, and better to
        # check if it is frozen.
        # assert not getattr(self, dataclasses._PARAMS).frozen
        for f in class_fields:
            value = getattr(self, f.name)
            if isinstance(value, list):
                setattr(self, f.name, tuple(value))

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        # try to fix: https://github.com/PaddlePaddle/PaddleNLP/issues/3355
        # when trying to get the keys of `OrderedDict`, `keys` method return empty values.
        # TODO(wj-Mcat): this bug should be fixed in Paddle framework
        tuples = ()
        for field in fields(self):
            if getattr(self, field.name, None) is None:
                continue
            tuples = tuples + (getattr(self, field.name),)

        return tuples


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BaseModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`paddle.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: paddle.Tensor = None
    pooler_output: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        cum_offsets (`tuple(paddle.Tensor)`, *optional*, needed when `return_full_hidden_states=True`:
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, 1)`.

            Offset of the current batch.
    """

    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None
    cum_offsets: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BaseModelOutputWithPastAndMTP(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mtp_outputs (`tuple(paddle.Tensor)`, *optional*):
            MTP Layers outputs, used to compute the mtp loss.
            heads.
    """

    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    mtp_outputs: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`paddle.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: paddle.Tensor = None
    pooler_output: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`paddle.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    start_logits: paddle.Tensor = None
    end_logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class MultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        loss (`paddle.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`paddle.Tensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `paddle.Tensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `paddle.Tensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class Seq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (`paddle.Tensor`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model, whose shape is `(batch_size, Sequence_length, hidden_size)`.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(paddle.Tensor))`, optional):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Returned when `use_cache=True` is passed or when `config.use_cache=True`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`.

            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Returned when `output_attentions=True` is passed or when `config.output_attentions=True`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Returned when `output_attentions=True` is passed or when `config.output_attentions=True`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`paddle.Tensor`, optional):
            Sequence of hidden-states at the output of the last layer of the encoder of the model whose shape is `(batch_size, sequence_length, hidden_size)`,
        encoder_hidden_states (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`.

            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Returned when `output_attentions=True` is passed or when `config.output_attentions=True`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    decoder_attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None
    encoder_last_hidden_state: Optional[paddle.Tensor] = None
    encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    encoder_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`paddle.Tensor`, optional):
            Language modeling loss whose shape is `(1,)`. Returned when `labels` is provided.
        logits (`paddle.Tensor`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) whose shape is `(batch_size, sequence_length, config.vocab_size)`).
        past_key_values (`tuple(tuple(paddle.Tensor))`, optional):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Returned when `use_cache=True` is passed or when `config.use_cache=True`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Returned when `output_attentions=True` is passed or when `config.output_attentions=True`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Returned when `output_attentions=True` is passed or when `config.output_attentions=True`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`paddle.Tensor`, optional):
            Sequence of hidden-states at the output of the last layer of the encoder of the model whose shape is `(batch_size, sequence_length, hidden_size)`.
        encoder_hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Returned when `output_attentions=True` is passed or when `config.output_attentions=True`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    decoder_attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None
    encoder_last_hidden_state: Optional[paddle.Tensor] = None
    encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    encoder_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class Seq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.
    Args:
        loss (`paddle.Tensor` ,optional):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
            A Tensor of shape `(1,)`, returned when `labels` is provided.
        start_logits (`paddle.Tensor`):
            Span-start scores (before SoftMax). Tensor of shape `(batch_size, sequence_length)`).
        end_logits (`paddle.Tensor`):
            Span-end scores (before SoftMax). Tensor of shape `(batch_size, sequence_length)`).
        past_key_values (`tuple(tuple(paddle.Tensor))`, optional):
            Tuple of `tuple(paddle.Tensor)` of length `n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Returned when `use_cache=True` is passed.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Returned when `output_hidden_states=True` is passed.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Returned when `output_attentions=True` is passed.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Returned when `output_attentions=True` is passed.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`paddle.Tensor` optional):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
            Tensor of shape `(batch_size, sequence_length, hidden_size)`.
        encoder_hidden_states (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Returned when `output_hidden_states=True` is passed.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Returned when `output_attentions=True` is passed.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[paddle.Tensor] = None
    start_logits: paddle.Tensor = None
    end_logits: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    decoder_attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None
    encoder_last_hidden_state: Optional[paddle.Tensor] = None
    encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    encoder_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class Seq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.
    Args:
        loss (`paddle.Tensor` optional):
            Classification (or regression if config.num_labels==1) loss of shape `(1,)`. Returned when `label` is provided).
        logits (`paddle.Tensor`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax) of shape `(batch_size, config.num_labels)`
        past_key_values (`tuple(tuple(paddle.Tensor))`, optional):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Returned when `use_cache=True` is passed.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Returned when `output_hidden_states=True` is passed.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Returned when `output_attentions=True` is passed.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Returned when `output_attentions=True` is passed.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`paddle.Tensor`, optional):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
            Tensor of shape `(batch_size, sequence_length, hidden_size)`.
        encoder_hidden_states (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Returned when `output_hidden_states=True` is passed.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Returned when `output_attentions=True` is passed.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    decoder_attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None
    encoder_last_hidden_state: Optional[paddle.Tensor] = None
    encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    encoder_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class SequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (`paddle.Tensor`, optional):
            Classification (or regression if config.num_labels==1) loss whose shape is `(1,)`.
            Returned when `labels` is provided.
        logits (`paddle.Tensor`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax)
            whose shape is `(batch_size, num_labels)`
        past_key_values (`tuple(tuple(paddle.Tensor))`, optional):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
            Returned when `use_cache=True` is passed or when `config.use_cache=True`).
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`).
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, optional):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Returned when `output_attentions=True` is passed or when `config.output_attentions=True`).
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BackboneOutput(ModelOutput):
    """
    Base class for outputs of backbones.

    Args:
        feature_maps (`tuple(paddle.Tensor)` of shape `(batch_size, num_channels, height, width)`):
            Feature maps of the stages.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, num_channels, height, width)`,
            depending on the backbone.

            Hidden-states of the model at the output of each stage plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Only applicable if the backbone uses attention.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    feature_maps: Tuple[paddle.Tensor] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`paddle.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: paddle.Tensor = None
    pooler_output: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class ImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`paddle.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class DepthEstimatorOutput(ModelOutput):
    """
    Base class for outputs of depth estimation models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        predicted_depth (`paddle.Tensor` of shape `(batch_size, height, width)`):
            Predicted depth for each pixel.

        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    predicted_depth: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class SemanticSegmenterOutput(ModelOutput):
    """
    Base class for outputs of semantic segmentation models.
    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`paddle.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.
            <Tip warning={true}>
            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.
            </Tip>
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class Seq2SeqSpectrogramOutput(ModelOutput):
    """
    Base class for sequence-to-sequence spectrogram outputs.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Spectrogram generation loss.
        spectrogram (`paddle.Tensor` of shape `(batch_size, sequence_length, num_bins)`):
            The predicted spectrogram.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[paddle.Tensor] = None
    spectrogram: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    decoder_attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None
    encoder_last_hidden_state: Optional[paddle.Tensor] = None
    encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    encoder_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class MoEModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        router_logits (`tuple(paddle.Tensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.
    """

    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    router_logits: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).

        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss (`paddle.Tensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.

        router_logits (`tuple(paddle.Tensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.

        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    aux_loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    router_logits: Optional[Tuple[paddle.Tensor]] = None
