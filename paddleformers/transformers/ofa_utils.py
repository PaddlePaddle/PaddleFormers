# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = [
    "prepare_qkv_ofa",
    "mha_ofa_forward",
    "encoder_ofa_forward",
    "encoder_layer_ofa_forward",
    "compute_neuron_head_importance",
    "reorder_neuron_head",
]


def prepare_qkv_ofa(self, query, key, value, cache=None):
    q = self.q_proj(query)
    if hasattr(self.q_proj, "fn") and self.q_proj.fn.cur_config["expand_ratio"] is not None:
        self.num_heads = int(self.num_heads * self.q_proj.fn.cur_config["expand_ratio"])
    q = paddle.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
    q = paddle.transpose(x=q, perm=[0, 2, 1, 3])

    if isinstance(cache, self.StaticCache):
        # for encoder-decoder attention in inference and has cached
        k, v = cache.k, cache.v
    else:
        k, v = self.compute_kv(key, value)

    if isinstance(cache, self.Cache):
        # for decoder self-attention in inference
        k = paddle.concat([cache.k, k], axis=2)
        v = paddle.concat([cache.v, v], axis=2)
        cache = self.Cache(k, v)

    return (q, k, v) if cache is None else (q, k, v, cache)


def mha_ofa_forward(self, query, key, value, attn_mask=None, cache=None):
    """
    monkey patch for MultiHeadAttention forward to accept head_mask
    attn_mask[0] = attn_mask, attn_mask[1] = head_mask
    """
    key = query if key is None else key
    value = query if value is None else value
    # compute q ,k ,v
    if cache is None:
        q, k, v = self._prepare_qkv(query, key, value, cache)
    else:
        q, k, v, cache = self._prepare_qkv(query, key, value, cache)

    # scale dot product attention
    product = paddle.matmul(x=q * (self.head_dim**-0.5), y=k, transpose_y=True)
    if attn_mask[0] is not None:
        # TODO(guosheng): support bool mask
        product = product + attn_mask[0]
    weights = F.softmax(product)
    if self.dropout:
        weights = F.dropout(weights, self.dropout, training=self.training, mode="upscale_in_train")

    if attn_mask[1] is not None:
        weights = weights * attn_mask[1]

    out = paddle.matmul(weights, v)

    # combine heads
    out = paddle.transpose(out, perm=[0, 2, 1, 3])
    out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

    # project to output
    out = self.out_proj(out)

    outs = [out]
    if self.need_weights:
        outs.append(weights)
    if cache is not None:
        outs.append(cache)
    if hasattr(self.q_proj, "fn") and self.q_proj.fn.cur_config["expand_ratio"] is not None:
        self.num_heads = int(float(self.num_heads) / self.q_proj.fn.cur_config["expand_ratio"])
    return out if len(outs) == 1 else tuple(outs)


def encoder_ofa_forward(
    self,
    src,
    src_mask=[None, None],
    cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=False,
):
    """
    monkey patch for TransformerEncoder forward to accept head_mask
    attn_mask[0] = attn_mask, attn_mask[1] = head_mask
    """
    output = src
    if src_mask[1] is not None:
        head_mask = src_mask[1]
        if len(head_mask.shape) == 1:
            head_mask = paddle.unsqueeze(paddle.unsqueeze(paddle.unsqueeze(paddle.unsqueeze(head_mask, 0), 0), -1), -1)
            head_mask = paddle.expand(head_mask, shape=[self.num_layers] + head_mask.shape[1:])
        elif len(head_mask.shape) == 2:
            head_mask = paddle.unsqueeze(paddle.unsqueeze(paddle.unsqueeze(head_mask, 1), -1), -1)
    else:
        head_mask = [None] * self.num_layers
    for i, mod in enumerate(self.layers):
        output = mod(output, src_mask=[src_mask[0], head_mask[i]])
    if self.norm is not None:
        output = self.norm(output)

    return output


def encoder_layer_ofa_forward(self, src, src_mask=None, cache=None, output_attentions=False):
    residual = src
    if self.normalize_before:
        src = self.norm1(src)
    # Add cache for encoder for the usage like UniLM
    if cache is None:
        src = self.self_attn(src, src, src, src_mask)
    else:
        src, incremental_cache = self.self_attn(src, src, src, src_mask, cache)

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
    return src if cache is None else (src, incremental_cache)


def reorder_head(layer, index):
    """
    Reorder head weights according index.
    Args:
         layer(paddle.nn.Layer): the instance of `paddle.nn.MultiHeadAttention` layer.
         index(list): the sort indices of multi-head.
    """
    assert isinstance(
        layer, nn.MultiHeadAttention
    ), "layer in reorder_head must be the instance of `paddle.nn.MultiHeadAttention`."
    n, a = layer.num_heads, layer.head_dim
    idx = paddle.reshape(
        paddle.index_select(paddle.reshape(paddle.arange(0, n * a, dtype="int64"), shape=[n, a]), index=index, axis=0),
        shape=[-1],
    )

    def reorder_head_matrix(linearLayer, index, dim=1):
        W = paddle.index_select(linearLayer.weight, index, axis=dim).detach()
        if linearLayer.bias is not None:
            if dim == 0:
                b = paddle.assign(linearLayer.bias).detach()
            else:
                b = paddle.assign(paddle.index_select(linearLayer.bias, index, axis=0)).detach()

        linearLayer.weight.stop_gradient = True
        linearLayer.weight.set_value(W)
        linearLayer.weight.stop_gradient = False
        if linearLayer.bias is not None:
            linearLayer.bias.stop_gradient = True
            linearLayer.bias.set_value(b)
            linearLayer.bias.stop_gradient = False

    reorder_head_matrix(layer.q_proj.fn if hasattr(layer.q_proj, "fn") else layer.q_proj, idx)
    reorder_head_matrix(layer.k_proj.fn if hasattr(layer.k_proj, "fn") else layer.k_proj, idx)
    reorder_head_matrix(layer.v_proj.fn if hasattr(layer.v_proj, "fn") else layer.v_proj, idx)
    reorder_head_matrix(layer.out_proj.fn if hasattr(layer.out_proj, "fn") else layer.out_proj, idx, dim=0)


def reorder_neuron(layer, index, dim=0):
    """
    Reorder feed-forward weights according index.
    Args:
         layer(paddle.nn.Layer): the instance of `paddle.nn.Linear` layer.
         index(list): the sort indices of feed-forward.
         dim(int): select weights according to the dim.
    """
    linearLayer = layer.fn if hasattr(layer, "fn") else layer
    W = paddle.index_select(linearLayer.weight, index, axis=dim).detach()
    if linearLayer.bias is not None:
        if dim == 0:
            b = paddle.assign(linearLayer.bias).detach()
        else:
            b = paddle.assign(paddle.index_select(linearLayer.bias, index, axis=0)).detach()
    linearLayer.weight.stop_gradient = True
    linearLayer.weight.set_value(W)
    linearLayer.weight.stop_gradient = False

    if linearLayer.bias is not None:
        linearLayer.bias.stop_gradient = True
        linearLayer.bias.set_value(b)
        linearLayer.bias.stop_gradient = False


def reorder_neuron_head(model, head_importance, neuron_importance):
    """
    Reorders weights according head importance and neuron importance
    """
    # Reorders heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # Reorders heads
        idx = paddle.argsort(head_importance[layer], descending=True)
        reorder_head(model.base_model.encoder.layers[layer].self_attn, idx)
        # Reorders neurons
        idx = paddle.argsort(paddle.to_tensor(current_importance), descending=True)
        reorder_neuron(model.base_model.encoder.layers[layer].linear1.fn, idx, dim=1)

        reorder_neuron(model.base_model.encoder.layers[layer].linear2.fn, idx, dim=0)


def compute_neuron_head_importance(
    model,
    data_loader,
    num_layers,
    num_heads,
    loss_fct=nn.loss.CrossEntropyLoss(),
    intermediate_name="linear1",
    output_name="linear2",
    label_names=None,
):
    """
    Computes the importance of multi-head attention and feed-forward  neuron in
    each transformer layer.

    Args:
        model(paddle.nn.Layer):
            The instance of transformer model.
        data_loader (DataLoader):
            An iterable data loader is used for evaluate. An instance of
            `paddle.io.Dataloader`.
        num_layers (int):
            Number of transformer layers.
        num_heads (int):
            Number of heads in each multi-head attention.
        loss_fct (Loss|optional):
            Loss function can be a `paddle.nn.Layer` instance. Default: `nn.loss.CrossEntropyLoss()`.
        intermediate_name (str|optional):
            The name of intermediate `Linear` layer in feed-forward.
            Defaults to `linear1`.
        output_name (str|optional):
            The name of output `Linear` layer in feed-forward.
            Defaults to `linear2`.
    """
    head_importance = paddle.zeros(shape=[num_layers, num_heads], dtype="float32")
    head_mask = paddle.ones(shape=[num_layers, num_heads], dtype="float32")
    head_mask.stop_gradient = False

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []

    for name, w in model.named_parameters():
        if intermediate_name in name:
            if len(w.shape) > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if output_name in name:
            if len(w.shape) > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(np.zeros(shape=[w.shape[1]], dtype="float32"))

    for i, batch in enumerate(data_loader):
        labels = None
        if isinstance(batch, list):
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids, attention_mask=[None, head_mask])
        else:
            if label_names is not None:
                labels = []
                for label in label_names:
                    labels.append(batch.pop(label))
                labels = tuple(labels)
            elif "labels" in batch:
                labels = batch.pop("labels")
                # For token cls tasks
                for key in ("length", "seq_len"):
                    if key in batch:
                        batch.pop(key)
            elif "start_positions" in batch and "end_positions" in batch:
                labels = (batch.pop("start_positions"), batch.pop("end_positions"))

            batch["attention_mask"] = [None, head_mask]
            logits = model(**batch)

        if loss_fct is not None:
            loss = loss_fct(logits, labels)
        else:
            raise NotImplementedError(
                "Model to be compressed is an instance of a custom class, "
                "so function `loss_fct(logits, labels)` should "
                "be implemented, and it should return a single float for precision "
                "value, such as acc."
            )

        loss.backward()
        head_importance += paddle.abs(paddle.to_tensor(head_mask.gradient()))
        for w1, b1, w2, current_importance in zip(
            intermediate_weight, intermediate_bias, output_weight, neuron_importance
        ):
            current_importance += np.abs((np.sum(w1.numpy() * w1.gradient(), axis=0) + b1.numpy() * b1.gradient()))
            current_importance += np.abs(np.sum(w2.numpy() * w2.gradient(), axis=1))
    return head_importance, neuron_importance
