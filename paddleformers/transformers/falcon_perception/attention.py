# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Dynamic Paddle reference attention; masks never allocate the full pixel Q x K matrix."""
import math
from dataclasses import dataclass, replace

import paddle
import paddle.nn.functional as F


@dataclass
class AttentionMask:
    tokens: paddle.Tensor
    pad_id: int
    eos_id: int
    soi_id: int
    eoi_id: int
    offset: int = 0

    def with_offset(self, offset):
        return replace(self, offset=offset)

    def materialize(self, q_len, kv_len):
        tokens = self.tokens[:, :kv_len]
        eos = tokens == self.eos_id
        # The source forces the final *padded allocation* position to be EOS.
        if kv_len == self.tokens.shape[1]:
            eos = eos.clone()
            eos[:, -1] = True
        doc = paddle.cumsum(eos.astype("int64"), axis=1)
        doc = paddle.concat([paddle.zeros_like(doc[:, :1]), doc[:, :-1]], axis=1)
        nonpad = paddle.cumsum((tokens != self.pad_id).astype("int64"), axis=1) > 0
        starts = paddle.cumsum((tokens == self.soi_id).astype("int64"), axis=1)
        ends = paddle.cumsum((tokens == self.eoi_id).astype("int64"), axis=1)
        image = (starts - ends) > 0
        image_ids = starts * image.astype("int64")
        sl = slice(self.offset, self.offset + q_len)
        q = paddle.arange(self.offset, self.offset + q_len).reshape([1, q_len, 1])
        k = paddle.arange(kv_len).reshape([1, 1, kv_len])
        causal = (q >= k) & (doc[:, sl, None] == doc[:, None, :]) & nonpad[:, None, :]
        visual = image[:, sl, None] & image[:, None, :] & (image_ids[:, sl, None] == image_ids[:, None, :])
        return (causal | visual).unsqueeze(1)


def create_batch_attention_mask(input_batch, *, pad_token_id, eos_token_id, soi_token_id, eoi_token_id, max_len=None):
    return AttentionMask(input_batch, pad_token_id, eos_token_id, soi_token_id, eoi_token_id)


def attention_with_lse(query, key, value, mask):
    """Return ordinary attention plus natural-log LSE, before learned sink scaling."""
    scores = paddle.matmul(query.astype("float32"), key.astype("float32").transpose([0, 1, 3, 2]))
    scores = scores * (query.shape[-1] ** -0.5)
    allowed = mask.materialize(query.shape[2], key.shape[2])
    scores = paddle.where(allowed, scores, paddle.full_like(scores, float("-inf")))
    lse = paddle.logsumexp(scores, axis=-1)
    # A completely masked row has zero output in FlexAttention.
    finite = paddle.isfinite(lse).unsqueeze(-1)
    probs = paddle.where(
        finite,
        paddle.exp(scores - paddle.where(finite, lse.unsqueeze(-1), paddle.zeros_like(lse.unsqueeze(-1)))),
        paddle.zeros_like(scores),
    )
    # FlexAttention casts probabilities to V's dtype before its dot product.
    output = paddle.matmul(probs.astype(value.dtype), value).astype(query.dtype)
    return output, lse


@dataclass
class WindowMask:
    height: int
    width: int
    low_height: int
    low_width: int
    ratio: float = 0.1
    device: object = None
    chunk_size: int = 1024

    def indices(self, begin, end):
        # Match window2d's float32 arithmetic and floor/ceil at image borders.
        q = paddle.arange(begin, end, dtype="int64")
        r = ((q // self.width).astype("float32") + 0.5) / self.height
        c = ((q % self.width).astype("float32") + 0.5) / self.width
        r0 = paddle.floor(paddle.clip(r - self.ratio, 0, 1) * self.low_height).astype("int64")
        r1 = paddle.ceil(paddle.clip(r + self.ratio, 0, 1) * self.low_height).astype("int64")
        c0 = paddle.floor(paddle.clip(c - self.ratio, 0, 1) * self.low_width).astype("int64")
        c1 = paddle.ceil(paddle.clip(c + self.ratio, 0, 1) * self.low_width).astype("int64")
        nr = min(self.low_height, math.ceil(2 * self.ratio * self.low_height) + 2)
        nc = min(self.low_width, math.ceil(2 * self.ratio * self.low_width) + 2)
        dr = paddle.arange(nr).reshape([1, nr, 1])
        dc = paddle.arange(nc).reshape([1, 1, nc])
        rows = r0[:, None, None] + dr
        cols = c0[:, None, None] + dc
        valid = (rows < r1[:, None, None]) & (cols < c1[:, None, None])
        indices = paddle.clip(rows, 0, self.low_height - 1) * self.low_width + paddle.clip(cols, 0, self.low_width - 1)
        return indices.reshape([end - begin, -1]), valid.reshape([end - begin, -1])


def window_attention(query, key, value, mask):
    """Gather only local K/V for each chunk of pixel queries (QK and V widths may differ)."""
    if not isinstance(mask, WindowMask):
        raise TypeError("AnyUp requires a WindowMask.")
    b, heads, length, dim = query.shape
    if length != mask.height * mask.width or key.shape[2] != mask.low_height * mask.low_width:
        raise ValueError("Window grid does not match attention tensor shapes.")
    key = key.transpose([0, 2, 1, 3])
    value = value.transpose([0, 2, 1, 3])
    output = paddle.empty([b, heads, length, value.shape[-1]], dtype=query.dtype)
    for begin in range(0, length, mask.chunk_size):
        end = min(length, begin + mask.chunk_size)
        indices, valid = mask.indices(begin, end)
        n, neighbors = indices.shape
        k = (
            paddle.index_select(key, indices.flatten(), axis=1)
            .reshape([b, n, neighbors, heads, dim])
            .transpose([0, 3, 1, 2, 4])
        )
        v = (
            paddle.index_select(value, indices.flatten(), axis=1)
            .reshape([b, n, neighbors, heads, value.shape[-1]])
            .transpose([0, 3, 1, 2, 4])
        )
        q = query[:, :, begin:end].astype("float32").unsqueeze(-2)
        scores = (q * k.astype("float32")).sum(axis=-1) * (dim**-0.5)
        scores = paddle.where(valid[None, None], scores, paddle.full_like(scores, float("-inf")))
        probs = F.softmax(scores, axis=-1)
        output[:, :, begin:end] = paddle.matmul(probs.astype(v.dtype).unsqueeze(-2), v).squeeze(-2)
    return output
