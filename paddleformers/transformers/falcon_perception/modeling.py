# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Adapted from TII Falcon Perception (Apache-2.0).
import math
from pathlib import Path

import einops as E
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import Tensor as T
from paddle import nn
from PIL import Image
from pycocotools import mask as mask_utils

from ..auto.tokenizer import AutoTokenizer
from .anyup import AnyUp
from .attention import WindowMask, attention_with_lse, create_batch_attention_mask
from .configuration import FalconPerceptionConfig
from .norm import RMSNorm, rms_norm
from .processing import load_image, process_batch
from .rope import (
    apply_3d_rotary_emb,
    apply_golden_freqs_cis_to_visual_pos,
    precompute_freqs_cis,
)


class FourierEncoder(nn.Layer):
    def __init__(self, in_dim: int, feat_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Linear(in_dim, feat_dim // 2, bias_attr=False)
        self.transform = nn.Linear(feat_dim, out_dim, bias_attr=False)

    def forward(self, x):
        f = 2 * math.pi * self.embed(x)
        f = paddle.cat([f.cos(), f.sin()], axis=-1)
        return self.transform(f)


class BboxDecoder(nn.Layer):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias_attr=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias_attr=False)

    def forward(self, x: T) -> T:
        return self.w2(F.relu(self.w1(x)).square())


class SegmDecoder(nn.Layer):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.LayerList([nn.Linear(in_dim, in_dim) for _ in range(num_layers - 1)])
        self.pixel_layer = nn.Linear(in_dim, out_dim, bias_attr=False)

    def forward(self, x) -> paddle.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x)).square()
        return self.pixel_layer(x)


def repeat_kv(x: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    B, S, H, D = x.shape
    if n_rep == 1:
        return x
    return paddle.unsqueeze(x, axis=3).expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)


class Attention(nn.Layer):
    def __init__(self, config: FalconPerceptionConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.n_rep = config.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim or config.dim // config.n_heads
        self.q_dim = config.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.wqkv = nn.Linear(config.dim, self.q_dim + 2 * self.kv_dim, bias_attr=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias_attr=False)
        self.sinks = nn.Parameter(paddle.zeros((config.n_heads,)))

    def _pre_attention_qkv(self, x) -> tuple[T, T, T]:
        qkv = self.wqkv(rms_norm(x, (x.size(-1),)))
        xq, xk, xv = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], axis=-1)
        xq = E.rearrange(xq, "b s (h d) -> b s h d", d=self.head_dim)
        xk = E.rearrange(xk, "b s (h d) -> b s h d", d=self.head_dim)
        xv = E.rearrange(xv, "b s (h d) -> b s h d", d=self.head_dim)
        xq = rms_norm(xq, (xq.size(-1),))
        xk = rms_norm(xk, (xk.size(-1),))
        xk = repeat_kv(xk, n_rep=self.n_rep)
        xv = repeat_kv(xv, n_rep=self.n_rep)
        return (xq, xk, xv)

    def _post_attention(self, output: T, lse: T) -> T:
        sinks_BHS = self.sinks.view(1, -1, 1)
        sink_scale = paddle.sigmoid(lse - sinks_BHS)
        output = (output * sink_scale.unsqueeze(-1)).to(output.dtype)
        output = output.permute(0, 2, 1, 3).contiguous().flatten(2)
        return self.wo(output)

    def forward(
        self,
        x: T,
        attention_masks,
        freqs_cis: T,
        freqs_cis_2d: T | None = None,
        pos_hw: T | None = None,
        kv_cache=None,
        input_pos=None,
        batch_idx=None,
        flex_attn_kernel_options=None,
    ):
        xq, xk, xv = self._pre_attention_qkv(x)
        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d, pos_hw)
        xq = E.rearrange(xq, "b s h d -> b h s d")
        xk = E.rearrange(xk, "b s h d -> b h s d")
        xv = E.rearrange(xv, "b s h d -> b h s d")
        xk, xv = kv_cache.insert_kv(self.layer_id, xk, xv, input_pos=input_pos, batch_idx=batch_idx)
        output, lse = attention_with_lse(xq, xk, xv, attention_masks)
        return self._post_attention(output, lse)


def squared_relu_gate(packed: T, hidden_dim: int) -> T:
    # Match the original kernel: the square is rounded to the input dtype
    # before multiplication by the interleaved up branch.
    gate = F.relu(packed[..., 0::2])
    return gate.square() * packed[..., 1::2]


class FeedForward(nn.Layer):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias_attr=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias_attr=False)
        self.hidden_dim = hidden_dim

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = rms_norm(x, (x.size(-1),))
        w13_out = self.w13(x)
        return self.w2(squared_relu_gate(w13_out, self.hidden_dim))


class TransformerBlock(nn.Layer):
    def __init__(self, layer_id: int, config: FalconPerceptionConfig):
        super().__init__()
        self.attention = Attention(config, layer_id)
        self.feed_forward = FeedForward(config.dim, config.ffn_dim)

    def forward(
        self,
        x: T,
        freqs_cis: T,
        freqs_cis_2d: T | None = None,
        pos_hw: T | None = None,
        attention_masks=None,
        kv_cache=None,
        input_pos=None,
        batch_idx=None,
        flex_attn_kernel_options=None,
    ):
        B, S, D = x.shape
        x = x + self.attention(
            x,
            freqs_cis=freqs_cis,
            freqs_cis_2d=freqs_cis_2d,
            pos_hw=pos_hw,
            attention_masks=attention_masks,
            kv_cache=kv_cache,
            input_pos=input_pos,
            batch_idx=batch_idx,
            flex_attn_kernel_options=flex_attn_kernel_options,
        )
        out = x + self.feed_forward(x)
        return out.reshape(B, S, D)


class KVCache:
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, num_layers):
        self.kv_shape = (num_layers, 2, max_batch_size, n_heads, max_seq_length, head_dim)
        self.kv_cache = None
        self.pos = 0
        self.pos_t: T | None = None

    def reset(self):
        self.pos = 0
        self.pos_t = None

    def get_pos(self):
        return self.pos

    def set_pos_t(self, pos_t):
        self.pos_t = pos_t

    def increment_and_get_pos_t(self):
        assert self.pos_t is not None
        self.pos_t += 1
        return self.pos_t

    def insert_kv(self, layer_id: int, k: T, v: T, **kwargs):
        del kwargs
        assert self.pos_t is not None
        if self.kv_cache is None:
            self.kv_cache = paddle.empty(self.kv_shape, dtype=k.dtype, device=k.place)
        B, H, T_add, D = k.size()
        t0, t1 = (self.pos, self.pos + T_add)
        self.kv_cache[layer_id, 0, :, :, t0:t1] = k
        self.kv_cache[layer_id, 1, :, :, t0:t1] = v
        key_view = self.kv_cache[layer_id, 0, :, :, :t1]
        value_view = self.kv_cache[layer_id, 1, :, :, :t1]
        if layer_id == self.kv_cache.size(0) - 1:
            self.pos = t1
        return (key_view, value_view)


@paddle.no_grad()
def sample_next_token(logits, rng, temperature=0.0, top_k=None):
    assert temperature >= 0.0
    if temperature == 0.0:
        return paddle.argmax(logits, axis=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = paddle.topk(logits, k, axis=-1)
        vals = vals / temperature
        probs = F.softmax(vals, axis=-1)
        choice = paddle.multinomial(
            probs,
            num_samples=1,
        )
        return idx.gather(1, choice)
    logits = logits / temperature
    probs = F.softmax(logits, axis=-1)
    return paddle.multinomial(
        probs,
        num_samples=1,
    )


class FalconPerceptionForSegmentation(nn.Layer):
    @property
    def place(self):
        return self.tok_embeddings.weight.place

    @property
    def dtype(self):
        return self.tok_embeddings.weight.dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, dtype="float32", **kwargs):
        """Load the original local HF safetensors, checking every key and shape.

        This single-card loader converts only actual Linear weights. No PyTorch
        installation or intermediate converted checkpoint is required.
        """
        from .loading import load_local_model

        return load_local_model(cls, pretrained_model_name_or_path, dtype=dtype, **kwargs)

    @classmethod
    def _from_config(cls, config, *, dtype=None, **kwargs):
        """Construct via AutoModel.from_config, without loading a checkpoint."""
        if kwargs:
            raise TypeError(f"Unsupported Falcon construction options: {sorted(kwargs)}")
        if not isinstance(config, FalconPerceptionConfig):
            raise TypeError("Falcon requires a FalconPerceptionConfig.")
        dtype = dtype or config.dtype or "float32"
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("Supported dtypes: float32, bfloat16 (experimental).")
        previous = paddle.get_default_dtype()
        try:
            paddle.set_default_dtype("float32")
            model = cls(config)
        finally:
            paddle.set_default_dtype(previous)
        if dtype != "float32":
            model.to(dtype=dtype)
        return model

    config_class = FalconPerceptionConfig
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, config: FalconPerceptionConfig):
        super().__init__()
        self.config = config
        img_in_dim = config.temporal_patch_size * config.spatial_patch_size**2 * config.channel_size
        self.img_projector = nn.Linear(img_in_dim, config.dim, bias_attr=False)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.LayerDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, config)
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias_attr=False)
        self.coord_encoder = FourierEncoder(2, config.coord_enc_dim, config.dim)
        self.coord_decoder = BboxDecoder(config.dim, config.coord_dec_dim, config.coord_out_dim)
        self.size_encoder = FourierEncoder(2, config.size_enc_dim, config.dim)
        self.size_decoder = BboxDecoder(config.dim, config.size_dec_dim, config.size_out_dim)
        if config.do_segmentation:
            self.itok_upsampler = AnyUp()
            self.proj_segm = SegmDecoder(config.dim, config.segm_out_dim, config.num_segm_layers)
            self.conv_segm = nn.Conv2D(config.dim, config.segm_out_dim, kernel_size=3, padding=1)
        rope_dim = config.head_dim // 2
        freqs_cis = precompute_freqs_cis(rope_dim, config.max_seq_len, config.rope_theta)
        freqs_cis_golden = paddle.zeros((config.n_heads, rope_dim // 2, 2), dtype=paddle.float32)
        self.register_buffer("freqs_cis", freqs_cis, persistable=False)
        self.register_buffer("freqs_cis_golden", freqs_cis_golden, persistable=True)
        self._weights_fused = False

    def _ensure_device_buffers(self):
        """Keep non-persistent rotary buffers on the model device."""
        if self._weights_fused:
            return
        device = self.tok_embeddings.weight.place
        c = self.config
        rope_dim = c.head_dim // 2
        freqs_cis = precompute_freqs_cis(rope_dim, c.max_seq_len, c.rope_theta).to(device)
        self.register_buffer("freqs_cis", freqs_cis, persistable=False)
        if self.freqs_cis_golden.place != device:
            self.freqs_cis_golden = self.freqs_cis_golden.to(device)
        self._weights_fused = True

    def _get_tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            import os

            path = self.config._name_or_path
            is_local = os.path.exists(path)
            self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=is_local, trust_remote_code=True)
            for token_name, token in self._tokenizer.special_tokens_map.items():
                if isinstance(token, str):
                    setattr(self._tokenizer, token_name, token)
                    setattr(self._tokenizer, token_name + "_id", self._tokenizer.convert_tokens_to_ids(token))
        return self._tokenizer

    def get_attention_mask(self, input_batch: T, max_len: int | None = None):
        return create_batch_attention_mask(
            input_batch,
            pad_token_id=getattr(self, "_pad_token_id", self.config.pad_token_id),
            eos_token_id=self.config.eos_id,
            soi_token_id=self.config.image_cls_token_id,
            eoi_token_id=self.config.img_end_id,
            max_len=max_len,
        )

    def get_upsampler_attn_mask(self, H, W, h, w, device):
        return WindowMask(H, W, h, w, device=device)

    def _scatter_img_tokens_with_projector(self, h_BSD, pixel_patches_NLC, pixel_masks_NTHW, tokens_BS):
        B, S, D = h_BSD.shape
        pixel_patch_mask = E.reduce(
            pixel_masks_NTHW,
            "n (t pt) (h ph) (w pw) -> (n t h w)",
            reduction="any",
            pt=self.config.temporal_patch_size,
            ph=self.config.spatial_patch_size,
            pw=self.config.spatial_patch_size,
        )
        pixel_patches_flat = E.rearrange(pixel_patches_NLC, "n p c -> (n p) c")
        valid_patches = pixel_patches_flat[pixel_patch_mask]
        valid_feats = self.img_projector(valid_patches)
        img_mask_h_BSD = E.repeat(tokens_BS == self.config.img_id, "b s -> b s d", d=D)
        assert valid_feats.numel() == img_mask_h_BSD.sum()
        return paddle.masked_scatter(h_BSD, img_mask_h_BSD, valid_feats)

    def _encode_coords(self, h_BSD: T, tokens_BS: T, all_xy: T):
        coord_tokens_mask = tokens_BS == self.config.coord_token_id
        if all_xy.numel() == 0:
            return h_BSD
        coord_tokens = self.coord_encoder(all_xy.reshape(-1, 2))
        if coord_tokens.shape[0] == h_BSD.shape[0]:
            h_BSD = paddle.where(
                coord_tokens_mask.unsqueeze(-1), coord_tokens.view(h_BSD.shape[0], -1, h_BSD.shape[-1]), h_BSD
            )
        else:
            h_BSD = h_BSD.masked_scatter_(coord_tokens_mask.unsqueeze(-1), coord_tokens)
        return h_BSD

    def _encode_sizes(self, h_BSD, tokens_BS, all_hw: T):
        size_tokens_mask = tokens_BS == self.config.size_token_id
        if all_hw.numel() == 0:
            return h_BSD
        size_tokens = self.size_encoder(all_hw.reshape(-1, 2))
        if size_tokens.shape[0] == h_BSD.shape[0]:
            h_BSD = paddle.where(
                size_tokens_mask.unsqueeze(-1), size_tokens.view(h_BSD.shape[0], -1, h_BSD.shape[-1]), h_BSD
            )
        else:
            h_BSD = h_BSD.masked_scatter_(size_tokens_mask.unsqueeze(-1), size_tokens)
        return h_BSD

    def decode_coords(self, h_BSD, labels):
        B, S, D = h_BSD.shape
        coord_masks = labels == self.config.coord_token_id
        coord_tokens = paddle.masked_select(h_BSD, coord_masks.unsqueeze(-1))
        coord_logits = self.coord_decoder(coord_tokens.reshape(-1, D))
        return E.rearrange(coord_logits, "b (two dim) -> b two dim", two=2)

    def decode_sizes(self, h_BSD, labels):
        B, S, D = h_BSD.shape
        size_masks = labels == self.config.size_token_id
        size_tokens = paddle.masked_select(h_BSD, size_masks.unsqueeze(-1))
        size_logits = self.size_decoder(size_tokens.reshape(-1, D))
        return E.rearrange(size_logits, "b (two dim) -> b two dim", two=2)

    def process_sizes(self, logits):
        num_bins = logits.shape[-1]
        pred = paddle.argmax(logits, axis=-1).float() / (num_bins - 1)
        min_size = paddle.log2(paddle.to_tensor(1 / num_bins))
        max_size = 0.0
        pred = pred * (max_size - min_size) + min_size
        return 2.0**pred

    def gather_img_tokens(self, h_BSD: T, tokens_BS: T, itok_masks_NTHW: T):
        B, S, D = h_BSD.shape
        itok_masks_BSD = E.repeat(tokens_BS == self.config.img_id, "b s -> b s d", d=D)
        itok_flatten = paddle.masked_select(h_BSD, itok_masks_BSD)
        itok_masks_NTHWD = E.repeat(itok_masks_NTHW, "n t h w -> n t h w d", d=D)
        itok_NTHWD = paddle.zeros_like(itok_masks_NTHWD, dtype=h_BSD.dtype, device=h_BSD.place)
        itok_NTHWD = itok_NTHWD.masked_scatter_(itok_masks_NTHWD, itok_flatten)
        return itok_NTHWD

    def upsample_img_features(self, h_BSD: T, tokens_BS: T, pixel_values_NTHWC: T, pixel_mask_NTHW: T):
        device = h_BSD.place
        c = self.config
        itok_masks_NTHW = E.reduce(
            pixel_mask_NTHW,
            "n (t pt) (h ph) (w pw) -> n t h w",
            reduction="any",
            pt=c.temporal_patch_size,
            ph=c.spatial_patch_size,
            pw=c.spatial_patch_size,
        )
        N, _, h, w = itok_masks_NTHW.shape
        _, _, H, W = pixel_mask_NTHW.shape
        images = E.rearrange(pixel_values_NTHWC, "n 1 h w c -> n c h w")
        lr_img_features = self.gather_img_tokens(h_BSD, tokens_BS, itok_masks_NTHW)
        lr_img_features = E.rearrange(lr_img_features, "n 1 h w d -> n d h w")
        lr_img_features = self.conv_segm(lr_img_features)
        upsampler_attn_mask = self.get_upsampler_attn_mask(H, W, h, w, device=device)
        hr_parts = []
        for i in range(N):
            hr_i = self.itok_upsampler(
                images=images[i : i + 1], features=lr_img_features[i : i + 1], attn_mask=upsampler_attn_mask
            )
            hr_parts.append(hr_i)
        return paddle.cat(hr_parts, axis=0) if N > 1 else hr_parts[0]

    @staticmethod
    def _mask_to_coco_rle(binary_masks: paddle.Tensor) -> list[dict]:
        C, H, W = binary_masks.shape
        has_any = E.reduce(binary_masks, "c h w -> c", reduction="any")
        binary_col = E.rearrange(binary_masks, "c h w -> c (w h)")
        diffs = binary_col[:, 1:] != binary_col[:, :-1]
        nz = paddle.nonzero(diffs, as_tuple=False)
        first_vals = binary_col[:, 0]
        nz_cpu = nz.cpu().numpy()
        has_any_cpu = has_any.cpu().numpy()
        first_vals_cpu = first_vals.cpu().numpy()
        del diffs, nz, binary_col, first_vals, has_any
        N_px = H * W
        if nz_cpu.shape[0] > 0:
            mask_ids = nz_cpu[:, 0]
            change_cols = nz_cpu[:, 1]
            uniq, grp_starts = np.unique(mask_ids, return_index=True)
            grp_ends = np.append(grp_starts[1:], len(mask_ids))
            mask_to_grp = {int(m): (int(gs), int(ge)) for m, gs, ge in zip(uniq, grp_starts, grp_ends)}
        else:
            change_cols = np.array([], dtype=np.intp)
            mask_to_grp = {}
        results = []
        for i in range(C):
            if not has_any_cpu[i]:
                continue
            if i in mask_to_grp:
                gs, ge = mask_to_grp[i]
                cidx = change_cols[gs:ge]
            else:
                cidx = np.array([], dtype=np.intp)
            num_runs = len(cidx) + 1
            starts = np.empty(num_runs, dtype=np.intp)
            starts[0] = 0
            if len(cidx) > 0:
                starts[1:] = cidx + 1
            counts = np.empty(num_runs, dtype=np.uint32)
            if num_runs > 1:
                counts[:-1] = np.diff(starts)
            counts[-1] = N_px - starts[-1]
            if first_vals_cpu[i]:
                counts = np.concatenate([[0], counts])
            rle = {"counts": counts.tolist(), "size": [H, W]}
            rle = mask_utils.frPyObjects(rle, H, W)
            rle["counts"] = rle["counts"].decode("utf-8")
            results.append(rle)
        return results

    def forward(
        self,
        tokens: T,
        attention_mask,
        kv_cache,
        rope_pos_t: T | None = None,
        rope_pos_hw: T | None = None,
        pixel_values: T | None = None,
        pixel_mask: T | None = None,
        coord_xy: T | None = None,
        size_hw: T | None = None,
    ):
        B, S = tokens.size()
        c = self.config
        block_mask = attention_mask
        T_pos = kv_cache.get_pos()
        is_prefill = S != 1
        if is_prefill:
            assert rope_pos_t is not None and rope_pos_hw is not None
            pos_t = rope_pos_t[:, T_pos : T_pos + S].long()
            kv_cache.pos_t = pos_t[:, -1:]
            freqs_cis = self.freqs_cis[pos_t]
            rope_pos_hw = rope_pos_hw[:, T_pos : T_pos + S]
            freqs_cis_golden = apply_golden_freqs_cis_to_visual_pos(self.freqs_cis_golden, rope_pos_hw)
        else:
            pos_t = kv_cache.increment_and_get_pos_t()
            freqs_cis = self.freqs_cis[pos_t]
            freqs_cis_golden = None
            block_mask = attention_mask.with_offset(T_pos)
        h_BSD = self.tok_embeddings(tokens)
        coord_xy = coord_xy if coord_xy is not None else h_BSD.new_empty(0)
        size_hw = size_hw if size_hw is not None else h_BSD.new_empty(0)
        h_BSD = self._encode_coords(h_BSD, tokens, coord_xy)
        h_BSD = self._encode_sizes(h_BSD, tokens, size_hw)
        if pixel_values is not None:
            assert pixel_mask is not None
            pixel_values = pixel_values.to(self.dtype)
            pixel_mask = pixel_mask.astype("bool")
            pixel_patches_NLC = E.rearrange(
                pixel_values,
                "n (t pt) (h ph) (w pw) c -> n (t h w) (pt ph pw c)",
                pt=c.temporal_patch_size,
                ph=c.spatial_patch_size,
                pw=c.spatial_patch_size,
            )
            h_BSD = self._scatter_img_tokens_with_projector(h_BSD, pixel_patches_NLC, pixel_mask, tokens)
        for layer in self.layers.values():
            h_BSD = layer(
                h_BSD,
                freqs_cis=freqs_cis,
                freqs_cis_2d=freqs_cis_golden,
                pos_hw=rope_pos_hw,
                attention_masks=block_mask,
                kv_cache=kv_cache,
            )
        h_BSD = self.norm(h_BSD)
        logits_BSV = self.output(h_BSD)
        return (logits_BSV, h_BSD)

    @paddle.no_grad()
    def generate(
        self,
        images,
        queries,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_k: int | None = None,
        min_dimension: int = 256,
        max_dimension: int = 1024,
        compile: bool = False,
        seed: int | None = 42,
        segm_threshold: float = 0.5,
    ) -> list[list[dict]]:
        """
        Segment objects in images matching the given queries.

        Args:
            images: One PIL image (or path/URL), optionally in a one-item list.
            queries: One query string, optionally in a one-item list.
            max_new_tokens: Maximum generation steps.
            temperature: Sampling temperature (0.0 = greedy).
            top_k: Top-k sampling (None = disabled).
            min_dimension: Min image side after resize.
            max_dimension: Max image side after resize.
            compile: Must be False for dynamic Paddle reference inference.
            seed: Random seed for reproducibility (None = non-deterministic).
            segm_threshold: Sigmoid threshold for binary mask.

        Returns:
            List (per image) of lists (per detection) of dicts::

                {
                    "xy": {"x": float, "y": float},
                    "hw": {"h": float, "w": float},
                    "mask_rle": {"counts": str, "size": [H, W]},
                }
        """
        self._ensure_device_buffers()
        if compile:
            raise ValueError("Paddle reference inference uses dynamic mode; pass compile=False.")
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        if isinstance(queries, str):
            queries = [queries]
        if len(images) != 1 or len(queries) != 1:
            raise ValueError("This reference implementation currently supports one image/query per call.")
        device = self.place
        tokenizer = self._get_tokenizer()
        self._pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
        stop_token_ids = [self.config.eos_id, tokenizer.convert_tokens_to_ids("<|end_of_query|>")]
        pil_images = [load_image(img).convert("RGB") for img in images]
        original_sizes = [(img.height, img.width) for img in pil_images]
        image_prompt_pairs = [
            (img, f"<|image|>Segment these expressions in the image:<|start_of_query|>{q}<|REF_SEG|>")
            for img, q in zip(pil_images, queries)
        ]
        batch_inputs = process_batch(
            tokenizer,
            self.config,
            image_prompt_pairs,
            max_length=4096,
            min_dimension=min_dimension,
            max_dimension=max_dimension,
        )
        batch_inputs = {k: v.to(device) if paddle.is_tensor(v) else v for k, v in batch_inputs.items()}
        tokens = batch_inputs["tokens"]
        B, L = tokens.size()
        block_size = 128
        S = (L + max_new_tokens + block_size - 1) // block_size * block_size
        assert S <= self.config.max_seq_len
        if seed is not None:
            paddle.seed(seed)
        rng = None
        kv_cache = KVCache(
            max_batch_size=B,
            max_seq_length=S,
            n_heads=self.config.n_heads,
            head_dim=self.config.head_dim,
            num_layers=self.config.n_layers,
        )
        padded_tokens = paddle.full((B, S), self._pad_token_id, dtype=tokens.dtype, device=device)
        padded_tokens[:, :L] = tokens
        attention_mask = self.get_attention_mask(padded_tokens, max_len=S)
        all_xy, all_hw = self._extract_coords([[]])
        coord_xy = all_xy.to(device=device, dtype=self.dtype)
        size_hw_t = all_hw.to(device=device, dtype=self.dtype)
        logits_BSV, h_BSD = self.forward(
            tokens=tokens,
            rope_pos_t=batch_inputs["pos_t"],
            rope_pos_hw=batch_inputs["pos_hw"],
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            pixel_values=batch_inputs["pixel_values"],
            pixel_mask=batch_inputs["pixel_mask"],
            coord_xy=coord_xy,
            size_hw=size_hw_t,
        )
        hr_img_features = self.upsample_img_features(
            h_BSD, tokens, batch_inputs["pixel_values"], batch_inputs["pixel_mask"]
        )
        aux_output_B = [[] for _ in range(B)]
        stop_ids = paddle.to_tensor(stop_token_ids).to(device)
        should_stop_B = paddle.full((B,), False, dtype=paddle.bool, device=device)
        while not paddle.all(should_stop_B) and (pos := kv_cache.get_pos()) < S:
            tokens_B1 = sample_next_token(logits_BSV[:, -1], rng, temperature, top_k)
            if paddle.any(should_stop_B):
                tokens_B1 = tokens_B1.clone()
                tokens_B1[should_stop_B, :] = self._pad_token_id
            padded_tokens[:, pos] = tokens_B1[:, -1]
            coord_logits = self.decode_coords(h_BSD[:, -1:], tokens_B1)
            sample_w_coord = paddle.where(tokens_B1 == self.config.coord_token_id)[0]
            num_bins = coord_logits.size(-1)
            coord_repeat_threshold = 0.01
            max_coord_attempts = 100
            xy_b2 = paddle.zeros([B, 2], device=device, dtype=self.dtype)
            for i, b in enumerate(sample_w_coord.tolist()):
                logits_b = coord_logits[i].clone()
                existing_coords = [
                    item for item in aux_output_B[b] if isinstance(item, dict) and "x" in item and ("y" in item)
                ]
                pred_x, pred_y = (0.0, 0.0)
                for _ in range(max_coord_attempts):
                    pred_bins = paddle.argmax(logits_b, axis=-1)
                    pred_x = pred_bins[0].item() / (num_bins - 1)
                    pred_y = pred_bins[1].item() / (num_bins - 1)
                    is_repeat = any(
                        (
                            abs(ec["x"] - pred_x) < coord_repeat_threshold
                            and abs(ec["y"] - pred_y) < coord_repeat_threshold
                            for ec in existing_coords
                        )
                    )
                    if not is_repeat:
                        break
                    logits_b[0, pred_bins[0]] = float("-inf")
                    logits_b[1, pred_bins[1]] = float("-inf")
                xy_b2[b, 0] = pred_x
                xy_b2[b, 1] = pred_y
                aux_output_B[b].append({"x": pred_x, "y": pred_y})
            size_logits = self.decode_sizes(h_BSD[:, -1:], tokens_B1)
            hw_b2 = self.process_sizes(size_logits)
            size_preds = [{"h": hw[0].item(), "w": hw[1].item()} for hw in hw_b2]
            sample_w_size = paddle.where(tokens_B1 == self.config.size_token_id)[0]
            for i, b in enumerate(sample_w_size.tolist()):
                aux_output_B[b].append(size_preds[i])
            sample_w_segm = paddle.where(tokens_B1 == self.config.seg_token_id)[0]
            segm_tokens = h_BSD[sample_w_segm, -1, :]
            segm_tokens = self.proj_segm(segm_tokens)
            segm_masks = paddle.einsum("kdhw,kd->khw", hr_img_features[sample_w_segm], segm_tokens)
            for i, b in enumerate(sample_w_segm):
                aux_output_B[b].append(segm_masks[i])
            logits_BSV, h_BSD = self.forward(
                tokens=tokens_B1,
                attention_mask=attention_mask,
                coord_xy=xy_b2.to(self.dtype),
                size_hw=hw_b2.to(self.dtype),
                kv_cache=kv_cache,
            )
            hit_stop_B = paddle.isin(tokens_B1, stop_ids).any(axis=-1)
            should_stop_B = should_stop_B.logical_or(hit_stop_B)
        pixel_mask_batch = batch_inputs["pixel_mask"][:, 0]
        results = []
        for b in range(B):
            dets = self._postprocess_aux(aux_output_B[b], pixel_mask_batch[b], original_sizes[b], segm_threshold)
            results.append(dets)
        return results

    def _extract_coords(self, coords_BO: list[list]):
        all_xy, all_hw = ([], [])
        for coords_O in coords_BO:
            if not coords_O:
                continue
            for coords in coords_O:
                for k, v in coords.items():
                    if k.startswith(("x", "y")):
                        all_xy.append(v)
                    elif k.startswith(("h", "w")):
                        all_hw.append(v)
        return (paddle.to_tensor(all_xy), paddle.to_tensor(all_hw))

    @staticmethod
    def _mask_nms(binary_masks: list[paddle.Tensor], iou_threshold: float = 0.6, nms_max_side: int = 256) -> list[int]:
        """
        Fast vectorised mask NMS on binary (H, W) tensors.

        Returns the list of kept indices ordered by descending mask score.
        The IoU matrix is computed via a single batched matmul; suppression
        uses one GPU boolean op per kept mask — no .item() in the inner loop.
        """
        N = len(binary_masks)
        if N <= 1:
            return list(range(N))
        device = binary_masks[0].place
        base_h, base_w = binary_masks[0].shape
        scale = min(1.0, nms_max_side / max(base_h, base_w))
        th = max(1, int(round(base_h * scale)))
        tw = max(1, int(round(base_w * scale)))
        resized = []
        for m in binary_masks:
            m = m.float()
            if m.shape != (th, tw):
                m = F.interpolate(m[None, None], size=(th, tw), mode="bilinear", align_corners=False).squeeze()
            resized.append(m)
        binary = paddle.stack(resized)
        flat = binary.view(N, -1)
        areas = flat.sum(axis=1)
        scores = areas
        intersection = flat @ flat.T
        union = areas[:, None] + areas[None, :] - intersection
        iou = intersection / union.clamp(min=1)
        order = scores.argsort(descending=True)
        suppressed = paddle.zeros(N, dtype=paddle.bool, device=device)
        keep = []
        for idx in order.tolist():
            if suppressed[idx]:
                continue
            keep.append(idx)
            suppressed |= iou[idx] > iou_threshold
        return keep

    def _postprocess_aux(
        self,
        aux_list: list,
        pixel_mask_hw: T,
        orig_hw: tuple[int, int],
        threshold: float,
        nms_iou_threshold: float = 0.6,
    ) -> list[dict]:
        """Convert raw aux outputs into structured detections with RLE masks."""
        orig_h, orig_w = orig_hw
        nonzero = paddle.nonzero(pixel_mask_hw, as_tuple=False)
        if len(nonzero) > 0:
            min_h, min_w = paddle.min(nonzero, axis=0)
            max_h, max_w = paddle.max(nonzero, axis=0)
            act_h = (max_h - min_h + 1).item()
            act_w = (max_w - min_w + 1).item()
        else:
            min_h = min_w = 0
            act_h = act_w = None
        candidates = []
        step = 3
        for i in range(0, len(aux_list), step):
            if i + 2 >= len(aux_list):
                break
            xy = aux_list[i]
            hw = aux_list[i + 1]
            mask_logits = aux_list[i + 2]
            if not isinstance(mask_logits, paddle.Tensor):
                continue
            if act_h is not None and act_w is not None:
                mask_logits = mask_logits[min_h : min_h + act_h, min_w : min_w + act_w]
            mask_logits = mask_logits.unsqueeze(0).unsqueeze(0).float()
            mask_logits = F.interpolate(mask_logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            mask_logits = mask_logits.squeeze(0).squeeze(0)
            binary_mask = (paddle.sigmoid(mask_logits) > threshold).bool()
            candidates.append({"xy": xy, "hw": hw, "binary_mask": binary_mask})
        if not candidates:
            return []
        keep_indices = self._mask_nms([c["binary_mask"] for c in candidates], iou_threshold=nms_iou_threshold)
        candidates = [candidates[i] for i in keep_indices]
        detections = []
        for c in candidates:
            rle_list = self._mask_to_coco_rle(c["binary_mask"].unsqueeze(0))
            mask_rle = rle_list[0] if rle_list else {"counts": "", "size": [orig_h, orig_w]}
            detections.append({"xy": c["xy"], "hw": c["hw"], "mask_rle": mask_rle})
        return detections
