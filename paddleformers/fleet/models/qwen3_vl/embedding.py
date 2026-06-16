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
from dataclasses import dataclass

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer
from paddle.nn import functional as F

from ...packed_seq_params import PackedSeqParams
from ...transformer import TransformerConfig
from ...transformer.layer import FleetLayer


@dataclass
class VisionEmbeddingSpec:
    rope_embedding: LayerSpec = None


class VisionEmbedding(FleetLayer):
    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: VisionEmbeddingSpec,
    ):
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = (
            self.spatial_merge_size * self.spatial_merge_size
        )
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.merge_hidden_size = self.embed_dim * (config.spatial_merge_size**2)

        kernel_size = [
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        ]
        self.patch_embed = nn.Conv3D(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.rotary_pos_emb = None
        if sublayers_spec.rope_embedding:
            self.rotary_pos_emb = build_spec_layer(
                sublayers_spec.rope_embedding,
            )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            hpos_ids = paddle.arange(h).unsqueeze(1).expand([-1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose(perm=[0, 2, 1, 3])
            hpos_ids = hpos_ids.flatten()

            wpos_ids = paddle.arange(w).unsqueeze(0).expand([h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3])
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                paddle.stack(x=[hpos_ids, wpos_ids], axis=-1).tile(
                    repeat_times=[t, 1]
                )
            )
        pos_ids = paddle.concat(x=pos_ids, axis=0)
        max_grid_size = int(grid_thw[:, 1:].max())
        # Get raw freqs [max_grid_size, head_dim//2] and index with 2D pos_ids
        freqs = self.rotary_pos_emb.get_freqs_non_repeated(max_grid_size)
        # pos_ids: [seq_len, 2], freqs: [max_grid_size, head_dim//2]
        # Index freqs with each position dim: freqs[pos_ids] -> [seq_len, 2, head_dim//2]
        rotary_pos_emb = freqs[pos_ids].flatten(start_axis=1)
        # rotary_pos_emb: [seq_len, head_dim//2] (2 spatial dims * dim//2 freqs)
        # Repeat to cover full head_dim so apply_rotary_pos_emb rotates all dims
        rotary_pos_emb = paddle.concat(
            [rotary_pos_emb, rotary_pos_emb], axis=-1
        )
        rotary_pos_emb = rotary_pos_emb[None, :, None, :]
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = (
            grid_thw[:, 0],
            grid_thw[:, 1],
            grid_thw[:, 2],
        )
        device = paddle.get_device()

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            t, h, w = int(t), int(h), int(w)
            h_idxs = paddle.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = paddle.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(
                max=self.num_grid_per_side - 1
            )
            w_idxs_ceil = (w_idxs.int() + 1).clip(
                max=self.num_grid_per_side - 1
            )

            dh = h_idxs - h_idxs_floor.astype("float32")
            dw = w_idxs - w_idxs_floor.astype("float32")

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = paddle.to_tensor(idx_list, dtype="int64")
        weight_tensor = paddle.to_tensor(
            weight_list, dtype=self.pos_embed.weight.dtype
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = (
            pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        )

        patch_pos_embeds = paddle.split(
            patch_pos_embeds,
            [int(h) * int(w) for h, w in zip(grid_hs, grid_ws)],
        )

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(
            patch_pos_embeds, grid_ts, grid_hs, grid_ws
        ):
            pos_embed = pos_embed.tile([int(t), 1])
            pos_embed = (
                pos_embed.reshape(
                    [
                        int(t),
                        int(h) // merge_size,
                        merge_size,
                        int(w) // merge_size,
                        merge_size,
                        -1,
                    ]
                )
                .transpose([0, 1, 3, 2, 4, 5])
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = paddle.concat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def get_packed_seq_params(
        self,
        grid_thw: paddle.Tensor,
    ):
        seqlens = paddle.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = seqlens.cumsum(axis=0).astype("int32")
        cu_seqlens = F.pad(cu_seqlens.unsqueeze(0), [1, 0], value=0).squeeze(0)

        max_seqlen = seqlens.max().item()
        total_seqlen = cu_seqlens[-1].item()

        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            total_seqlen_q=total_seqlen,
            total_seqlen_kv=total_seqlen,
            qkv_format="thd",
        )

    def forward(self, dict_args: dict):
        pixel_values = dict_args["pixel_values"]
        grid_thw = dict_args["grid_thw"]

        pixel_values = pixel_values.reshape(
            [
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ]
        )

        hidden_states = (
            self.patch_embed(pixel_values)
            .flatten(2)
            .transpose([0, 2, 1])
            .reshape([-1, self.embed_dim])
        )
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape([seq_len, -1])
        hidden_states = hidden_states.unsqueeze(0)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_cos = paddle.cos(rotary_pos_emb)
        rotary_pos_sin = paddle.sin(rotary_pos_emb)

        packed_seq_params = self.get_packed_seq_params(grid_thw)

        preproc_output = {
            "hidden_states": hidden_states,
            "attention_mask": dict_args.get("attention_mask", None),
            "rotary_pos_emb": rotary_pos_emb,
            "rotary_pos_cos": rotary_pos_cos,
            "rotary_pos_sin": rotary_pos_sin,
            "packed_seq_params": packed_seq_params,
        }

        return preproc_output
