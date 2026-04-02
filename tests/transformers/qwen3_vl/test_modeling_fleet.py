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
from __future__ import annotations

import types
import unittest

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

_FLEET_IMPORT_ERROR = None

try:
    from paddleformers.transformers.qwen3_vl.modeling import (
        Qwen3VLVisionRotaryEmbedding,
    )
    from paddleformers.transformers.qwen3_vl.modeling_fleet import (
        Qwen3VLVisionModel,
        safe_repeat_interleave_values,
    )
except Exception as error:
    Qwen3VLVisionModel = None
    Qwen3VLVisionRotaryEmbedding = None
    safe_repeat_interleave_values = None
    _FLEET_IMPORT_ERROR = error


@unittest.skipUnless(Qwen3VLVisionModel is not None, f"paddlefleet import failed: {_FLEET_IMPORT_ERROR}")
class Qwen3VLFleetPositionalEncodingTest(unittest.TestCase):
    FAST_POS_EMBED_ATOL = 2e-5

    def setUp(self):
        paddle.seed(2026)
        self.model = self._build_dummy_model()

    def _build_dummy_model(self):
        class DummyVisionModel:
            pass

        model = DummyVisionModel()
        model.spatial_merge_size = 2
        model.num_grid_per_side = 70
        model.pos_embed = nn.Embedding(model.num_grid_per_side**2, 8)
        weight = paddle.arange(model.num_grid_per_side**2 * 8, dtype="float32").reshape([-1, 8]) / 1000.0
        model.pos_embed.weight.set_value(weight)
        model.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(4)

        for name in [
            "_build_token_image_mapping",
            "rot_pos_emb",
            "fast_pos_embed_interpolate",
            "get_packed_seq_params",
        ]:
            setattr(model, name, types.MethodType(getattr(Qwen3VLVisionModel, name), model))

        return model

    def _legacy_rot_pos_emb(self, grid_thw):
        merge_size = self.model.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(height, width) for _, height, width in grid_thw_list)
        freq_table = self.model.rotary_pos_emb(max_hw)

        total_tokens = sum(int(frames * height * width) for frames, height, width in grid_thw_list)
        pos_ids = paddle.empty([total_tokens, 2], dtype="int64")

        offset = 0
        for num_frames, height, width in grid_thw_list:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = paddle.arange(merged_h)
            block_cols = paddle.arange(merged_w)
            intra_row = paddle.arange(merge_size)
            intra_col = paddle.arange(merge_size)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand([merged_h, merged_w, merge_size, merge_size]).reshape([-1])
            col_idx = col_idx.expand([merged_h, merged_w, merge_size, merge_size]).reshape([-1])

            coords = paddle.stack([row_idx, col_idx], axis=-1)
            if num_frames > 1:
                coords = coords.tile([num_frames, 1])

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(start_axis=1)

    def _legacy_fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            t, h, w = int(t), int(h), int(w)
            self.assertEqual(t, 1)

            h_idxs = paddle.linspace(0, self.model.num_grid_per_side - 1, h)
            w_idxs = paddle.linspace(0, self.model.num_grid_per_side - 1, w)

            h_floor = h_idxs.astype("int32")
            w_floor = w_idxs.astype("int32")
            h_ceil = (h_floor + 1).clip(max=self.model.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.model.num_grid_per_side - 1)

            dh = h_idxs - h_floor.astype("float32")
            dw = w_idxs - w_floor.astype("float32")

            base_h = h_floor * self.model.num_grid_per_side
            base_h_ceil = h_ceil * self.model.num_grid_per_side

            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
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
        weight_tensor = paddle.to_tensor(weight_list, dtype=self.model.pos_embed.weight.dtype)
        pos_embeds = self.model.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([int(h) * int(w) for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.model.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            h_merged = int(h) // merge_size
            w_merged = int(w) // merge_size
            pos_embed = (
                pos_embed.reshape([int(t), h_merged, merge_size, w_merged, merge_size, -1])
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return paddle.cat(patch_pos_embeds_permute)

    def test_fast_pos_embed_interpolate_matches_legacy_reference_with_expected_tolerance(self):
        for grid_thw in (
            paddle.to_tensor([[1, 6, 8], [1, 14, 14]], dtype="int64"),
            paddle.to_tensor([[1, 8, 6], [1, 4, 8]], dtype="int64"),
        ):
            with self.subTest(grid_thw=grid_thw.tolist()):
                actual = self.model.fast_pos_embed_interpolate(grid_thw)
                expected = self._legacy_fast_pos_embed_interpolate(grid_thw)

                self.assertEqual(list(actual.shape), list(expected.shape))
                self.assertTrue(
                    bool(
                        paddle.allclose(
                            actual,
                            expected,
                            rtol=0.0,
                            atol=self.FAST_POS_EMBED_ATOL,
                        )
                    )
                )

    def test_rot_pos_emb_matches_legacy_reference(self):
        for grid_thw in (
            paddle.to_tensor([[2, 4, 4], [1, 6, 8]], dtype="int64"),
            paddle.to_tensor([[1, 8, 6], [2, 4, 8]], dtype="int64"),
        ):
            with self.subTest(grid_thw=grid_thw.tolist()):
                actual = self.model.rot_pos_emb(grid_thw)
                expected = self._legacy_rot_pos_emb(grid_thw)

                self.assertTrue(bool(paddle.equal_all(actual, expected)))

    def test_get_packed_seq_params_matches_legacy_reference(self):
        grid_thw = paddle.to_tensor([[2, 4, 4], [1, 6, 8], [3, 2, 2]], dtype="int64")

        actual = self.model.get_packed_seq_params(grid_thw)

        seqlens = safe_repeat_interleave_values(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        expected_cu_seqlens = F.pad(seqlens.cumsum(axis=0, dtype=paddle.int32), (1, 0), value=0).contiguous()

        self.assertEqual(actual.max_seqlen_q, seqlens.max().item())
        self.assertEqual(actual.max_seqlen_kv, seqlens.max().item())
        self.assertEqual(actual.qkv_format, "thd")
        self.assertTrue(bool(paddle.equal_all(actual.cu_seqlens_q, expected_cu_seqlens)))
        self.assertTrue(bool(paddle.equal_all(actual.cu_seqlens_kv, expected_cu_seqlens)))


if __name__ == "__main__":
    unittest.main()
