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

"""Diagnostic equivalent position-id assignment; no repository source edit."""
import paddle


def embeddings_forward(self, pixel_values, patch_attention_mask):
    batch_size, _, max_im_h, max_im_w = pixel_values.shape
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose([0, 2, 1])
    boundaries = paddle.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side).astype(
        pixel_values.dtype
    )
    nb_patches_h = patch_attention_mask[:, :, 0].astype("int64").sum(axis=1)
    nb_patches_w = patch_attention_mask[:, 0, :].astype("int64").sum(axis=1)
    step_h = 1.0 / nb_patches_h.astype("float32")
    step_w = 1.0 / nb_patches_w.astype("float32")
    h_indices = paddle.arange(patch_attention_mask.shape[1], dtype="float32")
    w_indices = paddle.arange(patch_attention_mask.shape[2], dtype="float32")
    fractional_coords_h = paddle.clip(h_indices[None, :] * step_h[:, None], max=1.0 - 1e-6).astype(pixel_values.dtype)
    fractional_coords_w = paddle.clip(w_indices[None, :] * step_w[:, None], max=1.0 - 1e-6).astype(pixel_values.dtype)
    bucket_coords_h = paddle.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = paddle.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, :, None] * self.num_patches_per_side + bucket_coords_w[:, None, :]).reshape(
        [batch_size, -1]
    )
    flat_mask = patch_attention_mask.reshape([batch_size, -1]).astype("bool")
    position_ids = paddle.where(flat_mask, pos_ids, paddle.zeros_like(pos_ids))
    return embeddings + self.position_embedding(position_ids)
