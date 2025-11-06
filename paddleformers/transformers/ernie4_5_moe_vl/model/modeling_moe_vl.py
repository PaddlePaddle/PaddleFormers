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

"""Paddle Ernie VL model"""

import contextlib
import json
import re
from collections import defaultdict
from copy import deepcopy
from functools import partial
from types import MethodType
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.distributed.fleet.layers.mpu.mp_layers import VocabParallelEmbedding
from paddle.distributed.fleet.utils import recompute
from paddle.nn import functional as F

from paddleformers.utils.log import logger

from .configuration import Ernie4_5_VLMoeConfig
from .dfnrope.modeling import DFNRopeVisionTransformerPretrainedModel
from .distributed import RowSequenceParallelLinear, parallel_matmul
from .longcontext_ops import TensorBalanceByTokenType
from .modeling import Ernie4_5_LMHead
from .modeling import ErniePretrainingCriterion as ErniePretrainingCriterionBase
from .modeling import RMSNorm
from .modeling_moe import CausalLMOutputWithCrossAttentions, Ernie4_5_MoeForCausalLM
from .moe.moe_layer import manual_backward
from .sequence_parallel_utils import (
    AllGatherOp,
    GatherOp,
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)

try:
    from ..utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}


__all__ = [
    "Ernie4_5_VLMoeForConditionalGeneration",
]


class TokenType:
    """token type definition"""

    text = 0
    image = 1
    video = 2


IDTYPES_2_ID = {"text": 0, "image": 1, "video": 2}
IMAGETYPES_2_ID = {"image": 0, "video": 1, "padded_image": 2}


def monkey_patch_param_hook(param):
    """
    patch param hook
    """
    hook_list = []

    def _register_grad_hook(self, hook):
        nonlocal hook_list
        hook_list.append(hook)

    def hook_of_hook(g):
        nonlocal hook_list
        for h in hook_list:
            g = h(g)
        return g

    def hooks(self):
        nonlocal hook_list
        return hook_list

    def register_hook(self, hook, pos=None):
        nonlocal hook_list
        if pos is None:
            pos = len(hook_list)
        hook_list.insert(pos, hook)

        class _Remover:
            """hook remover"""

            def remove(self):
                """hook remover"""
                for i, h in enumerate(hook_list):
                    if h is hook:
                        break
                else:
                    logger.error(f"can not remove hook {hook} from: {hook_list}")
                    return False
                hook_list.pop(i)
                return True

        return _Remover()

    param._register_grad_hook(hook_of_hook)
    param._register_grad_hook = MethodType(_register_grad_hook, param)
    param.register_hook = MethodType(register_hook, param)
    param.hooks = MethodType(hooks, param)


def get_backbone_lm_param_regex(config):
    """
    return weight name regex from LLM backbone network
    """
    moe_rank = dist.get_rank(config.moe_group)
    moe_world_size = dist.get_world_size(config.moe_group)
    num_local_experts = (
        sum(config.moe_num_experts) // moe_world_size
        if config.moe_num_experts
        else config.moe_num_experts // moe_world_size
    )
    num_freeze_expert = config.moe_num_experts[0] if config.moe_num_experts else config.moe_num_experts

    freeze_part = [r"model\.norm.*", r"model\.layers.*norm.*"]  # freeze all norm
    # we do not include gate weight
    # gate weight detach modality
    freeze_part += [
        r"model\.layers\.(\d+)\.mlp\.(up_gate|gate|up|down)_proj\.*",
        r"model\.layers\.(\d+)\.mlp\.shared_experts\.(up_gate|gate|up|down)_proj\.*",
        r"model\.layers\.(\d+)\.self_attn.(q|k|v|o|qkv)_proj\.(weight|bias)",
        r"model\.layers\.(\d)+\.mlp\.gate\.weight$",
    ]
    logger.info(f"FREEZE_DEBUG: { moe_rank * num_local_experts} {num_freeze_expert}")
    freeze_part += [r"model\.embed_tokens\.weight"]
    freeze_part += [r"lm_head\.weight", r"lm_head\.bias"]

    assert freeze_part, f"not freeze any part, moe: {moe_rank}/{moe_world_size}"
    logger.info(f"freeze pattern: {freeze_part}, moe: {moe_rank}/{moe_world_size}")
    freeze_part = re.compile("|".join(freeze_part))
    return freeze_part


def create_freeze_hook(name, param, factor=0.0):
    """
    create hook to scale gradient
    """

    def _stopgrad_hook(g):
        with paddle.no_grad():
            return g.scale_(factor)  # using inplace operator

    return _stopgrad_hook


def create_partial_freeze_hook(name, param, factor, index):
    """
    create a hook to scale gradient for partial parameter
    """

    def _stopgrad_hook(g):
        with paddle.no_grad():
            g[:, :index] = g[:, :index] * factor
        return g

    return _stopgrad_hook


class ModalityDetach(PyLayer):
    """detach modality"""

    @staticmethod
    def forward(
        ctx,
        token_type_ids,
        *args,
        fn=None,
        is_first_fwd=False,
        freeze_context=None,
    ):
        """
        Args:
            if token_type_ids has no 0， add `freeze_context` to the backward of `fn`
        Returns:
            fn(token_type_ids, *args)
        """
        assert fn is not None
        if not is_first_fwd:
            ctx.fn = fn
        ctx.bwf, outputs = manual_backward(fn, is_first_fwd, token_type_ids, *args)
        should_freeze = token_type_ids.astype("bool").any().item()  # one image-token appears then freeze.
        if should_freeze:
            ctx.freeze_context = freeze_context
        else:
            ctx.freeze_context = contextlib.nullcontext
        return outputs

    @staticmethod
    def backward(ctx, *last_hidden_grad):
        """backward"""
        with ctx.freeze_context():
            input_embeds_grad = ctx.bwf(*last_hidden_grad)
        return input_embeds_grad


@paddle.no_grad()
def construct_types_for_video(image_mask, token_type_ids, image_type_ids):
    """
    construct_types_for_video,
    Args:
        image_mask: [B], 1 if is `im_patch_id` else 0
        token_type_ids: [B], see `IDTYPES_2_ID`
        image_type_ids: [B], see `IMAGETYPES_2_ID`
    Returns:
        image_is_video: shape as[image_B,], image_B is batch size of image_features.
                        Value is 0/1, 1 is video, 0 is image.
        compressed_image_indices: shape as [image_seq,],
                        image_seq the num of image_placeholder in input_ids.
                        Value is 0/1, 1 is video, 0 is image.
        video_images_with_placeholder: shape as [padded_video_b,],
                        padded_video_b is padding num of video, as batch size before temporal_linear.
                        Value is 0/1, 1 is video, 0 is pad.
    """
    if image_type_ids is not None:
        image_type_ids = image_type_ids[image_type_ids >= 0]  # remove padding
        # placeholder before conv3d
        video_images_with_placeholder = image_type_ids[image_type_ids != IMAGETYPES_2_ID["image"]]
        if video_images_with_placeholder.shape[0] != 0:
            video_images_with_placeholder = video_images_with_placeholder == IMAGETYPES_2_ID["video"]
            video_images_with_placeholder = video_images_with_placeholder.astype("int64")
        else:
            video_images_with_placeholder = None

        # image_is_video the type of visual feature to extract video and image from visual feature
        # 1 is video, 0 is image
        image_is_video = image_type_ids[image_type_ids != IMAGETYPES_2_ID["padded_image"]]

        assert image_is_video.shape[0] != 0, f"image_is_video is 0 shape, {image_is_video.shape}"

        image_is_video = image_is_video == IMAGETYPES_2_ID["video"]
        image_is_video = image_is_video.astype("int64")

    else:
        video_images_with_placeholder = None
        image_is_video = None

    # compressed_image_indices is type id after compressed visual feature，0 is image，1 is video after conv3d
    compressed_image_indices = token_type_ids[image_mask]
    compressed_image_indices = compressed_image_indices == TokenType.video
    compressed_image_indices = compressed_image_indices.astype("int64")

    return image_is_video, compressed_image_indices, video_images_with_placeholder


class VariableResolutionResamplerModel(nn.Layer):
    """
    VariableResolutionResamplerModel, support variable resolution
    """

    def __init__(self, in_dim, out_dim, spatial_conv_size, temporal_conv_size, config):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_recompute_resampler = config.use_recompute_resampler
        self.use_temporal_conv = config.use_temporal_conv
        self.tensor_parallel_degree = config.tensor_parallel_degree

        # compress spatial
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size
        # compress temporal
        self.temporal_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size * self.temporal_conv_size

        with paddle.utils.unique_name.guard("mm_resampler_"):

            self.spatial_linear = nn.Sequential(
                (
                    RowSequenceParallelLinear(
                        self.spatial_dim,
                        self.spatial_dim,
                        input_is_parallel=True,
                        has_bias=True,
                        fuse_matmul_bias=True,
                    )
                    if config.tensor_parallel_degree > 1
                    else nn.Linear(self.spatial_dim, self.spatial_dim)
                ),
                nn.GELU(),
                nn.Linear(self.spatial_dim, self.spatial_dim),
                nn.LayerNorm(self.spatial_dim, epsilon=1e-6),
            )

            if self.use_temporal_conv:
                self.temporal_linear = nn.Sequential(
                    nn.Linear(self.temporal_dim, self.spatial_dim),
                    nn.GELU(),
                    nn.Linear(self.spatial_dim, self.spatial_dim),
                    nn.LayerNorm(self.spatial_dim, epsilon=1e-6),
                )

            self.mlp = nn.Linear(self.spatial_dim, self.out_dim)

            out_config = deepcopy(config)
            out_config.hidden_size = out_dim
            # Note(GuoxiaWang): fuse can reduce gpu peak memory
            out_config.fuse_rms_norm = out_config.resampler_fuse_rms_norm
            self.after_norm = RMSNorm(out_config)

            if config.tensor_parallel_degree > 1:
                for idx in [2, 3]:
                    mark_as_sequence_parallel_parameter(self.spatial_linear[idx].weight)
                    mark_as_sequence_parallel_parameter(self.spatial_linear[idx].bias)

                if self.use_temporal_conv:
                    for idx in [0, 2, 3]:
                        mark_as_sequence_parallel_parameter(self.temporal_linear[idx].weight)
                        mark_as_sequence_parallel_parameter(self.temporal_linear[idx].bias)

                mark_as_sequence_parallel_parameter(self.mlp.weight)
                mark_as_sequence_parallel_parameter(self.mlp.bias)
                mark_as_sequence_parallel_parameter(self.after_norm.weight)

    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        reshape before linear to imitation conv
        """
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size**2)])
        return x

    def forward(self, x, image_mask, token_type_ids, image_type_ids, grid_thw):
        """
        x: image_features
        image_mask: [B]
        token_types_ids: [B]
        image_type_ids:  [B_image]
        grid_thw: [B_image, 3]
        """
        assert image_type_ids is not None

        def fwd_spatial(x):
            """
            x in the shape of [S, H]
            S is ordered in the following way: [ [patch_h*patch_w (row-major traversal)] * patch_time]
            H is simply hidden
            """
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            num_pad = 0
            if self.tensor_parallel_degree > 1:
                num_pad = (
                    x.shape[0] + self.tensor_parallel_degree - 1
                ) // self.tensor_parallel_degree * self.tensor_parallel_degree - x.shape[0]

            if num_pad > 0:
                x = paddle.nn.functional.pad(x, [0, num_pad, 0, 0])

            x = self.spatial_linear(x)

            if self.tensor_parallel_degree > 1:
                x = AllGatherOp.apply(x)

            if num_pad > 0:
                x = x[:-num_pad]
            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False):
            """
            x: [S, H]
            grid_thw: [S, 3]
                the second dimension: [t, h, w]
            """

            grid_thw_cpu = grid_thw.numpy()
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (self.spatial_conv_size**2)
            batch_offset = np.empty(tokens_per_img_or_vid.size, dtype=tokens_per_img_or_vid.dtype)
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            assert self.temporal_conv_size == 2, f"Hard Code: temporal_conv_size==2, got:{self.temporal_conv_size}"

            # TODO: support any temporal conv size
            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(grid_t, grid_hw_after_conv, batch_offset):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = paddle.to_tensor(np.concatenate(slice_offsets, axis=-1))

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(grid_t, grid_hw_after_conv, batch_offset):
                for temp_offset in range(1 if temporoal_size > 1 else 0, temporoal_size, 2):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = paddle.to_tensor(np.concatenate(slice_offsets2, axis=-1))

            x_timestep_1 = paddle.gather(x, slice_offsets, axis=0)
            x_timestep_2 = paddle.gather(x, slice_offsets2, axis=0)
            x = paddle.concat([x_timestep_1, x_timestep_2], axis=-1)

            return x

        def fwd_temporal(x):
            num_pad = 0
            if self.tensor_parallel_degree > 1:
                num_pad = (
                    x.shape[0] + self.tensor_parallel_degree - 1
                ) // self.tensor_parallel_degree * self.tensor_parallel_degree - x.shape[0]
            if num_pad > 0:
                x = paddle.nn.functional.pad(x, [0, num_pad, 0, 0])
            if self.tensor_parallel_degree > 1:
                x = ScatterOp.apply(x, axis=0)
            x = self.temporal_linear(x)

            if self.use_recompute_resampler:
                num_pad = paddle.to_tensor(num_pad)

            return x, num_pad

        def fwd_mlp(x):
            x = self.mlp(x)
            x = self.after_norm(x)
            if self.tensor_parallel_degree > 1:
                x = AllGatherOp.apply(x)
            return x

        num_pad = 0
        if self.use_recompute_resampler:
            x = recompute(fwd_spatial, x)
            if self.use_temporal_conv:
                x = recompute(fwd_placeholder, x, grid_thw)
                x, num_pad = recompute(fwd_temporal, x)
            x = recompute(fwd_mlp, x)
        else:
            x = fwd_spatial(x)
            if self.use_temporal_conv:
                x = fwd_placeholder(x, grid_thw)
                x, num_pad = fwd_temporal(x)
            x = fwd_mlp(x)
        if num_pad is not None and num_pad > 0:
            x = x[:-num_pad]
        return x

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )
        res = {"spatial_linear.0.weight": partial(fn, is_column=False)}  # row parallel
        return res


class ErniePretrainingCriterion(ErniePretrainingCriterionBase):
    """
    ErnieMoEVL -> ErnieMoE -> Ernie
    """

    def __init__(self, config):
        super().__init__(config)
        self.im_patch_id = config.im_patch_id
        self.max_text_id = config.max_text_id
        self.use_one_head = config.mm_vocab_size == 0

    def forward(
        self,
        scores_text,
        scores_image,
        labels,
        token_type_ids_shifted,
        token_type_ids_untouched,
        lm_weight=None,
        lm_bias=None,
        mm_weight=None,
        mm_bias=None,
        router_loss=None,
    ):
        """
        text-image separate Criterion, CE Loss for text only, other losses will be updated into global_training_logs.
        Args:
            score_text: text logits, only contains text data.
            scores_text_in_image: only contains text data logits。
            scores_image: only contains image sepecial-token logits。
            labels: original label，contains text/image special-token and ignored-index。
            token_type_ids_shifted: `labels` token-type。
            router_loss: router_loss
        Returns:
            loss: text-only CE loss
            loss_sum. text-only CE loss_sum
        """
        if self.config.use_recompute_loss_fn and self.config.use_fused_head_and_loss_fn:
            with paddle.no_grad():
                if token_type_ids_shifted.unique().shape[0] > 1:
                    labels, token_type_ids_shifted = TensorBalanceByTokenType.apply(
                        labels.squeeze(0),
                        token_type_ids_shifted,
                        is_tensor_sharded=False,
                    )
                else:
                    labels = ScatterOp.apply(labels, axis=-1)

        if self.use_one_head:
            if self.config.use_recompute_loss_fn or self.config.use_sparse_head_and_loss_fn:
                loss, loss_sum = super().forward((scores_text.unsqueeze(0), lm_weight, lm_bias), labels.unsqueeze(0))
            else:
                loss, loss_sum = super().forward(scores_text.unsqueeze(0), labels.unsqueeze(0))
            self.update_log(loss, token_type_ids_untouched)
            return loss, loss_sum

        image_mask_shifted = token_type_ids_shifted == TokenType.image
        text_pos_shifted = token_type_ids_shifted == TokenType.text

        assert scores_text is not None, f"no text or image token provided, text: {scores_text}"

        if scores_text is not None:
            labels_text = labels[text_pos_shifted]
            assert labels_text.size > 0, labels
            if self.config.use_recompute_loss_fn or self.config.use_sparse_head_and_loss_fn:
                assert lm_weight is not None and mm_weight is not None
                loss, loss_sum = super().forward(
                    (scores_text.unsqueeze(0), lm_weight, lm_bias),
                    labels_text.unsqueeze(0),
                )
            else:
                loss, loss_sum = super().forward(scores_text.unsqueeze(0), labels_text.unsqueeze(0))
            self.update_log(loss, token_type_ids_untouched)
        else:
            assert 0
            loss = paddle.zeros([], dtype="float32")
            loss.stop_gradient = False

        if scores_image is not None:
            labels_image = labels[image_mask_shifted]
            assert labels_image.size > 0, labels
            labels_image = paddle.where(
                labels_image >= 0, labels_image - self.max_text_id, labels_image
            )  # do not move ignored-index
            if self.config.use_recompute_loss_fn or self.config.use_sparse_head_and_loss_fn:
                assert mm_weight is not None and mm_bias is not None
                loss_image, _ = super().forward(
                    (scores_image.unsqueeze(0), mm_weight, mm_bias),
                    labels_image.unsqueeze(0),
                )
            else:
                loss_image, _ = super().forward(scores_image.unsqueeze(0), labels_image.unsqueeze(0))
            global_training_logs.update(image_special_token_loss=loss_image.detach())
            loss = loss + loss_image - loss_image.detach()

        if router_loss is not None and isinstance(router_loss, paddle.Tensor):
            global_training_logs.update(router_loss=router_loss.detach())
            loss = loss + router_loss - router_loss.detach()
        return loss, loss_sum

    def update_log(self, loss, token_type_ids_untouched):
        """update log"""
        pure_text = (
            token_type_ids_untouched == TokenType.text
        ).all()  # if all tokens are textual, it's a pure text dataset
        has_video = (token_type_ids_untouched == TokenType.video).any()  # if one token is video, then it's video data
        has_image = (token_type_ids_untouched == TokenType.image).any()  # if one token is image, then it's image data
        if pure_text:
            global_training_logs.update(lm_loss=loss.detach())
        elif has_video:
            global_training_logs.update(video_loss=loss.detach())
        elif has_image:
            global_training_logs.update(image_loss=loss.detach())
        else:
            raise RuntimeError(f"input token must be one of [text, video, image]: {token_type_ids_untouched}")
        return


def calc_multimodal_logits(
    last_hidden_state: paddle.Tensor,
    lm_head_weight: paddle.Tensor,
    lm_head_bias: paddle.Tensor,
    mm_head_weight: paddle.Tensor,
    mm_head_bias: paddle.Tensor,
    token_type_ids_shifted: paddle.Tensor,
    config: Ernie4_5_VLMoeConfig,
):
    """
    calculate logits for pure text, multimodal text, and image
    Args:
        last_hidden_state: The hidden of the last layer, in sequence-parallel, is in the split state.
        ...
        token_type_ids_shifted: # Non-sp split tensor
            The token-type-ids at the label position is used to select the lm-head corresponding to each token.
            Note: In the id sequence of alternating images and texts, the last text token will predict the image id,
            and vice versa, so it is necessary to select the lmhead weight corresponding to the label type.
    """
    # Align the type of ids with the type of label. For the last ids, assume that the token type remains unchanged.
    # TODO: Pass token-type-ids from reader
    # token_type_ids_shifted = paddle.concat([token_type_ids[:, 1:], token_type_ids[:, -1:]], 1)  #

    if config.use_recompute_loss_fn and config.use_fused_head_and_loss_fn:
        if config.sequence_parallel:
            if token_type_ids_shifted.unique().shape[0] > 1:  # Multimodal data
                last_hidden_state, token_type_ids_shifted = TensorBalanceByTokenType.apply(
                    last_hidden_state, token_type_ids_shifted
                )
            else:
                with paddle.no_grad():
                    token_type_ids_shifted = ScatterOp.apply(token_type_ids_shifted, axis=-1)
                    token_type_ids_shifted = token_type_ids_shifted.reshape([-1])
        else:
            token_type_ids_shifted = token_type_ids_shifted.reshape([-1])
    else:
        if config.sequence_parallel:
            last_hidden_state = GatherOp.apply(last_hidden_state)
            last_hidden_state = last_hidden_state.reshape([1, -1, last_hidden_state.shape[-1]])

        assert last_hidden_state.shape[:2] == token_type_ids_shifted.shape, (
            last_hidden_state.shape,
            token_type_ids_shifted.shape,
        )
    parallel_matmul_tp = partial(
        parallel_matmul,
        tensor_parallel_degree=config.tensor_parallel_degree,
        tensor_parallel_output=config.tensor_parallel_output,
        fuse_linear=config.fuse_linear,
        transpose_y=config.tie_word_embeddings,
    )

    if mm_head_weight is None:
        if config.use_recompute_loss_fn or config.use_sparse_head_and_loss_fn:
            return last_hidden_state, None, None
        score_text = parallel_matmul_tp(
            last_hidden_state,
            lm_head_weight,
            lm_head_bias,
            transpose_y=config.tie_word_embeddings,
        )
        return score_text, None, None

    image_mask_shifted = token_type_ids_shifted == TokenType.image
    text_pos_shifted = token_type_ids_shifted == TokenType.text

    if text_pos_shifted.any().item() > 0:
        if config.use_recompute_loss_fn or config.use_sparse_head_and_loss_fn:
            score_text = last_hidden_state[text_pos_shifted]
        else:
            score_text = parallel_matmul_tp(last_hidden_state[text_pos_shifted], lm_head_weight, lm_head_bias)
    else:
        score_text = None

    if mm_head_weight is not None and image_mask_shifted.any().item() > 0:
        if config.use_recompute_loss_fn or config.use_sparse_head_and_loss_fn:
            score_image = last_hidden_state[image_mask_shifted]
        else:
            score_image = parallel_matmul_tp(last_hidden_state[image_mask_shifted], mm_head_weight, mm_head_bias)
    else:
        score_image = None

    return score_text, score_image


class Ernie4_5_MoeVLHead(Ernie4_5_LMHead):
    """Ernie4_5_MoeVLHead"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sequence_parallel = config.sequence_parallel
        if config.mm_vocab_size > 0:
            mm_vocab_config = deepcopy(config)
            mm_vocab_config.vocab_size = config.mm_vocab_size
            assert mm_vocab_config.vocab_size > 0, mm_vocab_config
            assert mm_vocab_config.im_patch_id >= mm_vocab_config.max_text_id, mm_vocab_config
            self.mm_head = Ernie4_5_LMHead(mm_vocab_config)
        else:
            self.mm_head = None

    def forward(self, hidden_state, token_type_ids_labels, use_cache=False):
        """
        Args:
            hidden_state(paddle.Tensor): hidden state
            token_type_ids_labels(paddle.Tensor): token ids
            use_cache(bool): whether to use cache, default is False

        Returns:
            logits_text(paddle.Tensor): text logits
            logits_image(paddle.Tensor): image logits
        """
        if not use_cache:
            mm_head_weight = self.mm_head.weight if self.mm_head is not None else None
            mm_head_bias = self.mm_head.bias if self.mm_head is not None else None
            logits_text, logits_image, *_ = calc_multimodal_logits(  # note!!
                hidden_state,
                self.weight,
                self.bias,
                mm_head_weight,
                mm_head_bias,
                token_type_ids_labels,
                self.config,
            )
            return logits_text, logits_image
        else:
            if self.config.sequence_parallel:
                hidden_state = GatherOp.apply(hidden_state)
                logger.warning("you are trying to generate with sequence-parallel model")
                hidden_state = hidden_state.reshape([-1, self.config.max_sequence_length, hidden_state.shape[-1]])
            # assert not self.config.sequence_parallel, "generate is not supported in sequence-parallel mode"
            # TODO，support lm_head decode only
            return (
                parallel_matmul(
                    hidden_state[:, -1:, :],
                    self.weight,
                    self.bias,
                    transpose_y=self.config.tie_word_embeddings,
                    tensor_parallel_degree=self.config.tensor_parallel_degree,
                    tensor_parallel_output=False,
                    fuse_linear=self.config.fuse_linear,
                ),
                None,
            )


class Ernie4_5_VLMoeForConditionalGeneration(Ernie4_5_MoeForCausalLM):
    """Ernie4_5_VLMoeForConditionalGeneration"""

    config_class = Ernie4_5_VLMoeConfig
    main_input_name = "pixel_values"
    transpose_weight_keys = [
        "spatial_linear.0",
        "temporal_linear.0",
        "spatial_linear.2",
        "temporal_linear.2",
        "mlp",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # "gate",
        "proj",
        "qkv",
        "fc1",
        "fc2",
    ]

    def __init__(self, config: Ernie4_5_VLMoeConfig):
        """
        initialize Ernie4_5_VLMoeForConditionalGeneration

        Args:
            config(Ernie4_5_VLMoeConfig): Model configuration.
        """
        super().__init__(config)
        self.criterion = ErniePretrainingCriterion(config)
        self.modality_detach = config.modality_detach

        if config.mm_vocab_size > 0:
            if config.tensor_parallel_degree > 1:
                self.mm_embed_tokens = VocabParallelEmbedding(config.mm_vocab_size, config.hidden_size)
            else:
                self.mm_embed_tokens = nn.Embedding(config.mm_vocab_size, config.hidden_size)
        else:
            self.mm_embed_tokens = None

        self.model.resampler_model = VariableResolutionResamplerModel(
            config.vision_config.hidden_size,
            config.hidden_size,
            config.spatial_conv_size,
            config.temporal_conv_size,
            config=config,
        )

        self._modality_param_mapping = None
        self.image_preprocess = None
        self.lm_head = Ernie4_5_MoeVLHead(config)
        self.vision_model = DFNRopeVisionTransformerPretrainedModel(config=config.vision_config)

        self.tie_weights()  # maybe weight share

    def add_vision_model(
        self,
        encoder: nn.Layer,
    ):
        """add_vision_model"""
        self.vision_model = encoder
        self._set_modality_param_mapping()

    def add_image_preprocess(self, preprocess):
        """add image preprocess"""
        logger.info("image preprocess is set")
        self.image_preprocess = preprocess

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = Ernie4_5_MoeForCausalLM._get_tensor_parallel_mappings(config, is_split=is_split)
            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
        resampler_actions = VariableResolutionResamplerModel._get_tensor_parallel_mappings(config, is_split=is_split)
        mappings.update({f"model.resampler_model.{k}": v for k, v in resampler_actions.items()})

        if config.mm_vocab_size > 0:
            mappings.update(
                {
                    "mm_embed_tokens.weight": partial(fn, is_column=False),
                    "lm_head.mm_head.weight": partial(fn, is_column=True),
                    "lm_head.mm_head.bias": partial(fn, is_column=True),
                }
            )
        return mappings

    def _set_modality_param_mapping(self):
        """set modality parameter mapping"""
        lm_pattern = get_backbone_lm_param_regex(self.config)
        self._modality_param_mapping = defaultdict(lambda: [])
        for name, param in self.named_parameters():
            monkey_patch_param_hook(param)
            expert_type = getattr(param, "expert_type", None)
            if "vision_model" in name:
                self._modality_param_mapping["vit"].append((name, param, create_freeze_hook(name, param)))
                param.color = "vit"
            elif lm_pattern.match(name) or expert_type == "expert_type_0":
                self._modality_param_mapping["lm"].append((name, param, create_freeze_hook(name, param)))
                param.color = "lm"
            else:
                self._modality_param_mapping["mm"].append((name, param, create_freeze_hook(name, param)))
                param.color = "mm"
        debug_msg = {k: [i[0] for i in v] for k, v in self._modality_param_mapping.items()}
        logger.info(f"modality_param_mapping: {json.dumps(debug_msg, ensure_ascii=False, indent=2)}")

    def update_params_stat(self, param_group, stop_gradient):
        """freeze mm"""
        assert param_group in (
            "lm",
            "mm",
            "vit",
        ), "param_group must be in ('lm', 'mm', 'vit')"
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        if self._modality_param_mapping.get(param_group):
            for name, param, _ in self._modality_param_mapping[param_group]:
                # logger.info(f"mm: {name} set_stop_gradient to {stop_gradient}")
                param.stop_gradient = stop_gradient

    def freeze_vision(self):
        """freeze_vision"""
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        for name, param, _ in self._modality_param_mapping.get("vit", []):
            logger.info(f"Freezing vision parameter: {name}")
            param.stop_gradient = True
        self.vision_model.config.freeze_vision = True

    def vision_forward(
        self,
        images,
        image_position_ids,
        image_attention_mask,
        grid_thw,
    ):
        """vision_forward"""
        if self.image_preprocess is not None:
            assert images.dtype == paddle.uint8, images.dtype
            images = self.image_preprocess.rescale_factor * images.astype("float32")
            images = (images - self.image_preprocess.image_mean_tensor) / self.image_preprocess.image_std_tensor
            images = images.astype("bfloat16")
        else:
            assert images.dtype == paddle.bfloat16, images.dtype
        # logger.info(f"extract feature input - {images}--{grid_thw}")
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                paddle.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [0, 0, 1, 0],
                value=1,
            )
        image_features = self.vision_model.extract_feature(images, grid_thw)
        return image_features

    def vision_mapping_forward(
        self,
        token_type_ids,
        token_type_ids_w_video,
        input_ids,
        mm_input_ids,
        image_features,
        inputs_embeds,
        image_type_ids,
        grid_thw,
    ):
        """vision_mapping_forward"""
        if self.mm_embed_tokens is not None:
            mm_ids_features = self.mm_embed_tokens(mm_input_ids - self.config.max_text_id)
            inputs_embeds[token_type_ids == TokenType.image] = mm_ids_features[token_type_ids == TokenType.image]
        image_mask = input_ids == self.config.im_patch_id
        image_features = self.model.resampler_model(
            image_features,
            image_mask,
            token_type_ids_w_video,
            image_type_ids,
            grid_thw,
        )

        if image_features.dim == 2:
            B, N, C = image_features.shape
            image_features = image_features.reshape([B * N, C]).astype(inputs_embeds.dtype)
        # Will overwrite the part of `ids==im_patch_id` in `mm_ids_features`
        inputs_embeds[image_mask] = image_features
        # # TODO Normalize some parameters, detach some text, print image token
        # text_token_norm = inputs_embeds[input_ids != self.config.im_patch_id].norm(axis=-1).mean()
        # image_token_norm = image_features.norm(axis=-1).mean()
        # image_features /= paddle.sqrt(image_token_norm / text_token_norm)
        return inputs_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids,
        images=None,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        image_position_ids=None,
        image_attention_mask=None,
        token_type_ids=None,
        image_type_ids=None,
        grid_thw=None,
        **kwargs,
    ):
        """
        Prepare inputs for the decoder that can be used for generation.

        Args:
            input_ids (paddle.Tensor): Input ids.
            images (paddle.Tensor): Images. Default to None.
            use_cache (bool): Whether to use cache. Default to False.
            past_key_values (list): Past key values. Default to None.
            inputs_embeds (paddle.Tensor): Input embeddings. Default to None.
            image_position_ids (paddle.Tensor): Image position ids. Default to None.
            image_attention_mask (paddle.Tensor): Image attention mask. Default to None.
            token_type_ids (paddle.Tensor): Token type ids. Default to None.
            image_type_ids (paddle.Tensor): Image type ids. Default to None.
            grid_thw (paddle.Tensor): Grid thw. Default to None.
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            image_type_ids = image_type_ids[:, -1:] if image_type_ids is not None else None

        attention_mask = kwargs.get("attention_mask", None)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "return_dict": True,
                "images": images,
                "image_position_ids": image_position_ids,
                "image_attention_mask": image_attention_mask,
                "image_type_ids": image_type_ids,
                "token_type_ids": paddle.concat(
                    [
                        token_type_ids,
                        paddle.zeros([len(token_type_ids), 1], token_type_ids.dtype),
                    ],
                    axis=-1,
                ),
                "grid_thw": grid_thw,
            }
        )

        if self.config.rope_3d:
            model_inputs.update({"position_ids": kwargs["position_ids"]})

        return model_inputs

    def _post_init(self, original_init, *args, **kwargs):
        """
        Label all multimodal parameters in the model, only head and Embedding
        Experts parameters are already labeled
        """
        super()._post_init(self, original_init, *args, **kwargs)
        if self.mm_embed_tokens is not None:
            self.mm_embed_tokens.weight.expert_type = "expert_type_1"
        if self.lm_head.mm_head is not None:
            self.lm_head.mm_head.weight.expert_type = "expert_type_1"
        if getattr(self.lm_head.mm_head, "bias", None) is not None:
            self.lm_head.mm_head.bias.expert_type = "expert_type_1"

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[paddle.Tensor] = None,
        images: Optional[paddle.Tensor] = None,
        ignored_index: Optional[int] = 0,
        return_dict: Optional[bool] = None,
        image_position_ids: Optional[paddle.Tensor] = None,
        image_attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        image_type_ids: Optional[paddle.Tensor] = None,
        grid_thw: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        """
        Forward for Ernie4_5_VLMoeForConditionalGeneration

        Args:
            input_ids (paddle.Tensor): Input ids.
            position_ids (Optional[paddle.Tensor], optional): Position ids. Defaults to None.
            attention_mask (Optional[paddle.Tensor], optional): Attention mask. Defaults to None.
            past_key_values (Optional[List[paddle.Tensor]], optional): Past key values. Defaults to None.
            use_cache (Optional[bool], optional): Use cache. Defaults to None.
            output_attentions (Optional[bool], optional): Output attentions. Defaults to None.
            output_hidden_states (Optional[bool], optional): Output hidden states. Defaults to None.
            labels (Optional[paddle.Tensor], optional): Labels. Defaults to None.
            images (Optional[paddle.Tensor]): Images. Defaults to None.
            ignored_index (Optional[int], optional): Ignored index. Defaults to 0.
            return_dict (Optional[bool], optional): Return dict. Defaults to None.
            image_position_ids (Optional[paddle.Tensor], optional): Image position ids. Defaults to None.
            image_attention_mask (Optional[paddle.Tensor], optional): Image attention mask. Defaults to None.
            token_type_ids (Optional[paddle.Tensor], optional): Token type ids. Defaults to None.
            image_type_ids (Optional[paddle.Tensor], optional): Image type ids. Defaults to None.
            grid_thw (Optional[paddle.Tensor], optional): Grid thw. Defaults to None.
        """
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_mask = input_ids == self.config.im_patch_id

        if past_key_values is None:
            if images is not None:
                assert (image_mask).any().item(), (
                    image_mask.numpy().tolist(),
                    input_ids.numpy().tolist(),
                    self.config.im_patch_id,
                    images.shape,
                )
                image_features = self.vision_forward(
                    images,
                    image_position_ids,
                    image_attention_mask,
                    grid_thw,
                )
                if self.config.tensor_parallel_degree > 1:
                    S, C = image_features.shape
                    # When scatterOp cuts feature + 4-in-1, the features of the 4 tokens are merged together in advance.
                    image_features = image_features.reshape([-1, C * self.config.spatial_conv_size**2])
                    image_features = ScatterOp.apply(image_features, axis=-1)
                    image_features = image_features.reshape([S, -1])
            else:
                image_features = None  # no more faking
        else:
            image_features = None
        # inputs_embeds.stop_gradient = False
        # 0 == plain text, 1 == image, 1 will activate experts > 1
        if token_type_ids is None:
            # assert 0, f"using default token_type_ids: {token_type_ids}, image_type_ids: {image_type_ids}"
            token_type_ids = image_mask.astype("int64")
            token_type_ids_labels = paddle.concat([token_type_ids[:, 1:], token_type_ids[:, -1:]], 1)
        else:
            assert (
                token_type_ids.shape[1] == input_ids.shape[1] + 1
            ), f"token_type:{token_type_ids.shape}, ids:{input_ids.shape}"
            token_type_ids_labels = token_type_ids[..., 1:]
            # token_type_ids = token_type_ids[..., :-1]

        lm_input_ids = input_ids.clone()
        mm_input_ids = input_ids.clone()
        if self.mm_embed_tokens is not None:
            lm_input_ids[token_type_ids[..., :-1] == TokenType.image] = 0
            mm_input_ids[token_type_ids[..., :-1] == TokenType.text] = self.config.max_text_id
        # During embedding lookup, `max_text_id` will be subtracted uniformly.
        # The text part id is replaced with `max_text_id` + 1 to distinguish it from `im_patch_id`.
        # The replacement part will not be added to the final input_embeds so it doesn't matter.
        # assert self.config.max_text_id + 1 != self.config.im_patch_id,  \
        #      f'max_text_id:{self.config.max_text_id}, im_pach_id:{self.config.im_patch_id}'

        if self.training and self.modality_detach:
            assert not return_dict, "modality detach no support `return_dict`"
            assert not use_cache and past_key_values is None and not output_attentions and not output_hidden_states
            is_first_fwd = not framework._dygraph_tracer()._has_grad

            def fwdfn(
                token_type_ids,
                image_features,
                _,
                grid_thw,
            ):
                nonlocal input_ids, token_type_ids_labels, mm_input_ids, image_type_ids
                """During the backward of this function, the stop_graident attribute of param is reset"""
                inputs_embeds = self.model.embed_tokens(lm_input_ids).astype(self.embed_tokens.weight.dtype)
                token_type_ids_w_video = token_type_ids[..., :-1].clone()
                token_type_ids[token_type_ids == TokenType.video] = TokenType.image
                if images is not None:
                    inputs_embeds = self.vision_mapping_forward(
                        token_type_ids[..., :-1],
                        token_type_ids_w_video,
                        input_ids,  # cached
                        mm_input_ids,
                        image_features,
                        inputs_embeds,
                        image_type_ids,
                        grid_thw,
                    )
                else:
                    pass  # do nothing, should not hang under DygraphShardingOptimizerV2

                outputs = self.model(
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )

                logits_all = self.lm_head(
                    outputs.last_hidden_state,
                    token_type_ids_labels,
                    use_cache,
                )
                return logits_all + (outputs.router_loss,)

            @contextlib.contextmanager
            def freeze_context():
                assert (
                    self._modality_param_mapping
                ), "call `model.freeze_lm(stop_gradient=False) first before modality detach`"
                unfreeze_handler = [
                    p.register_hook(hook, 0) for n, p, hook in self._modality_param_mapping["lm"]
                ]  # param has been monkey-patched to have .hooks method
                yield  # backward fun
                for h in unfreeze_handler:
                    assert h.remove()

            t = paddle.zeros([])
            t.stop_gradient = False

            logits, logits_image, router_loss = ModalityDetach.apply(
                token_type_ids,
                image_features,
                t,
                grid_thw,
                fn=fwdfn,
                is_first_fwd=is_first_fwd,
                freeze_context=freeze_context,
            )
        else:
            inputs_embeds = self.model.embed_tokens(lm_input_ids)
            token_type_ids_w_video = token_type_ids[..., :-1].clone()
            token_type_ids[token_type_ids == TokenType.video] = TokenType.image

            if images is not None and image_features is not None:
                inputs_embeds = self.vision_mapping_forward(
                    token_type_ids[..., :-1],
                    token_type_ids_w_video,
                    input_ids,
                    mm_input_ids,
                    image_features,
                    inputs_embeds,
                    image_type_ids,
                    grid_thw,
                )
            else:
                pass  # do nothing, should not hang under DygraphShardingOptimizerV2

            outputs = self.model(
                position_ids=position_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,  #
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            logits_all = self.lm_head(
                outputs.last_hidden_state,
                token_type_ids_labels,
                use_cache,
            )
            logits, logits_image = logits_all
            router_loss = outputs.router_loss

        mm_head_weight = self.lm_head.mm_head.weight if self.lm_head.mm_head is not None else None
        mm_head_bias = self.lm_head.mm_head.bias if self.lm_head.mm_head is not None else None
        if return_dict:  # aka Generate Decoding
            if labels is not None:
                loss, _ = self.criterion(
                    logits,
                    None,
                    labels,
                    token_type_ids_labels,
                    token_type_ids,
                    self.lm_head.weight,
                    self.lm_head.bias,
                    mm_head_weight,
                    mm_head_bias,
                    router_loss=outputs.router_loss,
                )
            else:
                loss = None
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_loss=outputs.router_loss,
            )
        # Pretrain & Eval must have labels
        assert labels is not None
        loss = self.criterion(
            logits,
            logits_image,
            labels,
            token_type_ids_labels,
            token_type_ids,
            self.lm_head.weight,
            self.lm_head.bias,
            mm_head_weight,
            mm_head_bias,
            router_loss=router_loss,
        )
        return loss

    @staticmethod
    def _resolve_prefix_keys(state_keys_base, state_keys_real, ignore_error=False, base_model_prefix=None):
        """_resolve_prefix_keys"""
        # state_keys_map base to real
        state_keys_map = {}

        if base_model_prefix:
            for k in state_keys_real:
                if k.startswith("lm_head."):
                    continue
                # remove real key name `base_model_prefix` + '.'
                state_keys_map[k[len(base_model_prefix + ".") :]] = k
            return state_keys_map

        # sorted by length，match from long to short for A.key B.key ...
        state_keys_base = sorted(state_keys_base, key=lambda x: len(x), reverse=True)
        state_keys_real = set(state_keys_real)

        for key in state_keys_base:
            for x in state_keys_real:
                if "mm_embed_tokens" in x:
                    if "mm_embed_tokens" in key:
                        state_keys_map[key] = x
                        break
                elif x.endswith(key):
                    state_keys_map[key] = x
                    break
            if key not in state_keys_map:
                if not ignore_error:
                    logger.error(f"could not find name {key} in loaded state dict!")
            else:
                state_keys_real.remove(state_keys_map[key])

        return state_keys_map
