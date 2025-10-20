# !/usr/bin/env python3

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

"""
Vision Model trainable callback for training the vision model.
"""
import math
from types import MethodType

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.fleet import fleet
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients_with_group,
)
from paddle.nn import functional as F
from paddleformers.trainer.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from paddleformers.trainer.training_args import TrainingArguments
from paddleformers.utils.log import logger

from ernie.comm_utils import all_gather, mp_slice, profile, scatter_varlen


def showmem(msg):
    """_summary_

    Args:
        msg (_type_): _description_
    """
    logger.info(
        f" [Mem]: {msg} "
        f"Activated: {paddle.device.cuda.memory_allocated()/1024/1024:.3f} MB, "
        f"Reserved: {paddle.device.cuda.memory_reserved()/1024/1024:.3f} MB, "
        f"max-Activated: {paddle.device.cuda.max_memory_allocated()/1024/1024:.3f} MB, "
        f"max-Reserved: {paddle.device.cuda.max_memory_reserved()/1024/1024:.3f} MB "
    )


class VitTrainableCallback(TrainerCallback):
    """_summary_

    Args:
        TrainerCallback (_type_): _description_
    """

    def __init__(self, args, model, auto_cast_func=None, patches_per_image=256):
        """_summary_

        Args:
            args (_type_): _description_
            model (_type_): _description_
            auto_cast_func (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.vision_model = model.vision_model
        self.images_buffer = []
        self.images_features = []
        self.auto_cast_func = auto_cast_func
        self.meta = None
        self.patches_per_image = patches_per_image
        self.vit_second_fwd_batch_size = args.vit_second_fwd_batch_size
        assert (
            args.log_global_grad_norm
        ), "need hacked `log-global-grad-norm` to support pp non-dist-var gradnorm"
        for n, p in self.vision_model.named_parameters():
            p.pp_distributed = False  # used to avoid duplicate gradnorm

        hcg = fleet.get_hybrid_communicate_group()
        self.pp_group = hcg.get_pipe_parallel_group()
        self.sd_group = hcg.get_sharding_parallel_group()
        self.mp_group = hcg.get_model_parallel_group()
        self.mp_degree = max(1, hcg.get_model_parallel_world_size())
        logger.info(
            f"self.pp_group: {self.pp_group} - {args.pipeline_parallel_degree} -- {args.pipeline_parallel_rank}"
        )
        # hack extract_feature func
        assert hasattr(self.vision_model, "extract_feature")

        self.ori_extract_feature = self.vision_model.extract_feature

        def extract_feature_wrapper(inner_self, images, grid_thw, second_fwd=False):
            # recompute images
            if second_fwd:
                ori_freeze_vision = getattr(inner_self.config, "freeze_vision", True)
                inner_self.config.freeze_vision = False
            else:
                self.images_buffer.append((images.detach(), grid_thw))
            image_feature = self.ori_extract_feature(
                images, grid_thw, second_fwd=second_fwd
            )
            if self.meta is None:
                meta = image_feature.shape[1:]
                meta[-1] = meta[-1] // self.mp_degree
                self.meta = (meta, image_feature.dtype)

            if second_fwd:
                inner_self.config.freeze_vision = ori_freeze_vision
            return image_feature

        self.vision_model.extract_feature = MethodType(
            extract_feature_wrapper, self.vision_model
        )

        if args.pipeline_parallel_degree > 1:
            assert args.pp_need_data_degree == args.pipeline_parallel_degree
            # Rank 0 PP holds the full image_features.
            assert hasattr(model, "_prepare_pipeline_inputs_func")
            ori_prepare_pipeline_inputs_func = model._prepare_pipeline_inputs_func

            assert (
                not model.balanced_image_preprocess
            ), "不支持balanced_image_preprocess"

            def limao_huan_taizi_hook(
                p,
            ):
                def hook(g):
                    logger.info(f"limao hook called -- {p.name}")
                    p._clear_dataptr()

                return hook

            def _prepare_pipeline_inputs_func_wrapper(inner_self, data):
                def wrap(micro_data):
                    inputs, labels = micro_data
                    if args.pipeline_parallel_rank == 0 and inputs[2] is not None:
                        fea = inputs[2]
                        self.images_features.append(fea)
                        fea.stop_gradient = False
                        # fea.register_hook(limao_huan_taizi_hook(fea))
                    return (inputs, labels)

                return (wrap(i) for i in ori_prepare_pipeline_inputs_func(data))

            model._prepare_pipeline_inputs_func = MethodType(
                _prepare_pipeline_inputs_func_wrapper, model
            )
            logger.info("set prepare pipeline inputs func success.")

    @staticmethod
    def _enable_delay_scale_loss(args):
        """_summary_

        Args:
            args (_type_): _description_

        Returns:
            _type_: _description_
        """
        key = "enable_delay_scale_loss"
        if args.pipeline_parallel_degree > 1:
            return key in args.pipeline_parallel_config.split(" ")
        elif args.tensor_parallel_degree > 1:
            return key in args.tensor_parallel_config.split(" ")
        else:
            return False

    def do_partial_freeze(self, vision_param_mapping, freeze_config):
        """_summary_

        Args:
            vision_param_mapping (_type_): _description_
            freeze_config (_type_): _description_
        """
        freeze_vit = "freeze_vit" in freeze_config
        freeze_inception = "freeze_inception" in freeze_config
        assert not (freeze_vit and freeze_inception), "you should use freeze_vision"
        for name, param in vision_param_mapping:
            if "inception_" in name:
                if freeze_inception:
                    logger.info(f"freeze_inception--{name}")
                    param.stop_gradient = True
            else:
                if freeze_vit:
                    logger.info(f"freeze_vit--{name}")
                    param.stop_gradient = True

    def on_optimizer_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """_summary_

        Args:
            args (TrainingArguments): _description_
            state (TrainerState): _description_
            control (TrainerControl): _description_
        """
        if args.pipeline_parallel_degree <= 1:
            return
        if not self.images_buffer:
            return
        # showmem("____begin")
        if args.pipeline_parallel_rank == 0:
            image_features_grad = paddle.concat(
                [fea.grad for fea in self.images_features], axis=0
            )
            pp_data_balance = getattr(self.vision_model, "pp_data_balance", False)
            if pp_data_balance:
                with paddle.no_grad():
                    sorted_thw = self.vision_model.sorted_thw
                    sorted_idx = self.vision_model.sorted_idx
                    seq_idx_list = self.vision_model.seq_list
                    new_grads = []
                    for rank in range(self.pp_group.nranks):
                        # seq_idx_list = paddle.to_tensor(seq_idx_list, dtype='int64', place=image_features_grad.place)
                        seq_idx_list = np.array(seq_idx_list, dtype="int64")
                        sorted_thw = np.array(sorted_thw, dtype=np.int64)
                        sorted_idx = np.array(sorted_idx, dtype=np.int64)

                        rank_thw = sorted_thw[sorted_thw[:, -1] == rank]
                        rank_idx = sorted_idx[sorted_thw[:, -1] == rank]
                        new_grad = []
                        start_offset = seq_idx_list[rank_thw[:, -2]] + rank_idx
                        end_offset = (
                            seq_idx_list[rank_thw[:, -2]]
                            + rank_idx
                            + rank_thw[:, 1] * rank_thw[:, 2]
                        )
                        # index_list = [paddle.arange(start_offset[i],end_offset[i]) for i in range(len(rank_thw))]
                        index_list = [
                            np.arange(start_offset[i], end_offset[i])
                            for i in range(len(rank_thw))
                        ]
                        index_list = paddle.to_tensor(
                            np.concatenate(index_list, axis=-1), dtype=paddle.int64
                        )
                        new_grad = paddle.gather(image_features_grad, index_list)
                        new_grads.append(new_grad)
                    image_features_grad = paddle.concat(new_grads, axis=0)

        else:
            image_features_grad = None
        with profile("vit__grad_scatter"):
            # logger.info(f"image_features_grad-{image_features_grad}")
            seqlen = sum([im.shape[0] for im, _ in self.images_buffer])
            indices = []
            dist.all_gather(
                indices, paddle.to_tensor(seqlen, dtype="int32"), self.pp_group
            )
            # logger.info(f"INDICES_____{indices}")
            grads = paddle.empty(sum([self.meta[0]], [seqlen]), self.meta[1])
            # logger.info(f"INDICES_____{indices} -- GRADS____{grads.shape}")
            # grad 反广
            scatter_varlen(
                image_features_grad,
                grads,
                [i.item() for i in indices],
                0,
                self.pp_group,
            )
            grads = paddle.split(
                grads, [im.shape[0] for im, _ in self.images_buffer], axis=0
            )
        gather_grads = []
        if self.mp_degree > 1:
            for grad in grads:
                grad = all_gather(grad, axis=-1, group=self.mp_group)
                gather_grads.append(grad)
            grads = gather_grads

        # showmem("____after_scatter_grad")
        del image_features_grad
        del self.images_features

        def _innder_backward(im, thw, g):
            # showmem("____before_inner_forward")
            with self.auto_cast_func():
                # logger.info(f"2 forward_im_____{i.shape}")
                hi, indices = self.vision_model.extract_feature(
                    im, thw, second_fwd=True
                )
            # logger.info(f"BACKWARD_hi____{hi.shape}")
            # showmem("____before_inner_backward")
            if args.gradient_accumulation_steps > 1 and self._enable_delay_scale_loss(
                args
            ):
                g = g.scale_(1.0 / args.gradient_accumulation_steps)
            with paddle.no_grad():
                shard_head = getattr(self.vision_model, "attn_sep", False)
                if shard_head:
                    seqlen = g.shape[0]
                    num_pad = (
                        math.ceil(seqlen / self.mp_group.nranks) * self.mp_group.nranks
                        - seqlen
                    )
                    g = paddle.nn.functional.pad(g, [0, num_pad, 0, 0], value=0)
                g = mp_slice(g, indices, axis=0, group=self.mp_group)
            paddle.autograd.backward(hi, g)
            # showmem("____after_inner_backward")

        # showmem("____before_seconed_forward")
        with profile("vit__second_forward_backward"):
            for (im, thw), grad in zip(self.images_buffer, grads):
                # logger.info(f"BACKWARD_grad____{grad.shape}")
                # logger.info(f"BACKWARD_im____{im}; BACKWARD_thw____{thw}")
                # im 重切batch
                if self.vit_second_fwd_batch_size:
                    if thw is not None:
                        indices = []
                        thw_indices = []
                        thw_cumsum = F.pad(paddle.prod(thw, -1).cumsum(0), [1, 0])
                        s = 0
                        for i in range(1, len(thw)):
                            if (
                                (thw_cumsum[i] - thw_cumsum[s])
                                >= self.vit_second_fwd_batch_size
                                * self.patches_per_image
                            ):
                                indices.append(thw_cumsum[i] - thw_cumsum[s])
                                thw_indices.append(i - s)
                                s = i
                        if s < len(thw):
                            thw_indices.append(len(thw) - s)
                            indices.append(thw_cumsum[-1] - thw_cumsum[s])
                    else:
                        indices = [self.vit_second_fwd_batch_size] * (
                            im.shape[0] // self.vit_second_fwd_batch_size
                        )
                        if im.shape[0] % self.vit_second_fwd_batch_size != 0:
                            indices.append(im.shape[0] % self.vit_second_fwd_batch_size)
                    # logger.info(f"{indices}-{sum(indices)}--{thw_indices}-{sum(thw_indices)}")
                    for i, g, t in zip(
                        paddle.split(im, indices, axis=0),
                        paddle.split(grad, indices, axis=0),
                        (
                            paddle.split(thw, thw_indices, axis=0)
                            if thw is not None
                            else [None] * len(indices)
                        ),
                    ):
                        _innder_backward(i, t, g)
                else:
                    _innder_backward(im, thw, grad)

        del grads
        del self.images_buffer
        # pp reduce grads
        # showmem("____before_pp_reduce_grad")
        # logger.info(f"pp all_reduce_grad")
        with profile("vit__pp_reduce_grad"):
            for p in self.vision_model.parameters():
                if p.trainable and getattr(p, "main_grad", None) is None:
                    p.main_grad = paddle.zeros(p.shape, dtype="float32")
            fused_allreduce_gradients_with_group(
                self.vision_model.parameters(), self.pp_group
            )
            if self.sd_group.nranks > 1:
                fused_allreduce_gradients_with_group(
                    self.vision_model.parameters(), self.sd_group
                )
        # showmem("____after_pp_reduce_grad")

    def on_optimizer_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """_summary_

        Args:
            args (TrainingArguments): _description_
            state (TrainerState): _description_
            control (TrainerControl): _description_
        """
        if args.pipeline_parallel_degree <= 1:
            return
        self.images_buffer = []
        self.images_features = []
        # showmem("____optimizer_end")
