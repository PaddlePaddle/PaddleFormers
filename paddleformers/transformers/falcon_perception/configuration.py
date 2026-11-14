# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Adapted from TII Falcon Perception (Apache-2.0).
from ..configuration_utils import PretrainedConfig


class FalconPerceptionConfig(PretrainedConfig):
    model_type = "falcon_perception"

    def __init__(
        self,
        dim: int = 1024,
        n_layers: int = 28,
        n_heads: int = 16,
        head_dim: int = 128,
        n_kv_heads: int = 8,
        vocab_size: int = 65536,
        ffn_dim: int = 3072,
        norm_eps: float = 1e-05,
        max_seq_len: int = 8192,
        rope_theta: int = 10000,
        channel_size: int = 3,
        spatial_patch_size: int = 16,
        temporal_patch_size: int = 1,
        do_segmentation: bool = True,
        segm_out_dim: int = 256,
        num_segm_layers: int = 3,
        coord_enc_dim: int = 512,
        coord_dec_dim: int = 8192,
        coord_out_dim: int = 2048,
        coord_token_id: int = 240,
        size_enc_dim: int = 512,
        size_dec_dim: int = 8192,
        size_out_dim: int = 2048,
        size_token_id: int = 241,
        seg_token_id: int = 262,
        eos_id: int = 11,
        img_id: int = 227,
        image_cls_token_id: int = 244,
        image_reg_1_token_id: int = 245,
        image_reg_2_token_id: int = 246,
        image_reg_3_token_id: int = 247,
        image_reg_4_token_id: int = 248,
        img_end_id: int = 230,
        pad_token_id: int = 0,
        **kwargs
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.ffn_dim = ffn_dim
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.channel_size = channel_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.do_segmentation = do_segmentation
        self.segm_out_dim = segm_out_dim
        self.num_segm_layers = num_segm_layers
        self.coord_enc_dim = coord_enc_dim
        self.coord_dec_dim = coord_dec_dim
        self.coord_out_dim = coord_out_dim
        self.coord_token_id = coord_token_id
        self.size_enc_dim = size_enc_dim
        self.size_dec_dim = size_dec_dim
        self.size_out_dim = size_out_dim
        self.size_token_id = size_token_id
        self.seg_token_id = seg_token_id
        self.eos_id = eos_id
        self.img_id = img_id
        self.image_cls_token_id = image_cls_token_id
        self.image_reg_1_token_id = image_reg_1_token_id
        self.image_reg_2_token_id = image_reg_2_token_id
        self.image_reg_3_token_id = image_reg_3_token_id
        self.image_reg_4_token_id = image_reg_4_token_id
        self.img_end_id = img_end_id
        super().__init__(pad_token_id=pad_token_id, **kwargs)
