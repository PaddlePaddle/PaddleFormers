# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from contextlib import nullcontext
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlefleet import parallel_state, tensor_parallel
from paddlefleet.jit import jit_fuser
from paddlefleet.models.common.vision_layer.vision_layer import VisionLayer
from paddlefleet.packed_seq_params import PackedSeqParams
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.spec_utils import LayerSpec, build_layer
from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from paddlefleet.transformer.enums import ModelType
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.transformer_block import TransformerBlock, TransformerBlockSublayersSpec
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import WrappedTensor, deprecate_inference_params


class Qwen3VLPatchMerger(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        dim: int = None,
        context_dim: int = None,
        spatial_merge_size: int = None,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        context_dim = context_dim if context_dim is not None else config.hidden_size
        dim = dim if dim is not None else config.hidden_size
        spatial_merge_size = spatial_merge_size if spatial_merge_size is not None else config.spatial_merge_size
        
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = nn.LayerNorm(norm_dim, epsilon=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, dim)
    
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape([-1, self.hidden_size]))
            x = x.reshape([-1, self.hidden_size])
        else:
            x = self.norm(x)
            x = x.reshape([-1, self.hidden_size])
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3VLVisionTransformerBlock(TransformerBlock):
    """
    Qwen3-VL Vision Transformer Block.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerBlockSublayersSpec | LayerSpec,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection = None,
        vp_stage: int = None,
    ):
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=False,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLPatchMerger(config, use_postshuffle_norm=True)
            for _ in range(len(self.deepstack_visual_indexes))
        ])
        self.merger = Qwen3VLPatchMerger(
            config, dim=config.out_hidden_size, context_dim=config.hidden_size, spatial_merge_size=config.spatial_merge_size
        )
    
    def forward(
        self,
        hidden_states: paddle.Tensor | WrappedTensor,
        attention_mask: paddle.Tensor | None,
        context: paddle.Tensor | None = None,
        context_mask: paddle.Tensor | None = None,
        rotary_pos_emb: paddle.Tensor | None = None,
        rotary_pos_cos: paddle.Tensor | None = None,
        rotary_pos_sin: paddle.Tensor | None = None,
        attention_bias: paddle.Tensor | None = None,
        inference_context = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: paddle.Tensor | None = None,
        *,
        inference_params = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            packed_seq_params_full (PackedSeqParams, optional): Parameters for packed sequence
                processing for full attention.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)
        
        # Delete the obsolete reference to the initial input tensor if necessary.
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        
        if not self.pre_process:
            hidden_states = self.input_tensor
        
        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
    
        # hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        print("fleet vision 0 hidden_states", hidden_states._md5sum())
        
        with rng_context:
            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
                hidden_states, deepstack_feature_lists = hidden_states
            else:
                deepstack_feature_lists = []
                for l_no, layer in enumerate(self.layers):
                    packed_seq_params_now = packed_seq_params
                    input_dict = {
                        "hidden_states": hidden_states,
                        "attention_mask": attention_mask,
                        "context": context,
                        "rotary_pos_emb": rotary_pos_emb,
                        "rotary_pos_cos": rotary_pos_cos,
                        "rotary_pos_sin": rotary_pos_sin,
                        "attention_bias": attention_bias,
                        "packed_seq_params": packed_seq_params_now,
                    }
                    output = layer(input_dict)
                    hidden_states, context = output["hidden_states"], output["context"]
                    if (
                        paddle.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)
                    
                    if l_no in self.deepstack_visual_indexes:
                        deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(l_no)](hidden_states.squeeze(0))
                        deepstack_feature_lists.append(deepstack_feature)
                    print(f"fleet vision {l_no} hidden_states", hidden_states._md5sum())
        
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        
        hidden_states = self.merger(hidden_states.squeeze(0))
        
        return hidden_states, deepstack_feature_lists
    
    def _checkpointed_forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        context: paddle.Tensor,
        context_mask: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor,
        attention_bias: paddle.Tensor,
        packed_seq_params: PackedSeqParams,
    ):
        def custom(start: int, end: int):
            def custom_forwrad(hidden_states, attention_mask, context, context_mask, rotary_pos_emb):
                deepstack_feature_lists = []
                for index in range(start, end):
                    packed_seq_params_now = packed_seq_params
                    layer = self._get_layer(index)
                    input_dict = {
                        "hidden_states": hidden_states,
                        "attention_mask": attention_mask,
                        "context": context,
                        "rotary_pos_emb": rotary_pos_emb,
                        "attention_bias": attention_bias,
                        "inference_context": None,
                        "packed_seq_params": packed_seq_params_now,
                    }
                    output = layer(input_dict)
                    if index in self.deepstack_visual_indexes:
                        deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(index)](hidden_states)
                        deepstack_feature_lists.append(deepstack_feature)
                return (hidden_states, deepstack_feature_lists), context
            
            return custom_forwrad
        
        def checkpoint_handler(forward_func):
            """Determine whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`."""
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        deepstack_feature_lists = []
        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_index = 0
            while layer_index < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_index, layer_index + self.config.recompute_num_layers)
                )
                deepstack_feature_lists.extend(hidden_states[1])
                layer_index += self.config.recompute_num_layers
        
        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_index in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_index >= recompute_skip_num_layers
                    and layer_index < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_index, layer_index + 1))
                    deepstack_feature_lists.extend(hidden_states[1])
                else:
                    hidden_states, context = custom(layer_index, layer_index + 1)(
                        hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                    )
                    deepstack_feature_lists.extend(hidden_states[1])
        
        else:
            raise ValueError(f"Invalid activation recompute method: {self.config.recompute_method}.")
        
        return hidden_states[0], deepstack_feature_lists


class VisionRotaryEmbedding(nn.Module):
    inv_freq: paddle.Tensor
    
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs



class Qwen3VisionModel(VisionLayer):
    """Qwen3-VL vision model."""
    
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: LayerSpec,
    ):
        super().__init__(config=config)
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.merge_hidden_size = self.embed_dim * (config.spatial_merge_size ** 2)
        
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        
        self.conv1 = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )
        
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        
        self.model_type = ModelType.encoder_or_decoder
        
        self.decoder = Qwen3VLVisionTransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
        )
    
    @jit_fuser
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
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
            pos_ids.append(paddle.stack(x=[hpos_ids, wpos_ids], axis=-1).tile(repeat_times=[t, 1]))
        pos_ids = paddle.cat(x=pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb
    
    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = paddle.get_device()

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = paddle.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = paddle.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

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

        idx_tensor = paddle.tensor(idx_list, dtype=paddle.long, device=device)
        weight_tensor = paddle.tensor(weight_list, dtype=self.pos_embed.weight.dtype)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat([t, 1])
            pos_embed = (
                pos_embed.view([t, h // merge_size, merge_size, w // merge_size, merge_size, -1])
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = paddle.cat(patch_pos_embeds_permute)
        return patch_pos_embeds
    
    def get_packed_seq_params(
        self,
        grid_thw: paddle.Tensor,
    ):
        seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).contiguous()
        cu_seqlens = seqlens.cumsum(dim=0, dtype=paddle.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).contiguous()
        cu_seqlens = cu_seqlens.squeeze().contiguous()
        
        max_seqlen = seqlens.max().item()
        
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, attention_mask: paddle.Tensor | None = None, **kwargs) -> paddle.Tensor:
        # Pathed embedding
        target_dtype = self.conv1.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.conv1(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape([seq_len, -1])
        hidden_states = hidden_states.unsqueeze(0)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        rotary_pos_emb = paddle.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_cos = rotary_pos_emb.cos()
        rotary_pos_sin = rotary_pos_emb.sin()
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]
        rotary_pos_emb = rotary_pos_emb.transpose([1, 0])
        
        packed_seq_params = self.get_packed_seq_params(grid_thw)
        
        hidden_states = self.decoder(
            hidden_states,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
        )
        # hidden_states = hidden_states.sequeeze(1).view(-1, self.merge_hidden_size)
        return hidden_states