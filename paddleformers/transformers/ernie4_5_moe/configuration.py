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

""" Ernie4_5_Moe model configuration """
import json
from typing import Optional, Union

from ...utils.log import logger
from ..configuration_utils import PretrainedConfig

__all__ = ["Ernie4_5_MoeConfig"]


class Ernie4_5_MoeConfig(PretrainedConfig):
    """
    Configuration class for Ernie4_5_Moe model.

    This class stores the configuration of an Ernie4_5_Moe model, defining the model architecture.
    It inherits from PretrainedConfig and can be used to control model outputs.
    """

    model_type = "ernie4_5_moe"

    def __init__(
        self,
        vocab_size=103424,
        hidden_size=2560,
        intermediate_size=12288,
        max_position_embeddings=32768,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        use_flash_attention=True,
        use_rmsnorm=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_bias=False,
        rope_theta=10000,
        max_sequence_length=None,
        ignored_index=-100,
        attention_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        num_key_value_heads=None,
        micro_batch_size=-1,
        moe_num_experts: Optional[Union[int, list]] = 16,
        use_recompute_moe=False,
        moe_capacity=[64, 64, 64],
        moe_norm_min=1e-12,
        moe_aux_loss_lambda=1e-2,
        moe_z_loss_lambda=1e-4,
        moe_orthogonal_loss_lambda=1e-2,
        sinkhorn_2gate=True,
        sinkhorn_temp=3e-2,
        global_aux_loss=False,
        moe_dropout_prob=0.0,
        moe_group="dummy",
        moe_intermediate_size: Union[int, list] = 0,
        moe_num_shared_experts: int = 2,
        moe_layer_start_index=1,
        moe_layer_end_index=-1,
        moe_layer_interval=1,
        moe_reverse_token_drop: bool = False,
        moe_gate_act: str = "softmax",
        moe_norm_gate_logits=True,
        moe_all_to_all_dropout: float = 0.0,
        moe_k=2,
        moe_use_aux_free: bool = True,
        moe_group_experts: bool = False,
        moe_group_orthogonal_loss: bool = True,
        enable_delay_scale_loss: bool = True,
        num_acc_steps: int = 1,
        fuse_gate_detach_matmul: bool = False,
        moe_use_hard_gate=False,
        num_nextn_predict_layers=1,
        multi_token_pred_lambda=0.1,
        enable_mtp_magic_send=False,
        use_recompute_mtp=False,
        dpo_config=None,
        moe_multimodal_dispatch_use_allgather="",
        **kwargs,
    ):
        """
        Initialize Ernie4_5_Moe model configuration with default or specified parameters.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens)
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            intermediate_size (int): Dimensionality of the "intermediate" (feed-forward) layer
            max_position_embeddings (int): Maximum sequence length the model can handle
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder
            num_attention_heads (int): Number of attention heads for each attention layer
            head_dim (int): Dimensionality of each attention head
            hidden_act (str): Name of the activation function used in the feed-forward network
            rms_norm_eps (float): The epsilon used by the RMS normalization layers
            use_cache (bool): Whether to use caching for faster generation (decoding)
            use_flash_attention (bool): Whether to use FlashAttention for optimized attention computation
            recompute (bool): Whether to use gradient checkpointing to save memory
            recompute_granularity (str): Granularity of recomputation ("core_attn", "full", etc.)
            recompute_use_reentrant (bool): Whether to use reentrant checkpointing
            use_rmsnorm (bool): Whether to use RMSNorm instead of LayerNorm
            pad_token_id (int): Token ID used for padding sequences
            bos_token_id (int): Token ID used for beginning-of-sequence
            eos_token_id (int): Token ID used for end-of-sequence
            use_bias (bool): Whether to use bias terms in linear layers
            rope_theta (float): The base period of the RoPE embeddings
            max_sequence_length (int): Maximum sequence length for positional embeddings
            ignored_index (int): Target value that is ignored during loss computation
            attention_dropout_prob (float): Dropout probability for attention weights
            hidden_dropout_prob (float): Dropout probability for hidden layers
            num_key_value_heads (int): Number of key/value heads (for Grouped Query Attention)
            micro_batch_size (int): Size of micro batches (-1 for automatic)
            moe_num_experts: Number of experts in MoE layers
            use_recompute_moe: Whether to use recomputation for MoE layers
            moe_capacity: Capacity configuration for MoE layers
            moe_norm_min: Minimum value for routing normalization
            moe_layer_interval: Interval between MoE layers
            moe_layer_start_index: Starting layer index for MoE
            moe_layer_end_index: Ending layer index for MoE (-1 means last layer)
            moe_aux_loss_lambda: Weight for auxiliary loss
            moe_z_loss_lambda: Weight for z-loss
            moe_orthogonal_loss_lambda: Weight for orthogonal loss
            sinkhorn_2gate: Whether to use sinkhorn 2-gate routing
            sinkhorn_temp: Temperature for sinkhorn routing
            global_aux_loss: Whether to use global auxiliary loss
            moe_dropout_prob: Dropout probability for MoE layers
            moe_group: Group configuration for MoE experts
            moe_intermediate_size: Intermediate size for MoE layers
            moe_num_shared_experts: Number of shared experts
            moe_reverse_token_drop: Whether to use reverse token dropping
            moe_gate_act: Activation function for gating
            moe_norm_gate_logits: Whether to normalize gate logits
            moe_all_to_all_dropout: Dropout for all-to-all communication
            moe_k: Number of experts to route to
            moe_use_aux_free: Whether to use auxiliary-free routing
            moe_group_experts: Whether to group experts (requires hard gating)
            moe_group_orthogonal_loss: Whether to use group orthogonal loss
            enable_delay_scale_loss: Whether to enable delayed loss scaling
            num_acc_steps: Number of accumulation steps
            fuse_gate_detach_matmul: Whether to fuse gate detach matmul
            **kwargs: Additional keyword arguments passed to parent class

        Note:
            When use_recompute_moe is True, recompute_granularity will be changed to full_attn.
        """

        if use_recompute_moe:
            logger.warning(
                "set `use_recompute_moe`=True, disabling `recompute_granularity=full`, change to full_attn."
            )
            if kwargs["recompute"] and kwargs["recompute_granularity"] == "full":
                kwargs["recompute_granularity"] = "full_attn"

        # Set default for tied embeddings if not specified.
        if "tie_word_embeddings" not in kwargs:
            kwargs["tie_word_embeddings"] = True
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_flash_attention = use_flash_attention
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_rmsnorm = use_rmsnorm
        self.micro_batch_size = micro_batch_size
        self.max_sequence_length = max_sequence_length
        self.use_bias = use_bias
        self.rope_theta = rope_theta
        self.ignored_index = ignored_index
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_key_value_heads = num_key_value_heads
        self.moe_num_experts = moe_num_experts
        self.use_recompute_moe = use_recompute_moe
        self.moe_capacity = moe_capacity
        self.moe_norm_min = moe_norm_min
        self.moe_aux_loss_lambda = moe_aux_loss_lambda
        self.moe_z_loss_lambda = moe_z_loss_lambda
        self.moe_orthogonal_loss_lambda = moe_orthogonal_loss_lambda
        self.global_aux_loss = global_aux_loss
        self.sinkhorn_2gate = sinkhorn_2gate
        self.sinkhorn_temp = sinkhorn_temp
        self.moe_layer_interval = moe_layer_interval
        self.moe_dropout_prob = moe_dropout_prob
        self.moe_group = moe_group
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_layer_end_index = self.num_hidden_layers - 1 if moe_layer_end_index == -1 else moe_layer_end_index
        self.moe_layer_interval = moe_layer_interval
        self.moe_reverse_token_drop = moe_reverse_token_drop
        self.moe_k = moe_k
        self.moe_all_to_all_dropout = moe_all_to_all_dropout
        self.moe_group_experts = moe_group_experts
        self.moe_group_orthogonal_loss = moe_group_orthogonal_loss
        self.enable_delay_scale_loss = enable_delay_scale_loss
        self.num_acc_steps = num_acc_steps
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_layer_end_index = self.num_hidden_layers - 1 if moe_layer_end_index == -1 else moe_layer_end_index
        self.moe_gate_act = moe_gate_act
        self.moe_norm_gate_logits = moe_norm_gate_logits
        self.moe_use_aux_free = moe_use_aux_free
        self.fuse_gate_detach_matmul = fuse_gate_detach_matmul
        self.moe_use_hard_gate = moe_use_hard_gate
        self.moe_multimodal_dispatch_use_allgather = moe_multimodal_dispatch_use_allgather
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.multi_token_pred_lambda = multi_token_pred_lambda
        self.enable_mtp_magic_send = enable_mtp_magic_send
        self.use_recompute_mtp = use_recompute_mtp
        self.dpo_config = dpo_config
        self.register_unsavable_keys(
            [
                "disable_ffn_model_parallel",
                "num_acc_steps",
                "attention_dropout_prob",
                "dpo_config",
                "fuse_gate_detach_matmul",
                "global_aux_loss",
                "hidden_dropout_prob",
                "micro_batch_size",
                "max_sequence_length",
                "moe_group",
                "ignored_index",
                "use_recompute_moe",
                "use_rmsnorm",
                "use_recompute_mtp",
                "sinkhorn_2gate",
                "sinkhorn_temp",
                "enable_delay_scale_loss",
                "enable_mtp_magic_send",
                "moe_dropout_prob",
                "moe_use_aux_free",
                "moe_aux_loss_lambda",
                "moe_gate_act",
                "moe_group_experts",
                "moe_all_to_all_dropout",
                "moe_group_orthogonal_loss",
                "moe_norm_gate_logits",
                "moe_norm_min",
                "moe_orthogonal_loss_lambda",
                "moe_reverse_token_drop",
                "moe_use_hard_gate",
                "moe_z_loss_lambda",
                "moe_group_origin",
                "moe_rank",
                "moe_world_size",
                "multi_token_pred_lambda",
                "moe_multimodal_dispatch_use_allgather",
            ]
        )

    def to_json_string(self, use_diff: bool = True, saving_file=False) -> str:
        """
        Serialize the configuration to a JSON string with special handling for non-serializable objects.

        This method overrides the default JSON serialization to handle special objects like
        paddle.distributed.communication.group.Group that cannot be serialized normally.

        Args:
            use_diff (bool, optional): If True, only outputs the differences from the default configuration.
                                    If False, outputs the full configuration. Defaults to True.

        Returns:
            str: A JSON formatted string representation of the configuration, with proper indentation
                and handling for non-serializable objects.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict(saving_file=saving_file)
        else:
            config_dict = self.to_dict(saving_file=saving_file)

        def _serializer(obj):
            """
            Handle non-serializable objects during JSON conversion.

            Args:
                obj: The object to be serialized

            Returns:
                The serializable representation of the object

            """
            return repr(obj)

        return (
            json.dumps(
                config_dict,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=_serializer,
            )
            + "\n"
        )


__all__ = ["Ernie4_5_MoeConfig"]
