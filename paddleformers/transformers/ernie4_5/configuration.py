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

""" Ernie4_5 model configuration."""
from ..configuration_utils import PretrainedConfig

ERNIE_PRETRAINED_INIT_CONFIGURATION = {
    "ernie/tiny-random-ernie": {
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "model_type": "ernie",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": False,
        "recompute": False,
        "use_flash_attn": True,
        "use_pure_fp16": False,
    },
}


class Ernie4_5Config(PretrainedConfig):
    """
    Configuration class for Ernie4_5 model.

    This class stores the configuration of an Ernie4_5 model, defining the model architecture.
    It inherits from PretrainedConfig and can be used to control model outputs.
    """

    model_type = "ernie4_5"

    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=11008,
        max_position_embeddings=32768,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=None,
        scale_qk_coeff=1.0,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        use_flash_attention=True,
        use_sparse_flash_attn=True,
        use_var_len_flash_attn=False,
        recompute=False,
        recompute_granularity="core_attn",
        recompute_use_reentrant=False,
        use_rmsnorm=True,
        fuse_rms_norm=False,
        fuse_ln=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        fuse_swiglu=False,
        use_bias=False,
        rope_theta=10000,
        fuse_rope=False,
        fuse_softmax_mask=False,
        weight_share_add_bias=True,
        fuse_linear=False,
        max_sequence_length=None,
        ignored_index=-100,
        add_tail_layers=False,
        use_recompute_lm_head=False,
        use_recompute_loss_fn=False,
        refined_recompute=dict(),
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        compression_ratio: float = 1.0,
        num_key_value_heads=None,
        use_sparse_head_and_loss_fn=False,
        micro_batch_size=-1,
        use_fused_head_and_loss_fn=False,
        token_balance_loss=False,
        token_balance_seqlen=False,  # calculated based on batchsize and seqlen
        cachekv_quant: bool = False,
        pp_seg_method="layer:Ernie4_5DecoderLayer|EmptyLayer",
        dpo_config=None,
        **kwargs,
    ):
        """
        Initialize Ernie4_5 model configuration with default or specified parameters.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens)
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            intermediate_size (int): Dimensionality of the "intermediate" (feed-forward) layer
            max_position_embeddings (int): Maximum sequence length the model can handle
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder
            num_attention_heads (int): Number of attention heads for each attention layer
            rms_norm_eps (float): The epsilon used by the RMS normalization layers
            use_cache (bool): Whether to use caching for faster generation (decoding)
            use_flash_attention (bool): Whether to use FlashAttention for optimized attention computation
            use_sparse_flash_attn (bool): Whether to use sparse FlashAttention
            use_var_len_flash_attn (bool): Whether to use variable-length FlashAttention
            recompute (bool): Whether to use gradient checkpointing to save memory
            recompute_granularity (str): Granularity of recomputation ("core_attn", "full", etc.)
            recompute_use_reentrant (bool): Whether to use reentrant checkpointing
            use_rmsnorm (bool): Whether to use RMSNorm instead of LayerNorm
            fuse_rms_norm (bool): Whether to fuse RMSNorm operations for optimization
            fuse_ln (bool): Whether to fuse LayerNorm operations
            pad_token_id (int): Token ID used for padding sequences
            bos_token_id (int): Token ID used for beginning-of-sequence
            eos_token_id (int): Token ID used for end-of-sequence
            fuse_swiglu (bool): Whether to fuse SwiGLU operations
            use_bias (bool): Whether to use bias terms in linear layers
            rope_theta (float): The base period of the RoPE embeddings
            fuse_rope (bool): Whether to fuse RoPE operations
            weight_share_add_bias (bool): Whether to share bias weights in certain layers
            fuse_linear (bool): Whether to fuse linear operations
            max_sequence_length (int): Maximum sequence length for positional embeddings
            ignored_index (int): Target value that is ignored during loss computation
            add_tail_layers (int): Whether to add additional layers at the end
            use_recompute_lm_head (bool): Whether to recompute gradients for language model head
            use_recompute_loss_fn (bool): Whether to recompute gradients for loss function
            refined_recompute (dict): Dictionary specifying refined recomputation settings
            attention_probs_dropout_prob (float): Dropout probability for attention weights
            hidden_dropout_prob (float): Dropout probability for hidden layers
            compression_ratio (float): Ratio for KV cache compression (1.0 = no compression)
            num_key_value_heads (int): Number of key/value heads (for Grouped Query Attention)
            use_sparse_head_and_loss_fn (bool): Whether to use sparse attention head and loss function
            micro_batch_size (int): Size of micro batches (-1 for automatic)
            use_fused_head_loss_fn (bool): Whether to use fused head and loss function
            token_balance_loss (bool): Whether to balance loss by token count
            token_balance_seqlen (bool): Whether to balance sequence lengths
            cachekv_quant (bool): Whether to quantize key-value cache
            pp_seg_method (str): Method for pipeline parallel segmentation
            **kwargs: Additional keyword arguments passed to parent class
        """

        # Set default for tied embeddings if not specified.
        if "tie_word_embeddings" not in kwargs:
            kwargs["tie_word_embeddings"] = False
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
        self.head_dim = head_dim
        self.scale_qk_coeff = scale_qk_coeff
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.recompute = recompute
        self.recompute_granularity = recompute_granularity
        self.use_flash_attention = use_flash_attention
        self.use_sparse_flash_attn = use_sparse_flash_attn
        self.recompute_use_reentrant = recompute_use_reentrant
        self.use_var_len_flash_attn = use_var_len_flash_attn
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.fuse_swiglu = fuse_swiglu
        self.fuse_rms_norm = fuse_rms_norm
        self.fuse_ln = fuse_ln
        self.use_rmsnorm = use_rmsnorm
        self.micro_batch_size = micro_batch_size

        self.max_sequence_length = max_sequence_length
        self.use_bias = use_bias
        self.weight_share_add_bias = weight_share_add_bias
        self.rope_theta = rope_theta
        self.fuse_rope = fuse_rope
        self.fuse_softmax_mask = fuse_softmax_mask

        self.fuse_linear = fuse_linear
        self.ignored_index = ignored_index
        self.add_tail_layers = add_tail_layers
        self.use_recompute_lm_head = use_recompute_lm_head
        self.use_recompute_loss_fn = use_recompute_loss_fn

        self.refined_recompute = refined_recompute
        self.skip_recompute_ops = dict()
        """
            `refined_recompute` is a dictionary that specifies fine-grained gradient recomputation settings,
            which currently only takes effect in Pipeline Parallel (PP) mode.

            In PP mode, this dictionary populates `self.skip_recompute_ops` with the following structure:
            - Key (`op_name`): The operation name to configure, with possible values:
            * "mlp_row_ln" - MLP row-wise layer normalization
            * "flash_attn" - Flash attention operation
            * "attention_row_ln" - Attention row-wise layer normalization
            * "attention_column_ln" - Attention column-wise layer normalization
            * "mlp_column_ln" - MLP column-wise layer normalization

            - Value (`skip_num`): Controls how many times to skip recomputation:
            * 0: Never skip recomputation (minimum memory usage)
            * -1: Always skip recomputation (maximum memory usage)
            * [0,1,...,12]: Skip recomputation for specified number of times
            * ≥12: Equivalent to -1 (always skip recomputation)

            This allows precise control over memory/computation tradeoffs for different operations.
        """
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.compression_ratio = compression_ratio
        self.num_key_value_heads = num_key_value_heads
        self.use_sparse_head_and_loss_fn = use_sparse_head_and_loss_fn
        self.use_fused_head_and_loss_fn = use_fused_head_and_loss_fn
        self.token_balance_loss = token_balance_loss
        self.token_balance_seqlen = token_balance_seqlen
        self.cachekv_quant = cachekv_quant
        self.pp_seg_method = pp_seg_method
        self.dpo_config = dpo_config

        self.register_unsavable_keys(
            [
                "recompute",
                "recompute_use_reentrant",
                "refined_recompute",
                "recompute_granularity",
                "use_recompute_lm_head",
                "use_recompute_loss_fn",
                "pp_seg_method",
                "skip_recompute_ops",
                "use_sparse_flash_attn",
                "use_var_len_flash_attn",
                "use_sparse_head_and_loss_fn",
                "micro_batch_size",
                "fuse_softmax_mask",
                "cachekv_quant",
                "use_fused_head_and_loss_fn",
                "max_sequence_length",
                "dpo_config",
            ]
        )


__all__ = ["Ernie4_5Config"]
