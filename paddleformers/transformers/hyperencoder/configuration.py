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

"""The config class for HyperEncoder.

``AutoConfig.from_pretrained(model_path)`` reads ``model_path/config.json`` ->
this class -> :class:`HyperEncoderProvider` (``modeling_fleet.py``) -> builds
the network.

This file holds only the config class and its default values. The transfer of
fields to the Fleet side is done by the inherited
``TransformerConfig.from_config``, and the few values that need to be derived
live in ``HyperEncoderProvider.__post_init__``.
"""
from __future__ import annotations

from ..configuration_utils import PretrainedConfig

__all__ = ["HYPERENCODER_VOCAB_SIZE", "HYPERENCODER_HIDDEN_SIZE", "HyperEncoderConfig"]

# Special token ids, kept as a single source of truth on the Paddle side.
IM_PATCH_TOKEN = 128815
AUDIO_PATCH_TOKEN = 128829
CONTEXT_TOKEN = 128830
HYPERENCODER_VOCAB_SIZE = 129280
HYPERENCODER_HIDDEN_SIZE = 1280

# Named constants for the model geometry.
_NUM_HEADS = 10
_FFN_HIDDEN = 6848
_MOE_FFN_HIDDEN = 896
_NUM_MOE_EXPERTS = 64
_NUM_LAYERS = 12


# ---------------------------------------------------------------------------
# Serializable config class (`model_path/config.json` <-> `GPTConfig`)
# ---------------------------------------------------------------------------


class HyperEncoderConfig(PretrainedConfig):
    """The config class for HyperEncoder.

    ## What it does

    `AutoConfig.from_pretrained(model_path)` reads ``model_path/config.json`` ->
    this class -> :class:`HyperEncoderProvider` (``modeling_fleet.py``) -> builds
    the network. The entire architecture comes from `config.json`; the default
    values in the signature are only the fallback when a field is not provided.

    Transfer of fields to the Fleet side uses the framework's generic mechanism
    (``TransformerConfig.from_config``: ``object.__new__`` +
    ``register_attributes`` + ``__post_init__``). This class performs no
    translation of its own, so there is no possibility of "config.json was
    changed but had no effect". Non-default architecture values are the field
    defaults of ``HyperEncoderProvider``, and derivations live in its
    ``__post_init__``.

    ## Where fields come from (framework division of labor)

    | Source | Fields |
    |---|---|
    | ``config.json`` (travels with the checkpoint) | all architecture fields + ``hyperencoder_query_lengths`` / ``hyperencoder_seq_align`` + ``tensor_model_parallel_size`` |
    | yaml -> ``TrainingArguments`` -> ``LlmMetaConfig.set_llm_config`` | ``expert_model_parallel_size`` / ``pipeline_...`` / ``context_...`` / ``recompute_*`` / ``router_aux_loss_coef`` / ``moe_shared_expert_overlap`` |

    Mechanism: ``PretrainedConfig.__init__`` puts every field declared by
    ``LlmMetaConfig`` into ``_unsavable_keys`` (dropped when serializing), but
    explicitly discards ``tensor_model_parallel_size`` from that set -- because
    weights are saved sharded by TP, so the shard degree must travel with the
    weight directory.

    => Therefore parallelism / recompute / aux-loss are NOT in this class's
    ``__init__`` signature: they are received from ``**kwargs`` by
    ``set_expected_keys`` and given defaults. They can still be passed explicitly
    as a kwarg.

    Do NOT move them into the signature and assign them before
    ``super().__init__()`` -- ``set_expected_keys`` would overwrite them with
    defaults (for example ``expert_model_parallel_size`` 2->1,
    ``recompute_granularity`` 'full'->None). Architecture fields are unaffected
    because they live in the ``model_conf`` table, which ``_get_init()`` does not
    include.

    ## What changing the architecture does

    Following `config.json` works and does not raise -- this is the intended
    behavior of `AutoConfig`. But changing the architecture means changing the
    model, so any existing reference outputs no longer apply. Shape changes must
    first be made and re-validated on the reference side.
    """

    model_type = "hyperencoder"

    def __init__(
        self,
        vocab_size=HYPERENCODER_VOCAB_SIZE,
        hidden_size=HYPERENCODER_HIDDEN_SIZE,
        intermediate_size=_FFN_HIDDEN,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_HEADS,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        attention_dropout=0.0,
        hidden_dropout_prob=0.0,
        attention_bias=False,
        moe_intermediate_size=_MOE_FFN_HIDDEN,
        n_routed_experts=_NUM_MOE_EXPERTS,
        num_experts_per_tok=6,
        n_shared_experts=2,
        first_k_dense_replace=1,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        scoring_func="softmax",
        topk_method="greedy",
        tie_word_embeddings=False,
        # ---- HyperEncoder-specific geometry (exposed here as config fields) ----
        hyperencoder_query_lengths=(256, 8192),
        hyperencoder_seq_align: int = 128,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_bias = attention_bias

        # MoE
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method

        # HyperEncoder-specific geometry. json only carries a list, so normalize to a tuple.
        ql = tuple(int(v) for v in hyperencoder_query_lengths)
        if len(ql) != 2:
            raise ValueError(f"hyperencoder_query_lengths must be (short, long), got {ql}")
        self.hyperencoder_query_lengths = ql
        self.hyperencoder_seq_align = int(hyperencoder_seq_align)

        # `tie_word_embeddings` is popped from kwargs by the base class (default
        # True), so it must be put back into kwargs rather than set directly --
        # a direct setattr would be overwritten by the base class.
        kwargs.setdefault("tie_word_embeddings", tie_word_embeddings)
        super().__init__(**kwargs)
