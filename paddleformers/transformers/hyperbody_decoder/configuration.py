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

"""HyperBody decoder 的 HF-style config —— **几何与数值开关的单一真源**。

## 分工（与参考实现 ``fleet_formers`` 的 hyperencoder 同构）

| 侧 | 职责 |
|---|---|
| **本文件（PaddleFormers）** | 几何常量 + ``HyperBodyDecoderConfig``：``config.json`` 的落地形状，喂 ``AutoConfig`` / ``AutoModelForCausalLM`` |
| ``modeling.py``（PaddleFormers） | HF-style config → ``GPTConfig`` 的转换（``HyperBodyDecoderModelProvider``）+ 组网装配 |
| ``paddlefleet.models.hyperbody_decoder`` | 只提供组件：12 层 backbone 的 ``LayerSpec`` 列表 |

Megatron 侧的真源是 ``_make_language_config``
（``hyperbody/megatron_bridge_patch/recipes/hyperbody/hyperbody.py:61-147``）
以及紧随其后的 ``_build_model_specs``（提供 ``vocab_size``）。
下面每个字段都标了对应的源侧行号。

参考实现把 ``build_hyperencoder_config() -> GPTConfig`` 放在
``transformers/hyperencoder/configuration.py``，即「HF-style → GPTConfig」这件事
归 formers。decoder 走 ``paddleformers-cli``，这条转换由
``AutoConfig`` → 本类 → ``HyperBodyDecoderModelProvider.from_config`` 完成，
落点同样在 formers；fleet 侧因此**不再有** ``decoder_config.py``。

## 源侧有、Paddle 侧无对应字段的四项

* ``persist_layer_norm=True`` —— Paddle 的 norm 没有这个开关。
* ``moe_router_dtype="fp32"`` —— Paddle 侧 router 已强制 fp32（前提是不开
  ``use_accuracy_compatible``）。
* ``moe_router_pre_softmax=True`` —— Paddle 路由天生 pre-softmax，配
  ``scoring_func="softmax"`` + ``topk_method="greedy"`` + ``norm_topk_prob=False``
  即等价。
* ``attention_backend=AttnBackend.flash`` —— Paddle 侧对应 ``_attn_implementation``。

## 三处 decoder 与 encoder 真实不同（不是笔误）

* ``moe_expert_fusion=True``（源侧 ``moe_grouped_gemm=True``），encoder 是 ``False``。
* ``masked_softmax_fusion=True``，encoder 是 ``False``。
* 因果 vs 双向：decoder 不需要 encoder 那套 ``AttnMaskType.no_mask``
  + ``_attn_implementation="eager"`` 的绕行。

## ⚠️ 哪些字段会被 CLI 覆盖掉（写 yaml 时必须知道）

``paddleformers-cli`` 的链路是：

1. ``AutoConfig.from_pretrained`` → 走本类 ``__init__``；
2. ``PretrainedConfig.__init__`` 里的 ``set_expected_keys``（``configuration_utils.py:889``）
   把 ``LlmMetaConfig._get_init()`` 的**每一个** key 按它自己的默认值写到 config 上 ——
   所以本类 ``__init__`` 里先 ``self.x = ...`` 再 ``super().__init__()`` 的字段，
   凡属于 llm_meta 的都会被打回默认值。**对策：塞进 ``super().__init__(**)`` 的 kwargs**，
   ``set_expected_keys`` 会优先取 kwargs（``:206-207``）。
3. ``LlmMetaConfig.set_llm_config(model_config, training_args)``
   （``cli/train/sft/workflow.py:283``）再来一遍，这次值取自 ``SFTConfig``
   —— 而 ``SFTConfig`` 被 ``@llmmetaclass`` 装饰（``cli/train/sft/sft_config.py:28``），
   llm_meta 的字段在它上面全是 dataclass field。**这一步 config.json 拦不住**。

因此下面这几个源侧取值与 llm_meta 默认值**不一致**的字段，
必须同时写进 yaml，否则会被悄悄改掉：

| 字段 | 源侧 | llm_meta 默认 |
|---|---|---|
| ``moe_expert_fusion`` | ``True`` | ``False`` |
| ``router_aux_loss_coef`` | ``0.001`` | ``0.0`` |
| ``fp32_residual_connection`` | ``False``（Megatron 默认，源侧没设） | ``True`` |

``yaml/hyperbody_decoder_*.yaml`` 里这三条都显式写了，别删。

不属于 llm_meta、因此**本类说了算**的关键字段：``masked_softmax_fusion``、
``bias_activation_fusion``、``bias_dropout_fusion``、``cross_entropy_loss_fusion``、
``attention_softmax_in_fp32``、``calculate_per_token_loss``、``rms_norm_eps``、
``rope_theta``、``rotary_percent``、``head_dim``、``moe_layer_freq``、
``n_shared_experts``、``moe_intermediate_size``、``scoring_func``、``topk_method``、
``norm_topk_prob``、``routed_scaling_factor``、``n_group``、``topk_group``、
``use_cpu_initialization``、``use_accuracy_compatible``、``bf16``。
"""

from __future__ import annotations

from ..configuration_utils import PretrainedConfig

__all__ = [
    "HyperBodyDecoderConfig",
    "CONTEXT_TOKEN",
    "HYPERBODY_DECODER_VOCAB_SIZE",
    "HYPERBODY_DECODER_HIDDEN_SIZE",
    "HYPERBODY_DECODER_NUM_LAYERS",
    "HYPERBODY_DECODER_NUM_HEADS",
    "HYPERBODY_DECODER_FFN_HIDDEN",
    "HYPERBODY_DECODER_MOE_FFN_HIDDEN",
    "HYPERBODY_DECODER_NUM_MOE_EXPERTS",
    "HYPERBODY_DECODER_MOE_TOPK",
    "HYPERBODY_DECODER_NUM_SHARED_EXPERTS",
    "build_moe_layer_freq",
    "check_hyperbody_decoder_divisibility",
]

# 源侧 native_hyperbody_modality_submodules.py:41-46 的 special token。
# decoder 侧只用到 CONTEXT_TOKEN：LLM 在 input_ids 里出现 <context> 的位置把
# encoder 输出贴进 embedding 流（源侧 _build_model_specs :186-190）。
CONTEXT_TOKEN = 128830

HYPERBODY_DECODER_VOCAB_SIZE = 129280  # 源侧 :180 VOCAB_SIZE
HYPERBODY_DECODER_HIDDEN_SIZE = 1280  # 源侧 :65
HYPERBODY_DECODER_NUM_LAYERS = 12  # 源侧 :64
HYPERBODY_DECODER_NUM_HEADS = 10  # 源侧 :66
HYPERBODY_DECODER_FFN_HIDDEN = 6848  # 源侧 :71  只 layer 0 用
HYPERBODY_DECODER_MOE_FFN_HIDDEN = 896  # 源侧 :97
HYPERBODY_DECODER_NUM_MOE_EXPERTS = 64  # 源侧 :96
HYPERBODY_DECODER_MOE_TOPK = 6  # 源侧 :98
# 源侧 :99 给的是 moe_shared_expert_intermediate_size=1792，Paddle 侧用
# n_shared_experts × moe_intermediate_size 表达：2 × 896 = 1792
HYPERBODY_DECODER_NUM_SHARED_EXPERTS = 2


def build_moe_layer_freq(num_hidden_layers: int = HYPERBODY_DECODER_NUM_LAYERS) -> list[int]:
    """逐层的 dense/MoE 0-1 表：``[0] + [1]*(L-1)``（源侧 ``:100``）。

    ⚠️ 必须是 list。传 int 时 Paddle 走 ``i % N``（``gpt_layer_specs.py:803-807``），
    Megatron 走另一套取模语义，两边必然错位 —— encoder 那份 config 也标了同一个坑。
    """
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
    return [0] + [1] * (num_hidden_layers - 1)


def check_hyperbody_decoder_divisibility(config: "HyperBodyDecoderConfig") -> None:
    """并行度可除性检查（照 encoder 那份 ``check_hyperencoder_divisibility``）。

    源侧自己不查（Megatron 在 ``TransformerConfig.__post_init__`` 里统一查），
    这里提前查是为了在建模型**之前**报错，而不是等到某个 GEMM 形状对不上。
    读 config 上的真实几何而不是模块常量 —— smoke 那份 config.json 是缩小几何。
    """
    tp_size = max(getattr(config, "tensor_model_parallel_size", 1) or 1, 1)
    ep_size = max(getattr(config, "expert_model_parallel_size", 1) or 1, 1)
    for name in ("hidden_size", "num_attention_heads", "intermediate_size", "moe_intermediate_size"):
        value = getattr(config, name)
        if value % tp_size != 0:
            raise ValueError(f"decoder {name}={value} must be divisible by TP={tp_size}")
    if config.n_routed_experts % ep_size != 0:
        raise ValueError(f"decoder n_routed_experts={config.n_routed_experts} must be divisible by EP={ep_size}")


class HyperBodyDecoderConfig(PretrainedConfig):
    r"""HyperBody 的 LLM 主干（DeepSeekV2-Lite MoE，12 层 / 64 experts / top-6）。

    默认值即源侧 ``_make_language_config`` 的取值，所以 ``config.json`` 只需要
    ``{"model_type": "hyperbody_decoder", "architectures": [...]}`` 加上想改的字段。

    Args:
        moe_layer_freq: 逐层 dense/MoE 的 0-1 表。``None`` 时按
            ``build_moe_layer_freq(num_hidden_layers)`` 生成 ``[0] + [1]*(L-1)``。
            ⚠️ 传 int 会让 Paddle 走 ``i % N``（``gpt_layer_specs.py:803-807``），
            与 Megatron 的取模语义不同，必然错位。
        first_k_dense_replace: 只为了显式声明**必须是 ``None``** ——
            ``TransformerConfig.__post_init__``（``:2275-2297``）在它与非 int 的
            ``moe_layer_freq`` 同时给出时直接抛异常。
        use_cpu_initialization: 源侧 ``:67`` 是 ``True``（Megatron 的 TP 不变初始化：
            整张权重在 CPU 上建好再切），**这里刻意取 ``False``** —— 与 encoder
            侧同一处取舍（``hyperencoder_fleet/CLAUDE.json`` 的
            ``intentional_divergence``）。两个原因：
            1. Paddle 的 cpu-init 路径在 bf16 下本身是坏的：
               ``tensor_parallel/layers.py:1849`` 调
               ``_initialize_affine_weight_cpu`` 时**没有转发 ``params_dtype``**，
               master weight 停在该函数的 ``paddle.float32`` 默认值上，
               到 ``:269`` 的 ``weight.copy_(cpu_weight)`` 就炸
               ``dtype 16 != 10``（``VocabParallelEmbedding`` 那条路径转发了，
               所以只有 Column/RowParallelLinear 触发）。
            2. 反正对不上 torch 的 RNG，CPU 初始化换来的"TP 不变"对我们没有价值。
            代价：必须在 fleet RNG tracker 就绪之后建模型 ——
            分布式启动（``paddleformers-cli``）会做，裸脚本得自己先调
            ``paddlefleet.tensor_parallel.random.model_parallel_cuda_manual_seed(seed)``，
            否则报 ``cuda rng state model-parallel-rng is not added``。
    """

    model_type = "hyperbody_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # === 结构（源侧 :64-93）===
        vocab_size: int = HYPERBODY_DECODER_VOCAB_SIZE,
        hidden_size: int = HYPERBODY_DECODER_HIDDEN_SIZE,
        num_hidden_layers: int = HYPERBODY_DECODER_NUM_LAYERS,
        num_attention_heads: int = HYPERBODY_DECODER_NUM_HEADS,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        intermediate_size: int = HYPERBODY_DECODER_FFN_HIDDEN,
        hidden_act: str = "silu",
        gated_linear_unit: bool = True,
        multi_latent_attention: bool = False,
        use_qk_norm: bool = False,
        normalization: str = "RMSNorm",
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        # === 位置编码（源侧 :80-86）===
        position_embedding_type: str = "rope",
        rope_theta: float = 10000,
        rotary_percent: float = 1.0,
        max_position_embeddings: int = 8192,
        # === dropout / bias（源侧 :89-92）===
        attention_dropout: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        use_bias: bool = False,
        attention_bias: bool = False,
        # === MoE（源侧 :96-116）===
        n_routed_experts: int = HYPERBODY_DECODER_NUM_MOE_EXPERTS,
        moe_intermediate_size: int = HYPERBODY_DECODER_MOE_FFN_HIDDEN,
        num_experts_per_tok: int = HYPERBODY_DECODER_MOE_TOPK,
        n_shared_experts: int = HYPERBODY_DECODER_NUM_SHARED_EXPERTS,
        moe_layer_freq: list[int] | None = None,
        first_k_dense_replace: None = None,
        moe_token_dispatcher_type: str = "alltoall",
        moe_expert_fusion: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        router_aux_loss_coef: float = 0.001,
        scoring_func: str = "softmax",
        topk_method: str = "greedy",
        norm_topk_prob: bool = False,
        moe_router_load_balancing_type: str = "seq_aux_loss",
        moe_shared_expert_overlap: bool = True,
        routed_scaling_factor: float = 1.0,
        routed_scaling_factor_learnable: bool = False,
        # === dtype / 数值（源侧 :119-125）===
        bf16: bool = True,
        attention_softmax_in_fp32: bool = False,
        fp32_residual_connection: bool = False,
        variable_seq_lengths: bool = True,
        calculate_per_token_loss: bool = False,
        # === 融合开关（源侧 :128-133）===
        bias_activation_fusion: bool = True,
        masked_softmax_fusion: bool = True,
        bias_dropout_fusion: bool = True,
        apply_rope_fusion: bool = False,
        cross_entropy_loss_fusion: bool = False,
        # === 初始化 ===
        use_cpu_initialization: bool = False,
        use_accuracy_compatible: bool = False,
        # === 并行（运行期由 yaml 覆盖）===
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        pp_seg_method: str = "layer:TransformerLayer|EmptyLayer",
        # === 其它 ===
        tie_word_embeddings: bool = False,
        hyperbody_context_token_id: int = CONTEXT_TOKEN,
        **kwargs,
    ):
        # ---- 结构 ----
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # 源侧 :91 num_query_groups=10 == num_attention_heads ⇒ 纯 MHA，不是 GQA
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # ---- 位置编码 ----
        self.rope_theta = rope_theta
        self.rotary_percent = rotary_percent
        # 源侧 :85-86 把 seq_length 与 max_position_embeddings 设成同一个值。
        # 走 CLI 时这三个都会被 data_args.max_seq_len 覆盖
        # （cli/train/sft/workflow.py:338-339）。
        self.max_position_embeddings = max_position_embeddings
        self.max_sequence_length = max_position_embeddings
        self.seq_length = max_position_embeddings

        # ---- dropout / bias ----
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_bias = use_bias
        self.attention_bias = attention_bias

        # ---- MoE ----
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.moe_layer_freq = (
            build_moe_layer_freq(num_hidden_layers) if moe_layer_freq is None else list(moe_layer_freq)
        )
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        # :112 moe_router_pre_softmax=True 无对应字段 —— 下面三条即等价
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.routed_scaling_factor_learnable = routed_scaling_factor_learnable

        # ---- dtype / 数值 ----
        self.bf16 = bf16
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.variable_seq_lengths = variable_seq_lengths
        self.calculate_per_token_loss = calculate_per_token_loss

        # ---- 融合开关（都不属于 llm_meta，本类说了算）----
        self.bias_activation_fusion = bias_activation_fusion
        self.masked_softmax_fusion = masked_softmax_fusion
        self.bias_dropout_fusion = bias_dropout_fusion
        self.cross_entropy_loss_fusion = cross_entropy_loss_fusion

        # ---- 初始化 ----
        self.use_cpu_initialization = use_cpu_initialization
        self.use_accuracy_compatible = use_accuracy_compatible

        self.pp_seg_method = pp_seg_method
        # HyperBody 专属：LLM 在 <context> 位置把 encoder 输出贴进 embedding 流
        # （源侧 megatron_mimo_training_hyperbody.py:186-190）。decoder 单独训练时不用。
        self.hyperbody_context_token_id = hyperbody_context_token_id

        # ⚠️ 下面这些都属于 LlmMetaConfig，必须走 kwargs 才不会被
        # set_expected_keys 打回默认值（见模块 docstring 第 2 步）。
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            multi_latent_attention=multi_latent_attention,
            use_qk_norm=use_qk_norm,
            normalization=normalization,
            gated_linear_unit=gated_linear_unit,
            position_embedding_type=position_embedding_type,
            fp32_residual_connection=fp32_residual_connection,
            apply_rope_fusion=apply_rope_fusion,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_fusion=moe_expert_fusion,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_shared_expert_overlap=moe_shared_expert_overlap,
            router_aux_loss_coef=router_aux_loss_coef,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            context_parallel_size=context_parallel_size,
            sequence_parallel=sequence_parallel,
            **kwargs,
        )
