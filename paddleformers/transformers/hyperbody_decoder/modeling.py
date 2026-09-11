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

"""HyperBody decoder 的「HF-style config → ``GPTConfig``」转换 + 组网装配。

## 职责划分

照参考实现 ``fleet_formers`` 的 hyperencoder 摆：

| 侧 | 内容 |
|---|---|
| ``paddlefleet.models.hyperbody_decoder`` | **组件**：``get_hyperbody_decoder_layer_specs`` |
| ``configuration.py``（本包） | 几何常量 + ``HyperBodyDecoderConfig``（HF-style config 真源） |
| **本文件** | ① HF-style config → ``GPTConfig``（:class:`HyperBodyDecoderModelProvider`）② 组网装配（:func:`build_hyperbody_decoder_model`） |

参考实现里 ``HyperEncoderBlock(TransformerBlock)`` / ``HyperEncoderModel``
也都在 formers 侧，fleet 侧只留 ``layer_specs.py`` + ``modality_encoders.py``。

## 为什么组网要自己写一遍，而不是直接用 ``gpt_builder``

``paddlefleet.gpt_builders.gpt_builder`` 在 ``n_routed_experts`` 非空时会自己去调
``get_gpt_decoder_layers_spec``，**没有任何注入 layer spec 的口子**。要让「fleet 出
组件、formers 做装配」这条分工成立，就得由本文件直接调
``get_gpt_spec`` + ``build_spec_layer``，把 fleet 那份 spec 喂进去。

:func:`build_hyperbody_decoder_model` 是 ``gpt_builder`` 的窄化版本：只保留
HyperBody decoder 真正用到的路径（embedding + N 层 backbone + norm + lm_head +
LanguageLoss），其余分支（MTP、head/tail EmptyLayer、``separate_mtp_headloss``、
ringmoe 子组、meta-device 初始化）一律**显式拒绝**而不是静默跳过 —— 源侧
``_build_model_specs`` 一个都没用，静默跳过只会在将来打开某个开关时错得无声。

## ``__new__`` 而不是 ``__init__``

``AutoModelForCausalLM.from_config`` 走 ``model_utils.py:1349-1368`` 的
``with dtype_guard(dtype): model = cls(config)``，而我们要返回的是 fleet 的
``GPTModel``（一个 ``PipelineLayer``），不是本类的实例。仓里所有 fleet 模型都用
``__new__`` 返回别的对象（``kimi_k2``、``deepseek_v4``、``glm4_moe``），照办。

## ``_gen_aoa_config`` 为什么绕不过去

aoa 表面上只服务 **HF 格式互转**，常规 flex_checkpoint 存档根本不碰它。但
**CLI 训完必经一次 HF 导出**：``cli/train/sft/workflow.py:793`` 把
``last_fc_to_hf=True`` 写死了，没有 yaml 开关，缺 aoa 就在训练全部跑完之后抛
``RuntimeError: ... must implement either the _gen_inv_aoa_config ...``。

只写正向（HF → fleet）一张：``model_utils.py:3286-3289`` 会自己加
``aoa_config_reverse=True`` 反推导出保存方向。右侧的 fleet 结构化名不是猜的，
是从真存档 metadata 里读出来的：
``python scripts/dump_ckpt_keys.py <ckpt>/model_state/0.metadata``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from paddle.distributed.fleet.meta_parallel import build_spec_layer
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_spec
from paddlefleet.models.hyperbody_decoder import get_hyperbody_decoder_layer_specs

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ..gpt_provider import GPTModel, GPTModelProvider
from ..model_utils import PretrainedModel
from .configuration import HyperBodyDecoderConfig, check_hyperbody_decoder_divisibility

logger = logging.getLogger(__name__)

__all__ = [
    "HyperBodyDecoderForCausalLM",
    "HyperBodyDecoderForCausalLMPipe",
    "HyperBodyDecoderModelProvider",
    "build_hyperbody_decoder_model",
]


def build_hyperbody_decoder_model(config, *, num_stages: int, loss_fn=None):
    """组网：fleet 的 layer spec → ``get_gpt_spec`` → ``build_spec_layer``。

    这是 ``paddlefleet.gpt_builders.gpt_builder`` 的窄化版本。参数 ``config``
    是已经转好的 ``GPTConfig``（即 :class:`HyperBodyDecoderModelProvider` 自己）。

    Args:
        num_stages: pipeline 段数，等于 ``pipeline_model_parallel_size``。
        loss_fn: 为空时用 ``LanguageLoss(config)``（与 ``gpt_builder`` 一致）。
    """
    # 源侧 _build_model_specs 用不到这些分支；静默跳过会在将来某天错得无声。
    if getattr(config, "mtp_num_layers", None):
        raise NotImplementedError("HyperBody decoder 没有 MTP 层（源侧未启用）。")
    if getattr(config, "separate_mtp_headloss", False):
        raise NotImplementedError("HyperBody decoder 不用 separate_mtp_headloss。")
    if config.num_empty_layers_add_in_head or config.num_empty_layers_add_in_tail:
        raise NotImplementedError("HyperBody decoder 不插 EmptyLayer（pp 切分靠 seg_method）。")
    if getattr(config, "moe_token_dispatcher_type", None) == "ringmoe":
        raise NotImplementedError("ringmoe 需要 world 级子组初始化，本模型未支持。")
    if getattr(config, "init_model_with_meta_device", False):
        raise NotImplementedError("HyperBody decoder 不走 meta-device 初始化。")

    gpt_spec = get_gpt_spec(
        config=config,
        head_empty_layers_spec=[],
        # ★ 唯一与 gpt_builder 不同的一行：spec 来自 fleet 的 hyperbody_decoder 组件。
        transformer_layers_spec=get_hyperbody_decoder_layer_specs(config),
        tail_empty_layers_spec=[],
        mtp_layers_spec=None,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_sequence_length=config.max_sequence_length,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rope_theta,
        swa_rotary_base=config.swa_rope_theta,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
    )
    return build_spec_layer(
        gpt_spec,
        loss_fn=LanguageLoss(config) if loss_fn is None else loss_fn,
        num_stages=num_stages,
        # 与 GPTModelProvider.provide() 同一个切分口径。
        seg_method="layer:TransformerLayer|EmptyLayer",
    )


@dataclass
class HyperBodyDecoderModelProvider(GPTModelProvider):
    """HF-style config → ``GPTConfig`` 的落点（``GPTModelProvider`` 本身就是 ``GPTConfig``）。

    这里只列**必须钉死**的开关。几何（层数 / hidden / experts / topk …）与其余数值
    一律由 :class:`HyperBodyDecoderConfig` 通过 ``TransformerConfig.register_attributes``
    灌进来，不在本类重复声明 —— 重复声明会制造第二个真源，而真源在
    ``configuration.py``。

    ⚠️ ``from_config`` 走的是 ``object.__new__`` + ``register_attributes``
    （``transformer_config.py:1942-1948``），**不执行 dataclass 的 ``__init__``**。
    下面这些默认值之所以仍然生效，是因为无 ``default_factory`` 的 dataclass
    字段就是类属性，实例属性缺失时属性查找会落到类上。所以别把它们改成
    ``field(default_factory=...)``。
    """

    # ---- 注意力：MLA 全关，纯 MHA（源侧没设 ⇒ mcore 默认，Paddle 必须显式关）----
    multi_latent_attention: bool = False
    use_qk_norm: bool = False

    # ---- 位置编码：普通 RoPE，无任何 scaling（源侧 :80-86 只给了 rotary_base=10000）----
    # ⚠️ 必须显式写回 None：``GPTConfig`` 把继承来的 ``rope_scaling: dict = None``
    # 覆盖成了 ``float = 1.0``（``gpt_config.py:32`` vs ``transformer_config.py:485``），
    # 而 ``gpt_provider.py:210`` 只判 ``is not None``，于是 ``1.0`` 会掉进
    # ``:211`` 的 ``"mscale_all_dim" in self.rope_scaling`` 炸成
    # ``TypeError: argument of type 'float' is not iterable``。
    # 别的 fleet 模型躲过这一枪，是因为它们的 config.json 本来就带 rope_scaling dict。
    rope_scaling: dict = None

    # ---- FFN / norm（源侧 :73、:76）----
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"

    # ---- MoE（源侧 :103、:113、:114）----
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_shared_expert_overlap: bool = True

    # ---- 融合开关（源侧 :128-133）----
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True  # ⚠️ 与 encoder 的 False 不同
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = False
    cross_entropy_loss_fusion: bool = False

    # ---- 其它 ----
    # 源侧 :93 untie_embeddings_and_output_weights=True 的语义取反。
    # GPTModelProvider 的默认是 True，不覆盖会悄悄绑上 embedding 与 lm_head。
    tie_word_embeddings: bool = False
    # 源侧 :137-139 把重算三行注释掉了。
    recompute_granularity: str = None

    # ``dtype`` → ``params_dtype`` 其实已由 ``_process_attribute``
    # （``transformer_config.py:1981-1982``）处理，这里显式写一条只是
    # 与仓里其它 fleet provider 保持同一形状，便于对照。
    transform_rules = {
        **GPTModelProvider.transform_rules,
        "dtype": "params_dtype",
    }

    def provide(self, pre_process=None, post_process=None, vp_stage=None, loss_fn=None) -> GPTModel:
        """覆写父类的 ``provide()``，把组网换成 :func:`build_hyperbody_decoder_model`。

        父类那份会调 ``gpt_builder``（内部自选 layer spec），本模型要用 fleet
        ``hyperbody_decoder`` 提供的那份 spec，所以只能自己组。

        父类里另外两段在本模型上是死代码，不搬：``mtp_block_spec`` 算完从没传给
        ``gpt_builder``；``is_pipeline_asymmetric`` 算完也没人读。
        """
        # 父类的 rope 展平：``GPTConfig`` 可能带 ``rope_parameters`` 这层包装。
        if getattr(self, "rope_parameters", None):
            if self.rope_parameters.get("rope_type", "default") != "default":
                self.rope_type = self.rope_parameters["rope_type"]
            if "rope_theta" in self.rope_parameters:
                self.rope_theta = self.rope_parameters["rope_theta"]
        if isinstance(self.rope_scaling, dict) and "mscale_all_dim" in self.rope_scaling:
            self.mscale_all_dim = self.rope_scaling["mscale_all_dim"]

        fleet_model = build_hyperbody_decoder_model(
            self,
            num_stages=self.pipeline_model_parallel_size,
            loss_fn=loss_fn,
        )
        # 把 FleetGPTModel 转成 formers 的 GPTModel，好继承 PretrainedModel 的方法
        # （逐属性拷贝，与父类 gpt_provider.py:232-240 同一手法）。
        model = GPTModel.__new__(GPTModel)
        for attr_name in dir(fleet_model):
            if not attr_name.startswith("__"):
                try:
                    setattr(model, attr_name, getattr(fleet_model, attr_name))
                except Exception:  # 只读属性 / property，跳过即可
                    pass
        return model


class HyperBodyDecoderPretrainedModel(PretrainedModel):
    config_class = HyperBodyDecoderConfig
    base_model_prefix = "hyperbody_decoder"
    # gate 在 fleet 侧是 fp32（PaddleFleet 的 router 强制 fp32，前提是
    # use_accuracy_compatible=False），存档 metadata 里也确实是 float32。
    _keep_in_fp32_modules = ["mlp.gate.weight"]

    @classmethod
    def _gen_aoa_config(cls, config: HyperBodyDecoderConfig):
        """HF 权重名 → fleet 结构化名的正向映射表。

        右侧全部对齐真实存档（见模块 docstring 末尾的 dump 命令）：

        =========================================  ==================
        fleet 结构化名                              shape（smoke 几何）
        =========================================  ==================
        model.embedding.embed_tokens.weight        (V, H)
        model.layers.i.self_attn.qkv_proj.weight   (H, (nq+2nkv)·hd)
        model.layers.i.self_attn.o_proj.weight     (H, H)
        model.layers.i.mlp.up_gate_proj.weight     (H, 2I)   ← dense 层
        model.layers.i.mlp.down_proj.weight        (I, H)    ← dense 层
        model.layers.i.mlp.gate.weight             (E, H) fp32
        model.layers.i.mlp.experts.e.*             (H, 2mI) / (mI, H)
        model.layers.i.mlp.shared_experts.*        (H, 2·ns·mI) / (ns·mI, H)
        model.lm_head.weight                       (V, H)
        model.norm.weight                          (H,)
        =========================================  ==================

        ``^T`` 是因为 HF 的 nn.Linear 权重是 ``(out, in)``，Paddle 是
        ``(in, out)``；``embed_tokens`` / ``lm_head`` / ``gate`` / norm
        两侧同序，不带 ``^T``。

        映射的模板是 ``glm4_moe/modeling.py:829``（同为 MHA + 首层 dense +
        shared experts 的 MoE），三处刻意的差异：

        * 没有 ``gate.e_score_correction_bias`` —— 那是 noaux_tc 路由的产物，
          本模型是 ``scoring_func="softmax"`` + ``topk_method="greedy"``。
        * 没有 ``self_attn.{q,k}_norm`` —— ``use_qk_norm=False``。
        * dense 层集合从 ``moe_layer_freq`` 这张逐层表反推，**不能**读
          ``first_k_dense_replace``（本模型它必须是 ``None``）。
        """
        # 只有 ForCausalLM 两个类，fleet 侧结构化名统一带 "model." 前缀
        # （glm4_moe 那份靠 ``cls == cls.base_model_class`` 推，我们没有注册
        # base model 类，直接钉死更省事，也与 dump 出来的 key 一致）。
        pd_root = "model"
        num_experts = config.n_routed_experts
        moe_layer_freq = config.moe_layer_freq

        statements = [
            f"model.embed_tokens.weight -> {pd_root}.embedding.embed_tokens.weight",
            f"model.norm.weight -> {pd_root}.norm.weight",
        ]
        # 源侧 :93 untie ⇒ 正常走独立 lm_head 这条；tie 的分支只为完整性保留。
        if config.tie_word_embeddings:
            statements.append(f"model.embed_tokens.weight -> {pd_root}.lm_head.weight")
        else:
            statements.append(f"lm_head.weight -> {pd_root}.lm_head.weight")

        for layer_idx in range(config.num_hidden_layers):
            hf = f"model.layers.{layer_idx}"
            pd = f"{pd_root}.layers.{layer_idx}"
            statements += [
                f"{hf}.input_layernorm.weight -> {pd}.input_layernorm.weight",
                f"{hf}.post_attention_layernorm.weight -> {pd}.post_attention_layernorm.weight",
                f"{hf}.self_attn.o_proj.weight^T -> {pd}.self_attn.o_proj.weight",
                f"{hf}.self_attn.q_proj.weight^T, {hf}.self_attn.k_proj.weight^T, "
                f"{hf}.self_attn.v_proj.weight^T -> {pd}.self_attn.qkv_proj.weight, fused_qkv, "
                f"num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
            ]

            if not moe_layer_freq[layer_idx]:
                # dense 层（本模型只有 layer 0）：intermediate_size 那条 FFN。
                statements += [
                    f"{hf}.mlp.gate_proj.weight^T, {hf}.mlp.up_proj.weight^T "
                    f"-> {pd}.mlp.up_gate_proj.weight, fused_ffn",
                    f"{hf}.mlp.down_proj.weight^T -> {pd}.mlp.down_proj.weight",
                ]
                continue

            statements += [
                # gate 两侧 dtype 不同（HF bf16 / fleet fp32）。必须写成
                # src_dtype+dst_dtype 这对：单个 dtype=... 在反推方向上会被
                # aoa_engine.py:736 直接断言拦死。照 kimi_k2/modeling.py:82。
                f"{hf}.mlp.gate.weight -> {pd}.mlp.gate.weight, src_dtype='bfloat16',dst_dtype='float32'",
                f"{hf}.mlp.shared_experts.gate_proj.weight^T, {hf}.mlp.shared_experts.up_proj.weight^T "
                f"-> {pd}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                f"{hf}.mlp.shared_experts.down_proj.weight^T -> {pd}.mlp.shared_experts.down_proj.weight",
                # 逐专家：$EXPERT_ID 由 aoa 解析器展开成 0..num_experts-1。
                # 这里是 axis=1 而不是 fused_ffn —— routed expert 的 up/gate
                # 不参与 TP 切分，直接沿最后一维拼（照 glm4_moe:954 的 fleet 分支）。
                f"{hf}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, {hf}.mlp.experts.$EXPERT_ID.up_proj.weight^T "
                f"-> {pd}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=1",
                f"{hf}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {pd}.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]

            if config.moe_expert_fusion:
                # grouped GEMM：把逐专家的 2-D 权重再沿 axis=0 摞成 3-D
                # [E, ...]。左边是**已经映射好的 fleet 名**，即二段式。
                w1 = ",".join(f"{pd}.mlp.experts.{e}.up_gate_proj.weight" for e in range(num_experts))
                w2 = ",".join(f"{pd}.mlp.experts.{e}.down_proj.weight" for e in range(num_experts))
                statements += [
                    f"{w1} -> {pd}.mlp.grouped_gemm_experts.weight1, axis=0",
                    f"{w2} -> {pd}.mlp.grouped_gemm_experts.weight2, axis=0",
                ]

        return {"aoa_statements": statements}


def _build_hyperbody_decoder(cls, config):
    """两个 ``__new__`` 的公共体：规整 HF config → 建 provider → 组网。"""
    # 并行度兜底：yaml 没给时 LlmMetaConfig 可能塞进 0 或 -1，
    # 组网时拿到非正数会算出空的 pp 切分。
    for name in (
        "tensor_model_parallel_size",
        "context_parallel_size",
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "expert_model_parallel_size",
    ):
        setattr(config, name, max(getattr(config, name, 1) or 1, 1))

    # HyperBody decoder 的关键开关：即使 config.json / yaml 写错也强制纠回。
    # 这三条错了不会报错，只会静默换掉注意力实现或加上 QK norm。
    config.multi_latent_attention = False
    config.use_qk_norm = False
    config.gated_linear_unit = True

    # config 上若带着非 dict 的 rope_scaling（比如 config.json 抄来的 1.0），
    # register_attributes 会把它盖回 provider 的 None 之上，重新引爆
    # gpt_provider.py:211。HyperBody 没有 RoPE scaling，一律归 None。
    if not isinstance(getattr(config, "rope_scaling", None), dict):
        config.rope_scaling = None

    check_hyperbody_decoder_divisibility(config)

    # ★ HF-style config → GPTConfig 的那一步。
    provider = HyperBodyDecoderModelProvider.from_config(config)

    loss_fn = None
    if getattr(config, "dpo_config", None):
        loss_fn = CriterionLayerPipe(config, use_infohub=True)

    gpt_model = provider.provide(loss_fn=loss_fn)
    if not hasattr(config, "architectures"):
        config.architectures = [cls.__name__.replace("Pipe", "")]
    # provide() 返回的是 fleet 的 GPTModel，不是本类实例，所以 aoa 与 fp32
    # 白名单都得手动挂上去（save_pretrained 读的是这个对象的属性）。
    # 只挂正向表：model_utils.py:3286-3289 会自己反推保存方向。
    gpt_model._gen_aoa_config = cls._gen_aoa_config
    gpt_model._keep_in_fp32_modules = cls._keep_in_fp32_modules
    # Trainer 存档时读的是 config_to_save，不是 provider 那份被展平过的 config。
    gpt_model.config_to_save = config
    gpt_model.is_fleet = cls.is_fleet
    return gpt_model


class HyperBodyDecoderForCausalLM(HyperBodyDecoderPretrainedModel):
    is_fleet = True

    def __new__(cls, config):
        return _build_hyperbody_decoder(cls, config)


class HyperBodyDecoderForCausalLMPipe(HyperBodyDecoderPretrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        return _build_hyperbody_decoder(cls, config)
