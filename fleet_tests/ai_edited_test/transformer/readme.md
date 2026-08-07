# Transformer Tests / Transformer 模块测试

Unit tests for PaddleFleet transformer module including attention, MLP, MoE, dot product attention, multi-latent attention, and transformer layer/encoder.
PaddleFleet Transformer 模块的单元测试，包括注意力、MLP、MoE、点积注意力、多潜在注意力和 Transformer 层/编码器。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_attention.py` | Tests for SelfAttentionSublayersSpec dataclass / 测试自注意力子层规格 |
| `test_ai_attention_extra.py` | Tests for SelfAttentionSublayersSpec / 自注意力子层规格额外测试 |
| `test_ai_attention_extra_1.py` | Tests for SelfAttention gated_attention attribute / 测试自注意力门控属性 |
| `test_ai_attention_extra_2.py` | Tests for Attention.set_for_recompute_input_layernorm / 测试注意力重计算输入层归一化设置 |
| `test_ai_attention_extra_3.py` | Tests for _md5 helper function / 测试 MD5 辅助函数 |
| `test_ai_block_attn_res_extra.py` | Tests for BlockAttnResSublayersSpec / 测试块注意力残差子层规格 |
| `test_ai_dot_product_attention.py` | Tests for DotProductAttention constructor / 测试点积注意力构造器 |
| `test_ai_dot_product_attention_extra.py` | Tests for DotProductAttention construction / 点积注意力构造额外测试 |
| `test_ai_dot_product_attention_extra_1.py` | Tests for DotProductAttention construction / 点积注意力构造更多测试 |
| `test_ai_dot_product_attention_extra_2.py` | Tests for DotProductAttention with GQA / 测试组查询注意力下的点积注意力 |
| `test_ai_dot_product_attention_extra_3.py` | Tests for DotProductAttention initialization / 点积注意力初始化测试 |
| `test_ai_dot_product_attention_extra_4.py` | Tests for DotProductAttention with eager attention / 测试 eager 实现的点积注意力 |
| `test_ai_dsa_attention.py` | Tests for hadamard_transform function / 测试 Hadamard 变换函数 |
| `test_ai_fused_a2a_extra.py` | Tests for barrier_ep function / 测试 EP 屏障函数 |
| `test_ai_fusion_layer_utils_extra.py` | Tests for FP8_ALIGN constant / 测试 FP8 对齐常量 |
| `test_ai_gated_delta_net.py` | Tests for _l2norm helper function / 测试 L2 范数辅助函数 |
| `test_ai_mlp.py` | Tests for MLP constructor / 测试 MLP 构造器 |
| `test_ai_mlp_extra.py` | Tests for MLP with swiglu activation / 测试 SwiGLU 激活路径下的 MLP |
| `test_ai_mlp_extra_1.py` | Tests for MLP with bias_activation_fusion / 测试带偏置激活融合的 MLP |
| `test_ai_moe_expert_extra.py` | Tests for BMMFunction / 测试批量矩阵乘法函数 |
| `test_ai_moe_expert_extra_2.py` | Tests for GroupedMLPExpert initialization / 测试分组 MLP 专家初始化 |
| `test_ai_moe_fp8_utils_extra.py` | Tests for FP8_ALIGN constant / FP8 对齐常量额外测试 |
| `test_ai_moe_fp8_utils_extra_1.py` | Tests for FP8_ALIGN constant / FP8 对齐常量更多测试 |
| `test_ai_moe_layer_extra_2.py` | Tests for MoELayer method existence / 测试 MoE 层方法存在性 |
| `test_ai_moe_layer_extra_3.py` | Tests for MoELayer module import and structure / MoE 层导入与结构测试 |
| `test_ai_moe_layer_transformer_extra.py` | Tests for MoESublayers dataclass / 测试 MoE 子层数据类 |
| `test_ai_moe_utils_extra.py` | Tests for permute function / 测试 permute 函数 |
| `test_ai_moe_utils_extra_1.py` | Tests for AddAuxiliaryLoss / 测试辅助损失加法 |
| `test_ai_moe_utils_extra_2.py` | Tests for barrier_ep function / EP 屏障函数额外测试 |
| `test_ai_moe_utils_extra_3.py` | Tests for RandomSTE / 测试随机直通估计器 |
| `test_ai_moe_utils_extra_4.py` | Tests for is_tensor function / 测试张量判断函数 |
| `test_ai_moe_utils_extra_5.py` | Tests for barrier_ep function / EP 屏障函数更多测试 |
| `test_ai_moe_utils_extra_6.py` | Tests for RandomSTE PyLayer / 随机直通估计器 PyLayer 测试 |
| `test_ai_moe_utils_extra_7.py` | Tests for manual_backward function / 测试手动反向传播函数 |
| `test_ai_mtp_mask_experimental_dataflow.py` | Tests for experimental_dataflow config field / 测试 experimental_dataflow 配置字段 |
| `test_ai_multi_latent_attention.py` | Tests for MLASelfAttentionSublayersSpec / 测试多潜在自注意力子层规格 |
| `test_ai_multi_latent_attention_extra.py` | Tests for MLASelfAttention backward_dw / 测试 MLA 自注意力反向权重梯度 |
| `test_ai_multi_latent_attention_extra_2.py` | Tests for _ec_compatible_rope_apply / 测试 EC 兼容旋转位置编码应用 |
| `test_ai_multi_latent_attention_extra_3.py` | Tests for MLASelfAttention.backward_dw / MLA 自注意力反向权重梯度额外测试 |
| `test_ai_paddle_norm.py` | Tests for RMSNorm layer / 测试 RMS 归一化层 |
| `test_ai_paddle_norm_extra.py` | Tests for RMSNorm forward / RMS 归一化前向传播测试 |
| `test_ai_paddle_norm_extra_1.py` | Tests for WrappedPaddleNorm normalization type / 测试封装 Paddle 归一化类型选择 |
| `test_ai_token_dispatcher_extra.py` | Tests for _DispatchManager abstract interface / 测试调度管理器抽象接口 |
| `test_ai_token_dispatcher_extra_1.py` | Tests for _DispatchManager abstract class / 调度管理器抽象类额外测试 |
| `test_ai_token_dispatcher_extra_2.py` | Tests for _DeepepManager setup_metadata / 测试 DeepEP 管理器元数据设置 |
| `test_ai_transformer_block.py` | Tests for TransformerBlockSublayersSpec / 测试 Transformer 块子层规格 |
| `test_ai_transformer_block_extra.py` | Tests for TransformerBlock forward / Transformer 块前向传播测试 |
| `test_ai_transformer_config.py` | Tests for TransformerConfig defaults / 测试 Transformer 配置默认值 |
| `test_ai_transformer_config_extra.py` | Tests for TransformerConfig defaults / Transformer 配置默认值额外测试 |
| `test_ai_transformer_encoder.py` | Tests for build_overlapped_nodes / 测试重叠节点构建 |
| `test_ai_transformer_encoder_extra.py` | Tests for TransformerEncoder helper methods / Transformer 编码器辅助方法测试 |
| `test_ai_transformer_encoder_extra_1.py` | Tests for build_overlapped_nodes with layer nodes / 带层节点的重叠节点构建测试 |
| `test_ai_transformer_encoder_extra_2.py` | Tests for _set_pipeline_name_mapping / 测试流水线名称映射设置 |
| `test_ai_transformer_encoder_extra_3.py` | Tests for build_overlapped_nodes when no overlap layers / 无重叠层时的节点构建测试 |
| `test_ai_transformer_encoder_extra_4.py` | Tests for TransformerEncoder.set_state_dict / Transformer 编码器状态字典设置测试 |
| `test_ai_transformer_encoder_extra_5.py` | Tests for TransformerEncoder.get_layer_desc_list modal prefix / Transformer 编码器模态前缀测试 |
| `test_ai_transformer_layer.py` | Tests for tensors_clone utility / 测试张量克隆工具函数 |
| `test_ai_transformer_layer_extra.py` | Tests for tensors_clone utility / 张量克隆工具函数额外测试 |
| `test_ai_transformer_layer_extra_1.py` | Tests for tensors_clone with nested structures / 嵌套结构张量克隆测试 |
| `test_ai_transformer_layer_extra_2.py` | Tests for tensors_clone with edge cases / 张量克隆边界情况测试 |
| `test_ai_transformer_layer_extra_3.py` | Tests for TransformerLayerNode schedule / Transformer 层节点调度测试 |
| `test_ai_transformer_utils.py` | Tests for get_default_causal_mask / 测试默认因果掩码获取 |
| `test_ai_transformer_utils_extra.py` | Tests for AttnMaskType enum / 测试注意力掩码类型枚举 |
