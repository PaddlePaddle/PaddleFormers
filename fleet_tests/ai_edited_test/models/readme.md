# Model Tests / 模型模块测试

Unit tests for PaddleFleet model components including GPT, CLIP, LLaVA, Qwen, Kimi, and multimodal models.
PaddleFleet 模型组件的单元测试，包括 GPT、CLIP、LLaVA、Qwen、Kimi 和多模态模型。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_backends.py` | Tests for BackendSpecProvider protocol and LocalSpecProvider / 测试后端规格提供者协议与本地实现 |
| `test_ai_clip_vit_model.py` | Tests for get_num_image_embeddings with CLIP, SigLIP, InternViT / 测试 CLIP/SigLIP/InternViT 图像嵌入数量获取 |
| `test_ai_clip_vit_model_extra.py` | Tests for CLIPViTModel initialization variants / CLIP ViT 模型初始化变体测试 |
| `test_ai_context_parallel.py` | Tests for get_padding with SP, CP, and FP8 configurations / 测试不同并行配置下的 padding 获取 |
| `test_ai_empty_layer.py` | Tests for EmptyLayer initialization, forward, and FleetLayer inheritance / 测试空层的初始化、前向传播与继承 |
| `test_ai_gpt_config.py` | Tests for GPTConfig defaults, custom values, and TransformerConfig inheritance / 测试 GPT 配置默认值与继承 |
| `test_ai_gpt_embedding.py` | Tests for GPTEmbeddingSpec and get_placeholder_mask / 测试 GPT 嵌入规格与占位符 mask |
| `test_ai_gpt_model.py` | Tests for build_overlapped_nodes, GPTSublayersSpec, add_sequential_layer / 测试重叠节点构建与子层规格 |
| `test_ai_gpt_model_extra.py` | Tests for GPTSublayersSpec, build_overlapped_nodes helpers / GPT 模型辅助函数额外测试 |
| `test_ai_gpt_model_extra_1.py` | Tests for GPTModel.get_layer_desc_list with various prefixes / 测试不同前缀下的层描述列表获取 |
| `test_ai_gpt_model_extra_2.py` | Tests for GPTSublayersSpec defaults and add/get_sequential_layers / GPT 子层规格默认值与顺序层操作测试 |
| `test_ai_gpt_model_extra_3.py` | Tests for GPTModel.set_state_dict key remapping / 测试 GPT 模型状态字典键重映射 |
| `test_ai_gpt_model_extra_4.py` | Tests for GPTModel build_overlapped_nodes and overlapped_forward_backward / GPT 模型重叠前向反向传播测试 |
| `test_ai_gpt_model_utils.py` | Tests for GPTModelEstimator parameter, FLOPs, and MFU / 测试 GPT 模型参数量、FLOPs 与 MFU 估算 |
| `test_ai_kimi_k25_embedding.py` | Tests for VisionEmbeddingSpec and get_1d_sincos_pos_embed / 测试视觉嵌入规格与正弦位置编码 |
| `test_ai_kimi_k25_model.py` | Tests for KimiK25 vision transformer layer/model classes / 测试 Kimi K25 视觉 Transformer 层与模型 |
| `test_ai_kimi_k25_model_extra.py` | Tests for KimiK25 sublayers and vision transformer forward / Kimi K25 子层与前向传播额外测试 |
| `test_ai_kimi_k25_sd2_tpool_merge.py` | Tests for KimiK25VisionSd2TpoolMerger / 测试 Kimi K25 视觉池化合并器 |
| `test_ai_language_loss.py` | Tests for subbatch utility and LanguageLoss initialization / 测试子批次工具与语言损失初始化 |
| `test_ai_language_loss_extra.py` | Tests for LanguageLoss init, forward_impl, forward with MTP / 语言损失初始化与前向传播测试 |
| `test_ai_language_loss_extra_1.py` | Tests for subbatch with use_recompute=True / 启用重计算的子批次测试 |
| `test_ai_language_loss_extra_2.py` | Tests for LanguageLoss with various parallel configurations / 不同并行配置下的语言损失测试 |
| `test_ai_language_loss_extra_3.py` | Tests for subbatch function with keyword arguments / 关键字参数子批次测试 |
| `test_ai_language_loss_extra_4.py` | Tests for MTPLanguageLoss.forward / 测试多 Token 预测语言损失前向传播 |
| `test_ai_llava_model.py` | Tests for LLaVA model module-level constants / 测试 LLaVA 模块级常量 |
| `test_ai_llava_model_extra.py` | Tests for pixel_shuffle standalone function / 测试像素重排函数 |
| `test_ai_llava_model_extra_1.py` | Tests for LLaVAModel set_input_tensor additional paths / LLaVA 模型输入张量设置额外测试 |
| `test_ai_llava_model_extra_2.py` | Tests for pixel_shuffle function correctness / 像素重排正确性测试 |
| `test_ai_llava_model_extra_3.py` | Tests for pixel_shuffle with version parameter / 带版本参数的像素重排测试 |
| `test_ai_llava_spec.py` | Tests for decoder_model_with_local_default_spec / 测试本地默认规格解码模型函数 |
| `test_ai_llava_spec_extra.py` | Tests for LLaVA spec components / LLaVA 规格组件测试 |
| `test_ai_lm_head.py` | Tests for GPTLMHead forward method / 测试 GPT 语言模型头前向传播 |
| `test_ai_moe_layer_specs.py` | Tests for get_moe_layer_spec_for_backend / 测试后端 MoE 层规格获取 |
| `test_ai_mtp_head_loss.py` | Tests for separate_mtp_headloss branches / 测试独立 MTP 头损失分支 |
| `test_ai_multimodal_context_parallel.py` | Tests for multimodal get_padding with SP, CP, FP8 / 多模态上下文并行 padding 测试 |
| `test_ai_multimodal_projector.py` | Tests for MultimodalProjector with various projector types / 测试多模态投影器 |
| `test_ai_multimodal_projector_extra.py` | Tests for MultimodalProjector with real MLP projector / 多模态 MLP 投影器测试 |
| `test_ai_qwen3_5_model.py` | Tests for Qwen3_5RMSNorm class / 测试 Qwen3.5 RMS 归一化层 |
| `test_ai_qwen3_5_model_extra.py` | Tests for Qwen3_5RMSNorm initialization / Qwen3.5 RMS 归一化初始化测试 |
| `test_ai_qwen3_vl_model.py` | Tests for Qwen3VL vision model sublayers and layer structure / 测试 Qwen3VL 视觉模型子层与层结构 |
| `test_ai_qwen3_vl_patch_merger.py` | Tests for Qwen3VLVisionPatchMergerSpec dataclass / 测试 Qwen3VL 视觉 patch 合并规格 |
| `test_ai_radio.py` | Tests for RADIOViTModel initialization with mocked internals / 测试 RADIO ViT 模型初始化 |
| `test_ai_radio_extra.py` | Tests for RADIOViTModel.get_pos_enc in eval mode / RADIO ViT 位置编码评估模式测试 |
| `test_ai_rope_utils_extra.py` | Tests for _apply_rotary_pos_emb_bshd function / 旋转位置编码应用函数测试 |
| `test_ai_rope_utils_extra_2.py` | Tests for _rotate_half function / 旋转半张量函数测试 |
| `test_ai_vision_layer.py` | Tests for VisionLayer initialization / 测试视觉层初始化 |
| `test_ai_vit_layer_specs.py` | Tests for get_vit_layer_with_local_spec / 测试本地规格 ViT 层获取 |
