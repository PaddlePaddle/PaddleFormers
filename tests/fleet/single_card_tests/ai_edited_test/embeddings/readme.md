# Embedding Tests / 嵌入模块测试

Unit tests for PaddleFleet embedding modules including language model embedding and rotary position embedding.
PaddleFleet 嵌入模块的单元测试，包括语言模型嵌入和旋转位置编码。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_language_model_embedding.py` | Tests for LanguageModelEmbedding init, forward, sequence parallel / 测试语言模型嵌入的初始化、前向传播与序列并行 |
| `test_ai_language_model_embedding_extra.py` | Tests for LanguageModelEmbedding init, embedding_weight, forward / 语言模型嵌入的额外测试 |
| `test_ai_rope_utils.py` | Tests for get_pos_emb_on_this_cp_rank, _rotate_half, _apply_rotary_pos_emb_bshd / 测试旋转位置编码工具函数 |
| `test_ai_rotary_pos_embedding.py` | Tests for RotaryEmbedding initialization, forward, get_cos_sin, scaling / 测试旋转位置编码的初始化与缩放 |
| `test_ai_rotary_pos_embedding_extra.py` | Tests for RotaryEmbedding._apply_scaling and MultimodalRotaryEmbedding / 测试旋转编码缩放与多模态旋转编码 |
| `test_ai_yarn_roary_pos_embedding_extra.py` | Tests for YARN rotary embedding helper functions / 测试 YARN 长上下文旋转编码辅助函数 |
| `test_ai_yarn_rotary_pos_embedding.py` | Tests for YARN rotary embedding helpers for long context scaling / 测试 YARN 长上下文缩放旋转编码辅助函数 |
