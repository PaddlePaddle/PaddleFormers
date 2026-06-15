# Fusions Tests / 融合算子模块测试

Unit tests for PaddleFleet fusion operators including bias activation, layer norm, RMS norm, softmax, and SwiGLU.
PaddleFleet 融合算子的单元测试，包括偏置激活、层归一化、RMS 归一化、softmax 和 SwiGLU。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_bias_dropout_extra.py` | Tests for _bias_dropout_add_func with training/eval modes / 测试偏置 dropout 加法函数的训练/推理模式 |
| `test_ai_bias_geglu_extra4.py` | Tests for GEGLU activation functions including back functions / 测试 GEGLU 激活函数及其反向传播 |
| `test_ai_bias_geglu_extra5.py` | Tests for BiasGeGLUFunction, GeGLUFunction, and WeightedQuickGeGLUFunction PyLayers / 测试 GEGLU 系列 PyLayer |
| `test_ai_bias_gelu_extra2.py` | Tests for bias_gelu, bias_gelu_back, and GeLUFunction PyLayer / 测试 GeLU 偏置激活与反向传播 |
| `test_ai_bias_swiglu_extra2.py` | Tests for SwiGLU activation functions / 测试 SwiGLU 激活函数 |
| `test_ai_bias_swiglu_extra3.py` | Tests for BiasSwiGLUFunction and SwiGLUFunction PyLayers / 测试 SwiGLU 系列 PyLayer |
| `test_ai_fused_bias_dropout.py` | Tests for _bias_dropout_add_func, bias_dropout_add_unfused / 测试融合偏置 dropout 加法 |
| `test_ai_fused_bias_geglu_extra.py` | Additional correctness tests for GEGLU functions / GEGLU 函数正确性额外测试 |
| `test_ai_fused_bias_geglu_extra2.py` | Tests for geglu, bias_geglu, and quick_geglu output shapes / 测试 GEGLU 输出形状 |
| `test_ai_fused_bias_geglu_extra3.py` | Tests for geglu and bias_geglu with various input dimensions / 不同输入维度下的 GEGLU 测试 |
| `test_ai_fused_bias_gelu_extra.py` | Additional tests for bias_gelu and GeLUFunction autograd / GeLU 自动微分额外测试 |
| `test_ai_fused_bias_swiglu.py` | Tests for swiglu, bias_swiglu, and weighted_swiglu output shapes / 测试 SwiGLU 输出形状 |
| `test_ai_fused_bias_swiglu_extra.py` | Tests for swiglu_back and bias_swiglu_back CUDA and CPU paths / 测试 SwiGLU 反向传播的 CPU/CUDA 路径 |
| `test_ai_fused_layer_norm.py` | Tests for FusedLayerNorm initialization and reset_parameters / 测试融合层归一化初始化与参数重置 |
| `test_ai_fused_layer_norm_extra.py` | Tests for FusedLayerNorm with various config options / 不同配置下的融合层归一化测试 |
| `test_ai_fused_rms_norm.py` | Tests for FusedRmsNorm initialization and sequence parallel flag / 测试融合 RMS 归一化初始化与序列并行 |
| `test_ai_fused_rms_norm_extra.py` | Tests for FusedRmsNorm with various config options / 不同配置下的融合 RMS 归一化测试 |
| `test_ai_fused_softmax.py` | Tests for SoftmaxOne initialization and forward pass / 测试 SoftmaxOne 初始化与前向传播 |
| `test_ai_fused_softmax_extra.py` | Tests for SoftmaxOne with mixed precision and causal handling / 混合精度与因果掩码下的 SoftmaxOne 测试 |
| `test_ai_fused_softmax_extra2.py` | Tests for SoftmaxOne forward and FusedScaleMaskSoftmax initialization / SoftmaxOne 前向与融合缩放掩码 softmax 测试 |
| `test_ai_fused_swiglu_scale.py` | Tests for fused_swiglu_scale_forward CPU fallback and CUDA paths / 测试融合 SwiGLU 缩放的前向传播 |
| `test_ai_fused_swiglu_scale_extra.py` | Tests for fused_swiglu_scale_forward and backward CPU/CUDA / 测试融合 SwiGLU 缩放的前向与反向传播 |
| `test_ai_fused_swiglu_scale_extra2.py` | Tests for fused_swiglu_scale_forward and backward CPU fallback / 融合 SwiGLU 缩放 CPU 回退路径测试 |
| `test_ai_fusions_module_structure.py` | Tests for fusions module imports / 测试融合模块导入结构 |
| `test_ai_quick_geglu.py` | Tests for quick_gelu function output shape, range, and dtype / 测试 quick_gelu 输出形状、范围与数据类型 |
