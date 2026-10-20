# Falcon Perception 单测

采用与 MiniCPM4.1、Qwen2.5-VL 相同的 `unittest.TestCase`、`ModelTester`、`ConfigTester` 组织方式。Falcon 当前提供专用推理接口，因此直接测试其 `tokens`、KV cache 和元组输出，不继承包含训练、标准文本生成和模型保存假设的通用测试 mixin。

| 文件 | 覆盖内容 |
| --- | --- |
| `test_configuration.py` | 公共配置序列化、保存加载、参数设置与 Falcon 感知配置字段 |
| `test_modeling.py` | 小模型输出形状与有限值、文本及图文缓存解码等价性、因果注意力、图像 padding、几何输入和解码、缓存重置、分割后处理与输入限制 |
| `test_processing.py` | 本地图片读取、缩放、归一化、RGB 转换、padding mask、图像 token 展开、多图顺序、BOS、整块截断、时空位置及批处理左填充 |
| `test_generation.py` | 结构化实例生成、AnyUp、RLE 输出、两种停止 token，以及连续调用的状态隔离 |
| `test_auto.py` | Auto 注册、原始 HF 权重及子目录加载、配置回退和缓存推理 |
| `test_reference.py` | attention、RoPE、FFN、权重转置和后处理的独立参考检查 |
| `test_acceptance.py` | 精度验收工具是否检查完整 logits 和实际生成 token |

新增测试使用小模型、合成图片及本地构造的 tokenizer，无需下载权重或访问网络。生成测试只固定采样器的输出 token，实际运行模型前向、缓存、几何头、AnyUp 和分割后处理；它验证生成流程，不代表真实权重精度验收。

从 PaddleFormers 仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=7 FLAGS_allow_tf32_cublas=0 FLAGS_allow_tf32_cudnn=0 \
PYTHONPATH="$PWD" python -m pytest -q tests/transformers/falcon_perception
```

使用已配置依赖的 `paddle_falcon` 环境，也可将物理卡改为 6。模型前向与生成测试使用仓库的 `gpu_device_initializer`，无可用 GPU 时跳过；当前 Paddle 环境的 RMSNorm 内核需要 GPU。配置、预处理和原有参考检查使用 CPU。
