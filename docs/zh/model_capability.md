# 训练能力支持
|模型|PT / CPT|SFT|SFT-LoRA|DPO|DPO-LoRA|
|-|-|-|-|-|-|
|Cohere|✓|✓|✓|✓|✓|
|DeepSeekv3|✓|✓|✓|✓|✓|
|🏛️ERNIE-4.5|✓|✓|✓|✓|✓|
|Gemma3|✓|✓|✓|✓|✓|
|GLM-4.5|✓|✓|✓|✓|✓|
|GPT-OSS|✓|✓|✓|x|x|
|Seed-OSS|✓|✓|✓|x|x|
|Granite|✓|✓|✓|✓|✓|
|LLaMA3|✓|✓|✓|✓|✓|
|LLaMA4-Text|✓|✓|✓|x|x|
|MiniMax-Text-01|✓|✓|✓|x|x|
|OLMo2|✓|✓|✓|✓|✓|
|OLMo3|待端到端验证|待端到端验证|待验证|待验证|待验证|
|Phi4|✓|✓|✓|✓|✓|
|Qwen2|✓|✓|✓|✓|✓|
|Qwen3|✓|✓|✓|✓|✓|
|Qwen3-Next|✓|✓|✓|✓|✓|
|Llama Guard 4|x|✓|✓|x|x|
|🏛️ERNIE-4.5-VL|x|✓|✓|x|x|
|Phi-4-multimodal|x|✓|✓|x|x|
|🏛️PaddleOCR-VL|x|✓|✓|x|x|
|Molmo|x|✓|✓|x|x|
|Qwen2.5-VL|x|✓|✓|x|x|
|Qwen3-VL|x|✓|✓|x|x|

# 分布式能力支持
|模型|TP + SP|PP|EP|CP|DP|FSDP|
|-|-|-|-|-|-|-|
|Cohere|✓|✓|-|x|✓|✓|
|DeepSeekv3|✓|✓|✓|x|✓|✓|
|🏛️ERNIE-4.5|✓|✓|✓|x|✓|✓|
|Gemma3|x|✓|-|x|✓|✓|
|GLM-4.5|✓|✓|✓|✓|✓|✓|
|GPT-OSS|✓|✓|x|x|✓|✓|
|Seed-OSS|✓|✓|-|x|✓|✓|
|Granite|✓|✓|-|x|✓|✓|
|LLaMA3|✓|✓|-|x|✓|✓|
|LLaMA4-Text|x|x|x|x|✓|✓|
|MiniMax-Text-01|x|x|x|x|✓|✓|
|OLMo2|✓|✓|-|x|✓|✓|
|OLMo3|已接入，待多卡验证|已接入，待多卡验证|-|x|待多卡验证|待多卡验证|
|Phi4|✓|✓|-|x|✓|✓|
|Qwen2|✓|✓|x|x|✓|✓|
|Qwen3|✓|✓|✓|✓|✓|✓|
|Qwen3-Next|✓|✓|✓|x|✓|✓|
|Llama Guard 4|x|x|-|x|x|x|
|🏛️ERNIE-4.5-VL|✓|✓|✓|x|✓|✓|
|Phi-4-multimodal|x|x|-|x|✓|✓|
|🏛️PaddleOCR-VL|x|x|-|x|✓|✓|
|Molmo|x|x|x|x|✓|✓|
|Qwen2.5-VL|✓|x|-|x|✓|✓|
|Qwen3-VL|x|x|✓|x|✓|✓|

## OLMo3 验证状态与分布式回归

OLMo3 的单卡基础回归及原生 Paddle、flex_checkpoint 保存重载已验证。
包含真实模型 logits、十 token 对齐、聊天模板生成的 Olmo2/Olmo3 完整单卡回归已通过；JUnit 记录 94 项、0 失败、0 错误、6 项公共框架跳过。
固定参考数组已在 Transformers 4.57.0、PyTorch 2.9.0 的 CPU SDPA BF16 路径精确复现。
7B 模型使用 Paddle GPU SDPA BF16 时，前 20 个 logits 的平均绝对误差为 `0.008935546875`，达到 `< 1e-2` 要求，前 10 个生成 token 完全一致。
多卡验收未完成；单卡 stage 对照、输出及梯度测试不替代 TP/PP 多卡实测。
分布式回归沿用公共脚本 `tests/transformers/test_tensor_parallel.py`，在具备相应 GPU 数量的环境执行：

```bash
# TP=2: MHA/GQA and SP output/gradient comparisons
PADDLEFORMERS_TESTING=1 ~/miniconda3/envs/paddleformers/bin/python -m paddle.distributed.launch --devices 0,1 tests/transformers/test_tensor_parallel.py
# PP=2: tied weights, recompute, inference and optimizer updates
PADDLEFORMERS_TESTING=1 ~/miniconda3/envs/paddleformers/bin/python -m paddle.distributed.launch --devices 0,1 tests/transformers/test_tensor_parallel.py --pipeline-parallel-size 2
# TP=2, PP=2
PADDLEFORMERS_TESTING=1 ~/miniconda3/envs/paddleformers/bin/python -m paddle.distributed.launch --devices 0,1,2,3 tests/transformers/test_tensor_parallel.py --pipeline-parallel-size 2
```

# 多硬件训练支持

|模型|昆仑芯 P800|天数智芯天垓150|沐曦 C550|摩尔线程 S5000|
|-|-|-|-|-|
|🏛️PaddleOCR-VL|✓|✓|x|✓|
|🏛️ERNIE-4.5|✓|✓|✓|✓|
|DeepSeekv3|✓|x|x|x|
