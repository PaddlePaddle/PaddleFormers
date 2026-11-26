# 训练能力支持
|模型|PT / CPT|SFT|SFT-LoRA|DPO|DPO-LoRA|
|-|-|-|-|-|-|
|Cohere|✓|✓|✓|✓|✓|
|StableLM|✓|✓|✓|✓|✓|
|DeepSeekv3|✓|✓|✓|✓|✓|
|🏛️ERNIE-4.5|✓|✓|✓|✓|✓|
|Gemma3|✓|✓|✓|✓|✓|
|GLM-4.5|✓|✓|✓|✓|✓|
|GPT-OSS|✓|✓|✓|x|x|
|Seed-OSS|✓|✓|✓|x|x|
|Granite|✓|✓|✓|✓|✓|
|Hunyuan Dense V1|✓|✓|✓|x|x|
|LLaMA3|✓|✓|✓|✓|✓|
|LLaMA4-Text|✓|✓|✓|x|x|
|MiniMax-Text-01|✓|✓|✓|x|x|
|OLMo2|✓|✓|✓|✓|✓|
|Phi4|✓|✓|✓|✓|✓|
|Qwen2|✓|✓|✓|✓|✓|
|Qwen3|✓|✓|✓|✓|✓|
|Qwen3-Next|✓|✓|✓|✓|✓|
|LFM2-VL|x|✓|✓|x|x|
|Llama Guard 4|x|✓|✓|x|x|
|🏛️ERNIE-4.5-VL|x|✓|✓|x|x|
|Phi-4-multimodal|x|✓|✓|x|x|
|AyaVision|x|✓|✓|x|x|
|🏛️PaddleOCR-VL|x|✓|✓|x|x|
|Molmo|x|✓|✓|x|x|
|Qwen2.5-VL|x|✓|✓|x|x|
|Qwen3-VL|x|✓|✓|x|x|

LFM2-VL 的单卡 SFT/LoRA 已有前向、loss 和梯度回归；本地原生 HF 小模型采用 FP32/eager 对齐。文本 TP/SP 已实现列/行并行投影、分片 ShortConv、序列切分及归一化梯度归约，但尚待双 GPU/NCCL 实测，因此下表暂不标记 TP/SP 通过。双卡检查命令：`python -m tests.transformers.lfm2_vl.parallel_regression`，比较 TP=2、TP=2+SP 的 logits 和两次梯度累积，覆盖 tied/untied embedding。

# 分布式能力支持
|模型|TP + SP|PP|EP|CP|DP|FSDP|
|-|-|-|-|-|-|-|
|Cohere|✓|✓|-|x|✓|✓|
|StableLM|✓|✓|-|x|✓|✓|
|DeepSeekv3|✓|✓|✓|x|✓|✓|
|🏛️ERNIE-4.5|✓|✓|✓|x|✓|✓|
|Gemma3|x|✓|-|x|✓|✓|
|GLM-4.5|✓|✓|✓|✓|✓|✓|
|GPT-OSS|✓|✓|x|x|✓|✓|
|Seed-OSS|✓|✓|-|x|✓|✓|
|Granite|✓|✓|-|x|✓|✓|
|Hunyuan Dense V1|✓|✓|-|x|✓|✓|
|LLaMA3|✓|✓|-|x|✓|✓|
|LLaMA4-Text|x|x|x|x|✓|✓|
|MiniMax-Text-01|x|x|x|x|✓|✓|
|OLMo2|✓|✓|-|x|✓|✓|
|Phi4|✓|✓|-|x|✓|✓|
|Qwen2|✓|✓|x|x|✓|✓|
|Qwen3|✓|✓|✓|✓|✓|✓|
|Qwen3-Next|✓|✓|✓|x|✓|✓|
|LFM2-VL|x|x|-|x|x|x|
|Llama Guard 4|x|x|-|x|x|x|
|🏛️ERNIE-4.5-VL|✓|✓|✓|x|✓|✓|
|Phi-4-multimodal|x|x|-|x|✓|✓|
|AyaVision|x|x|-|x|✓|✓|
|🏛️PaddleOCR-VL|x|x|-|x|✓|✓|
|Molmo|x|x|x|x|✓|✓|
|Qwen2.5-VL|✓|x|-|x|✓|✓|
|Qwen3-VL|x|x|✓|x|✓|✓|

# 多硬件训练支持
|模型|昆仑芯 P800|天数智芯天垓150|沐曦 C550|摩尔线程 S5000|
|-|-|-|-|-|
|🏛️PaddleOCR-VL|✓|✓|x|✓|
|🏛️ERNIE-4.5|✓|✓|✓|✓|
|DeepSeekv3|✓|x|x|x|
