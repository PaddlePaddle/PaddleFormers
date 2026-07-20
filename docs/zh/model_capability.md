# 训练能力支持
|模型|PT / CPT|SFT|SFT-LoRA|DPO|DPO-LoRA|
|-|-|-|-|-|-|
|DeepSeekv3|✓|✓|✓|✓|✓|
|🏛️ERNIE-4.5|✓|✓|✓|✓|✓|
|Gemma3|✓|✓|✓|✓|✓|
|GLM-4.5|✓|✓|✓|✓|✓|
|GPT-OSS|✓|✓|✓|x|x|
|LLaMA3|✓|✓|✓|✓|✓|
|OLMo2|✓|✓|✓|✓|✓|
|Phi4|✓|✓|✓|✓|✓|
|Qwen2|✓|✓|✓|✓|✓|
|Qwen3|✓|✓|✓|✓|✓|
|Qwen3-Next|✓|✓|✓|✓|✓|
|🏛️ERNIE-4.5-VL|x|✓|✓|x|x|
|🏛️PaddleOCR-VL|x|✓|✓|x|x|
|Qwen2.5-VL|x|✓|✓|x|x|
|Qwen3-VL|x|✓|✓|x|x|

# 分布式能力支持
|模型|TP + SP|PP|EP|CP|DP|FSDP|
|-|-|-|-|-|-|-|
|DeepSeekv3|✓|✓|✓|x|✓|✓|
|🏛️ERNIE-4.5|✓|✓|✓|x|✓|✓|
|Gemma3|x|✓|-|x|✓|✓|
|GLM-4.5|✓|✓|✓|✓|✓|✓|
|GPT-OSS|✓|✓|x|x|✓|✓|
|LLaMA3|✓|✓|-|x|✓|✓|
|OLMo2|✓|✓|-|x|✓|✓|
|Phi4|✓|✓|-|x|✓|✓|
|Qwen2|✓|✓|x|x|✓|✓|
|Qwen3|✓|✓|✓|✓|✓|✓|
|Qwen3-Next|✓|✓|✓|x|✓|✓|
|🏛️ERNIE-4.5-VL|✓|✓|✓|x|✓|✓|
|🏛️PaddleOCR-VL|x|x|-|x|✓|✓|
|Qwen2.5-VL|✓|x|-|x|✓|✓|
|Qwen3-VL|x|x|✓|x|✓|✓|

# 多硬件训练支持
|模型|昆仑芯 P800|天数智芯天垓150|沐曦 C550|摩尔线程 S5000|
|-|-|-|-|-|
|🏛️PaddleOCR-VL|✓|✓|x|✓|
|🏛️ERNIE-4.5|✓|✓|✓|✓|
|DeepSeekv3|✓|x|x|x|
