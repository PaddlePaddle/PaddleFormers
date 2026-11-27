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
|OpenELM|✓|✓|✓|✓|✓|
|OLMo2|✓|✓|✓|✓|✓|
|OLMo3|✓|✓|✓|x|✓|
|Phi4|✓|✓|✓|✓|✓|
|Qwen2|✓|✓|✓|✓|✓|
|Qwen3|✓|✓|✓|✓|✓|
|Qwen3-Next|✓|✓|✓|✓|✓|
|DeepSeek-OCR-2|x|✓|✓|x|x|
|Llama Guard 4|x|✓|✓|x|x|
|🏛️ERNIE-4.5-VL|x|✓|✓|x|x|
|Phi-4-multimodal|x|✓|✓|x|x|
|AyaVision|x|✓|✓|x|x|
|🏛️PaddleOCR-VL|x|✓|✓|x|x|
|Molmo|x|✓|✓|x|x|
|Qwen2.5-VL|x|✓|✓|x|x|
|Qwen3-VL|x|✓|✓|x|x|

DeepSeek-OCR-2 新增原生 `DeepseekOcr2*` 接口及 `deepseek_ocr2_native` 模板；视觉分支参与反向传播。历史 `DeepseekOCR2*` checkpoint 使用原模板。原生路径的本地回归覆盖小模型 FP32/eager 前向、视觉梯度、缓存生成和权重往返；不能视为完整预训练模型或分布式训练验收。

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
|OpenELM|x|✓|-|x|✓|✓|
|OLMo2|✓|✓|-|x|✓|✓|
|OLMo3|✓|✓|-|x|✓|x|
|Phi4|✓|✓|-|x|✓|✓|
|Qwen2|✓|✓|x|x|✓|✓|
|Qwen3|✓|✓|✓|✓|✓|✓|
|Qwen3-Next|✓|✓|✓|x|✓|✓|
|DeepSeek-OCR-2|x|x|-|x|x|x|
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
