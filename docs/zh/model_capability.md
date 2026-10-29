# 训练能力支持
|模型|PT / CPT|SFT|SFT-LoRA|DPO|DPO-LoRA|
|-|-|-|-|-|-|
|Cohere|✓|✓|✓|✓|✓|
|StableLM|✓|✓|✓|待验证|待验证|
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
|StableLM|待验证|待验证|-|x|待验证|待验证|
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

StableLM 已接入 TP、SP 和 PP（`StableLmForCausalLMPipe`）训练路径；当前仅有单卡验证，多卡训练与推理仍待实测。PP 暂不支持带持久 KV Cache 的自回归生成。DPO、DPO-LoRA、DP 和 FSDP 尚未完成端到端验证。

# 多硬件训练支持
|模型|昆仑芯 P800|天数智芯天垓150|沐曦 C550|摩尔线程 S5000|
|-|-|-|-|-|
|🏛️PaddleOCR-VL|✓|✓|x|✓|
|🏛️ERNIE-4.5|✓|✓|✓|✓|
|DeepSeekv3|✓|x|x|x|
