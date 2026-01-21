# 模型列表和使用说明

|Model Name|Training Method|Context Length|Quantization|XPUs Required|Deployment Commands|Applicable Version|
|-|-|-|-|-|-|-|
|ERNIE-4.5-21B-A3B|SFT|32K|BF16|8|paddleformers-cli train examples/config/xpu/ERNIE-4.5-21B-A3B/sft/full_32k.yaml|1.0|
||SFT-LoRA|32K|BF16|4|bash train examples/config/xpu/ERNIE-4.5-21B-A3B/sft/run_lora_32k.sh|1.0|
|ERNIE-4.5-0.3B|SFT|8k|BF16|1|paddleformers-cli train examples/config/xpu/ERNIE-4.5-0.3B/sft/full_8k.yaml|1.0|
||SFT-LoRA|8k|BF16|1|paddleformers-cli train examples/config/xpu/ERNIE-4.5-0.3B/sft/lora_8k.yaml|1.0|
|PaddleOCR-VL-0.9B|SFT-Full|16K|BF16|1|bash examples/config/xpu/PaddleOCR-VL/sft/run_paddleocr-vl_full_16k.sh|1.0|
||SFT-LoRA|16K|BF16|1|bash examples/config/xpu/PaddleOCR-VL/sft/run_paddleocr-vl_lora_16k.sh|1.0|
|DeepSeek-V3|SFT-Full|32K|BF16|128|mpirun bash ./examples/config/xpu/DeepseekV3/sft/run_full_32k.sh|1.0|
