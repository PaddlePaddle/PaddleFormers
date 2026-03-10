# 模型列表和使用说明

|Model Name|Training Method|Context Length|Quantization|ILUVATAR-GPUs Required|Deployment Commands|Applicable Version|
|-|-|-|-|-|-|-|
|ERNIE-4.5-21B-A3B-PT|SFT-Full|8K|BF16|8卡16芯|bash examples/config/iluvatar/ERNIE-4.5-21B-A3B-PT/sft/run_full_8k.sh|1.0|
|ERNIE-4.5-21B-A3B-PT|SFT-LoRA|8K|BF16|4卡8芯|bash examples/config/iluvatar/ERNIE-4.5-21B-A3B-PT/sft/run_lora_8k.sh|1.0|
|ERNIE-4.5-0.3B-PT|SFT-Full|8K|BF16|单卡单芯|bash examples/config/iluvatar/ERNIE-4.5-0.3B-PT/sft/run_full_8k.sh|1.0|
|ERNIE-4.5-0.3B-PT|SFT-LoRA|8K|BF16|单卡单芯|bash examples/config/iluvatar/ERNIE-4.5-0.3B-PT/sft/run_lora_8k.sh|1.0|
|PaddleOCR-VL-0.9B|SFT-Full|16K|BF16|单卡单芯|bash bash examples/config/iluvatar/PaddleOCR-VL/sft/run_paddleocr-vl_full_16k.sh|1.0|
|PaddleOCR-VL-0.9B|SFT-LoRA|16K|BF16|单卡单芯|bash bash examples/config/iluvatar/PaddleOCR-VL/sft/run_paddleocr-vl_lora_16k.sh|1.0|
