# MiMo-VL-7B-SFT Conversion Report

## 1. Purpose

This report records the MiMo-VL-7B-SFT checkpoint migration from HuggingFace safetensors to a Paddle-compatible checkpoint.

MiMo-VL-7B-SFT does not introduce a new PaddleFormers model implementation. The checkpoint follows the Qwen2.5-VL architecture and can be loaded by the existing `Qwen2_5_VLForConditionalGeneration` implementation.

Therefore, this migration focuses on checkpoint conversion and validation instead of adding a new `paddleformers/transformers/mimo_vl` modeling directory.

## 2. Conversion Command

```bash
python tests/integration_test/convert_mimo_vl_to_paddle.py \
  --source-dir /path/to/MiMo-VL-7B-SFT \
  --output-dir /path/to/MiMo-VL-7B-SFT-paddle \
  --dtype bfloat16 \
  --load-checkpoint-format legacy \
  --device cpu \
  --verify-paddle-load \
  --overwrite
```

Notes:

- `convert_from_hf=True` is used internally when loading the original HuggingFace checkpoint.
- `load_checkpoint_format="legacy"` is used for stable local loading.
- Tokenizer, processor, generation config, and auxiliary metadata files are copied to the output directory.

## 3. Converted Checkpoint Loading

The converted checkpoint is loaded with:

```python
from paddleformers.transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/path/to/MiMo-VL-7B-SFT-paddle",
    convert_from_hf=True,
    use_safetensors=True,
    load_checkpoint_format="legacy",
    low_cpu_mem_usage=True,
    dtype="bfloat16",
)
```

`convert_from_hf=True` is required because the saved safetensors checkpoint uses HF-style key names.

## 4. Load Validation

Result:

```text
All model checkpoint weights were used when initializing Qwen2_5_VLForConditionalGeneration.
verify_load_pass
```

## 5. Text-Only Smoke Inference

Result:

```text
smoke_pass=True
gen_time_sec=1.682
```

## 6. HF vs Paddle Logits Consistency

The same text input was tokenized once and fed to both the original HuggingFace checkpoint and the converted Paddle checkpoint.

Command:

```bash
CUDA_VISIBLE_DEVICES=7 LOAD_STATE_DICT_THREAD_NUM=8 \
python tests/integration_test/verify_mimo_vl_hf_vs_paddle.py \
  --hf-dir /path/to/MiMo-VL-7B-SFT \
  --paddle-dir /path/to/MiMo-VL-7B-SFT-paddle \
  --device gpu \
  --hf-dtype bfloat16 \
  --paddle-dtype bfloat16 \
  --paddle-load-checkpoint-format legacy \
  --low-cpu-mem-usage \
  --strict
```

Result:

```json
{
  "shape": [1, 8, 151680],
  "max_abs_diff": 0.0005908012390136719,
  "mean_abs_diff": 2.8572936571436003e-05,
  "p99_abs_diff": 0.000240325927734375,
  "allclose": true,
  "atol": 0.001,
  "rtol": 0.001,
  "hf_top1_token_id": 56177,
  "paddle_top1_token_id": 56177,
  "top1_match": true
}
```

## 7. Conclusion

MiMo-VL-7B-SFT is supported as a Qwen2.5-VL checkpoint variant. The converted Paddle checkpoint can be loaded by the existing `Qwen2_5_VLForConditionalGeneration` implementation, passes text-only smoke inference, and is numerically aligned with the original HuggingFace checkpoint under `atol=1e-3, rtol=1e-3`.
