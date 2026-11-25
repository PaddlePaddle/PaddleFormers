# MiniCPM upstream configuration fixtures

The `config.json` files are unmodified upstream metadata from:

- `legacy`: `openbmb/MiniCPM4-0.5B` (the existing `MiniCPM` implementation).
- `minicpm4_1`: `openbmb/MiniCPM4.1-8B` (the new `MiniCPM4_1` implementation).

Neither configuration contains `fuse_attention_qkv` or `fuse_attention_ffn`.
The 4.1 checkpoint declares `model_type: minicpm` and `MiniCPMForCausalLM`
in `architectures`, so registration alone cannot distinguish the families.
The legacy checkpoint also covers metadata that omits `model_type` entirely.

`tests/transformers/auto/test_minicpm.py` loads the original configurations
without constructing full-size models. Its offline checkpoint tests retain
that metadata and reduce tensor dimensions to exercise real HF safetensors
loading and FlexCheckpoint conversion without downloads or large allocations.
They check model selection, converted non-square K-projection weights, fusion
field defaults, and finite forward outputs through both `AutoModel` and
`AutoModelForCausalLM`.

The slow tests load the complete pretrained weights for both upstream models.
Use the existing `RUN_SLOW_TEST=True` switch to run them. For local weights,
set `MINICPM_LEGACY_CHECKPOINT` and `MINICPM4_1_CHECKPOINT` to their respective
checkpoint directories; otherwise the upstream model IDs are used.
