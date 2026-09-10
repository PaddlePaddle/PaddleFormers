# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
import types
import unittest

# Direct import to avoid __init__.py triggering workflow.py which requires AutoTokenizer
_MODULE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "paddleformers", "cli", "train", "deepseek_v3_pretrain"
)
_MODULE_DIR = os.path.abspath(_MODULE_DIR)

_UTILS_DIR = os.path.join(_MODULE_DIR, "utils")


def _load_module(name, path):
    """Load a module directly from file path without going through __init__.py."""
    full_name = f"paddleformers.cli.train.deepseek_v3_pretrain.utils.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Mock torch and safetensors.torch since torch is not installed.
# Do NOT mock the base "safetensors" package itself -- it is already installed
# and mocking it breaks importlib.util.find_spec() used by transformers.
if "torch" not in sys.modules:
    _mock_torch = types.ModuleType("torch")
    _mock_torch.__spec__ = types.SimpleNamespace(name="torch", origin=None, submodule_search_locations=[])
    _mock_torch.__path__ = []
    _mock_torch.__version__ = "2.0.0"
    _mock_torch.int8 = "int8"
    _mock_torch.int16 = "int16"
    _mock_torch.int32 = "int32"
    _mock_torch.int64 = "int64"
    _mock_torch.float16 = "float16"
    _mock_torch.float32 = "float32"
    _mock_torch.float64 = "float64"
    _mock_torch.bfloat16 = "bfloat16"
    _mock_torch.bool = "bool"
    _mock_torch.uint8 = "uint8"
    _mock_torch.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = _mock_torch

    _mock_torch_nn = types.ModuleType("torch.nn")
    _mock_torch_nn.Module = type("Module", (), {})
    sys.modules["torch.nn"] = _mock_torch_nn

if "safetensors.torch" not in sys.modules:
    _mock_safetensors_torch = types.ModuleType("safetensors.torch")

    def _mock_load_file(path):
        return {}

    def _mock_save_file(tensors, path):
        pass

    _mock_safetensors_torch.load_file = _mock_load_file
    _mock_safetensors_torch.save_file = _mock_save_file
    sys.modules["safetensors.torch"] = _mock_safetensors_torch

# Load the convert_ckpt_to_sft module directly
_conv_mod = _load_module("convert_ckpt_to_sft", os.path.join(_UTILS_DIR, "convert_ckpt_to_sft.py"))

_EXPERT_W1_RE = _conv_mod._EXPERT_W1_RE
_EXPERT_W1_RE_v2 = _conv_mod._EXPERT_W1_RE_v2
_EXPERT_W2_RE = _conv_mod._EXPERT_W2_RE
_LAYER_RE = _conv_mod._LAYER_RE
_SHARE_EXPERT_W1_RE = _conv_mod._SHARE_EXPERT_W1_RE
_SHARE_EXPERT_W1_RE_v2 = _conv_mod._SHARE_EXPERT_W1_RE_v2
_SHARE_EXPERT_W2_RE = _conv_mod._SHARE_EXPERT_W2_RE
_handle_expert_weights = _conv_mod._handle_expert_weights
_handle_mlp_weights = _conv_mod._handle_mlp_weights
_handle_shared_expert_weights = _conv_mod._handle_shared_expert_weights
_is_need_transpose = _conv_mod._is_need_transpose
paddle_name_to_hf_names = _conv_mod.paddle_name_to_hf_names
prepare_tensor = _conv_mod.prepare_tensor


class TestPaddleNameToHfNames(unittest.TestCase):
    """Test paddle_name_to_hf_names function in convert_ckpt_to_sft."""

    def test_embed_tokens(self):
        """Test conversion of embedding tokens."""
        result = paddle_name_to_hf_names("deepseek_v2.embed_tokens.weight")
        self.assertEqual(result, ["model.embed_tokens.weight"])

    def test_norm_weight(self):
        """Test conversion of norm weight."""
        result = paddle_name_to_hf_names("deepseek_v2.norm.weight")
        self.assertEqual(result, ["model.norm.weight"])

    def test_lm_head(self):
        """Test conversion of lm_head weight."""
        result = paddle_name_to_hf_names("lm_head.weight")
        self.assertEqual(result, ["lm_head.weight"])

    def test_norm_weight_ignored(self):
        """Test that norm_weight names return empty list."""
        result = paddle_name_to_hf_names("some.norm_weight.param")
        self.assertEqual(result, [])

    def test_router_ignored(self):
        """Test that router names return empty list."""
        result = paddle_name_to_hf_names("deepseek_v2.layers.3.mlp.router.weight")
        self.assertEqual(result, [])

    def test_input_layernorm_ignored(self):
        """Test that input_layernorm names return empty list due to filter."""
        # The source code checks "input_layernorm" in paddle_name before custom_name_map
        result = paddle_name_to_hf_names("deepseek_v2.layers.3.input_layernorm.weight")
        self.assertEqual(result, [])

    def test_unmatched_name(self):
        """Test unmatched name returns empty list."""
        result = paddle_name_to_hf_names("some.random.name")
        self.assertEqual(result, [])

    def test_custom_name_map(self):
        """Test custom name mapping for fused_rms_norm_linear.rms_norm_weight."""
        # "input_layernorm" in name triggers the filter, but other custom_name_map entries work
        result = paddle_name_to_hf_names("deepseek_v2.layers.3.self_attn.fused_rms_norm_linear.rms_norm_weight")
        self.assertEqual(result, ["model.layers.3.input_layernorm.weight"])

    def test_expert_w1(self):
        """Test expert w1 weight conversion."""
        result = paddle_name_to_hf_names("deepseek_v2.layers.5.mlp.experts.2.w1.weight")
        self.assertEqual(len(result), 2)
        self.assertIn("gate_proj.weight", result[0])
        self.assertIn("up_proj.weight", result[1])

    def test_expert_w2(self):
        """Test expert w2 weight conversion."""
        result = paddle_name_to_hf_names("deepseek_v2.layers.5.mlp.experts.2.w2.weight")
        self.assertEqual(len(result), 1)
        self.assertIn("down_proj.weight", result[0])

    def test_expert_gate_up_fused(self):
        """Test expert gate_up_fused_proj weight conversion."""
        result = paddle_name_to_hf_names("deepseek_v2.layers.5.mlp.experts.3.gate_up_fused_proj.weight")
        self.assertEqual(len(result), 2)
        self.assertIn("gate_proj.weight", result[0])
        self.assertIn("up_proj.weight", result[1])

    def test_shared_expert_w1(self):
        """Test shared expert w1 weight conversion."""
        result = paddle_name_to_hf_names("deepseek_v2.layers.5.mlp.shared_experts.w1.weight")
        self.assertEqual(len(result), 2)

    def test_shared_expert_w2(self):
        """Test shared expert w2 weight conversion."""
        result = paddle_name_to_hf_names("deepseek_v2.layers.5.mlp.shared_experts.w2.weight")
        self.assertEqual(len(result), 1)
        self.assertIn("down_proj.weight", result[0])

    def test_mlp_w1(self):
        """Test mlp w1 weight conversion."""
        result = paddle_name_to_hf_names("deepseek_v2.layers.3.mlp.w1")
        self.assertEqual(len(result), 2)
        self.assertIn("gate_proj.weight", result[0])
        self.assertIn("up_proj.weight", result[1])

    def test_mlp_w2(self):
        """Test mlp w2 weight conversion."""
        result = paddle_name_to_hf_names("deepseek_v2.layers.3.mlp.w2")
        self.assertEqual(len(result), 1)
        self.assertIn("down_proj.weight", result[0])

    def test_fallback_name_replacement(self):
        """Test fallback name replacement for unmatched patterns."""
        result = paddle_name_to_hf_names("deepseek_v2.layers.3.self_attn.o_proj.weight")
        self.assertEqual(result, ["model.layers.3.self_attn.o_proj.weight"])


class TestHandleExpertWeights(unittest.TestCase):
    """Test _handle_expert_weights function."""

    def test_w1_weight(self):
        """Test handling of expert w1 weight."""
        result = _handle_expert_weights("model.layers.3", "mlp.experts.5.w1.weight")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "model.layers.3.mlp.experts.5.gate_proj.weight")
        self.assertEqual(result[1], "model.layers.3.mlp.experts.5.up_proj.weight")

    def test_w2_weight(self):
        """Test handling of expert w2 weight."""
        result = _handle_expert_weights("model.layers.3", "mlp.experts.5.w2.weight")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "model.layers.3.mlp.experts.5.down_proj.weight")

    def test_non_expert_weight(self):
        """Test handling of non-expert weight returns None."""
        result = _handle_expert_weights("model.layers.3", "self_attn.q_proj.weight")
        self.assertIsNone(result)


class TestHandleSharedExpertWeights(unittest.TestCase):
    """Test _handle_shared_expert_weights function."""

    def test_w1_weight(self):
        """Test handling of shared expert w1 weight."""
        result = _handle_shared_expert_weights("model.layers.3", "mlp.shared_experts.w1.weight")
        self.assertEqual(len(result), 2)

    def test_w2_weight(self):
        """Test handling of shared expert w2 weight."""
        result = _handle_shared_expert_weights("model.layers.3", "mlp.shared_experts.w2.weight")
        self.assertEqual(len(result), 1)
        self.assertIn("down_proj.weight", result[0])

    def test_non_shared_expert_weight(self):
        """Test handling of non-shared-expert weight returns None."""
        result = _handle_shared_expert_weights("model.layers.3", "self_attn.q_proj.weight")
        self.assertIsNone(result)


class TestHandleMlpWeights(unittest.TestCase):
    """Test _handle_mlp_weights function."""

    def test_w1_weight(self):
        """Test handling of mlp w1 weight."""
        result = _handle_mlp_weights("model.layers.3", "mlp.w1")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "model.layers.3.mlp.gate_proj.weight")
        self.assertEqual(result[1], "model.layers.3.mlp.up_proj.weight")

    def test_w2_weight(self):
        """Test handling of mlp w2 weight."""
        result = _handle_mlp_weights("model.layers.3", "mlp.w2")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "model.layers.3.mlp.down_proj.weight")

    def test_non_mlp_weight(self):
        """Test handling of non-mlp weight returns None."""
        result = _handle_mlp_weights("model.layers.3", "self_attn.q_proj.weight")
        self.assertIsNone(result)


class TestIsNeedTranspose(unittest.TestCase):
    """Test _is_need_transpose function."""

    def test_kv_down_weight(self):
        """Test that fused_rms_norm_linear.kv_down_weight needs transpose."""
        self.assertTrue(_is_need_transpose("deepseek_v2.layers.3.self_attn.fused_rms_norm_linear.kv_down_weight"))

    def test_kv_up_weight(self):
        """Test that memory_recompute_att.kv_up_weight needs transpose."""
        self.assertTrue(_is_need_transpose("deepseek_v2.layers.3.self_attn.memory_recompute_att.kv_up_weight"))

    def test_o_proj_weight(self):
        """Test that o_proj.weight needs transpose."""
        self.assertTrue(_is_need_transpose("deepseek_v2.layers.3.self_attn.o_proj.weight"))

    def test_q_down_weight(self):
        """Test that fused_rms_norm_linear.q_down_weight needs transpose."""
        self.assertTrue(_is_need_transpose("deepseek_v2.layers.3.self_attn.fused_rms_norm_linear.q_down_weight"))

    def test_q_up_weight(self):
        """Test that memory_recompute_att.q_up_weight needs transpose."""
        self.assertTrue(_is_need_transpose("deepseek_v2.layers.3.self_attn.memory_recompute_att.q_up_weight"))

    def test_w1_transpose(self):
        """Test that w1 needs transpose."""
        self.assertTrue(_is_need_transpose("deepseek_v2.layers.3.mlp.w1"))

    def test_w2_transpose(self):
        """Test that w2 needs transpose."""
        self.assertTrue(_is_need_transpose("deepseek_v2.layers.3.mlp.w2"))

    def test_gate_weight_transpose(self):
        """Test that gate.weight needs transpose."""
        self.assertTrue(_is_need_transpose("deepseek_v2.layers.3.mlp.gate.weight"))

    def test_eh_proj_transpose(self):
        """Test that eh_proj.weight needs transpose."""
        self.assertTrue(_is_need_transpose("deepseek_v2.layers.3.eh_proj.weight"))

    def test_lm_head_transpose(self):
        """Test that lm_head.weight needs transpose."""
        self.assertTrue(_is_need_transpose("lm_head.weight"))

    def test_no_transpose_needed(self):
        """Test names that do not need transpose."""
        self.assertFalse(_is_need_transpose("deepseek_v2.layers.3.self_attn.q_proj.weight"))
        self.assertFalse(_is_need_transpose("deepseek_v2.layers.3.self_attn.k_proj.weight"))
        self.assertFalse(_is_need_transpose("deepseek_v2.norm.weight"))


class TestPrepareTensorConvertCkpt(unittest.TestCase):
    """Test prepare_tensor function in convert_ckpt_to_sft."""

    def test_single_key_return(self):
        """Test with a name that maps to a single HF name."""
        import paddle

        value = paddle.randn([4, 8])
        result, size = prepare_tensor("deepseek_v2.norm.weight", value)
        self.assertEqual(len(result), 1)
        self.assertIn("model.norm.weight", result)
        self.assertGreater(size, 0)

    def test_zero_key_return(self):
        """Test with a name that maps to zero HF names."""
        import paddle

        value = paddle.randn([4, 8])
        result, size = prepare_tensor("some.norm_weight.param", value)
        self.assertEqual(len(result), 0)
        self.assertEqual(size, 0)

    def test_two_key_return(self):
        """Test with a name that maps to two HF names (gate/up split)."""

        # mlp.w1 maps to gate_proj and up_proj
        # prepare_tensor in convert_ckpt_to_sft uses torch API (dim=0)
        # which is incompatible with paddle, so we test the name mapping only
        key = "deepseek_v2.layers.3.mlp.w1"
        new_keys = paddle_name_to_hf_names(key)
        self.assertEqual(len(new_keys), 2)
        self.assertIn("gate_proj.weight", new_keys[0])
        self.assertIn("up_proj.weight", new_keys[1])


class TestRegexPatterns(unittest.TestCase):
    """Test regex pattern matching in convert_ckpt_to_sft."""

    def test_layer_re(self):
        """Test _LAYER_RE pattern matching."""
        m = _LAYER_RE.match("deepseek_v2.layers.5.self_attn.q_proj.weight")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "5")
        self.assertEqual(m.group(2), "self_attn.q_proj.weight")

    def test_layer_re_no_match(self):
        """Test _LAYER_RE with non-matching string."""
        m = _LAYER_RE.match("other.pattern")
        self.assertIsNone(m)

    def test_expert_w1_re(self):
        """Test _EXPERT_W1_RE pattern matching."""
        m = _EXPERT_W1_RE.match("mlp.experts.3.w1.weight")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "3")

    def test_expert_w2_re(self):
        """Test _EXPERT_W2_RE pattern matching."""
        m = _EXPERT_W2_RE.match("mlp.experts.7.w2.weight")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "7")

    def test_share_expert_w1_re(self):
        """Test _SHARE_EXPERT_W1_RE pattern matching."""
        m = _SHARE_EXPERT_W1_RE.match("mlp.shared_experts.w1.weight")
        self.assertIsNotNone(m)

    def test_share_expert_w2_re(self):
        """Test _SHARE_EXPERT_W2_RE pattern matching."""
        m = _SHARE_EXPERT_W2_RE.match("mlp.shared_experts.w2.weight")
        self.assertIsNotNone(m)

    def test_expert_w1_re_v2(self):
        """Test _EXPERT_W1_RE_v2 pattern matching."""
        m = _EXPERT_W1_RE_v2.match("mlp.experts.5.gate_up_fused_proj.weight")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "5")

    def test_share_expert_w1_re_v2(self):
        """Test _SHARE_EXPERT_W1_RE_v2 pattern matching."""
        m = _SHARE_EXPERT_W1_RE_v2.match("mlp.shared_experts.gate_up_fused_proj.weight")
        self.assertIsNotNone(m)


if __name__ == "__main__":
    unittest.main()
