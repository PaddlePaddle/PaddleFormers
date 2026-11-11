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
"""Tests for quantization/checkpoint_dequant.py and quantization/hf_checkpoint.py."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import paddle
from paddle.distributed.flex_checkpoint.dcp.metadata import (
    LocalTensorMetadata,
    Metadata,
)

import paddleformers.quantization.checkpoint_dequant as checkpoint_dequant
from paddleformers.quantization.checkpoint_dequant import (
    FP8BlockCheckpointDequantizer,
    MXFP4GroupCheckpointDequantizer,
    get_checkpoint_dequantizer,
    register_checkpoint_dequantizer,
)
from paddleformers.quantization.hf_checkpoint import (
    HFDequantLoadTransform,
    HFQuantizationGroupSpec,
    HFQuantizedWeightSpec,
    QuanDescriptor,
    QuanMetadata,
    build_hf_dequant_load_transform,
    hf_checkpoint_is_quantized,
)

# Physical metadata file DCP looks for in an HF checkpoint directory.
PADDLE_METADATA_FILE_NAME = "flex-ckpt.auto_generated.metadata"
HF_CONFIG_FILE_NAME = "config.json"
SAFETENSORS_FILE_NAME = "model-00001-of-00001.safetensors"
# Byte width of every safetensors storage format used below.
SAFETENSORS_ITEM_SIZE = {"F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BF16": 2}

WEIGHT_SUFFIX = ".weight"
SCALE_SUFFIX = ".scale"
FP8_WEIGHT = "layers.0.attn.wq_a.weight"
FP8_SCALE = "layers.0.attn.wq_a.scale"
MXFP4_WEIGHT = "layers.0.mlp.experts.0.w1.weight"
MXFP4_SCALE = "layers.0.mlp.experts.0.w1.scale"
UNQUANTIZED_WEIGHT = "layers.0.norm.weight"
# ue8m0 exponent codes: 127 is 2**0, so 125..129 cover 0.25 .. 4.0.
UE8M0_ONE = 127
# e4m3 code for 1.0 (exponent bias 7, zero mantissa).
E4M3_ONE = 0x38


def fp8_group(**overrides):
    """Return a fresh fp8_block descriptor group so tests never share mutable state."""
    group = {
        "name": "fp8",
        "targets": [r"re:.*\.attn\.wq_a\.weight$"],
        "quant_method": "fp8_block",
        "value_format": "e4m3",
        "scale_format": "ue8m0",
        "block_shape": [2, 2],
    }
    group.update(overrides)
    return group


def mxfp4_group(**overrides):
    """Return a fresh mxfp4_group descriptor group."""
    group = {
        "name": "mxfp4",
        "targets": [r"re:.*\.experts\.[0-9]+\.w[123]\.weight$"],
        "quant_method": "mxfp4_group",
        "value_format": "e2m1",
        "scale_format": "ue8m0",
        "block_shape": [2],
    }
    group.update(overrides)
    return group


def descriptor_dict(*group_dicts, **overrides):
    """Build a descriptor payload; defaults to a single fp8_block group."""
    payload = {
        "schema_version": 1,
        "component_pairing": {"weight_suffix": WEIGHT_SUFFIX, "scale_suffix": SCALE_SUFFIX},
        "logic_name_suffix": WEIGHT_SUFFIX,
        "groups": list(group_dicts) or [fp8_group()],
    }
    payload.update(overrides)
    return payload


def physical_metadata(entries):
    """Build Metadata shaped like what create_hf_ckpt_metadata() returns for an HF checkpoint."""
    return Metadata(
        state_dict_metadata={
            key: [
                LocalTensorMetadata(
                    global_offset=(0,) * len(shape),
                    local_shape=tuple(shape),
                    global_shape=tuple(shape),
                    dtype=dtype,
                )
            ]
            for key, (shape, dtype) in entries.items()
        },
        storage_metadata={},
    )


def fp8_physical_metadata(weight_shape=(4, 4), scale_shape=(2, 2)):
    """Physical metadata for one fp8_block weight/scale pair."""
    return physical_metadata({FP8_WEIGHT: (weight_shape, "uint8"), FP8_SCALE: (scale_shape, "uint8")})


def fp8_group_spec(block_axes=(0, 1), block_shape=(2, 2)):
    """Build a group spec directly, bypassing descriptor parsing."""
    spec = HFQuantizationGroupSpec(
        name="fp8",
        targets=(),
        quant_method="fp8_block",
        value_format="e4m3",
        scale_format="ue8m0",
        block_shape=tuple(block_shape),
        dequantizer=get_checkpoint_dequantizer("fp8_block").configure_formats("e4m3", "ue8m0"),
    )
    if block_axes is not None:
        spec.configure_geometry(block_axes)
    return spec


def fp8_weight_spec(logical_shape=(4, 4), components=None, group_name="fp8", logical_name=FP8_WEIGHT):
    """Build a weight relation directly, bypassing descriptor parsing."""
    return HFQuantizedWeightSpec(
        group_name=group_name,
        logical_name=logical_name,
        logical_shape=tuple(logical_shape),
        components={"qweight": FP8_WEIGHT, "scale": FP8_SCALE} if components is None else components,
    )


def write_safetensors(path, tensors):
    """Write a safetensors file whose payload is zeroed; only the header is read here."""
    header = {}
    offset = 0
    for key, storage_format, shape in tensors:
        nbytes = SAFETENSORS_ITEM_SIZE[storage_format]
        for dim in shape:
            nbytes *= dim
        header[key] = {
            "dtype": storage_format,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    with open(path, "wb") as file:
        file.write(len(raw_header).to_bytes(8, byteorder="little"))
        file.write(raw_header)
        file.write(bytes(offset))


def configured_dequantizer(method, value_format, scale_format, block_axes, block_shape):
    """Fully configure a registered dequantizer prototype."""
    return (
        get_checkpoint_dequantizer(method)
        .configure_formats(value_format, scale_format)
        .configure_geometry(block_axes, block_shape)
    )


def target_shard(global_shape, local_shape, global_offset, dtype="bfloat16"):
    """Describe the logical shard a rank wants to load."""
    return LocalTensorMetadata(
        global_offset=tuple(global_offset),
        local_shape=tuple(local_shape),
        global_shape=tuple(global_shape),
        dtype=dtype,
    )


class CPUDequantTestCase(unittest.TestCase):
    """Pin tensor-producing tests to CPU so results never depend on the CI device."""

    def setUp(self):
        self._original_device = paddle.get_device()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.set_device(self._original_device)


class TestRawFormatDecoding(CPUDequantTestCase):
    """Raw 8-bit safetensors codes are reinterpreted by checkpoint_dequant, not by paddle."""

    def test_e4m3_decodes_zero_subnormal_normal_max_and_nan(self):
        """0x00/0x01/0x38/0x7E/0xFE/0x7F cover every interesting e4m3 region."""
        qweight = paddle.to_tensor([[0x00, 0x01, E4M3_ONE, 0x7E, 0xFE, 0x7F]], dtype="uint8")
        scale = paddle.to_tensor([[UE8M0_ONE]], dtype="uint8")
        dequantizer = configured_dequantizer("fp8_block", "e4m3", "ue8m0", (0, 1), (1, 6))

        output = dequantizer.dequantize({"qweight": qweight, "scale": scale}, paddle.float32).numpy()

        np.testing.assert_array_equal(output[0, :5], [0.0, 2.0**-9, 1.0, 448.0, -448.0])
        self.assertTrue(np.isnan(output[0, 5]))

    def test_ue8m0_scale_codes_decode_to_powers_of_two(self):
        """Every scale element is a biased exponent, so codes 125..129 give 0.25 .. 4.0."""
        qweight = paddle.full([1, 5], E4M3_ONE, dtype="uint8")
        scale = paddle.to_tensor([[125, 126, 127, 128, 129]], dtype="uint8")
        dequantizer = configured_dequantizer("fp8_block", "e4m3", "ue8m0", (0, 1), (1, 1))

        output = dequantizer.dequantize({"qweight": qweight, "scale": scale}, paddle.float32)

        np.testing.assert_array_equal(output.numpy(), [[0.25, 0.5, 1.0, 2.0, 4.0]])

    def test_int8_storage_is_reinterpreted_as_unsigned_codes(self):
        """Safetensors I8 tensors deliver codes >= 128 as negative int8 values."""
        qweight = paddle.to_tensor([[E4M3_ONE, -2]], dtype="int8")  # -2 is the byte 0xFE
        scale = paddle.full([1, 2], UE8M0_ONE, dtype="uint8")
        dequantizer = configured_dequantizer("fp8_block", "e4m3", "ue8m0", (0, 1), (1, 1))

        output = dequantizer.dequantize({"qweight": qweight, "scale": scale}, paddle.float32)

        np.testing.assert_array_equal(output.numpy(), [[1.0, -448.0]])

    def test_e2m1_unpacks_the_low_nibble_before_the_high_nibble(self):
        """0x21 holds codes (1, 2); 0xA9 holds codes (9, 10), i.e. the negative half."""
        qweight = paddle.to_tensor([[0x21, -87]], dtype="int8")  # -87 is the byte 0xA9
        scale = paddle.to_tensor([[127, 128]], dtype="uint8")
        dequantizer = configured_dequantizer("mxfp4_group", "e2m1", "ue8m0", (1,), (2,))

        output = dequantizer.dequantize({"qweight": qweight, "scale": scale}, paddle.float32)

        np.testing.assert_array_equal(output.numpy(), [[0.5, 1.0, -1.0, -2.0]])

    def test_non_8bit_qweight_storage_is_rejected(self):
        """A float qweight means the safetensors reader lost the raw bytes."""
        dequantizer = configured_dequantizer("fp8_block", "e4m3", "ue8m0", (0, 1), (1, 1))

        with self.assertRaisesRegex(TypeError, "raw uint8/int8 storage"):
            dequantizer.dequantize(
                {
                    "qweight": paddle.zeros([1, 1], dtype="float32"),
                    "scale": paddle.full([1, 1], UE8M0_ONE, dtype="uint8"),
                },
                paddle.float32,
            )


class TestFP8BlockDequantizer(CPUDequantTestCase):
    """FP8 block dequantization math, geometry contracts and configuration immutability."""

    def setUp(self):
        super().setUp()
        self.dequantizer = configured_dequantizer("fp8_block", "e4m3", "ue8m0", (0, 1), (2, 2))

    def test_block_scales_are_broadcast_across_each_block(self):
        """A (1, 2) scale grid covers a 2x4 weight in two 2x2 blocks."""
        qweight = paddle.full([2, 4], E4M3_ONE, dtype="uint8")
        scale = paddle.to_tensor([[127, 128]], dtype="uint8")

        output = self.dequantizer.dequantize({"qweight": qweight, "scale": scale}, paddle.float32)

        np.testing.assert_array_equal(output.numpy(), [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]])

    def test_partial_trailing_block_is_cropped(self):
        """A 3x3 weight needs a 2x2 grid; the expanded grid is cropped back to 3x3."""
        qweight = paddle.full([3, 3], E4M3_ONE, dtype="uint8")
        scale = paddle.to_tensor([[127, 128], [129, 130]], dtype="uint8")

        output = self.dequantizer.dequantize({"qweight": qweight, "scale": scale}, paddle.float32)

        np.testing.assert_array_equal(output.numpy(), [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [4.0, 4.0, 8.0]])

    def test_output_dtype_follows_the_request(self):
        qweight = paddle.full([2, 2], E4M3_ONE, dtype="uint8")
        scale = paddle.full([1, 1], UE8M0_ONE, dtype="uint8")

        output = self.dequantizer.dequantize({"qweight": qweight, "scale": scale}, paddle.bfloat16)

        self.assertEqual(output.dtype, paddle.bfloat16)

    def test_scale_grid_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid scale grid shape"):
            self.dequantizer.dequantize(
                {
                    "qweight": paddle.full([2, 4], E4M3_ONE, dtype="uint8"),
                    "scale": paddle.full([1, 1], UE8M0_ONE, dtype="uint8"),
                },
                paddle.float32,
            )

    def test_scale_rank_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Scale rank must match"):
            self.dequantizer.dequantize(
                {
                    "qweight": paddle.full([2, 4], E4M3_ONE, dtype="uint8"),
                    "scale": paddle.full([2], UE8M0_ONE, dtype="uint8"),
                },
                paddle.float32,
            )

    def test_missing_scale_component_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "Missing checkpoint quantization components"):
            self.dequantizer.dequantize({"qweight": paddle.full([2, 2], E4M3_ONE, dtype="uint8")}, paddle.float32)

    def test_dequantize_requires_block_geometry(self):
        """Formats alone are not enough; block geometry is inferred from the checkpoint."""
        without_geometry = get_checkpoint_dequantizer("fp8_block").configure_formats("e4m3", "ue8m0")

        with self.assertRaisesRegex(ValueError, "requires block geometry"):
            without_geometry.dequantize(
                {
                    "qweight": paddle.full([2, 2], E4M3_ONE, dtype="uint8"),
                    "scale": paddle.full([1, 1], UE8M0_ONE, dtype="uint8"),
                },
                paddle.float32,
            )

    def test_configure_formats_rejects_unsupported_formats(self):
        prototype = get_checkpoint_dequantizer("fp8_block")

        with self.assertRaisesRegex(ValueError, "value_format"):
            prototype.configure_formats("e2m1", "ue8m0")
        with self.assertRaisesRegex(ValueError, "scale_format"):
            prototype.configure_formats("e4m3", "int8")

    def test_configuring_never_mutates_the_registered_prototype(self):
        """The registry hands out shared singletons, so configuration must copy."""
        prototype = get_checkpoint_dequantizer("fp8_block")

        configured = prototype.configure_formats("E4M3 ", " UE8M0").configure_geometry((0, 1), (2, 2))

        self.assertIsNot(configured, prototype)
        self.assertIsNone(prototype.value_format)
        self.assertIsNone(prototype.block_axes)
        self.assertEqual((configured.value_format, configured.scale_format), ("e4m3", "ue8m0"))
        self.assertEqual((configured.block_axes, configured.block_shape), ((0, 1), (2, 2)))

    def test_invalid_block_geometry_is_rejected(self):
        prototype = get_checkpoint_dequantizer("fp8_block").configure_formats("e4m3", "ue8m0")
        cases = [
            ((), ()),  # no axes at all
            ((0,), (2, 2)),  # axis count and block count disagree
            ((0, 0), (2, 2)),  # duplicated axis
            ((-1,), (2,)),  # negative axis
            ((0,), (0,)),  # non-positive block size
        ]

        for block_axes, block_shape in cases:
            with self.subTest(block_axes=block_axes, block_shape=block_shape):
                with self.assertRaisesRegex(ValueError, "Invalid block geometry"):
                    prototype.configure_geometry(block_axes, block_shape)

    def test_shard_alignment_allows_block_edges_and_tensor_bounds(self):
        """Local reads are only safe when both shard bounds land on a block or tensor edge."""
        cases = [
            ((0, 0), (2, 4), True),
            ((2, 0), (2, 4), True),
            ((2, 2), (2, 2), True),
            ((0, 0), (4, 4), True),
            ((0, 0), (3, 4), False),  # end 3 is neither a block edge nor the bound
            ((1, 0), (2, 4), False),  # unaligned start
            ((0, 0), (4, 3), False),  # unaligned end on the last axis
        ]

        for global_offset, local_shape, expected in cases:
            with self.subTest(global_offset=global_offset, local_shape=local_shape):
                self.assertEqual(
                    self.dequantizer.logical_shard_is_aligned((4, 4), local_shape, global_offset),
                    expected,
                )

    def test_logical_and_physical_shapes_are_identical(self):
        """FP8 stores one byte per logical value, unlike the packed MXFP4 layout."""
        self.assertEqual(self.dequantizer.logical_shape((4, 4)), (4, 4))
        self.assertEqual(self.dequantizer.physical_qweight_shape((4, 4)), (4, 4))
        self.assertEqual(self.dequantizer.physical_qweight_slice((2, 0), (2, 4)), ((2, 0), (2, 4)))


class TestMXFP4GroupDequantizer(CPUDequantTestCase):
    """MXFP4 packs two logical values per stored byte, which shifts every shape."""

    def setUp(self):
        super().setUp()
        self.dequantizer = configured_dequantizer("mxfp4_group", "e2m1", "ue8m0", (1,), (2,))

    def test_logical_shape_doubles_the_packed_axis(self):
        self.assertEqual(self.dequantizer.logical_shape((2, 3)), (2, 6))

        with self.assertRaisesRegex(ValueError, "Invalid MXFP4 weight shape"):
            self.dequantizer.logical_shape(())

    def test_physical_qweight_shape_halves_the_packed_axis(self):
        self.assertEqual(self.dequantizer.physical_qweight_shape((2, 6)), (2, 3))

        with self.assertRaisesRegex(ValueError, "Invalid MXFP4 logical weight shape"):
            self.dequantizer.physical_qweight_shape((2, 5))

    def test_physical_qweight_slice_halves_the_packed_axis(self):
        self.assertEqual(self.dequantizer.physical_qweight_slice((0, 4), (2, 4)), ((0, 2), (2, 2)))

        with self.assertRaisesRegex(ValueError, "even last-axis offsets and sizes"):
            self.dequantizer.physical_qweight_slice((0, 1), (2, 4))

    def test_shard_alignment_also_requires_even_packed_bounds(self):
        self.assertTrue(self.dequantizer.logical_shard_is_aligned((2, 8), (2, 4), (0, 0)))
        self.assertFalse(self.dequantizer.logical_shard_is_aligned((2, 8), (2, 3), (0, 0)))

    def test_group_scales_are_applied_after_unpacking(self):
        """Each stored byte yields two logical values, and every group of 2 shares a scale."""
        qweight = paddle.to_tensor([[0x21, 0x43], [0x21, 0x43]], dtype="uint8")
        scale = paddle.to_tensor([[127, 128], [128, 127]], dtype="uint8")

        output = self.dequantizer.dequantize({"qweight": qweight, "scale": scale}, paddle.float32)

        np.testing.assert_array_equal(output.numpy(), [[0.5, 1.0, 3.0, 4.0], [1.0, 2.0, 1.5, 2.0]])


class TestDequantizerRegistry(unittest.TestCase):
    """Method lookup and registration contracts."""

    def test_lookup_returns_the_prototype_of_each_registered_method(self):
        self.assertIsInstance(get_checkpoint_dequantizer("fp8_block"), FP8BlockCheckpointDequantizer)
        self.assertIsInstance(get_checkpoint_dequantizer("mxfp4_group"), MXFP4GroupCheckpointDequantizer)

    def test_lookup_normalizes_the_method_name(self):
        self.assertIs(get_checkpoint_dequantizer("  FP8_Block "), get_checkpoint_dequantizer("fp8_block"))

    def test_unknown_method_reports_the_registered_methods(self):
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint quantization method 'int4_block'"):
            get_checkpoint_dequantizer("int4_block")

    def test_invalid_method_names_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be a string"):
            get_checkpoint_dequantizer(None)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            get_checkpoint_dequantizer("   ")

    def test_registration_makes_a_method_available_under_its_normalized_name(self):
        dequantizer = FP8BlockCheckpointDequantizer()
        # There is no public unregister hook, so restore the global registry explicitly.
        self.addCleanup(checkpoint_dequant._CHECKPOINT_DEQUANTIZERS.pop, "fp8_block_probe", None)

        register_checkpoint_dequantizer("  FP8_Block_Probe ", dequantizer)

        self.assertIs(get_checkpoint_dequantizer("fp8_block_probe"), dequantizer)

    def test_duplicate_registration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "is already registered"):
            register_checkpoint_dequantizer("fp8_block", FP8BlockCheckpointDequantizer())

    def test_object_without_a_dequantize_method_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "callable dequantize"):
            register_checkpoint_dequantizer("not_a_dequantizer", object())

        self.assertNotIn("not_a_dequantizer", checkpoint_dequant._CHECKPOINT_DEQUANTIZERS)


class TestQuanDescriptorValidation(unittest.TestCase):
    """Descriptor payloads are validated before any checkpoint is opened."""

    def test_valid_descriptor_round_trips(self):
        payload = descriptor_dict()

        self.assertEqual(QuanDescriptor.from_dict(payload).to_dict(), payload)

    def test_targets_without_the_re_prefix_are_matched_literally(self):
        descriptor = QuanDescriptor.from_dict(descriptor_dict(fp8_group(targets=[FP8_WEIGHT])))

        metadata = descriptor.build_metadata(fp8_physical_metadata())

        self.assertEqual(set(metadata.relations), {FP8_WEIGHT})

    def test_literal_targets_do_not_match_longer_names(self):
        """A literal target names one exact tensor; ``re:`` stays the way to cover a family."""
        cases = [
            (
                "extra prefix on the checkpoint key",
                FP8_WEIGHT,
                {"prefix." + FP8_WEIGHT: ((4, 4), "uint8"), "prefix." + FP8_SCALE: ((2, 2), "uint8")},
            ),
            (
                "layer indices sharing a numeric prefix",
                "layers.1",
                {"layers.11.attn.wq_a.weight": ((4, 4), "uint8"), "layers.11.attn.wq_a.scale": ((2, 2), "uint8")},
            ),
            (
                "target stopping short of the weight suffix",
                FP8_WEIGHT[: -len(WEIGHT_SUFFIX)],
                {FP8_WEIGHT: ((4, 4), "uint8"), FP8_SCALE: ((2, 2), "uint8")},
            ),
        ]

        for name, target, entries in cases:
            with self.subTest(name):
                descriptor = QuanDescriptor.from_dict(descriptor_dict(fp8_group(targets=[target])))

                with self.assertRaisesRegex(ValueError, "did not match all quantized weight/scale pairs"):
                    descriptor.build_metadata(physical_metadata(entries))

    def test_invalid_descriptors_are_rejected(self):
        cases = [
            ("unsupported schema version", descriptor_dict(schema_version=2), "schema_version"),
            ("empty groups", descriptor_dict(groups=[]), "non-empty groups list"),
            ("missing component_pairing", descriptor_dict(component_pairing=None), "component_pairing as an object"),
            (
                "non-string suffix",
                descriptor_dict(component_pairing={"weight_suffix": 1, "scale_suffix": SCALE_SUFFIX}),
                "string weight_suffix and scale_suffix",
            ),
            (
                "empty suffix",
                descriptor_dict(component_pairing={"weight_suffix": "", "scale_suffix": SCALE_SUFFIX}),
                "non-empty strings",
            ),
            ("empty logic_name_suffix", descriptor_dict(logic_name_suffix=""), "logic_name_suffix"),
            (
                "duplicate group names",
                descriptor_dict(fp8_group(), fp8_group(targets=[r"re:.*\.attn\.wq_b\.weight$"])),
                "group names must be unique",
            ),
            ("blank group name", descriptor_dict(fp8_group(name=" ")), "non-empty name"),
            ("missing quant_method", descriptor_dict(fp8_group(quant_method="")), "must define quant_method"),
            ("missing value_format", descriptor_dict(fp8_group(value_format="")), "must define value_format"),
            ("missing scale_format", descriptor_dict(fp8_group(scale_format="")), "must define scale_format"),
            ("unknown quant_method", descriptor_dict(fp8_group(quant_method="int4")), "Invalid quan_desc formats"),
            ("unsupported value_format", descriptor_dict(fp8_group(value_format="e2m1")), "Invalid quan_desc formats"),
            ("empty block_shape", descriptor_dict(fp8_group(block_shape=[])), "positive integer block_shape"),
            ("zero block size", descriptor_dict(fp8_group(block_shape=[2, 0])), "positive integer block_shape"),
            ("empty targets", descriptor_dict(fp8_group(targets=[])), "non-empty string targets list"),
            ("broken target regex", descriptor_dict(fp8_group(targets=["re:["])), "Invalid quan_desc target pattern"),
        ]

        for name, payload, message in cases:
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, message):
                    QuanDescriptor.from_dict(payload)


class TestQuanDescriptorMetadata(unittest.TestCase):
    """build_metadata() binds descriptor rules to the tensors actually present in a checkpoint."""

    def build(self, entries, *group_dicts, output_dtype=paddle.bfloat16):
        descriptor = QuanDescriptor.from_dict(descriptor_dict(*group_dicts))
        return descriptor.build_metadata(physical_metadata(entries), output_dtype=output_dtype)

    def test_block_axes_are_inferred_from_the_scale_grid(self):
        """The descriptor only states block sizes; axes come from the stored scale shape."""
        metadata = self.build(
            {
                FP8_WEIGHT: ((2, 4), "uint8"),
                FP8_SCALE: ((1, 2), "uint8"),
                MXFP4_WEIGHT: ((2, 2), "int8"),
                MXFP4_SCALE: ((2, 1), "uint8"),
            },
            fp8_group(),
            mxfp4_group(block_shape=[4]),
        )

        self.assertEqual(set(metadata.relations), {FP8_WEIGHT, MXFP4_WEIGHT})
        self.assertEqual(metadata.groups["fp8"].block_axes, (0, 1))
        self.assertEqual(metadata.groups["mxfp4"].block_axes, (1,))
        self.assertEqual(metadata.relations[FP8_WEIGHT].logical_shape, (2, 4))
        # MXFP4 packs two values per byte, so the logical weight is twice as wide.
        self.assertEqual(metadata.relations[MXFP4_WEIGHT].logical_shape, (2, 4))

    def test_one_group_is_shared_by_every_weight_it_matches(self):
        entries = {
            "layers.0.mlp.experts.0.w1.weight": ((2, 2), "int8"),
            "layers.0.mlp.experts.0.w1.scale": ((2, 1), "uint8"),
            "layers.0.mlp.experts.0.w2.weight": ((2, 2), "int8"),
            "layers.0.mlp.experts.0.w2.scale": ((2, 1), "uint8"),
        }

        metadata = self.build(entries, mxfp4_group(block_shape=[4]))

        w1 = metadata.relations["layers.0.mlp.experts.0.w1.weight"]
        w2 = metadata.relations["layers.0.mlp.experts.0.w2.weight"]
        self.assertEqual((w1.group_name, w2.group_name), ("mxfp4", "mxfp4"))
        self.assertIs(metadata.groups[w1.group_name], metadata.groups[w2.group_name])
        self.assertEqual(
            w1.components,
            {
                "qweight": "layers.0.mlp.experts.0.w1.weight",
                "scale": "layers.0.mlp.experts.0.w1.scale",
            },
        )

    def test_logical_metadata_uses_the_requested_output_dtype(self):
        metadata = self.build(
            {FP8_WEIGHT: ((2, 4), "uint8"), FP8_SCALE: ((1, 2), "uint8")},
            output_dtype=paddle.float16,
        )

        logical = metadata.logical_metadata[FP8_WEIGHT]
        self.assertEqual(logical.dtype, "float16")
        self.assertEqual(tuple(logical.global_shape), (2, 4))
        self.assertEqual(tuple(logical.global_offset), (0, 0))

    def test_unquantized_weights_stay_out_of_the_logical_view(self):
        metadata = self.build(
            {
                FP8_WEIGHT: ((2, 4), "uint8"),
                FP8_SCALE: ((1, 2), "uint8"),
                UNQUANTIZED_WEIGHT: ((4,), "bfloat16"),
            }
        )

        self.assertEqual(set(metadata.logical_metadata), {FP8_WEIGHT})

    def test_weight_without_its_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "paired scale .* is missing"):
            self.build({FP8_WEIGHT: ((2, 4), "uint8")})

    def test_weight_matching_two_groups_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "matches multiple quan_desc groups"):
            self.build(
                {FP8_WEIGHT: ((2, 4), "uint8"), FP8_SCALE: ((1, 2), "uint8")},
                fp8_group(),
                fp8_group(name="fp8_overlapping_rule"),
            )

    def test_partially_covered_checkpoint_is_rejected(self):
        """Silently skipping a quantized pair would load garbage weights."""
        with self.assertRaisesRegex(ValueError, "did not match all quantized"):
            self.build(
                {
                    FP8_WEIGHT: ((2, 4), "uint8"),
                    FP8_SCALE: ((1, 2), "uint8"),
                    "layers.0.attn.wq_b.weight": ((2, 4), "uint8"),
                    "layers.0.attn.wq_b.scale": ((1, 2), "uint8"),
                }
            )

    def test_descriptor_matching_nothing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "matched no quantized weight/scale pairs"):
            self.build({UNQUANTIZED_WEIGHT: ((4,), "bfloat16")})

    def test_scale_grid_that_fits_no_axis_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cannot infer block axes"):
            self.build({FP8_WEIGHT: ((2, 4), "uint8"), FP8_SCALE: ((1, 3), "uint8")})

    def test_empty_checkpoint_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contains no physical tensors"):
            QuanDescriptor.from_dict(descriptor_dict()).build_metadata(physical_metadata({}))


class TestQuanMetadataValidation(unittest.TestCase):
    """QuanMetadata rejects inconsistent relations at construction time."""

    def quan_metadata(self, spec, group=None, relation_key=None):
        group = fp8_group_spec() if group is None else group
        return QuanMetadata(
            groups={group.name: group},
            relations={relation_key or spec.logical_name: spec},
            logical_metadata={},
            physical_metadata=fp8_physical_metadata(),
        )

    def test_relation_pointing_at_an_unknown_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown group"):
            self.quan_metadata(fp8_weight_spec(group_name="missing"))

    def test_relation_key_must_match_the_logical_name(self):
        with self.assertRaisesRegex(ValueError, "does not match logical_name"):
            self.quan_metadata(fp8_weight_spec(), relation_key="layers.0.attn.wq_b.weight")

    def test_invalid_logical_shapes_are_rejected(self):
        for logical_shape in ((), (4, 0), (4, -1)):
            with self.subTest(logical_shape=logical_shape):
                with self.assertRaisesRegex(ValueError, "Invalid logical shape"):
                    self.quan_metadata(fp8_weight_spec(logical_shape=logical_shape))

    def test_relation_without_a_qweight_component_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not define a qweight component"):
            self.quan_metadata(fp8_weight_spec(components={"scale": FP8_SCALE}))

    def test_empty_component_source_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty component role or source key"):
            self.quan_metadata(fp8_weight_spec(components={"qweight": FP8_WEIGHT, "scale": ""}))

    def test_group_without_block_axes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not define block_axes"):
            self.quan_metadata(fp8_weight_spec(), group=fp8_group_spec(block_axes=None))

    def test_block_axes_outside_the_logical_rank_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside logical shape"):
            self.quan_metadata(fp8_weight_spec(logical_shape=(4,)), group=fp8_group_spec(block_axes=(0, 1)))


class TestHFDequantLoadTransform(CPUDequantTestCase):
    """Read planning and dequantization for a 4x4 fp8_block weight with 2x2 blocks."""

    def setUp(self):
        super().setUp()
        self.transform = HFDequantLoadTransform(
            QuanDescriptor.from_dict(descriptor_dict()).build_metadata(fp8_physical_metadata())
        )

    def test_logical_metadata_is_returned_as_a_copy(self):
        logical = self.transform.logical_metadata()
        self.assertEqual(set(logical), {FP8_WEIGHT})

        logical.clear()

        self.assertEqual(set(self.transform.logical_metadata()), {FP8_WEIGHT})

    def test_source_keys_list_the_qweight_before_the_scale(self):
        self.assertEqual(self.transform.source_keys(FP8_WEIGHT), [FP8_WEIGHT, FP8_SCALE])

    def test_unmanaged_logical_key_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "is not managed by this load transform"):
            self.transform.source_keys("layers.0.attn.wq_b.weight")

    def test_block_aligned_shard_reads_local_slices(self):
        plan = self.transform.read_plan(FP8_WEIGHT, target_shard((4, 4), (2, 4), (0, 0)))

        self.assertEqual(plan.mode, "local")
        self.assertEqual(plan.logical_global_shape, (4, 4))
        self.assertEqual(plan.logical_local_shape, (2, 4))
        self.assertEqual(plan.logical_global_offset, (0, 0))
        qweight = plan.source_slices[FP8_WEIGHT]
        self.assertEqual((tuple(qweight.global_offset), tuple(qweight.local_shape)), ((0, 0), (2, 4)))
        # Two logical rows map onto a single row of the 2x2 scale grid.
        scale = plan.source_slices[FP8_SCALE]
        self.assertEqual(tuple(scale.global_shape), (2, 2))
        self.assertEqual((tuple(scale.global_offset), tuple(scale.local_shape)), ((0, 0), (1, 2)))

    def test_offset_shard_maps_onto_the_matching_scale_rows(self):
        plan = self.transform.read_plan(FP8_WEIGHT, target_shard((4, 4), (2, 4), (2, 0)))

        self.assertEqual(plan.mode, "local")
        scale = plan.source_slices[FP8_SCALE]
        self.assertEqual((tuple(scale.global_offset), tuple(scale.local_shape)), ((1, 0), (1, 2)))

    def test_whole_tensor_shard_uses_global_mode(self):
        plan = self.transform.read_plan(FP8_WEIGHT, target_shard((4, 4), (4, 4), (0, 0)))

        self.assertEqual(plan.mode, "global")
        self.assertEqual(tuple(plan.source_slices[FP8_SCALE].local_shape), (2, 2))

    def test_unaligned_shard_falls_back_to_global_mode(self):
        """An unaligned shard has no well-defined scale origin, so the full tensor is read."""
        plan = self.transform.read_plan(FP8_WEIGHT, target_shard((4, 4), (2, 4), (1, 0)))

        self.assertEqual(plan.mode, "global")
        self.assertEqual(plan.logical_local_shape, (4, 4))
        self.assertEqual(plan.logical_global_offset, (0, 0))
        self.assertEqual(tuple(plan.source_slices[FP8_WEIGHT].local_shape), (4, 4))

    def test_force_global_overrides_an_aligned_shard(self):
        plan = self.transform.read_plan(FP8_WEIGHT, target_shard((4, 4), (2, 4), (0, 0)), force_global=True)

        self.assertEqual(plan.mode, "global")

    def test_shard_with_the_wrong_global_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Target shape mismatch"):
            self.transform.read_plan(FP8_WEIGHT, target_shard((8, 4), (2, 4), (0, 0)))

    def test_checkpoint_shape_disagreeing_with_the_descriptor_is_rejected(self):
        transform = HFDequantLoadTransform(
            QuanMetadata(
                groups={"fp8": fp8_group_spec()},
                relations={FP8_WEIGHT: fp8_weight_spec()},
                logical_metadata={FP8_WEIGHT: target_shard((4, 4), (4, 4), (0, 0))},
                physical_metadata=fp8_physical_metadata(scale_shape=(4, 4)),
            )
        )

        with self.assertRaisesRegex(ValueError, "Read plan shape mismatch"):
            transform.read_plan(FP8_WEIGHT, target_shard((4, 4), (4, 4), (0, 0)))

    def test_mxfp4_plan_halves_the_packed_axis(self):
        transform = HFDequantLoadTransform(
            QuanDescriptor.from_dict(descriptor_dict(mxfp4_group())).build_metadata(
                physical_metadata({MXFP4_WEIGHT: ((2, 2), "int8"), MXFP4_SCALE: ((2, 2), "uint8")})
            )
        )

        plan = transform.read_plan(MXFP4_WEIGHT, target_shard((2, 4), (1, 4), (0, 0)))

        self.assertEqual(plan.mode, "local")
        qweight = plan.source_slices[MXFP4_WEIGHT]
        self.assertEqual(tuple(qweight.global_shape), (2, 2))
        self.assertEqual((tuple(qweight.global_offset), tuple(qweight.local_shape)), ((0, 0), (1, 2)))

    def test_apply_dequantizes_the_whole_tensor(self):
        self.transform.read_plan(FP8_WEIGHT, target_shard((4, 4), (4, 4), (0, 0)))

        output = self.transform.apply(
            FP8_WEIGHT,
            {
                FP8_WEIGHT: paddle.full([4, 4], E4M3_ONE, dtype="uint8"),
                FP8_SCALE: paddle.full([2, 2], 128, dtype="uint8"),
            },
            paddle.float32,
        )

        np.testing.assert_array_equal(output.numpy(), np.full((4, 4), 2.0, dtype="float32"))

    def test_apply_dequantizes_a_local_shard(self):
        self.transform.read_plan(FP8_WEIGHT, target_shard((4, 4), (2, 4), (0, 0)))

        output = self.transform.apply(
            FP8_WEIGHT,
            {
                FP8_WEIGHT: paddle.full([2, 4], E4M3_ONE, dtype="uint8"),
                FP8_SCALE: paddle.full([1, 2], 128, dtype="uint8"),
            },
            paddle.float32,
        )

        np.testing.assert_array_equal(output.numpy(), np.full((2, 4), 2.0, dtype="float32"))

    def test_apply_without_a_plan_expects_the_whole_logical_tensor(self):
        """No cached plan means no local read happened, so a shard is a bug."""
        with self.assertRaisesRegex(ValueError, "expected \\(4, 4\\), got \\(2, 4\\)"):
            self.transform.apply(
                FP8_WEIGHT,
                {
                    FP8_WEIGHT: paddle.full([2, 4], E4M3_ONE, dtype="uint8"),
                    FP8_SCALE: paddle.full([1, 2], 128, dtype="uint8"),
                },
                paddle.float32,
            )

    def test_apply_rejects_sources_that_contradict_the_cached_plan(self):
        self.transform.read_plan(FP8_WEIGHT, target_shard((4, 4), (2, 4), (0, 0)))

        with self.assertRaisesRegex(ValueError, "Invalid dequantized shape"):
            self.transform.apply(
                FP8_WEIGHT,
                {
                    FP8_WEIGHT: paddle.full([4, 4], E4M3_ONE, dtype="uint8"),
                    FP8_SCALE: paddle.full([2, 2], 128, dtype="uint8"),
                },
                paddle.float32,
            )


class TestHFCheckpointIsQuantized(unittest.TestCase):
    """Detection of quantized checkpoints from their own config.json."""

    def setUp(self):
        self.checkpoint_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.checkpoint_path, True)

    def write_hf_config(self, config):
        with open(os.path.join(self.checkpoint_path, HF_CONFIG_FILE_NAME), "w", encoding="utf-8") as file:
            json.dump(config, file)

    def test_fp8_block_checkpoint_is_detected(self):
        """The quantization_config a DeepSeek-V4 FP8 release carries."""
        self.write_hf_config(
            {
                "model_type": "deepseek_v4",
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                },
            }
        )
        self.assertTrue(hf_checkpoint_is_quantized(self.checkpoint_path))

    def test_compressed_tensors_checkpoint_is_detected(self):
        """The quantization_config a Kimi-K3 MXFP4 release carries."""
        self.write_hf_config(
            {
                "model_type": "kimi_k3",
                "quantization_config": {
                    "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
                    "format": "pack-quantized",
                    "ignore": ["lm_head"],
                    "quant_method": "compressed-tensors",
                },
            }
        )
        self.assertTrue(hf_checkpoint_is_quantized(self.checkpoint_path))

    def test_unquantized_checkpoint_is_not_detected(self):
        self.write_hf_config({"model_type": "deepseek_v4", "dtype": "bfloat16"})
        self.assertFalse(hf_checkpoint_is_quantized(self.checkpoint_path))

    def test_empty_quantization_config_is_not_detected(self):
        """An empty block states no quantization, so it must not enable the transform."""
        self.write_hf_config({"quantization_config": {}})
        self.assertFalse(hf_checkpoint_is_quantized(self.checkpoint_path))

    def test_non_object_quantization_config_is_not_detected(self):
        self.write_hf_config({"quantization_config": "fp8"})
        self.assertFalse(hf_checkpoint_is_quantized(self.checkpoint_path))

    def test_checkpoint_without_a_config_file_is_not_detected(self):
        self.assertFalse(hf_checkpoint_is_quantized(self.checkpoint_path))


class TestBuildHFDequantLoadTransform(unittest.TestCase):
    """End-to-end construction from a checkpoint directory."""

    def setUp(self):
        self.checkpoint_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.checkpoint_path, True)

    def write_checkpoint(self, tensors=None):
        write_safetensors(
            os.path.join(self.checkpoint_path, SAFETENSORS_FILE_NAME),
            tensors
            if tensors is not None
            else [
                (FP8_WEIGHT, "F8_E4M3", (4, 4)),
                (FP8_SCALE, "F8_E8M0", (2, 2)),
                (UNQUANTIZED_WEIGHT, "BF16", (4,)),
            ],
        )

    def write_paddle_metadata(self, entries=None):
        """State the physical metadata the checkpoint directory carries.

        ``create_hf_ckpt_metadata()`` cannot describe a quantized checkpoint: its
        dtype table has no F8_* entry, so it is the checkpoint producer that has
        to supply this file.  Raw 8-bit formats appear as ``uint8`` because that
        is what leaves their meaning to the transform.
        """
        if entries is None:
            entries = {
                FP8_WEIGHT: ((4, 4), "uint8"),
                FP8_SCALE: ((2, 2), "uint8"),
                UNQUANTIZED_WEIGHT: ((4,), "bfloat16"),
            }
        paddle.save(
            physical_metadata(entries),
            os.path.join(self.checkpoint_path, PADDLE_METADATA_FILE_NAME),
        )

    def test_checkpoint_without_a_model_descriptor_returns_none(self):
        """Quantization rules come from the model definition, never from the checkpoint."""
        self.write_checkpoint()

        self.assertIsNone(build_hf_dequant_load_transform(self.checkpoint_path))

    def test_transform_is_built_from_the_checkpoint_metadata(self):
        self.write_checkpoint()
        self.write_paddle_metadata()

        transform = build_hf_dequant_load_transform(self.checkpoint_path, descriptor_dict())

        self.assertIsInstance(transform, HFDequantLoadTransform)
        logical = transform.logical_metadata()
        self.assertEqual(set(logical), {FP8_WEIGHT})
        self.assertEqual(tuple(logical[FP8_WEIGHT].global_shape), (4, 4))
        self.assertEqual(logical[FP8_WEIGHT].dtype, "bfloat16")
        # Raw 8-bit formats stay uint8 so the transform owns their meaning.
        physical = transform.quan_metadata.physical_metadata.state_dict_metadata
        self.assertEqual(physical[FP8_WEIGHT][0].dtype, "uint8")
        self.assertEqual(physical[FP8_SCALE][0].dtype, "uint8")

    def test_paddle_metadata_is_read_instead_of_the_safetensors_files(self):
        self.write_paddle_metadata()

        # No safetensors file was ever written, so the build can only have come
        # from the metadata file.
        transform = build_hf_dequant_load_transform(self.checkpoint_path, descriptor_dict())

        self.assertEqual(set(transform.logical_metadata()), {FP8_WEIGHT})

    def test_unquantized_checkpoint_with_a_descriptor_is_rejected(self):
        self.write_checkpoint([(UNQUANTIZED_WEIGHT, "BF16", (4,))])
        self.write_paddle_metadata({UNQUANTIZED_WEIGHT: ((4,), "bfloat16")})

        with self.assertRaisesRegex(ValueError, "matched no quantized weight/scale pairs"):
            build_hf_dequant_load_transform(self.checkpoint_path, descriptor_dict())


if __name__ == "__main__":
    unittest.main()
