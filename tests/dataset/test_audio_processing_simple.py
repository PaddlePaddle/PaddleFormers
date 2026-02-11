#!/usr/bin/env python3

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

"""
Simple script to test audio processing against Swift reference output.

Usage:
    python test_audio_processing_simple.py --model-path <model_path>
"""

import argparse

import numpy as np

from paddleformers.datasets.template.mm_plugin import Qwen2OmniPlugin
from paddleformers.transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="Test audio processing")
    parser.add_argument(
        "--model-path",
        type=str,
        # default="/root/paddlejob/workspace/env_run/chenxuran/customQwen3-Omni-30B-A3B-Instruct",
        required=True,
        help="Path to the Qwen2.5-Omni model (or HuggingFace model ID)",
    )
    parser.add_argument(
        "--audio-input",
        type=str,
        default="share-audio_input.npy",
        help="Path to audio input .npy file",
    )
    parser.add_argument(
        "--expected-output",
        type=str,
        default="swift_audio_res-post_processor.npy",
        help="Path to expected output .npy file from Swift",
    )
    args = parser.parse_args()

    print(f"Loading processor from: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading audio input from: {args.audio_input}")
    audio_input = np.load(args.audio_input)

    # Convert to list format (as expected by the plugin)
    if audio_input.ndim == 1:
        audios = [audio_input.tolist()]
    elif audio_input.ndim == 2:
        audios = [audio.tolist() for audio in audio_input]
    else:
        raise ValueError(f"Unexpected audio shape: {audio_input.shape}")

    print(f"Audio shape: {audio_input.shape}, Number of audios: {len(audios)}")

    # Initialize plugin
    Qwen2OmniPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>", audio_token="<|audio_pad|>")

    # Extract features (exact same as selected code)
    mm_inputs = {}
    mm_inputs.update(
        processor.feature_extractor(
            audios,
            sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pd",
        )
    )

    # Get actual output
    actual_input_features = mm_inputs["input_features"]
    if hasattr(actual_input_features, "numpy"):
        actual_input_features = actual_input_features.numpy()

    actual_input_features = actual_input_features.squeeze()

    print(f"Actual output shape: {actual_input_features.shape}")

    # Load expected output
    print(f"Loading expected output from: {args.expected_output}")
    expected_output = np.load(args.expected_output)
    target_width = expected_output.shape[1]
    print(f"Expected output shape: {expected_output.shape}")

    # 裁切掉定长Tensor的padding部分
    if actual_input_features.shape[1] > target_width:
        actual_input_features = actual_input_features[:, :target_width]
        print(f"✅ Success: Cropped to {actual_input_features.shape}")
    else:
        print(f"ℹ️ No cropping needed: Current shape is {actual_input_features.shape}")

    # Compare
    print("\n" + "=" * 50)
    print("Comparing outputs...")
    print("=" * 50)

    # Check shapes
    if actual_input_features.shape != expected_output.shape:
        print("❌ Shape mismatch!")
        print(f"   Actual:   {actual_input_features.shape}")
        print(f"   Expected: {expected_output.shape}")
        return 1

    # Check values
    max_diff = np.max(np.abs(actual_input_features - expected_output))
    mean_diff = np.mean(np.abs(actual_input_features - expected_output))

    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    # Use a tolerance appropriate for audio features (might be larger due to differences)
    atol = 0
    rtol = 0

    try:
        np.testing.assert_allclose(actual_input_features, expected_output, rtol=rtol, atol=atol)
        print(f"\n✅ PASSED! Outputs match within tolerance (rtol={rtol}, atol={atol})")
        return 0
    except AssertionError as e:
        print(f"\n❌ FAILED! {e}")
        return 1


if __name__ == "__main__":
    exit(main())
