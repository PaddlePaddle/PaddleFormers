# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import unittest

from paddleformers.cli.hparams.preprocess_args import (
    BasePreprocessArguments,
    CoarseProcessorArguments,
    End2EndProcessorArguments,
    End2EndProcessorArgumentsHelper,
    ImageModificationProcessorArguments,
    InputIdsMassageArguments,
    UtteranceProcessorArguments,
)


class TestBasePreprocessArguments(unittest.TestCase):
    """Tests for BasePreprocessArguments."""

    def test_post_init(self):
        """Test __post_init__ does not raise."""
        args = BasePreprocessArguments()
        # Should not raise any exception
        args.__post_init__()


class TestUtteranceProcessorArguments(unittest.TestCase):
    """Tests for UtteranceProcessorArguments."""

    def test_defaults(self):
        """Test default values."""
        args = UtteranceProcessorArguments()
        self.assertIsNone(args.tokenizer)
        self.assertIsNone(args.tokenizer_name)

    def test_tokenizer_name_sets_tokenizer(self):
        """Test that tokenizer_name propagates to tokenizer when tokenizer is None."""
        args = UtteranceProcessorArguments(tokenizer_name="/path/to/tokenizer")
        self.assertEqual(args.tokenizer, "/path/to/tokenizer")
        self.assertEqual(args.tokenizer_name, "/path/to/tokenizer")

    def test_tokenizer_takes_precedence(self):
        """Test that explicit tokenizer value is not overridden by tokenizer_name."""
        args = UtteranceProcessorArguments(
            tokenizer="/explicit/path",
            tokenizer_name="/name/path",
        )
        self.assertEqual(args.tokenizer, "/explicit/path")


class TestCoarseProcessorArguments(unittest.TestCase):
    """Tests for CoarseProcessorArguments."""

    def test_defaults(self):
        """Test default values."""
        args = CoarseProcessorArguments()
        self.assertEqual(args.video_fps, 2)
        self.assertEqual(args.video_min_frames, 16)
        self.assertEqual(args.video_max_frames, 480)
        self.assertEqual(args.video_target_frames, -1)
        self.assertEqual(args.video_frames_sample, "middle")

    def test_custom_video_fps(self):
        """Test custom video_fps."""
        args = CoarseProcessorArguments(video_fps=5)
        self.assertEqual(args.video_fps, 5)

    def test_frames_sample_middle(self):
        """Test video_frames_sample 'middle' is accepted."""
        args = CoarseProcessorArguments(video_frames_sample="middle")
        self.assertEqual(args.video_frames_sample, "middle")

    def test_frames_sample_rand(self):
        """Test video_frames_sample 'rand' is accepted."""
        args = CoarseProcessorArguments(video_frames_sample="rand")
        self.assertEqual(args.video_frames_sample, "rand")

    def test_frames_sample_leading(self):
        """Test video_frames_sample 'leading' is accepted."""
        args = CoarseProcessorArguments(video_frames_sample="leading")
        self.assertEqual(args.video_frames_sample, "leading")

    def test_frames_sample_case_insensitive(self):
        """Test video_frames_sample is case-insensitive."""
        args = CoarseProcessorArguments(video_frames_sample="MIDDLE")
        self.assertEqual(args.video_frames_sample, "middle")

    def test_frames_sample_invalid_raises(self):
        """Test invalid video_frames_sample raises AssertionError."""
        with self.assertRaises(AssertionError):
            CoarseProcessorArguments(video_frames_sample="invalid")


class TestInputIdsMassageArguments(unittest.TestCase):
    """Tests for InputIdsMassageArguments."""

    def test_defaults(self):
        """Test default values."""
        args = InputIdsMassageArguments()
        self.assertIsNone(args.corpus_name)
        self.assertEqual(args.im_prefix_length, 64)
        self.assertTrue(args.use_pic_id)
        self.assertEqual(args.prompt_dir, "./")
        self.assertTrue(args.serialize_output)
        self.assertFalse(args.one_sample_in_one_seq)
        self.assertFalse(args.variable_resolution)
        self.assertEqual(args.spatial_conv_size, 2)
        self.assertIsNone(args.adaptive_max_imgtoken_option)
        self.assertIsNone(args.adaptive_max_imgtoken_rate)
        self.assertIsNone(args.max_pixels)
        self.assertIsNone(args.min_pixels)
        self.assertFalse(args.drop_untrainble_sample)
        self.assertEqual(args.chat_template, "ernie_vl")

    def test_custom_values(self):
        """Test custom values."""
        args = InputIdsMassageArguments(
            corpus_name="test_corpus",
            im_prefix_length=128,
            use_pic_id=False,
            prompt_dir="/prompts",
            drop_untrainble_sample=True,
        )
        self.assertEqual(args.corpus_name, "test_corpus")
        self.assertEqual(args.im_prefix_length, 128)
        self.assertFalse(args.use_pic_id)
        self.assertEqual(args.prompt_dir, "/prompts")
        self.assertTrue(args.drop_untrainble_sample)

    def test_adaptive_max_imgtoken_parsing(self):
        """Test adaptive_max_imgtoken_option and rate are parsed correctly."""
        args = InputIdsMassageArguments(
            adaptive_max_imgtoken_option="1,2,3",
            adaptive_max_imgtoken_rate="0.1,0.2,0.3",
        )
        self.assertEqual(args.adaptive_max_imgtoken_option, [1, 2, 3])
        self.assertEqual(args.adaptive_max_imgtoken_rate, [0.1, 0.2, 0.3])

    def test_adaptive_max_imgtoken_none(self):
        """Test adaptive_max_imgtoken_option stays None when only rate is set."""
        args = InputIdsMassageArguments(
            adaptive_max_imgtoken_rate="0.1,0.2",
        )
        # Only one set, so parsing is skipped
        self.assertIsInstance(args.adaptive_max_imgtoken_rate, str)

    def test_video_pixel_defaults(self):
        """Test video pixel defaults."""
        args = InputIdsMassageArguments()
        self.assertIsNone(args.video_max_pixels)
        self.assertIsNone(args.video_min_pixels)


class TestImageModificationProcessorArguments(unittest.TestCase):
    """Tests for ImageModificationProcessorArguments."""

    def test_defaults(self):
        """Test default values."""
        args = ImageModificationProcessorArguments()
        self.assertEqual(args.image_token_len, 64)
        self.assertEqual(args.image_dtype, "uint8")
        self.assertFalse(args.render_timestamp)
        self.assertFalse(args.sft_shift_by_one)

    def test_custom_values(self):
        """Test custom values."""
        args = ImageModificationProcessorArguments(
            image_token_len=128,
            image_dtype="float32",
            render_timestamp=True,
            sft_shift_by_one=True,
        )
        self.assertEqual(args.image_token_len, 128)
        self.assertEqual(args.image_dtype, "float32")
        self.assertTrue(args.render_timestamp)
        self.assertTrue(args.sft_shift_by_one)


class TestEnd2EndProcessorArgumentsHelper(unittest.TestCase):
    """Tests for End2EndProcessorArgumentsHelper."""

    def test_defaults(self):
        """Test default values."""
        args = End2EndProcessorArgumentsHelper()
        self.assertFalse(args.load_args_from_api)

    def test_custom_values(self):
        """Test custom values."""
        args = End2EndProcessorArgumentsHelper(load_args_from_api=True)
        self.assertTrue(args.load_args_from_api)


class TestEnd2EndProcessorArguments(unittest.TestCase):
    """Tests for End2EndProcessorArguments combined class."""

    def test_defaults(self):
        """Test default values from all parent classes."""
        args = End2EndProcessorArguments()
        # From UtteranceProcessorArguments
        self.assertIsNone(args.tokenizer)
        self.assertIsNone(args.tokenizer_name)
        # From CoarseProcessorArguments
        self.assertEqual(args.video_fps, 2)
        self.assertEqual(args.video_frames_sample, "middle")
        # From InputIdsMassageArguments
        self.assertEqual(args.im_prefix_length, 64)
        self.assertTrue(args.use_pic_id)
        # From ImageModificationProcessorArguments
        self.assertEqual(args.image_token_len, 64)
        # From End2EndProcessorArgumentsHelper
        self.assertFalse(args.load_args_from_api)

    def test_tokenizer_name_sets_tokenizer(self):
        """Test tokenizer_name propagation in combined class."""
        args = End2EndProcessorArguments(tokenizer_name="/path/to/tokenizer")
        self.assertEqual(args.tokenizer, "/path/to/tokenizer")


if __name__ == "__main__":
    unittest.main()
