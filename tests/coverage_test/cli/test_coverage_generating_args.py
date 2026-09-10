# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import unittest

from paddleformers.cli.hparams.generating_args import GeneratingArguments, StreamOptions


class TestStreamOptions(unittest.TestCase):
    """Tests for StreamOptions class."""

    def test_default_values(self):
        """Test StreamOptions default values."""
        opts = StreamOptions()
        self.assertEqual(opts.count, 20)
        self.assertEqual(opts.ranked, "newest")
        self.assertFalse(opts.unreadOnly)
        self.assertIsNone(opts.newerThan)
        self.assertEqual(opts._max_count, 100)
        self.assertIsNone(opts.continuation)

    def test_custom_max_count(self):
        """Test StreamOptions with custom max_count."""
        opts = StreamOptions(max_count=50)
        self.assertEqual(opts._max_count, 50)

    def test_custom_count(self):
        """Test StreamOptions count attribute can be changed."""
        opts = StreamOptions()
        opts.count = 30
        self.assertEqual(opts.count, 30)

    def test_custom_ranked(self):
        """Test StreamOptions ranked attribute can be changed."""
        opts = StreamOptions()
        opts.ranked = "oldest"
        self.assertEqual(opts.ranked, "oldest")

    def test_custom_unread_only(self):
        """Test StreamOptions unreadOnly attribute can be changed."""
        opts = StreamOptions()
        opts.unreadOnly = True
        self.assertTrue(opts.unreadOnly)


class TestGeneratingArguments(unittest.TestCase):
    """Tests for GeneratingArguments dataclass."""

    def test_defaults(self):
        """Test GeneratingArguments default values."""
        args = GeneratingArguments()
        self.assertEqual(args.max_new_tokens, 1024)
        self.assertEqual(args.min_tokens, 0)
        self.assertAlmostEqual(args.temperature, 0.95)
        self.assertAlmostEqual(args.top_p, 0.7)
        self.assertAlmostEqual(args.frequency_penalty, 0.0)
        self.assertAlmostEqual(args.presence_penalty, 0.0)
        self.assertAlmostEqual(args.repetition_penalty, 1.0)

    def test_default_stream(self):
        """Test default stream is True."""
        args = GeneratingArguments()
        self.assertTrue(args.stream)

    def test_default_stream_options_is_none(self):
        """Test default stream_options is None."""
        args = GeneratingArguments()
        self.assertIsNone(args.stream_options)

    def test_default_enable_thinking(self):
        """Test default enable_thinking is False."""
        args = GeneratingArguments()
        self.assertFalse(args.enable_thinking)

    def test_custom_max_new_tokens(self):
        """Test custom max_new_tokens."""
        args = GeneratingArguments(max_new_tokens=2048)
        self.assertEqual(args.max_new_tokens, 2048)

    def test_custom_min_tokens(self):
        """Test custom min_tokens."""
        args = GeneratingArguments(min_tokens=10)
        self.assertEqual(args.min_tokens, 10)

    def test_custom_temperature(self):
        """Test custom temperature."""
        args = GeneratingArguments(temperature=0.5)
        self.assertAlmostEqual(args.temperature, 0.5)

    def test_custom_top_p(self):
        """Test custom top_p."""
        args = GeneratingArguments(top_p=0.9)
        self.assertAlmostEqual(args.top_p, 0.9)

    def test_custom_frequency_penalty(self):
        """Test custom frequency_penalty."""
        args = GeneratingArguments(frequency_penalty=0.5)
        self.assertAlmostEqual(args.frequency_penalty, 0.5)

    def test_custom_presence_penalty(self):
        """Test custom presence_penalty."""
        args = GeneratingArguments(presence_penalty=0.3)
        self.assertAlmostEqual(args.presence_penalty, 0.3)

    def test_custom_repetition_penalty(self):
        """Test custom repetition_penalty."""
        args = GeneratingArguments(repetition_penalty=1.2)
        self.assertAlmostEqual(args.repetition_penalty, 1.2)

    def test_stream_false(self):
        """Test setting stream to False."""
        args = GeneratingArguments(stream=False)
        self.assertFalse(args.stream)

    def test_custom_stream_options(self):
        """Test setting custom stream_options."""
        opts = StreamOptions(max_count=50)
        args = GeneratingArguments(stream_options=opts)
        self.assertIsNotNone(args.stream_options)
        self.assertEqual(args.stream_options._max_count, 50)

    def test_enable_thinking_true(self):
        """Test setting enable_thinking to True."""
        args = GeneratingArguments(enable_thinking=True)
        self.assertTrue(args.enable_thinking)

    def test_multiple_custom_values(self):
        """Test setting multiple custom values at once."""
        args = GeneratingArguments(
            max_new_tokens=512,
            min_tokens=5,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            stream=False,
            enable_thinking=True,
        )
        self.assertEqual(args.max_new_tokens, 512)
        self.assertEqual(args.min_tokens, 5)
        self.assertAlmostEqual(args.temperature, 0.8)
        self.assertAlmostEqual(args.top_p, 0.95)
        self.assertAlmostEqual(args.repetition_penalty, 1.1)
        self.assertFalse(args.stream)
        self.assertTrue(args.enable_thinking)


if __name__ == "__main__":
    unittest.main()
