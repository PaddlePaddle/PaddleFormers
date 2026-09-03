# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import unittest

from paddleformers.cli.hparams.data_args import DataArguments


class TestDataArguments(unittest.TestCase):
    """Tests for DataArguments dataclass."""

    def test_default_values(self):
        """Test DataArguments default values."""
        args = DataArguments()
        self.assertEqual(args.dataset_type, "iterable")
        self.assertIsNone(args.input_dir)
        self.assertEqual(args.split, "950,50")
        self.assertIsNone(args.train_dataset_type)
        self.assertIsNone(args.train_dataset_path)
        self.assertIsNone(args.train_dataset_prob)
        self.assertEqual(args.eval_dataset_type, "erniekit")
        self.assertEqual(args.eval_dataset_path, "examples/data/sft-eval.jsonl")
        self.assertEqual(args.eval_dataset_prob, "1.0")

    def test_default_sequence_lengths(self):
        """Test default max_seq_len and max_prompt_len."""
        args = DataArguments()
        self.assertEqual(args.max_seq_len, 4096)
        self.assertEqual(args.max_prompt_len, 2048)

    def test_default_strategy_values(self):
        """Test default data strategy values."""
        args = DataArguments()
        self.assertTrue(args.greedy_intokens)
        self.assertEqual(args.buffer_size, 500)
        self.assertFalse(args.packing)
        self.assertFalse(args.padding_free)
        self.assertEqual(args.mix_strategy, "concat")

    def test_default_template_values(self):
        """Test default template values."""
        args = DataArguments()
        self.assertTrue(args.encode_one_turn)
        self.assertTrue(args.use_template)
        self.assertIsNone(args.template)
        self.assertFalse(args.split_multi_turn)
        self.assertEqual(args.template_backend, "custom")

    def test_custom_dataset_type_map(self):
        """Test setting dataset_type to 'map'."""
        args = DataArguments(dataset_type="map")
        self.assertEqual(args.dataset_type, "map")

    def test_custom_max_seq_len(self):
        """Test setting custom max_seq_len."""
        args = DataArguments(max_seq_len=8192)
        self.assertEqual(args.max_seq_len, 8192)

    def test_custom_packing_and_padding_free(self):
        """Test enabling packing and padding_free."""
        args = DataArguments(packing=True, padding_free=True)
        self.assertTrue(args.packing)
        self.assertTrue(args.padding_free)

    def test_custom_template_backend(self):
        """Test setting template_backend to jinja."""
        args = DataArguments(template_backend="jinja", split_multi_turn=True)
        self.assertEqual(args.template_backend, "jinja")
        self.assertTrue(args.split_multi_turn)

    def test_default_cache_and_warmup(self):
        """Test default caching and warmup values."""
        args = DataArguments()
        self.assertEqual(args.data_impl, "mmap")
        self.assertTrue(args.skip_warmup)
        self.assertFalse(args.warmup_only_rank0)
        self.assertIsNone(args.data_cache)

    def test_default_truncation(self):
        """Test default truncation values."""
        args = DataArguments()
        self.assertEqual(args.truncation_strategy, "delete")
        self.assertTrue(args.truncate_packing)

    def test_default_output_values(self):
        """Test default output-related values."""
        args = DataArguments()
        self.assertEqual(args.dataset_output_dir, "./dataset_output")
        self.assertIsNone(args.new_special_tokens_path)
        self.assertIsNone(args.custom_register_path)
        self.assertFalse(args.make_offline_data)

    def test_default_processor_values(self):
        """Test default processor-related values."""
        args = DataArguments()
        self.assertIsNone(args.processor_use_fast)
        self.assertTrue(args.binpacking)
        self.assertEqual(args.packing_interval, 1000)

    def test_custom_packed_idx_cache_dir(self):
        """Test setting packed_idx_cache_dir."""
        args = DataArguments(packed_idx_cache_dir="/tmp/cache")
        self.assertEqual(args.packed_idx_cache_dir, "/tmp/cache")

    def test_eval_with_do_generation(self):
        """Test eval_with_do_generation default and custom value."""
        args = DataArguments()
        self.assertFalse(args.eval_with_do_generation)
        args = DataArguments(eval_with_do_generation=True)
        self.assertTrue(args.eval_with_do_generation)

    def test_share_folder(self):
        """Test share_folder default and custom value."""
        args = DataArguments()
        self.assertFalse(args.share_folder)
        args = DataArguments(share_folder=True)
        self.assertTrue(args.share_folder)

    def test_random_shuffle(self):
        """Test random_shuffle default and custom value."""
        args = DataArguments()
        self.assertTrue(args.random_shuffle)
        args = DataArguments(random_shuffle=False)
        self.assertFalse(args.random_shuffle)

    def test_num_samples_each_epoch(self):
        """Test num_samples_each_epoch default and custom value."""
        args = DataArguments()
        self.assertEqual(args.num_samples_each_epoch, 6000000)
        args = DataArguments(num_samples_each_epoch=100000)
        self.assertEqual(args.num_samples_each_epoch, 100000)

    def test_dataclass_repr(self):
        """Test DataArguments string representation."""
        args = DataArguments()
        repr_str = repr(args)
        self.assertIn("DataArguments", repr_str)

    def test_multiple_custom_values(self):
        """Test setting multiple custom values at once."""
        args = DataArguments(
            dataset_type="map",
            max_seq_len=2048,
            packing=True,
            padding_free=True,
            template="default",
            mix_strategy="interleave",
        )
        self.assertEqual(args.dataset_type, "map")
        self.assertEqual(args.max_seq_len, 2048)
        self.assertTrue(args.packing)
        self.assertTrue(args.padding_free)
        self.assertEqual(args.template, "default")
        self.assertEqual(args.mix_strategy, "interleave")


if __name__ == "__main__":
    unittest.main()
