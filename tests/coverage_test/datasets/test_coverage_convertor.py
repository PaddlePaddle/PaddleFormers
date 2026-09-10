# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest

from paddleformers.datasets.reader.convertor import (
    convert_dpo_txt_data,
    convert_mm_data,
    convert_pretraining_data,
    convert_txt_data,
    erniekit_convertor,
    messages_convertor,
)


class TestConvertDpoTxtData(unittest.TestCase):
    """Tests for convert_dpo_txt_data function."""

    def test_basic_dpo_conversion(self):
        data = {
            "src": ["question1", "question2"],
            "tgt": ["answer1"],
            "response": ["good_response", "bad_response"],
            "sort": [1, 0],
        }
        result = convert_dpo_txt_data(data)
        self.assertIn("chosen_response", result)
        self.assertIn("rejected_response", result)
        self.assertIn("messages", result)
        # sort[0]=1 > sort[1]=0, so response[0] is chosen
        self.assertEqual(result["chosen_response"][0]["content"], "good_response")
        self.assertEqual(result["rejected_response"][0]["content"], "bad_response")

    def test_dpo_reversed_sort(self):
        data = {
            "src": ["question1", "question2"],
            "tgt": ["answer1"],
            "response": ["bad_response", "good_response"],
            "sort": [0, 1],
        }
        result = convert_dpo_txt_data(data)
        # sort[1]=1 > sort[0]=0, so response[1] is chosen
        self.assertEqual(result["chosen_response"][0]["content"], "good_response")
        self.assertEqual(result["rejected_response"][0]["content"], "bad_response")

    def test_dpo_string_src_tgt(self):
        data = {
            "src": "question",
            "tgt": [],
            "response": ["good", "bad"],
            "sort": [1, 0],
        }
        result = convert_dpo_txt_data(data)
        # String src should be converted to list, empty list tgt stays empty
        self.assertIsInstance(result["src"], list)
        self.assertIsInstance(result["tgt"], list)

    def test_dpo_string_response(self):
        data = {
            "src": ["question", "followup"],
            "tgt": ["answer"],
            "response": "ab",
            "sort": [1, 0],
        }
        result = convert_dpo_txt_data(data)
        # String response of length 2 gets converted: response[0]="a" (str), response[1]="b" (str)
        # Both are strings so they become [["a"], ["b"]]
        self.assertIsInstance(result["response"], list)
        self.assertIsInstance(result["response"][0], list)

    def test_dpo_multi_turn(self):
        data = {
            "src": ["question1", "question2"],
            "tgt": ["answer1"],
            "response": [["resp1", "query2", "resp2"], ["resp1_bad", "query2_bad", "resp2_bad"]],
            "sort": [1, 0],
        }
        result = convert_dpo_txt_data(data)
        # Should have multi-turn messages
        self.assertGreater(len(result["messages"]), 0)

    def test_dpo_src_tgt_length_mismatch(self):
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1", "a2", "a3"],
            "response": ["good", "bad"],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_dpo_response_length_not_2(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": ["only_one"],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_dpo_sort_length_not_2(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": ["good", "bad"],
            "sort": [1],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_dpo_sort_same_values(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": ["good", "bad"],
            "sort": [1, 1],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_dpo_response_not_list(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": ["good", 123],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_dpo_response_even_count(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": [["r1", "r2"], ["r1_bad", "r2_bad"]],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_dpo_response_empty_string(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": [" ", "bad"],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_dpo_response_empty_list(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": [[], ["bad"]],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_dpo_with_system_is_system_default(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": ["good", "bad"],
            "sort": [1, 0],
        }
        result = convert_dpo_txt_data(data)
        self.assertEqual(result["is_system"], 0)

    def test_dpo_with_system_is_system_1(self):
        data = {
            "src": ["system_prompt", "q", "q2"],
            "tgt": ["a1", "a"],
            "response": ["good", "bad"],
            "sort": [1, 0],
            "is_system": 1,
        }
        result = convert_dpo_txt_data(data)
        self.assertEqual(result["system"], "system_prompt")
        self.assertNotIn("system_prompt", result["src"])

    def test_dpo_with_system_field(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": ["good", "bad"],
            "sort": [1, 0],
            "system": "You are helpful.",
        }
        result = convert_dpo_txt_data(data)
        self.assertIn("messages", result)
        # First message should be system
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][0]["content"], "You are helpful.")

    def test_dpo_system_not_string_raises(self):
        data = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": ["good", "bad"],
            "sort": [1, 0],
            "system": 123,
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_dpo_messages_structure(self):
        data = {
            "src": ["question", "followup"],
            "tgt": ["answer"],
            "response": ["good_response", "bad_response"],
            "sort": [1, 0],
        }
        result = convert_dpo_txt_data(data)
        messages = result["messages"]
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "question")
        chosen = result["chosen_response"]
        self.assertEqual(chosen[0]["role"], "assistant")
        self.assertEqual(chosen[0]["content"], "good_response")
        rejected = result["rejected_response"]
        self.assertEqual(rejected[0]["role"], "assistant")
        self.assertEqual(rejected[0]["content"], "bad_response")

    def test_dpo_multi_turn_chosen_rejected_structure(self):
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": [["a1_good", "q2", "a2_good"], ["a1_bad", "q2", "a2_bad"]],
            "sort": [1, 0],
        }
        result = convert_dpo_txt_data(data)
        chosen = result["chosen_response"]
        # First is assistant (idx 0, even)
        self.assertEqual(chosen[0]["role"], "assistant")
        # Second is user (idx 1, odd)
        self.assertEqual(chosen[1]["role"], "user")
        # Third is assistant (idx 2, even)
        self.assertEqual(chosen[2]["role"], "assistant")


class TestConvertTxtData(unittest.TestCase):
    """Tests for convert_txt_data function."""

    def test_basic_sft_conversion(self):
        item = {
            "src": ["question"],
            "tgt": ["answer"],
        }
        result = convert_txt_data(item)
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][1]["role"], "assistant")

    def test_sft_string_src_tgt(self):
        item = {
            "src": "question",
            "tgt": "answer",
        }
        result = convert_txt_data(item)
        self.assertIsInstance(result["messages"], list)

    def test_sft_multi_turn(self):
        item = {
            "src": ["q1", "q2"],
            "tgt": ["a1", "a2"],
        }
        result = convert_txt_data(item)
        self.assertEqual(len(result["messages"]), 4)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][1]["role"], "assistant")
        self.assertEqual(result["messages"][2]["role"], "user")
        self.assertEqual(result["messages"][3]["role"], "assistant")

    def test_sft_empty_src_raises(self):
        item = {
            "src": [],
            "tgt": ["answer"],
        }
        with self.assertRaises(ValueError):
            convert_txt_data(item)

    def test_sft_empty_tgt_raises(self):
        item = {
            "src": ["question"],
            "tgt": [],
        }
        with self.assertRaises(ValueError):
            convert_txt_data(item)

    def test_sft_empty_string_in_src_raises(self):
        item = {
            "src": [" "],
            "tgt": ["answer"],
        }
        with self.assertRaises(ValueError):
            convert_txt_data(item)

    def test_sft_empty_string_in_tgt_raises(self):
        item = {
            "src": ["question"],
            "tgt": ["  "],
        }
        with self.assertRaises(ValueError):
            convert_txt_data(item)

    def test_sft_label_auto_added(self):
        item = {
            "src": ["q"],
            "tgt": ["a"],
        }
        convert_txt_data(item)
        self.assertEqual(item["label"], [1])

    def test_sft_src_tgt_label_length_mismatch(self):
        item = {
            "src": ["q1", "q2"],
            "tgt": ["a"],
            "label": [1, 1],
        }
        with self.assertRaises(ValueError):
            convert_txt_data(item)

    def test_sft_with_system(self):
        item = {
            "src": ["q"],
            "tgt": ["a"],
            "system": "You are helpful.",
        }
        result = convert_txt_data(item)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][0]["content"], "You are helpful.")

    def test_sft_is_system_1(self):
        item = {
            "src": ["system_q", "q"],
            "tgt": ["system_a", "a"],
            "is_system": 1,
        }
        result = convert_txt_data(item)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][0]["content"], "system_q")

    def test_sft_system_not_string_raises(self):
        item = {
            "src": ["q"],
            "tgt": ["a"],
            "system": 123,
        }
        with self.assertRaises(ValueError):
            convert_txt_data(item)

    def test_sft_empty_system_not_in_messages(self):
        item = {
            "src": ["q"],
            "tgt": ["a"],
            "system": "",
        }
        result = convert_txt_data(item)
        # Empty system should not add a system message
        self.assertTrue(all(m["role"] != "system" for m in result["messages"]))

    def test_sft_with_label_field(self):
        item = {
            "src": ["q"],
            "tgt": ["a"],
            "label": [1],
        }
        result = convert_txt_data(item)
        self.assertEqual(len(result["messages"]), 2)


class TestConvertMmData(unittest.TestCase):
    """Tests for convert_mm_data function."""

    def test_basic_text_only(self):
        item = {
            "text_info": [{"tag": "no_mask", "text": "Hello"}],
        }
        result = convert_mm_data(item)
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "assistant")

    def test_text_with_system(self):
        item = {
            "system": "You are helpful.",
            "text_info": [{"tag": "no_mask", "text": "Hello"}],
        }
        result = convert_mm_data(item)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][0]["content"], "You are helpful.")

    def test_image_single(self):
        item = {
            "text_info": [
                {"tag": "mask", "text": "What is this?"},
                {"tag": "no_mask", "text": "This is a cat"},
            ],
            "image_info": [{"image_url": "http://img.jpg", "matched_text_index": 0}],
        }
        result = convert_mm_data(item)
        self.assertIn("images", result)
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0], "http://img.jpg")

    def test_video_single(self):
        item = {
            "text_info": [
                {"tag": "mask", "text": "What is this?"},
                {"tag": "no_mask", "text": "This is a video"},
            ],
            "video_info": [{"image_url": "http://vid.mp4", "matched_text_index": 0}],
        }
        result = convert_mm_data(item)
        self.assertIn("videos", result)
        self.assertEqual(len(result["videos"]), 1)

    def test_with_tools(self):
        item = {
            "text_info": [{"tag": "no_mask", "text": "Hello"}],
            "tools": [{"name": "calculator"}],
        }
        result = convert_mm_data(item)
        self.assertIn("tools", result)

    def test_image_and_video_with_order(self):
        item = {
            "text_info": [
                {"tag": "no_mask", "text": "Describe this."},
            ],
            "image_info": [{"image_url": "img.jpg", "matched_text_index": 0}],
            "video_info": [{"image_url": "vid.mp4", "matched_text_index": 0}],
            "order": {"type": ["image", "text"], "index": [0, 0]},
        }
        result = convert_mm_data(item)
        self.assertIn("images", result)

    def test_image_and_video_without_order_raises(self):
        item = {
            "text_info": [{"tag": "no_mask", "text": "Describe"}],
            "image_info": [{"image_url": "img.jpg", "matched_text_index": 0}],
            "video_info": [{"image_url": "vid.mp4", "matched_text_index": 0}],
        }
        with self.assertRaises(AssertionError):
            convert_mm_data(item)

    def test_no_mask_with_tool_calls(self):
        item = {
            "text_info": [
                {"tag": "no_mask", "text": "Let me calculate", "tool_calls": '{"fn": "calc"}'},
            ],
        }
        result = convert_mm_data(item)
        # Should have tool_calls in the message
        has_tool_calls = any("tool_calls" in m for m in result["messages"])
        self.assertTrue(has_tool_calls)

    def test_no_mask_with_tool_calls_list(self):
        item = {
            "text_info": [
                {"tag": "no_mask", "text": "Tools", "tool_calls": [{"fn": "calc"}]},
            ],
        }
        result = convert_mm_data(item)
        has_tool_calls = any("tool_calls" in m for m in result["messages"])
        self.assertTrue(has_tool_calls)

    def test_mask_with_tool_response(self):
        item = {
            "text_info": [
                {"tag": "mask", "text": "What is 2+2?"},
                {"tag": "no_mask", "text": "4"},
                {"tag": "mask", "text": "The answer is 4", "tool_response": True},
            ],
        }
        result = convert_mm_data(item)
        # Should have observation role for tool response
        has_observation = any(m["role"] == "observation" for m in result["messages"])
        self.assertTrue(has_observation)

    def test_no_mask_without_tool_calls(self):
        item = {
            "text_info": [
                {"tag": "no_mask", "text": "Hello world"},
            ],
        }
        result = convert_mm_data(item)
        # Should not have tool_calls in the message
        has_tool_calls = any("tool_calls" in m for m in result["messages"])
        self.assertFalse(has_tool_calls)

    def test_empty_text_info(self):
        item = {}
        result = convert_mm_data(item)
        self.assertEqual(result["messages"], [])

    def test_mask_end_tag(self):
        """Test when last text is mask (user observation)."""
        item = {
            "text_info": [
                {"tag": "mask", "text": "Describe this image."},
            ],
            "image_info": [{"image_url": "img.jpg", "matched_text_index": 0}],
        }
        result = convert_mm_data(item)
        # The last message should be from user (mask)
        last_msg = result["messages"][-1]
        self.assertEqual(last_msg["role"], "user")
        self.assertIn("<image>", last_msg["content"])


class TestConvertPretrainingData(unittest.TestCase):
    """Tests for convert_pretraining_data function."""

    def test_basic_pretraining(self):
        data = {"text": "Hello world"}
        result = convert_pretraining_data(data)
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "assistant")
        self.assertEqual(result["messages"][0]["content"], "Hello world")

    def test_pretraining_text_as_list(self):
        data = {"text": ["Hello world"]}
        result = convert_pretraining_data(data)
        self.assertEqual(result["messages"][0]["content"], "Hello world")

    def test_pretraining_empty_text_raises(self):
        data = {"text": "   "}
        with self.assertRaises(ValueError):
            convert_pretraining_data(data)

    def test_pretraining_not_string_raises(self):
        data = {"text": 123}
        with self.assertRaises(AssertionError):
            convert_pretraining_data(data)


class TestErniekitConvertor(unittest.TestCase):
    """Tests for erniekit_convertor function."""

    def test_convert_dpo_data(self):
        item = {
            "src": ["q", "q2"],
            "tgt": ["a"],
            "response": ["good", "bad"],
            "sort": [1, 0],
        }
        result = erniekit_convertor(item)
        self.assertIn("chosen_response", result)
        self.assertIn("rejected_response", result)

    def test_convert_sft_data(self):
        item = {
            "src": ["q"],
            "tgt": ["a"],
        }
        result = erniekit_convertor(item)
        self.assertIn("messages", result)

    def test_convert_pretraining_data(self):
        item = {"text": "Hello"}
        result = erniekit_convertor(item)
        self.assertIn("messages", result)

    def test_convert_mm_data(self):
        item = {
            "text_info": [{"tag": "no_mask", "text": "Hello"}],
        }
        result = erniekit_convertor(item)
        self.assertIn("messages", result)


class TestMessagesConvertor(unittest.TestCase):
    """Tests for messages_convertor function."""

    def test_passthrough(self):
        item = {"messages": [{"role": "user", "content": "Hello"}]}
        result = messages_convertor(item)
        self.assertIs(result, item)

    def test_passthrough_no_messages(self):
        item = {"data": "value"}
        result = messages_convertor(item)
        self.assertIs(result, item)


if __name__ == "__main__":
    unittest.main()
