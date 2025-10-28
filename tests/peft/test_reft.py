# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


import unittest

import paddle

# from llm.utils.data import convert_example_for_reft
from paddleformers.data import DataCollatorForSeq2Seq
from paddleformers.peft.reft import (
    LoreftIntervention,
    LowRankRotateLayer,
    ReFTConfig,
    ReftDataCollator,
    ReFTModel,
    TinyIntervention,
)
from paddleformers.peft.reft.modeling_utils import (
    count_parameters,
    get_type_from_string,
    set_seed,
)
from paddleformers.transformers import AutoModelForCausalLM, AutoTokenizer


class TestReftDataCollator(unittest.TestCase):
    def test_call(self):
        model_name = "Paddleformers/tiny-random-llama"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=512,
            padding_side="right",
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
        )
        reft_data_collator = ReftDataCollator(data_collator)
        instances = [
            {
                "input_ids": [[1, 2, 3], [4, 5, 6]],
                "intervention_locations": [[0, 1, 0], [1, 0, 1]],
            },
            {
                "input_ids": [[7, 8, 9], [10, 11, 12]],
                "intervention_locations": [[1, 0, 1], [0, 1, 0]],
            },
        ]

        batch_inputs = reft_data_collator(instances)

        self.assertIn("input_ids", batch_inputs)
        self.assertIn("intervention_locations", batch_inputs)
        self.assertIsInstance(batch_inputs["input_ids"], paddle.Tensor)
        self.assertIsInstance(batch_inputs["intervention_locations"], paddle.Tensor)


class TestBasicUtils(unittest.TestCase):
    def test_get_type_from_string(self):
        class_str = "paddleformers.peft.reft.LoreftIntervention"
        cls = get_type_from_string(class_str)
        self.assertIsInstance(cls, type(LoreftIntervention))

    def test_set_seed(self):
        set_seed(42)
        set_seed(66)

    def test_count_param(self):
        model = AutoModelForCausalLM.from_pretrained("Paddleformers/tiny-random-llama")
        count_parameters(model)


class TestReftConfig(unittest.TestCase):
    def test_reft_config(self):
        layers = [0, 1, 2]
        representations = [
            {
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": LoreftIntervention(
                    embed_dim=768,
                    low_rank_dimension=4,
                    dropout=0.00,
                    dtype="float32",
                    act_fn="linear",
                    device="gpu",
                    add_bias=False,
                ),
            }
            for l in layers
        ]
        reft_config = ReFTConfig(representations=representations)
        reft_config.__str__()


class TestLoReftIntervention(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "embed_dim": 64,
            "low_rank_dimension": 4,
            "dtype": paddle.float32,
            "dropout": 0.1,
            "act_fn": "linear",
        }

    def test_initialization(self):
        intervention = LoreftIntervention(**self.kwargs)
        self.assertIsInstance(intervention.rotate_layer, LowRankRotateLayer)
        self.assertIsInstance(intervention.learned_source, paddle.nn.Linear)
        self.assertEqual(intervention.dropout.p, self.kwargs["dropout"])

    def test_forward(self):
        base = paddle.randn([10, self.kwargs["embed_dim"]])
        intervention = LoreftIntervention(**self.kwargs)
        output = intervention.forward(base)
        self.assertEqual(output.shape, base.shape)
        self.assertEqual(output.dtype, self.kwargs["dtype"])

    def test_load_state_dict(self):
        model = LoreftIntervention(**self.kwargs)
        state_dict = {
            "learned_source.weight": paddle.randn([64, 4]),
            "learned_source.bias": paddle.zeros([4]),
            "rotate_layer.weight": paddle.randn([64, 4]),
        }
        model.load_state_dict(state_dict)
        self.assertTrue(paddle.allclose(model.learned_source.weight.data, state_dict["learned_source.weight"]))
        self.assertTrue(paddle.allclose(model.learned_source.bias.data, state_dict["learned_source.bias"]))
        self.assertTrue(
            paddle.allclose(
                model.rotate_layer.weight[:, : state_dict["rotate_layer.weight"].shape[-1]],
                state_dict["rotate_layer.weight"],
            )
        )


class TestTinyIntervention(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "embed_dim": 768,
            "low_rank_dimension": 4,
            "dtype": paddle.float32,
            "dropout": 0.1,
            "act_fn": "relu",
        }

    def test_initialization(self):
        intervention = TinyIntervention(**self.kwargs)
        self.assertEqual(intervention.rank, self.kwargs["low_rank_dimension"])
        self.assertEqual(intervention.hidden_size, self.kwargs["embed_dim"])
        self.assertEqual(intervention.param_A.shape, [self.kwargs["embed_dim"], self.kwargs["low_rank_dimension"]])
        self.assertEqual(intervention.param_B.shape, [self.kwargs["low_rank_dimension"], self.kwargs["embed_dim"]])
        self.assertEqual(intervention.param_a.shape, [self.kwargs["low_rank_dimension"]])
        self.assertEqual(intervention.param_b.shape, [self.kwargs["embed_dim"]])

    def test_forward(self):
        base = paddle.randn([10, self.kwargs["embed_dim"]])
        intervention = TinyIntervention(**self.kwargs)
        output = intervention.forward(base)
        self.assertEqual(output.shape, base.shape)
        self.assertEqual(output.dtype, self.kwargs["dtype"])

    def test_load_state_dict(self):
        model = TinyIntervention(**self.kwargs)
        state_dict = {
            "param_A": paddle.randn([768, 4]),
            "param_B": paddle.randn([4, 768]),
            "param_a": paddle.randn([4]),
            "param_b": paddle.randn([768]),
        }
        model.load_state_dict(state_dict)
        self.assertTrue(paddle.allclose(model.param_A, state_dict["param_A"]))
        self.assertTrue(paddle.allclose(model.param_B, state_dict["param_B"]))
        self.assertTrue(paddle.allclose(model.param_a, state_dict["param_a"]))
        self.assertTrue(paddle.allclose(model.param_b, state_dict["param_b"]))


class TestReftModel(unittest.TestCase):
    def test_get_reft_model(self):
        model = AutoModelForCausalLM.from_pretrained("Paddleformers/tiny-random-llama")
        layers = [0]
        representations = [
            {
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": LoreftIntervention(
                    embed_dim=768,
                    low_rank_dimension=4,
                    dropout=0.00,
                    dtype="float32",
                    act_fn="linear",
                    device="gpu",
                    add_bias=False,
                ),
            }
            for l in layers
        ]
        reft_config = ReFTConfig(representations=representations)
        reft_model = ReFTModel(reft_config, model)
        reft_model.print_trainable_parameters()
        self.assertTrue(type(reft_model), ReFTModel)

    def test_reft_model_forward(self):
        model = AutoModelForCausalLM.from_pretrained("Paddleformers/tiny-random-llama")

        layers = [0]
        representations = [
            {
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": LoreftIntervention(
                    embed_dim=768,
                    low_rank_dimension=4,
                    dropout=0.00,
                    dtype="float32",
                    act_fn="linear",
                    device="gpu",
                    add_bias=False,
                ),
            }
            for l in layers
        ]
        reft_config = ReFTConfig(representations=representations)
        reft_model = ReFTModel(reft_config, model)
        reft_model.print_trainable_parameters()
        outputs = reft_model.model(**{"input_ids": paddle.randint(low=1, high=100, shape=(5, 10))})
        self.assertTrue(outputs[0].shape, [5, 10, 32000])


if __name__ == "__main__":
    unittest.main()
