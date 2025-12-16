# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# from __future__ import annotations

# import unittest

# import paddle

# from paddleformers.transformers import (
#     Ernie4_5_MoeConfig,
#     Ernie4_5_MoeForCausalLM,
#     Ernie4_5_MoeModel,
# )
# from tests.testing_utils import require_gpu
# from tests.transformers.test_configuration_common import ConfigTester
# from tests.transformers.test_generation_utils import GenerationTesterMixin
# from tests.transformers.test_modeling_common import (
#     ModelTesterMixin,
#     ids_tensor,
#     random_attention_mask,
# )

# require_at_least_one_gpu = require_gpu(1)


# class Ernie4_5_MoeModelTester:
#     def __init__(
#         self,
#         parent,
#         vocab_size=64,
#         hidden_size=64,
#         intermediate_size=320,
#         max_position_embeddings=32768,
#         num_hidden_layers=3,
#         num_attention_heads=8,
#         num_key_value_heads=2,
#         head_dim=None,
#         hidden_act="silu",
#         initializer_range=0.02,
#         rms_norm_eps=1e-6,
#         use_cache=False,
#         pad_token_id=0,
#         bos_token_id=1,
#         eos_token_id=2,
#         fuse_swiglu=False,
#         use_bias=False,
#         rope_theta=10000,
#         max_sequence_length=8,
#         ignored_index=-100,
#         attention_dropout_prob=0.0,
#         hidden_dropout_prob=0.0,
#         compression_ratio=1.0,
#         micro_batch_size=-1,
#         moe_num_experts=4,
#         use_recompute_moe=False,
#         moe_capacity=[64, 64, 64],
#         moe_norm_min=1e-12,
#         moe_aux_loss_lambda=1e-2,
#         moe_z_loss_lambda=1e-4,
#         moe_orthogonal_loss_lambda=1e-2,
#         sinkhorn_2gate=True,
#         sinkhorn_temp=3e-2,
#         global_aux_loss=False,
#         moe_dropout_prob=0.0,
#         moe_group="dummy",
#         moe_intermediate_size=32,
#         moe_num_shared_experts=1,
#         moe_layer_start_index=1,
#         moe_layer_end_index=-1,
#         moe_layer_interval=1,
#         moe_reverse_token_drop=False,
#         moe_gate_act="softmax",
#         moe_norm_gate_logits=True,
#         moe_all_to_all_dropout=0.0,
#         moe_k=2,
#         moe_use_aux_free=True,
#         moe_group_experts=False,
#         moe_group_orthogonal_loss=True,
#         enable_delay_scale_loss=True,
#         num_acc_steps=1,
#         fuse_gate_detach_matmul=False,
#         moe_use_hard_gate=False,
#         num_nextn_predict_layers=0,
#         multi_token_pred_lambda=0.1,
#         enable_mtp_magic_send=False,
#         use_recompute_mtp=False,
#         is_training=True,
#         batch_size=2,
#         seq_length=10,
#         use_input_mask=True,
#         use_labels=True,
#         return_dict=False,
#         type_sequence_label_size=2,
#         num_labels=3,
#         num_choices=4,
#     ):
#         self.parent: Ernie4_5_MoeModelTest = parent
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.max_position_embeddings = max_position_embeddings
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
#         self.hidden_act = hidden_act
#         self.initializer_range = initializer_range
#         self.rms_norm_eps = rms_norm_eps
#         self.use_cache = use_cache
#         self.pad_token_id = pad_token_id
#         self.bos_token_id = bos_token_id
#         self.eos_token_id = eos_token_id
#         self.fuse_swiglu = fuse_swiglu
#         self.micro_batch_size = micro_batch_size
#         self.max_sequence_length = max_sequence_length
#         self.use_bias = use_bias
#         self.rope_theta = rope_theta
#         self.ignored_index = ignored_index
#         self.attention_dropout_prob = attention_dropout_prob
#         self.hidden_dropout_prob = hidden_dropout_prob
#         self.compression_ratio = compression_ratio
#         self.num_key_value_heads = num_key_value_heads
#         self.moe_num_experts = moe_num_experts
#         self.use_recompute_moe = use_recompute_moe
#         self.moe_capacity = moe_capacity
#         self.moe_norm_min = moe_norm_min
#         self.moe_aux_loss_lambda = moe_aux_loss_lambda
#         self.moe_z_loss_lambda = moe_z_loss_lambda
#         self.moe_orthogonal_loss_lambda = moe_orthogonal_loss_lambda
#         self.global_aux_loss = global_aux_loss
#         self.sinkhorn_2gate = sinkhorn_2gate
#         self.sinkhorn_temp = sinkhorn_temp
#         self.moe_layer_interval = moe_layer_interval
#         self.moe_dropout_prob = moe_dropout_prob
#         self.moe_group = moe_group
#         self.moe_intermediate_size = moe_intermediate_size
#         self.moe_num_shared_experts = moe_num_shared_experts
#         self.moe_layer_start_index = moe_layer_start_index
#         self.moe_layer_end_index = self.num_hidden_layers - 1 if moe_layer_end_index == -1 else moe_layer_end_index
#         self.moe_layer_interval = moe_layer_interval
#         self.moe_reverse_token_drop = moe_reverse_token_drop
#         self.moe_k = moe_k
#         self.moe_all_to_all_dropout = moe_all_to_all_dropout
#         self.moe_group_experts = moe_group_experts
#         self.moe_group_orthogonal_loss = moe_group_orthogonal_loss
#         self.enable_delay_scale_loss = enable_delay_scale_loss
#         self.num_acc_steps = num_acc_steps
#         self.moe_layer_start_index = moe_layer_start_index
#         self.moe_layer_end_index = self.num_hidden_layers - 1 if moe_layer_end_index == -1 else moe_layer_end_index
#         self.moe_gate_act = moe_gate_act
#         self.moe_norm_gate_logits = moe_norm_gate_logits
#         self.moe_use_aux_free = moe_use_aux_free
#         self.fuse_gate_detach_matmul = fuse_gate_detach_matmul
#         self.moe_use_hard_gate = moe_use_hard_gate
#         self.num_nextn_predict_layers = num_nextn_predict_layers
#         self.multi_token_pred_lambda = multi_token_pred_lambda
#         self.enable_mtp_magic_send = enable_mtp_magic_send
#         self.use_recompute_mtp = use_recompute_mtp
#         self.is_training = is_training
#         self.batch_size = batch_size
#         self.seq_length = seq_length
#         self.use_input_mask = use_input_mask
#         self.use_labels = use_labels
#         self.return_dict = return_dict
#         self.type_sequence_label_size = type_sequence_label_size
#         self.num_labels = num_labels
#         self.num_choices = num_choices

#     def prepare_config_and_inputs(self):
#         input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

#         input_mask = None
#         if self.use_input_mask:
#             input_mask = random_attention_mask([self.batch_size, self.seq_length])

#         sequence_labels = None
#         token_labels = None
#         choice_labels = None
#         if self.use_labels:
#             sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
#             token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
#             choice_labels = ids_tensor([self.batch_size], self.num_choices)

#         config = self.get_config()
#         return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

#     def get_config(self) -> Ernie4_5_MoeConfig:
#         return Ernie4_5_MoeConfig(
#             vocab_size=self.vocab_size,
#             hidden_size=self.hidden_size,
#             intermediate_size=self.intermediate_size,
#             max_position_embeddings=self.max_position_embeddings,
#             num_hidden_layers=self.num_hidden_layers,
#             num_attention_heads=self.num_attention_heads,
#             head_dim=self.head_dim,
#             hidden_act=self.hidden_act,
#             initializer_range=self.initializer_range,
#             rms_norm_eps=self.rms_norm_eps,
#             use_cache=self.use_cache,
#             pad_token_id=self.pad_token_id,
#             bos_token_id=self.bos_token_id,
#             eos_token_id=self.eos_token_id,
#             fuse_swiglu=self.fuse_swiglu,
#             micro_batch_size=self.micro_batch_size,
#             max_sequence_length=self.max_sequence_length,
#             use_bias=self.use_bias,
#             rope_theta=self.rope_theta,
#             ignored_index=self.ignored_index,
#             attention_dropout_prob=self.attention_dropout_prob,
#             hidden_dropout_prob=self.hidden_dropout_prob,
#             compression_ratio=self.compression_ratio,
#             num_key_value_heads=self.num_key_value_heads,
#             moe_num_experts=self.moe_num_experts,
#             use_recompute_moe=self.use_recompute_moe,
#             moe_capacity=self.moe_capacity,
#             moe_norm_min=self.moe_norm_min,
#             moe_aux_loss_lambda=self.moe_aux_loss_lambda,
#             moe_z_loss_lambda=self.moe_z_loss_lambda,
#             moe_orthogonal_loss_lambda=self.moe_orthogonal_loss_lambda,
#             global_aux_loss=self.global_aux_loss,
#             sinkhorn_2gate=self.sinkhorn_2gate,
#             sinkhorn_temp=self.sinkhorn_temp,
#             moe_layer_interval=self.moe_layer_interval,
#             moe_dropout_prob=self.moe_dropout_prob,
#             moe_group=self.moe_group,
#             moe_intermediate_size=self.moe_intermediate_size,
#             moe_num_shared_experts=self.moe_num_shared_experts,
#             moe_layer_start_index=self.moe_layer_start_index,
#             moe_layer_end_index=self.num_hidden_layers - 1
#             if self.moe_layer_end_index == -1
#             else self.moe_layer_end_index,
#             moe_reverse_token_drop=self.moe_reverse_token_drop,
#             moe_k=self.moe_k,
#             moe_all_to_all_dropout=self.moe_all_to_all_dropout,
#             moe_group_experts=self.moe_group_experts,
#             moe_group_orthogonal_loss=self.moe_group_orthogonal_loss,
#             enable_delay_scale_loss=self.enable_delay_scale_loss,
#             num_acc_steps=self.num_acc_steps,
#             moe_gate_act=self.moe_gate_act,
#             moe_norm_gate_logits=self.moe_norm_gate_logits,
#             moe_use_aux_free=self.moe_use_aux_free,
#             fuse_gate_detach_matmul=self.fuse_gate_detach_matmul,
#             moe_use_hard_gate=self.moe_use_hard_gate,
#             num_nextn_predict_layers=self.num_nextn_predict_layers,
#             multi_token_pred_lambda=self.multi_token_pred_lambda,
#             enable_mtp_magic_send=self.enable_mtp_magic_send,
#             use_recompute_mtp=self.use_recompute_mtp,
#         )

#     def create_and_check_model(
#         self, config: Ernie4_5_MoeConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
#     ):
#         model = Ernie4_5_MoeModel(config)
#         model.eval()
#         result = model(input_ids)
#         self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

#     def create_and_check_model_attention_mask(self, config: Ernie4_5_MoeConfig, input_ids):
#         model = Ernie4_5_MoeModel(config)
#         model.eval()
#         attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length])
#         result_2d = model(input_ids, attention_mask=attn_mask_2d)[0]
#         batch, seq_length = input_ids.shape
#         causal_mask = paddle.tril(paddle.ones((batch, seq_length, seq_length), dtype=attn_mask_2d.dtype))
#         attn_mask_3d = causal_mask & attn_mask_2d.unsqueeze(-1)
#         result_3d = model(input_ids, attention_mask=attn_mask_3d)[0]
#         attn_mask_4d = attn_mask_3d.unsqueeze(1)
#         result_4d = model(input_ids, attention_mask=attn_mask_4d)[0]
#         result_no_attention_mask = model(input_ids, attention_mask=None)[0]
#         # Assert non-padding tokens have the same logits with different attention_mask shape
#         self.parent.assertTrue((result_2d[attn_mask_2d] == result_3d[attn_mask_2d]).all())
#         self.parent.assertTrue((result_2d[attn_mask_2d] == result_4d[attn_mask_2d]).all())
#         self.parent.assertFalse((result_2d[attn_mask_2d] == result_no_attention_mask[attn_mask_2d]).all())

#     def create_and_check_model_as_decoder(
#         self,
#         config,
#         input_ids,
#         input_mask,
#         sequence_labels,
#         token_labels,
#         choice_labels,
#     ):
#         config.add_cross_attention = True
#         model = Ernie4_5_MoeModel(config)
#         model.eval()
#         result = model(
#             input_ids,
#             attention_mask=input_mask,
#         )
#         result = model(
#             input_ids,
#             attention_mask=input_mask,
#         )
#         result = model(input_ids, attention_mask=input_mask)
#         self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

#     def create_and_check_for_causal_lm(
#         self,
#         config,
#         input_ids,
#         input_mask,
#         sequence_labels,
#         token_labels,
#         choice_labels,
#     ):
#         model = Ernie4_5_MoeForCausalLM(config=config)
#         model.eval()
#         result = model(input_ids, attention_mask=input_mask, labels=token_labels, return_dict=True)
#         self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

#     def prepare_config_and_inputs_for_common(self):
#         config_and_inputs = self.prepare_config_and_inputs()
#         (
#             config,
#             input_ids,
#             input_mask,
#             sequence_labels,
#             token_labels,
#             choice_labels,
#         ) = config_and_inputs
#         inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
#         return config, inputs_dict

#     def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
#         model = Ernie4_5_MoeModel(config)
#         model.eval()

#         model(
#             input_ids,
#             use_cache=True,
#             labels=input_ids if self.use_labels else None,
#             return_dict=self.return_dict,
#         )

#     def check_model_position_ids(self, config, input_ids, input_mask, *args):
#         model = Ernie4_5_MoeForCausalLM(config)
#         model.eval()

#         result_no_position_id = model(
#             input_ids,
#             labels=input_ids if self.use_labels else None,
#             return_dict=self.return_dict,
#         )
#         batch_size, seq_len = input_ids.shape
#         position_ids = paddle.arange(seq_len).expand((batch_size, seq_len))
#         result_position_id = model(
#             input_ids,
#             position_ids,
#             labels=input_ids if self.use_labels else None,
#             return_dict=self.return_dict,
#         )
#         if self.use_labels:
#             self.parent.assertTrue((result_position_id[1] == result_no_position_id[1]).all())
#         else:
#             self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())


# class Ernie4_5_MoeModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
#     base_model_class = Ernie4_5_MoeModel
#     return_dict = True
#     use_labels = False
#     use_test_model_name_list = False

#     all_model_classes = (Ernie4_5_MoeModel, Ernie4_5_MoeForCausalLM)
#     all_generative_model_classes = {Ernie4_5_MoeForCausalLM: (Ernie4_5_MoeModel, "ernie4_5_moe")}

#     def setUp(self):
#         super().setUp()

#         self.model_tester = Ernie4_5_MoeModelTester(self)
#         self.config_tester = ConfigTester(self, config_class=Ernie4_5_MoeConfig, vocab_size=256, hidden_size=24)

#     def _get_input_ids_and_config(self):
#         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

#         input_ids = inputs_dict[self.input_name]
#         attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)

#         max_batch_size = 2
#         sequence_length = input_ids.shape[-1] // 2
#         input_ids = input_ids[:max_batch_size, :sequence_length]
#         attention_mask = attention_mask[:max_batch_size, :sequence_length]
#         max_length = 3

#         return config, input_ids, attention_mask, max_length

#     @require_at_least_one_gpu
#     def test_model(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_model(*config_and_inputs)

#     @require_at_least_one_gpu
#     def test_model_attention_mask(self):
#         config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
#         self.model_tester.create_and_check_model_attention_mask(config, input_dict["input_ids"])

#     @require_at_least_one_gpu
#     def test_model_position_ids(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.check_model_position_ids(*config_and_inputs)

#     @require_at_least_one_gpu
#     def test_generate_without_input_ids(self):
#         # this requires 4-D attention mask logic, which is not supported yet
#         pass

#     @require_at_least_one_gpu
#     def test_model_decoder_model(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

#     @require_at_least_one_gpu
#     def test_model_lm_head_model(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

#     @require_at_least_one_gpu
#     def test_model_causal_lm(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

#     @require_at_least_one_gpu
#     def test_attention_outputs(self):
#         super().test_attention_outputs()

#     @require_at_least_one_gpu
#     def test_beam_search_generate(self):
#         super().test_beam_search_generate()

#     @require_at_least_one_gpu
#     def test_determinism(self):
#         super().test_determinism()

#     @require_at_least_one_gpu
#     def test_greedy_generate(self):
#         super().test_greedy_generate()

#     @require_at_least_one_gpu
#     def test_group_beam_search_generate(self):
#         super().test_group_beam_search_generate()

#     @require_at_least_one_gpu
#     def test_hidden_states_output(self):
#         super().test_hidden_states_output()

#     @require_at_least_one_gpu
#     def test_resize_tokens_embeddings(self):
#         super().test_resize_tokens_embeddings()

#     @require_at_least_one_gpu
#     def test_resize_position_vector_embeddings(self):
#         super().test_resize_position_vector_embeddings()

#     @require_at_least_one_gpu
#     def test_inputs_embeds(self):
#         super().test_inputs_embeds()

#     @require_at_least_one_gpu
#     def test_pretrained_config_save_load(self):
#         super().test_pretrained_config_save_load()

#     @require_at_least_one_gpu
#     def test_training(self):
#         super().test_training()

#     @require_at_least_one_gpu
#     def test_training_gradient_checkpointing(self):
#         super().test_training_gradient_checkpointing()

#     @require_at_least_one_gpu
#     def test_sample_generate(self):
#         super().test_sample_generate()

#     @require_at_least_one_gpu
#     def test_save_load(self):
#         super().test_save_load()


# class Ernie4_5_MoeCompatibilityTest(unittest.TestCase):
#     @classmethod
#     @require_package("transformers", "torch")
#     def setUpClass(cls) -> None:
#         from transformers import Ernie4_5_MoeConfig, Ernie4_5_MoeForCausalLM

#         # when python application is done, `TemporaryDirectory` will be free
#         cls.torch_model_path = tempfile.TemporaryDirectory().name
#         config = Ernie4_5_MoeConfig(
#             hidden_size=16, intermediate_size=320, moe_intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_nextn_predict_layers = 0
#         )
#         model = Ernie4_5_MoeForCausalLM(config)
#         model.save_pretrained(cls.torch_model_path)

#     @require_package("transformers", "torch")
#     def test_Ernie4_5_Moe_converter(self):
#         # 1. create common input
#         input_ids = np.random.randint(100, 200, [1, 20])

#         # 2. forward the paddle model
#         from paddleformers.transformers import Ernie4_5_MoeModel

#         paddle_model = Ernie4_5_MoeModel.from_pretrained(self.torch_model_path, convert_from_hf=True, dtype="float32")
#         paddle_model.eval()
#         paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=True).last_hidden_state

#         # 3. forward the torch  model
#         import torch
#         from transformers import Ernie4_5_MoeModel

#         torch_model = Ernie4_5_MoeModel.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
#         torch_model.eval()
#         torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]
#         self.assertTrue(
#             np.allclose(
#                 paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
#                 torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
#                 atol=1e-2,
#                 rtol=1e-2,
#             )
#         )

#     @require_package("transformers", "torch")
#     def test_Ernie4_5_Moe_converter_from_local_dir(self):
#         with tempfile.TemporaryDirectory() as tempdir:

#             # 1. create common input
#             input_ids = np.random.randint(100, 200, [1, 20])

#             # 2. forward the torch  model
#             import torch
#             from transformers import Ernie4_5_MoeModel

#             torch_model = Ernie4_5_MoeModel.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
#             torch_model.eval()
#             torch_model.save_pretrained(tempdir)
#             torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

#             # 2. forward the paddle model
#             from paddleformers.transformers import Ernie4_5_MoeModel

#             paddle_model = Ernie4_5_MoeModel.from_pretrained(tempdir, convert_from_hf=True, dtype="float32")
#             paddle_model.eval()
#             paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=True).last_hidden_state

#             self.assertTrue(
#                 np.allclose(
#                     paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
#                     torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
#                     atol=1e-2,
#                     rtol=1e-2,
#                 )
#             )

#     @parameterized.expand([("Ernie4_5_MoeModel",), ("Ernie4_5_MoeForCausalLM",)])
#     @require_package("transformers", "torch")
#     def test_Ernie_4_5_Moe_classes_from_local_dir(self, class_name, pytorch_class_name: str | None = None):
#         pytorch_class_name = pytorch_class_name or class_name
#         with tempfile.TemporaryDirectory() as tempdir:

#             # 1. create common input
#             input_ids = np.random.randint(100, 200, [1, 20])

#             # 2. forward the torch model
#             import torch
#             import transformers

#             torch_model_class = getattr(transformers, pytorch_class_name)
#             torch_model = torch_model_class.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
#             torch_model.eval()

#             torch_model.save_pretrained(tempdir)
#             torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

#             # 3. forward the paddle model
#             from paddleformers import transformers
#             paddle_model_class = getattr(transformers, class_name)
#             paddle_model = paddle_model_class.from_pretrained(tempdir, convert_from_hf=True, dtype="float32")
#             paddle_model.eval()

#             paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=True).last_hidden_state

#             self.assertTrue(
#                 np.allclose(
#                     paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
#                     torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
#                     atol=1e-2,
#                     rtol=1e-2,
#                 )
#             )


# if __name__ == "__main__":
#     unittest.main()
