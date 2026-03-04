# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import copy
import inspect
import os
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from typing import Optional, Tuple, Type

import numpy as np
import paddle
from paddle.distributed.utils.launch_utils import (
    TrainerProc,
    find_free_ports,
    get_cluster,
    watch_local_trainers,
)

from paddleformers.transformers import AutoModelForCausalLM, AutoTokenizer
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.transformers.model_utils import PretrainedModel
from paddleformers.utils.env import CONFIG_NAME, LEGACY_CONFIG_NAME  # MODEL_HOME,

from ..testing_utils import slow


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
    return configs_no_init


def get_cluster_from_args(selected_gpus):
    cluster_node_ips = "127.0.0.1"
    node_ip = "127.0.0.1"

    node_ips = [x.strip() for x in cluster_node_ips.split(",")]

    node_ips.index(node_ip)

    free_ports = None

    free_ports = find_free_ports(len(selected_gpus))
    if free_ports is not None:
        free_ports = list(free_ports)

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, selected_gpus)


def get_gpus(selected_gpus):
    selected_gpus = [x.strip() for x in selected_gpus.split(",")]
    return selected_gpus


def start_local_trainers_cpu(trainer_endpoints, training_script, training_script_args, log_dir=None):
    current_env = copy.copy(os.environ.copy())
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    n_rank = len(trainer_endpoints)
    print(trainer_endpoints)
    for rank_id, endpoint in enumerate(trainer_endpoints):
        proc_env = {
            "PADDLE_DISTRI_BACKEND": "gloo",
            "PADDLE_TRAINER_ID": "%d" % rank_id,
            "PADDLE_CURRENT_ENDPOINT": "%s" % endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % n_rank,
            "PADDLE_TRAINER_ENDPOINTS": ",".join(trainer_endpoints),
        }

        current_env.update(proc_env)

        print("trainer proc env:{}".format(current_env))

        assert os.getenv("WITH_COVERAGE", "OFF") == "OFF", "Gloo don't support WITH_COVERAGE."
        cmd = "python -u " + training_script

        print("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None

        proc = subprocess.Popen(cmd.split(" "), env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = rank_id
        tp.log_fn = fn
        tp.cmd = cmd

        procs.append(tp)

    return procs


def start_local_trainers(
    cluster,
    pod,
    training_script,
    training_script_args="",
    eager_mode=True,
    allocator_strategy="auto_growth",
    log_dir=None,
    without_http_proxy=True,
):
    current_env = copy.copy(os.environ.copy())
    # paddle broadcast ncclUniqueId use socket, and
    # proxy maybe make trainers unreachable, so delete them.
    # if we set them to "", grpc will log error message "bad uri"
    # so just delete them.

    # current_env.pop("http_proxy", None)
    # current_env.pop("https_proxy", None)

    # parse args
    if isinstance(training_script_args, dict):
        training_script_args = [f"--{k} {v}" for k, v in training_script_args.items()]

    if isinstance(training_script_args, list):
        training_script_args = " ".join(training_script_args)

    procs = []
    for t in pod.trainers:
        proc_env = {
            "FLAGS_selected_gpus": "%s" % ",".join([str(g) for g in t.gpus]),
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % t.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints()),
        }

        proc_env["FLAGS_allocator_strategy"] = allocator_strategy
        if allocator_strategy == "auto_growth":
            proc_env["FLAGS_fraction_of_gpu_memory_to_use"] = "0.1"

        current_env.update(proc_env)

        if os.getenv("WITH_COVERAGE", "OFF") == "ON":
            cmd = "python -m coverage run --branch -p " + training_script
        else:
            cmd = f"python -u {training_script} {training_script_args}"

        print("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None

        proc = subprocess.Popen(cmd.split(" "), env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.log_fn = fn
        tp.cmd = cmd

        procs.append(tp)

    return procs


def ids_tensor(shape, vocab_size, dtype="int32"):
    #  Creates a random int32 tensor of the shape within the vocab size
    return paddle.randint(low=1, high=vocab_size, dtype=dtype, shape=shape)


def random_attention_mask(shape, dtype="int32"):
    attn_mask = ids_tensor(shape, vocab_size=2, dtype=dtype)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0):
    """Creates a random float32 tensor"""
    return scale * paddle.randn(shape, dtype="float32")


def check_two_model_parameter(first_model: PretrainedModel, second_model: PretrainedModel):
    assert len(set(first_model.state_dict().keys()) - set(second_model.state_dict().keys())) == 0

    # random choice the keys to compare
    key = random.choice(list(first_model.state_dict().keys()))
    diff = first_model.state_dict()[key] - second_model.state_dict()[key]
    assert diff.sum().item() == 0


class ModelTesterMixin:
    model_tester = None
    base_model_class: Optional[Type[PretrainedModel]] = None
    all_model_classes: Tuple[Type[PretrainedModel]] = ()
    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_mismatched_shapes = True
    test_missing_keys = True
    test_model_compatibility_keys = True
    test_tie_weights = False
    use_test_inputs_embeds = False
    use_test_model_name_list = True
    is_encoder_decoder = False
    has_attentions = True
    model_split_percents = [0.5, 0.7, 0.9]

    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class.__name__.endswith("ForMultipleChoice"):
            inputs_dict = {
                k: v.unsqueeze(1).expand(shape=[-1, self.model_tester.num_choices, -1])
                if isinstance(v, paddle.Tensor) and v.ndim > 1
                else v
                for k, v in inputs_dict.items()
            }
        return inputs_dict

    def _make_model_instance(self, config, model_class):
        if isinstance(config, PretrainedConfig):
            return model_class(config)
        if model_class == self.base_model_class:
            return model_class(**config)

        return model_class(self.base_model_class(**config))

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.numpy()
            out_2[np.isnan(out_2)] = 0

            out_1 = out1.numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, save_to_hf=False, save_checkpoint_format="")
                model = model_class.from_pretrained(tmpdirname, convert_from_hf=False, load_checkpoint_format="")
                model.eval()
                with paddle.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            # support tuple of tensor
            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["input_ids"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @unittest.skip("Not implemented yet")
    def test_training(self):
        # TODO(guosheng): add more tests for training if loss is implemented
        pass

    @unittest.skip("Not implemented yet")
    def test_training_gradient_checkpointing(self):
        pass

    def test_attention_outputs(self):
        if not self.has_attentions:
            return
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            signature = inspect.signature(model_class.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            if not all(name in arg_names for name in ["output_attentions", "output_hidden_states", "return_dict"]):
                continue
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            inputs_dict["return_dict"] = True
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if self.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # TODO(guosheng): check that output_attentions also work using config

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                # Question Answering model returns start_logits and end_logits
                if model_class.__name__.endswith("ForQuestionAnswering"):
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if self.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = self._make_model_instance(config, model_class)
            model.eval()

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if self.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if self.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict["return_dict"] = True
        for model_class in self.all_model_classes:
            signature = inspect.signature(model_class.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            if not all(name in arg_names for name in ["output_attentions", "output_hidden_states", "return_dict"]):
                continue
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            # TODO(guosheng): check that output_hidden_states also work using config

    @unittest.skip("Not implemented")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_resize_position_vector_embeddings(self):
        if not self.test_resize_position_embeddings:
            return

        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = self._make_model_instance(config, model_class)

            if self.model_tester.is_training is False:
                model.eval()

            max_position_embeddings = config.max_position_embeddings

            # Retrieve the embeddings and clone theme
            if self.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                encoder_cloned_embeddings = encoder_model_embed.weight.clone()
                decoder_cloned_embeddings = decoder_model_embed.weight.clone()
            else:
                model_embed = model.get_position_embeddings()
                cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the position embeddings with a larger max_position_embeddings increases
            # the model's position embeddings size
            model.resize_position_embeddings(max_position_embeddings + 10)
            self.assertEqual(model.config.max_position_embeddings, max_position_embeddings + 10)

            # Check that it actually resizes the embeddings matrix
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                self.assertEqual(encoder_model_embed.weight.shape[0], encoder_cloned_embeddings.shape[0] + 10)
                self.assertEqual(decoder_model_embed.weight.shape[0], decoder_cloned_embeddings.shape[0] + 10)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the position embeddings with a smaller max_position_embeddings decreases
            # the model's max_position_embeddings
            model.resize_position_embeddings(max_position_embeddings - 5)
            self.assertEqual(model.base_model.config["max_position_embeddings"], max_position_embeddings - 5)

            # Check that it actually resizes the embeddings matrix
            if self.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                self.assertEqual(encoder_model_embed.weight.shape[0], encoder_cloned_embeddings.shape[0] - 5)
                self.assertEqual(decoder_model_embed.weight.shape[0], decoder_cloned_embeddings.shape[0] - 5)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 5)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True

            if model.config.is_encoder_decoder:
                for p1, p2 in zip(encoder_cloned_embeddings, encoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
                for p1, p2 in zip(decoder_cloned_embeddings, decoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
            else:
                for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

    def test_resize_tokens_embeddings(self):
        if not self.test_resize_embeddings:
            return
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = self._make_model_instance(config, model_class)
            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.get_text_config().vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"] = paddle.clip(inputs_dict["input_ids"], max=model_vocab_size - 15 - 1)

            # make sure that decoder_input_ids are resized as well
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"] = paddle.clip(
                    inputs_dict["decoder_input_ids"], max=model_vocab_size - 15 - 1
                )
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            # Get the new vocab size from the resized embedding matrix
            new_vocab_size = model_embed.weight.shape[0]
            original_slice_to_compare = cloned_embeddings[0:new_vocab_size]
            models_equal_tensor = paddle.equal_all(original_slice_to_compare, model_embed.weight)
            self.assertTrue(models_equal_tensor.item())

    def _compare_tensor(self, tensor1, tensor2, rtol=1e-04, atol=1e-04):
        if tensor1.dtype != tensor2.dtype:
            return False

        if tensor1.dtype in [paddle.float32, paddle.float64]:
            return paddle.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        else:
            return paddle.equal_all(tensor1, tensor2)

    def test_inputs_embeds(self):
        # pass the test if don't need to test inputs embeddings
        if not self.use_test_inputs_embeds:
            return
        # get config for model and inputs_dict for model forward
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        # test all model classes
        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            with paddle.no_grad():
                ids_output = model(**inputs)

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with paddle.no_grad():
                embeds_output = model(**inputs)

            if isinstance(embeds_output, paddle.Tensor):
                self.assertTrue(self._compare_tensor(ids_output, embeds_output))
            else:
                for ids_item, embeds_item in zip(ids_output, embeds_output):
                    self.assertTrue(self._compare_tensor(ids_item, embeds_item))

    # def test_model_name_list(self):
    #     if not self.use_test_model_name_list:
    #         return
    #     config = self.model_tester.get_config()
    #     if isinstance(config, PretrainedConfig):
    #         model = self.base_model_class(config)
    #     else:
    #         model = self.base_model_class(**config)
    #     self.assertTrue(len(model.model_name_list) != 0)

    def test_pretrained_config_save_load(self):
        if self.base_model_class is None or not self.base_model_class.constructed_from_pretrained_config():
            return

        config_class = self.base_model_class.config_class
        with tempfile.TemporaryDirectory() as tempdir:
            config = config_class()

            config.save_pretrained(tempdir)

            # check the file exist
            self.assertFalse(os.path.exists(os.path.join(tempdir, LEGACY_CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, CONFIG_NAME)))

            # rename the CONFIG_NAME
            shutil.move(os.path.join(tempdir, CONFIG_NAME), os.path.join(tempdir, LEGACY_CONFIG_NAME))

            loaded_config = config.__class__.from_pretrained(tempdir)
            for key in config.__dict__.keys():
                if key == "paddleformers_version" and config.paddleformers_version is None:
                    continue
                elif isinstance(getattr(config, key), PretrainedConfig):
                    self.assertEqual(getattr(config, key).to_dict(), getattr(loaded_config, key).to_dict())
                else:
                    self.assertEqual(getattr(config, key), getattr(loaded_config, key))

    def random_choice_pretrained_config_field(self) -> Optional[str]:
        if self.base_model_class is None or not self.base_model_class.constructed_from_pretrained_config():
            return None

        config = self.base_model_class.config_class()
        fields = [key for key, value in config.to_dict() if value]
        return random.choice(fields)

    def test_for_missed_attribute(self):
        if not self.test_model_compatibility_keys:
            self.skipTest(f"Do not test model_compatibility_keys on {self.base_model_class}")
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if not model_class.constructed_from_pretrained_config():
                continue

            model = self._make_model_instance(config, model_class)

            all_maps: dict = copy.deepcopy(model_class.config_class.attribute_map)

            for old_attribute, new_attribute in all_maps.items():
                old_value = getattr(model.config, old_attribute)
                new_value = getattr(model.config, new_attribute)

                # eg: dropout can be an instance of nn.Dropout, so we should check it attribute
                if type(new_value) != type(old_value):
                    continue

                self.assertEqual(old_value, new_value)

    def test_tie_weight(self):
        # test whether id of input_embeding equal id of output_embeding ?
        if not self.test_tie_weights:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if "CausalLM" not in model_class.__name__ and "MaskedLM" not in model_class.__name__:
                continue

            model = self._make_model_instance(config, model_class)

            if not model.config.tie_word_embeddings:
                continue

            if hasattr(model, "get_input_embeddings") and hasattr(model, "get_output_embeddings"):
                try:
                    input_embeddings = model.get_input_embeddings()
                except NotImplementedError:
                    continue

                try:
                    output_embeddings = model.get_output_embeddings()
                except NotImplementedError:
                    continue

                if input_embeddings is not None and output_embeddings is not None:
                    if hasattr(output_embeddings, "weight"):
                        output_embeddings_weight = output_embeddings.weight
                    else:
                        output_embeddings_weight = output_embeddings

                    if hasattr(input_embeddings, "weight"):
                        input_embeddings_weight = input_embeddings.weight
                    else:
                        input_embeddings_weight = input_embeddings
                    print(
                        input_embeddings_weight,
                        output_embeddings_weight,
                    )
                    print(
                        "model name :{},id is{},{}".format(
                            model_class, id(output_embeddings_weight), id(input_embeddings_weight)
                        )
                    )
                    self.assertEqual(id(output_embeddings_weight), id(input_embeddings_weight))


class ModelTesterPretrainedMixin:
    base_model_class: PretrainedModel = None
    hf_remote_test_model_path: str = None
    paddlehub_remote_test_model_path: str = None

    @slow
    def test_model_from_pretrained_hf_hub(self):
        if self.hf_remote_test_model_path is None or self.base_model_class is None:
            return
        model = self.base_model_class.from_pretrained(
            self.hf_remote_test_model_path,
            download_hub="huggingface",
            convert_from_hf=False,
            load_checkpoint_format="",
        )
        self.assertIsNotNone(model)

    def test_model_from_pretrained_paddle_hub(self):
        if self.paddlehub_remote_test_model_path is None or self.base_model_class is None:
            return
        model = self.base_model_class.from_pretrained(
            self.paddlehub_remote_test_model_path, convert_from_hf=False, load_checkpoint_format=""
        )
        self.assertIsNotNone(model)

    def test_model_from_config_paddle_hub(self):
        if self.paddlehub_remote_test_model_path is None or self.base_model_class is None:
            return
        config = self.base_model_class.config_class.from_pretrained(self.paddlehub_remote_test_model_path)
        model = self.base_model_class.from_config(config)
        self.assertIsNotNone(model)

    @slow
    def test_model_from_pretrained_with_cache_dir(self):
        for model_name in list(self.base_model_class.pretrained_init_configuration)[:1]:
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = str(tempdir)

                model = self.base_model_class.from_pretrained(
                    model_name, cache_dir=tempdir, convert_from_hf=False, load_checkpoint_format=""
                )
                self.assertIsNotNone(model)
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(tempdir, model_name, self.base_model_class.resource_files_names["model_state"])
                    )
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(tempdir, model_name, self.base_model_class.model_config_file))
                )

    @slow
    def test_pretrained_save_and_load(self):
        """test the pretrained model save and load with two different ways: url-file-name & model_state name

        eg: `bert-base-uncased.pdparams` and `model_state.pdparams`
        """
        for model_name in list(self.base_model_class.pretrained_init_configuration)[:1]:
            model = self.base_model_class.from_pretrained(model_name, convert_from_hf=False, load_checkpoint_format="")
            self.assertIsNotNone(model)

            # 1. save and load
            with tempfile.TemporaryDirectory() as tempdir:
                tempdirname = str(tempdir)
                model.save_pretrained(tempdirname, save_to_hf=False, save_checkpoint_format="")

                loaded_model = self.base_model_class.from_pretrained(
                    tempdirname, convert_from_hf=False, load_checkpoint_format=""
                )

                check_two_model_parameter(model, loaded_model)


class DistributedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gpus = "0,1"

    def get_world_size(self):
        return len(self.gpus.split(","))

    def run_on_gpu(
        self,
        training_script,
        training_script_args=None,
        gpus: str = "0,1",
        eager_mode=True,
        allocator_strategy="auto_growth",
    ):
        if not paddle.framework.core.is_compiled_with_cuda() or paddle.framework.core.get_cuda_device_count() == 0:
            return

        cluster = None
        pod = None

        cluster, pod = get_cluster_from_args(get_gpus(gpus))

        procs = start_local_trainers(
            cluster,
            pod,
            eager_mode=eager_mode,
            allocator_strategy=allocator_strategy,
            training_script=training_script,
            training_script_args=training_script_args,
        )

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_endpoints())

            if not alive:
                print("Local procs complete, POD info:{}".format(pod))
                break
            time.sleep(3)


class GenerationD2STestMixin:
    article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
    internal_testing_model = None

    TokenizerClass = AutoTokenizer
    CausalLMClass = AutoModelForCausalLM
    max_new_tokens = 20

    def setUp(self):
        paddle.disable_static()
        super().setUp()

    def tearDown(self):
        paddle.disable_static()
        super().setUp()
