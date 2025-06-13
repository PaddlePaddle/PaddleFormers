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

from contextlib import nullcontext

import paddle
from paddle.base import core
from paddle.distributed import fleet

from ..trainer import Trainer
from ..transformers.contrastive_loss import (
    MatryoshkaContrastiveLoss,
    MatryoshkaInfclLoss,
    SimpleContrastiveLoss,
    SimpleInfclLoss,
)
from ..transformers.embedding_utils import dist_gather_tensor_with_gradient
from ..utils import empty_device_cache

__all__ = ["EmbeddingTrainer"]


class EmbeddingTrainer(Trainer):
    def __init__(self, model_args, **kwargs):
        super().__init__(**kwargs)

        self.model_args = model_args
        self.embedding_negatives_cross_device = model_args.embedding_negatives_cross_device
        self.accum_data = []
        self.accum_freq = 0
        self.accum_q_features = []
        self.accum_p_features = []
        self.accum_rng_states = {}
        self.accum_rng_states["cpu"] = []
        self.accum_rng_states["cuda"] = []
        self.accum_rng_states["hybrid"] = []

        if model_args.embedding_matryoshka_dims is not None and len(model_args.embedding_matryoshka_dims) > 0:
            if model_args.loss_type == "inf_cl":
                self.embedding_negatives_cross_device = False
                self.loss_fn = MatryoshkaInfclLoss(model_args.embedding_matryoshka_dims, model_args.inf_cl_head_dim)
            elif model_args.loss_type == "contrastive":
                self.loss_fn = MatryoshkaContrastiveLoss(
                    model_args.embedding_temperature, model_args.embedding_matryoshka_dims
                )
        else:
            if model_args.loss_type == "inf_cl":
                self.embedding_negatives_cross_device = False
                self.loss_fn = SimpleInfclLoss(model_args.inf_cl_head_dim)
            elif model_args.loss_type == "contrastive":
                self.loss_fn = SimpleContrastiveLoss(model_args.embedding_temperature)

    def clear_memory(self):
        self.accum_q_features.clear()
        self.accum_p_features.clear()
        empty_device_cache()

    def clear_state(self):
        self.accum_data.clear()
        self.accum_rng_states["cpu"].clear()
        self.accum_rng_states["cuda"].clear()
        self.accum_rng_states["hybrid"].clear()
        self.accum_freq = 0

    @paddle.no_grad()
    def forward_no_grad(self, model, inputs):
        # Step1: graph-less forward
        self.accum_data.append(inputs)
        inputs = self._prepare_inputs(inputs)
        with self.autocast_smart_context_manager():
            # collect rand states
            self.accum_rng_states["cpu"].append(paddle.framework.core.default_cpu_generator().get_state())
            self.accum_rng_states["cuda"].append(paddle.get_rng_state())
            if self.args.use_hybrid_parallel:
                self.accum_rng_states["hybrid"].append(
                    fleet.meta_parallel.get_rng_state_tracker().get_states_tracker()
                )

            query_reps, passage_reps = model(**inputs, return_encode=True)

            if self.embedding_negatives_cross_device:
                query_reps = dist_gather_tensor_with_gradient(query_reps)
                passage_reps = dist_gather_tensor_with_gradient(passage_reps)

            self.accum_q_features.append(query_reps)
            self.accum_p_features.append(passage_reps)

        self.accum_freq += 1

    def get_current_rng_state(self):
        return {
            "cpu": [paddle.framework.core.default_cpu_generator().get_state()],
            "cuda": [paddle.get_rng_state()],
            "hybrid": [fleet.meta_parallel.get_rng_state_tracker().get_states_tracker()]
            if self.args.use_hybrid_parallel
            else [],
        }

    def reset_rng_state(self, states, index=0):
        # set random states
        if len(states) != 3:
            raise ValueError("The length of state should be 3")
        cpu_state = states["cpu"][index]
        cuda_state = states["cuda"][index]
        paddle.framework.core.default_cpu_generator().set_state(cpu_state)
        # TODO(daisiming): support xpu and other custom devices.
        if core.is_compiled_with_cuda():
            for j in range(core.get_cuda_device_count()):
                core.default_cuda_generator(j).set_state(cuda_state[j])
        if self.args.use_hybrid_parallel:
            hybrid_state = states["hybrid"][index]
            fleet.meta_parallel.get_rng_state_tracker().set_states_tracker(hybrid_state)

    def accum_forward_backward(self, model):
        # Step2: representation gradient computation and caching
        for i in range(len(self.accum_q_features)):
            self.accum_q_features[i].stop_gradient = False
        q_reps = paddle.concat(self.accum_q_features, axis=0)
        for i in range(len(self.accum_p_features)):
            self.accum_p_features[i].stop_gradient = False
        p_reps = paddle.concat(self.accum_p_features, axis=0)

        loss = self.loss_fn(q_reps, p_reps)
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        # get representation gradient cache
        accum_q_grads = [q.grad for q in self.accum_q_features]
        accum_p_grads = [p.grad for p in self.accum_p_features]
        del q_reps, p_reps

        # clear trash memory
        self.clear_memory()

        current_rng_state = self.get_current_rng_state()
        # Step3: sub-batch gradient accumulation
        for i in range(self.accum_freq):
            inputs = self.accum_data[i]
            inputs = self._prepare_inputs(inputs)

            sync_context = model.no_sync() if i != self.accum_freq - 1 and hasattr(model, "no_sync") else nullcontext()
            with sync_context:
                self.reset_rng_state(self.accum_rng_states, index=i)

                with self.autocast_smart_context_manager():
                    query_reps, passage_reps = model(**inputs, return_encode=True)

                if self.embedding_negatives_cross_device:
                    query_reps = dist_gather_tensor_with_gradient(query_reps)
                    passage_reps = dist_gather_tensor_with_gradient(passage_reps)

                _loss = paddle.dot(query_reps.flatten(), accum_q_grads[i].flatten()) + paddle.dot(
                    passage_reps.flatten(), accum_p_grads[i].flatten()
                )
                _loss.backward()

        self.reset_rng_state(current_rng_state)
        self.clear_state()
        return loss.detach()

    def training_step(
        self,
        model,
        inputs,
        step_control=0,
    ):
        if self.args.pipeline_parallel_degree > 1:
            raise NotImplementedError("Cannot support pipeline parallel for Embedding training now.")

        if self.args.gradient_accumulation_steps == 1:
            return super().training_step(model, inputs)
        else:
            self.forward_no_grad(model, inputs)

            # if (step_control + 1) % self.args.gradient_accumulation_steps is not zero, move on to next batch.
            if (step_control + 1) % self.args.gradient_accumulation_steps != 0:
                return 0.0

            loss = self.accum_forward_backward(model)
        return loss
