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

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import contextlib
import logging
import random

import numpy as np
import paddle
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.recompute.recompute import (
    custom_state_manager,
    detach_variable,
    switch_rng_state_tracker,
)
from paddle.framework import core

from ..parallel_state import (
    get_expert_model_parallel_rank,
    get_expert_tensor_parallel_rank,
    get_tensor_model_parallel_rank,
)

HAVE_TE = False


# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"
_EXPERT_PARALLEL_RNG_TRACKER_NAME = "expert-parallel-rng"
_DATA_PARALLEL_RNG_TRACKER_NAME = "data-parallel-rng"


def _get_cuda_rng_state(device: int | None = None, clone: bool = False, graph_safe: bool = False) -> paddle.Tensor:
    """Return the random number generator state of the specified GPU.

    Arguments:
        device (int): The device id to retrieve the rng state,
            If None, use current device, specified by ``set_device``
            function.
        clone (bool): Whether to also clone the retrieved RNG state
        graph_safe (bool): Get the rng state in a graph safe manner.

    This function is adapted from paddle.get_rng_state()"""

    assert graph_safe is False, "graph_safe is not supported yet"
    # if not using cuda graphs, just use the builtin function
    if not graph_safe:
        return paddle.cuda.get_rng_state(device)


def _set_cuda_rng_state(new_state, device: int | None = None, graph_safe: bool = False):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (GeneratorState): The desired state
        device (int): The device id to retrieve the rng state,
            If None, use current device, specified by ``set_device``
            function.
        graph_safe (bool): Set the rng state in a graph safe manner.

    """
    assert graph_safe is False, "graph_safe is not supported yet"
    with paddle.LazyGuard():
        paddle.cuda.set_rng_state(new_state, device=device)


def get_expert_parallel_rng_tracker_name():
    """Get the expert parallel rng tracker name"""
    global _EXPERT_PARALLEL_RNG_TRACKER_NAME
    return _EXPERT_PARALLEL_RNG_TRACKER_NAME


def get_data_parallel_rng_tracker_name():
    """Get the data parallel rng tracker name"""
    global _DATA_PARALLEL_RNG_TRACKER_NAME
    return _DATA_PARALLEL_RNG_TRACKER_NAME


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self, use_cudagraphable_rng=False, is_inference_rng_tracker=False):
        assert use_cudagraphable_rng is False, "use_cudagraphable_rng is not supported yet"
        self.reset()
        self.use_cudagraphable_rng = use_cudagraphable_rng
        self.is_inference_rng_tracker = is_inference_rng_tracker

    def is_initialized(self):
        """Checks if the internal RNG state has been set with set_states()."""
        return self._is_initialized

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False

        # Map from a string name to the cuda rng state.
        self.states_ = {}

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self._is_initialized = True
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        self._is_initialized = True
        # Check seed is not already used.
        if seed in self.seeds_:
            raise ValueError(f"seed {seed} already exists")
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise ValueError(f"cuda rng state {name} already exists")

        # If available, create the state in a graph safe manner
        if self.use_cudagraphable_rng:
            new_state = _get_cuda_rng_state(clone=True, graph_safe=True)
            new_state.manual_seed(seed)
            self.states_[name] = new_state
        else:
            # Get the current rng state.
            orig_rng_state = paddle.cuda.get_rng_state()
            # Set the new state and store it.
            paddle.cuda.manual_seed(seed)
            self.states_[name] = paddle.cuda.get_rng_state()
            # Reset rng state to what it was.
            _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception(f"cuda rng state {name} is not added")
        # Store current rng state.
        orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name], graph_safe=self.use_cudagraphable_rng)
        # Record cpu RNG state
        cpu_rng_state = paddle.get_rng_state("cpu")
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Throw a warning if cpu RNG state changed
            if not cpu_rng_state == paddle.get_rng_state("cpu"):
                if not hasattr(self.__class__, "_cpu_rng_warning_issued"):
                    logging.getLogger(__name__).warning("CPU RNG state changed within GPU RNG context")
                    self.__class__._cpu_rng_warning_issued = True
            # Update the current rng state for later use.
            self.states_[name] = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state, graph_safe=self.use_cudagraphable_rng)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = None
_CUDA_RNG_STATE_TRACKER_INITIALIZED = False


def initialize_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    force_reset: bool = False,
):
    """Create the RNG tracker.
    In particular, TransformerEngine's implementation is cudagraphable and supports FP8.
    """
    assert use_cudagraphable_rng is False, "use_cudagraphable_rng is not supported yet"
    assert use_te_rng_tracker is False, "use_te_rng_tracker is not supported yet"
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED
    if force_reset:
        _CUDA_RNG_STATE_TRACKER = None
        _CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:
        return

    # Get the base tracker class
    base_tracker = None
    # if HAVE_TE and use_te_rng_tracker:
    #     if not is_te_min_version("1.5.0"):
    #         raise RuntimeError("use_te_rng_tracker requires TransformerEngine version >= 1.5")
    #     from ..extensions.transformer_engine import TECudaRNGStatesTracker

    #     base_tracker = TECudaRNGStatesTracker
    #     tracker_kwargs = {"is_inference_rng_tracker": inference_rng_tracker}
    # else:
    base_tracker = CudaRNGStatesTracker
    tracker_kwargs = {
        "use_cudagraphable_rng": use_cudagraphable_rng,
        "is_inference_rng_tracker": inference_rng_tracker,
    }

    if inference_rng_tracker:

        class InferenceCudaRNGStatesTracker(base_tracker):  # type: ignore[valid-type, misc]
            """RNG tracker for inference."""

            def add(self, name, seed):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def set_states(self, states):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
                """Mirrors the interface from the training RNG tracker."""
                return contextlib.nullcontext()

        tracker_class = InferenceCudaRNGStatesTracker
    else:
        tracker_class = base_tracker

    _CUDA_RNG_STATE_TRACKER = tracker_class(**tracker_kwargs)
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True


def get_cuda_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    assert use_cudagraphable_rng is False, "use_cudagraphable_rng is not supported yet"
    assert use_te_rng_tracker is False, "use_te_rng_tracker is not supported yet"
    """Get cuda rng tracker."""
    initialize_rng_tracker(use_te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
    return _CUDA_RNG_STATE_TRACKER


def get_all_rng_states():
    """Returns all generator states used by the current `CudaRNGStatesTracker`."""

    assert (
        _CUDA_RNG_STATE_TRACKER_INITIALIZED
    ), "Tried getting all rng states but RNG Tracker has not been initialized!"

    if isinstance(_CUDA_RNG_STATE_TRACKER, CudaRNGStatesTracker):
        return _CUDA_RNG_STATE_TRACKER.states_
    else:
        return {}


def model_parallel_cuda_manual_seed(
    seed: int,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    tp_rank: int | None = None,
    ep_rank: int | None = None,
    etp_rank: int | None = None,
):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no paddle.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Three set of RNG states are tracked:
    default state: This is for data parallelism and is the same among a set of model parallel GPUs
    but different across different model parallel groups. This is used for example for dropout
    in the non-tensor-model-parallel regions.
    tensor-model-parallel state: This state is different among a set of model parallel GPUs,
    but the same across data parallel groups. This is used for example for dropout
    in model parallel regions.
    expert-parallel-seed: This state is only used for the expert layer of MoE models.
    It is different among expert-tensor and expert-model parallel GPUs, and the same
    across expert-data parallel groups.
    """
    assert te_rng_tracker is False, "te_rng_tracker is not supported yet"
    assert use_cudagraphable_rng is False, "use_cudagraphable_rng is not supported yet"
    if tp_rank is None:
        tp_rank = get_tensor_model_parallel_rank()
    if ep_rank is None:
        ep_rank = get_expert_model_parallel_rank()
    if etp_rank is None:
        etp_rank = get_expert_tensor_parallel_rank()
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    tensor_model_parallel_seed = offset + tp_rank
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    initialize_rng_tracker(te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    paddle.cuda.manual_seed(data_parallel_seed)
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)

    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)

    expert_parallel_seed = seed + 1024 + 100 * ep_rank + etp_rank
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)


def _get_all_rng_states():
    """Get all the rng states."""
    cpu_rng_state = paddle.get_rng_state("cpu")
    cuda_rng_state = _get_cuda_rng_state()
    cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
    return cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker


def _set_all_rng_states(cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
    """Set all the rng states."""
    paddle.set_rng_state(cpu_rng_state, device="cpu")
    _set_cuda_rng_state(cuda_rng_state)
    get_cuda_rng_tracker().set_states(cuda_rng_state_tracker)


@contextlib.contextmanager
def _fork_rng():
    """Fork the rng state."""
    # Store the current states.
    current_states = _get_all_rng_states()
    try:
        yield
    finally:
        # Set the states back to what it was at the start of this function.
        _set_all_rng_states(*current_states)


def checkpoint(function, *args, **kwargs):
    """Checkpoint a model or part of the model."""
    pass


@contextlib.contextmanager
def enable_share_grad_holder():
    """Temporarily enable grad-holder sharing instead of copying the grad."""
    flag = "FLAGS_share_tensor_for_grad_tensor_holder"
    old_value = paddle.get_flags([flag])[flag]
    if not old_value:
        paddle.set_flags({flag: True})
    try:
        yield
    finally:
        if not old_value:
            paddle.set_flags({flag: False})


class RecomputeWithoutOutputFunction(paddle.autograd.PyLayer):
    """Autograd wrapper for RecomputeWithoutOutput."""

    @staticmethod
    def forward(ctx, run_function, owner, *args):
        """forward"""
        with paddle.no_grad():
            outputs = run_function(*args)
        ctx.save_for_backward(*detach_variable(args))
        owner.ctx = ctx
        return outputs

    @staticmethod
    def backward(ctx, *output_grads):
        """backward"""
        inputs = ctx.inputs
        outputs = ctx.outputs
        share_grad_holder = enable_share_grad_holder() if ctx.share_grad_holder else contextlib.nullcontext()
        with paddle.amp.auto_cast(enable=False), share_grad_holder:
            paddle.autograd.backward(outputs, output_grads)
        ctx.outputs = None
        ctx.inputs = None
        grads = tuple(inp.grad for inp in inputs if isinstance(inp, paddle.Tensor))
        return grads


class RecomputeWithoutOutput:
    """Manage forward-output discard and backward-time recompute to save memory."""

    def __init__(self):
        self.run_function = None
        self.ctx = None
        self.outputs = None

    def recompute(
        self,
        run_function,
        *args,
        preserve_rng_state=True,
        share_grad_holder=False,
    ):
        """
        Recompute without saving outputs to save memory.

        Args:
            run_function: The forward function that will be executed again during backward.
            *args: Positional tensor and non-tensor inputs passed to `run_function`.
            preserve_rng_state: Whether to preserve RNG states for deterministic recomputation.
            share_grad_holder: Whether to share the grad holder instead of copying the grad in backward.
        """
        self.run_function = run_function
        self.preserve_rng_state = preserve_rng_state

        if preserve_rng_state:
            self.fw_rng_state = paddle.get_rng_state()
            self.fwd_rng_state_tracker = get_rng_state_tracker().get_states_tracker()
            self.fwd_numpy_state = np.random.get_state()
            self.fwd_random_state = random.getstate()

            if custom_state_manager.custom_get_state_func is None:
                assert custom_state_manager.custom_set_state_func is None
                custom_get_state_func = lambda x=None: None
                custom_set_state_func = lambda x=None: None
            else:
                custom_get_state_func = custom_state_manager.custom_get_state_func
                custom_set_state_func = custom_state_manager.custom_set_state_func

            self.fwd_custom_state = custom_get_state_func()
            self.custom_get_state_func = custom_get_state_func
            self.custom_set_state_func = custom_set_state_func

        tracer = paddle.base.framework._dygraph_tracer()
        self.is_fw_autocast = False if tracer._amp_level == core.AmpLevel.O0 else True
        if tracer._amp_level == core.AmpLevel.O2:
            self.amp_level = "O2"
        elif tracer._amp_level in (core.AmpLevel.O1, core.AmpLevel.O0):
            self.amp_level = "O1"
        else:
            raise ValueError(f"unsupported amp level: {tracer._amp_level}")

        if tracer._amp_dtype == "float16":
            self.amp_dtype = "float16"
        elif tracer._amp_dtype in ("bfloat16", "float32"):
            self.amp_dtype = "bfloat16"
        else:
            raise ValueError(f"unsupported amp dtype: {tracer._amp_dtype}")

        self.amp_white_list, self.amp_black_list = tracer._get_amp_op_list()

        outputs = RecomputeWithoutOutputFunction.apply(run_function, self, *args)
        self.outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        self.ctx.share_grad_holder = share_grad_holder
        return outputs

    def _recompute(self, grad):
        """Re-run the saved forward under restored states when the backward hook is triggered."""
        if self.preserve_rng_state:
            rng_ctx = switch_rng_state_tracker(
                self.fw_rng_state,
                self.fwd_rng_state_tracker,
                self.fwd_numpy_state,
                self.fwd_random_state,
                self.fwd_custom_state,
                self.custom_get_state_func,
                self.custom_set_state_func,
            )
        else:
            rng_ctx = contextlib.nullcontext()

        amp_ctx = paddle.amp.auto_cast(
            enable=self.is_fw_autocast,
            custom_white_list=self.amp_white_list,
            custom_black_list=self.amp_black_list,
            level=self.amp_level,
            dtype=self.amp_dtype,
        )

        inputs = detach_variable(self.ctx.saved_tensor())

        with rng_ctx, amp_ctx:
            outputs = self.run_function(*inputs)

        if isinstance(outputs, paddle.Tensor):
            outputs = (outputs,)

        for stale_output, recomputed_output in zip(self.outputs, outputs):
            recomputed_output._share_buffer_to(stale_output)

        self.ctx.inputs = inputs
        self.ctx.outputs = outputs
        self.outputs = None
        self.ctx = None
        self.run_function = None

    def discard_output_and_register_recompute(self, hook_tensor):
        """Clear saved output data and register the recomputation hook on the target tensor."""
        for output in self.outputs:
            output._clear_data()

        if not hook_tensor.stop_gradient:
            hook_tensor.register_hook(self._recompute)
