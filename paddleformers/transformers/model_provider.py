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

# Refer to NVIDIA Megatron-Bridge https://github.com/NVIDIA-NeMo/Megatron-Bridge
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import abc
import os
from typing import Callable, Generic, TypeVar, Union

try:
    from typing import Unpack
except ImportError:
    try:
        from typing_extensions import Unpack
    except ImportError:
        from unittest.mock import MagicMock

        Unpack = MagicMock()

import paddle
import paddle.distributed as dist

from ..utils.import_utils import is_paddlefleet_available

# This module requires paddlefleet to be installed
if not is_paddlefleet_available():
    raise ImportError(
        "paddlefleet is required for model_provider. "
        "Please install paddlefleet to use this module. "
        "You can install it with: pip install paddlefleet"
    )

from paddlefleet import parallel_state, tensor_parallel
from paddlefleet.transformer.enums import ModelType
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.utils import get_model_config

try:
    from paddlefleet.fp8_utils import correct_amax_history_if_needed
except ImportError:
    correct_amax_history_if_needed = None

ModelT = TypeVar("ModelT", bound=FleetLayer)


# TODO: Support DistributedDataParallelConfig
class DistributedDataParallelConfig:
    def __init__(self):
        pass


class ModelProviderMixin(abc.ABC, Generic[ModelT]):
    """A mixin that implements the ModelProvider pattern for PaddleFormers.

    The ModelProvider pattern solves ecosystem fragmentation by providing a standardized
    way to instantiate models. This mixin provides a consistent `provide_distributed_model()` method
    that handles the complexity of distributed training setup.

    For advanced customization, multiple hooks can be registered via `register_pre_wrap_hook`
    and `register_post_wrap_hook`. These hooks allow modifying the model before and after
    it's wrapped for distributed training (e.g., freezing layers, logging). The composed
    hooks can be accessed via the `pre_wrap_hook` and `post_wrap_hook` properties.

    Subclasses must implement the `provide` method to define the model architecture.
    """

    CONFIG_NAME = "mhub_model.json"
    DEFAULT_CONFIG_FORMAT = "json"

    @abc.abstractmethod
    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> ModelT:
        """Abstract method to provide the model instance.

        Subclasses must implement this method to return the specific Megatron model
        (e.g., `GPTModel`) with its configuration. This method is called by `get_model`
        to obtain the base model before it is wrapped for distributed training.

        Args:
            pre_process (bool, optional): Whether to include the embedding layer (used with pipeline parallelism).
            post_process (bool, optional): Whether to include the output layer (used with pipeline parallelism).
            vp_stage (int, optional): The virtual pipeline stage of the model.

        Returns:
            ModelT: The FleetLayer instance.
        """
        pass

    def provide_distributed_model(
        self,
        ddp_config: DistributedDataParallelConfig | None = None,
        model_type=ModelType.encoder_or_decoder,
        overlap_param_gather_with_optimizer_step: bool = False,
        fp16: bool | None = None,
        bf16: bool | None = None,
        use_paddle_fsdp: bool = False,
        use_paddle_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        use_cpu_initialization: None | bool = False,
        init_model_with_meta_device: bool | None = None,
        pre_wrap_hook: Union[
            Callable[[list[FleetLayer]], list[FleetLayer]],
            list[Callable[[list[FleetLayer]], list[FleetLayer]]],
        ]
        | None = None,
        post_wrap_hook: Callable[[list[FleetLayer]], list[FleetLayer]] | None = None,
    ) -> list[ModelT]:
        """Instantiate and wrap the model for distributed training.

        This method retrieves the model from `provide` and sets up the distributed
        environment, including data-parallel and model-parallel configurations.
        It's the primary entry point for creating a model that's ready for use
        in the PaddleFormers ecosystem.

        Args:
            ddp_config: Configuration for distributed data parallel.
            model_type: Type of model (encoder, decoder, or both).
            overlap_param_gather_with_optimizer_step: Whether to overlap param gathering.
            fp16: Override FP16 setting.
            bf16: Override BF16 setting.
            use_paddle_fsdp: Use paddle's Fully Sharded Data Parallel
            use_paddle_fsdp2: Use Paddle FSDP2 instead of custom DDP.
            wrap_with_ddp: Whether to wrap model with DDP.
            data_parallel_random_init: Initialize parameters randomly across data parallel ranks.
            use_cpu_initialization: Initialize model on CPU.
            init_model_with_meta_device: Initialize model on meta device.
            pre_wrap_hook: A single callable or list of callables to modify the model before it's wrapped.
                If provided, this will override all hooks registered via `register_pre_wrap_hook`.
                If a list is provided, hooks will be executed in order.
            post_wrap_hook: A single callable to modify the model after it's wrapped. If provided,
                this will override all hooks registered via `register_post_wrap_hook`.

        Returns:
            A list containing the wrapped model instance.
        """
        if wrap_with_ddp and not ddp_config:
            raise ValueError("ddp_config is required when wrap_with_ddp is True")

        if not paddle.distributed.is_initialized():
            os.environ["RANK"] = os.environ.get("RANK", "0")
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
            paddle.cuda.set_device(dist.get_rank())
            paddle.distributed.init_process_group("nccl")

        if not parallel_state.is_initialized():
            print("Model parallel not initialized, initializing...")
            self.initialize_model_parallel(seed=0)

        # Convert list of hooks to a single composed callable
        if isinstance(pre_wrap_hook, list):

            def composed_pre_wrap_hook(model: list[FleetLayer]) -> list[FleetLayer]:
                for hook in pre_wrap_hook:
                    model = hook(model)
                return model

            final_pre_wrap_hook = composed_pre_wrap_hook
        else:
            final_pre_wrap_hook = pre_wrap_hook or self.pre_wrap_hook
        final_post_wrap_hook = post_wrap_hook or self.post_wrap_hook

        model = get_model(
            self,
            ddp_config=ddp_config,
            model_type=model_type,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            fp16=fp16,
            bf16=bf16,
            use_paddle_fsdp=use_paddle_fsdp,
            use_paddle_fsdp2=use_paddle_fsdp2,
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
            use_cpu_initialization=use_cpu_initialization,
            init_model_with_meta_device=init_model_with_meta_device,
            pre_wrap_hook=final_pre_wrap_hook,
        )

        if final_post_wrap_hook:
            _model = final_post_wrap_hook(model)
            if _model is not None:
                model = _model

        return model

    def initialize_model_parallel(
        self, seed: int | None = None, seed_kwargs: dict | None = None, **model_parallel_kwargs
    ) -> None:
        """Initializes model parallelism and sets the random seed.

        This is a convenience method that sets up tensor, pipeline, and other
        forms of model parallelism based on the attributes of the provider instance.

        Args:
            seed: The random seed for model parallel RNG.
            seed_kwargs: Additional arguments for `model_parallel_cuda_manual_seed`.
            **model_parallel_kwargs: Additional arguments for `parallel_state.initialize_model_parallel`.
        """
        if not paddle.distributed.is_initialized():
            paddle.cuda.set_device(dist.get_rank())
            paddle.distributed.init_process_group("nccl")

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=getattr(self, "tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=getattr(self, "pipeline_model_parallel_size", 1),
            virtual_pipeline_model_parallel_size=getattr(self, "virtual_pipeline_model_parallel_size", None),
            context_parallel_size=getattr(self, "context_parallel_size", 1) or 1,
            expert_model_parallel_size=getattr(self, "expert_model_parallel_size", 1) or 1,
            expert_tensor_parallel_size=getattr(self, "expert_tensor_parallel_size", None),
            **model_parallel_kwargs,
        )
        # TODO(pkuzyc): Support model_parallel_cuda_manual_seed for PaddleFleet
        # if seed is not None:
        #    model_parallel_cuda_manual_seed(seed, **(seed_kwargs or {}))

    @property
    def meta_model(self) -> list[ModelT]:
        """Returns the model instantiated on the meta device for inspection.

        This is useful for examining the model architecture without allocating
        GPU memory.
        """
        return self(wrap_with_ddp=False, init_model_with_meta_device=True)

    @property
    def pre_wrap_hook(self) -> Callable[[list[FleetLayer]], list[FleetLayer]] | None:
        """A composed callable of all registered pre-wrap hooks.

        This read-only property returns a single function that executes all registered
        pre-wrap hooks in order. The hook is applied before the model is passed to the DDP
        wrapper and can be used for tasks like freezing layers or altering model structure.

        Use `register_pre_wrap_hook` to add a hook to the execution chain.

        Returns:
            A callable that executes all registered pre-wrap hooks in order, or None if no
            hooks are registered.
        """
        if not hasattr(self, "_pre_wrap_hooks") or not self._pre_wrap_hooks:
            return None

        def composed_hook(model: list[FleetLayer]) -> list[FleetLayer]:
            for hook in self._pre_wrap_hooks:
                model = hook(model)
            return model

        return composed_hook

    def register_pre_wrap_hook(
        self,
        hook: Callable[[list[FleetLayer]], list[FleetLayer]],
        prepend: bool = False,
    ) -> None:
        """Registers a hook to be executed before the model is wrapped.

        The hook should be a callable that accepts a list of `FleetLayer` instances
        and returns a (potentially modified) list of `FleetLayer` instances.

        Args:
            hook: The hook to register.
            prepend: If True, the hook is inserted at the beginning of the execution
                chain. Otherwise, it is appended to the end.
        """
        if not hasattr(self, "_pre_wrap_hooks"):
            self._pre_wrap_hooks = []
        if prepend:
            self._pre_wrap_hooks.insert(0, hook)
        else:
            self._pre_wrap_hooks.append(hook)

    @property
    def post_wrap_hook(self) -> Callable[[list[FleetLayer]], list[FleetLayer]] | None:
        """A composed callable of all registered post-wrap hooks.

        This read-only property returns a single function that executes all registered
        post-wrap hooks in order. The hook is applied after the model has been wrapped by
        DDP and is useful for tasks like logging or attaching custom attributes.

        Use `register_post_wrap_hook` to add a hook to the execution chain.

        Returns:
            A callable that executes all registered post-wrap hooks in order, or None if no
            hooks are registered.
        """
        if not hasattr(self, "_post_wrap_hooks") or not self._post_wrap_hooks:
            return None

        def composed_hook(model: list[FleetLayer]) -> list[FleetLayer]:
            for hook in self._post_wrap_hooks:
                model = hook(model)
            return model

        return composed_hook

    def register_post_wrap_hook(
        self,
        hook: Callable[[list[FleetLayer]], list[FleetLayer]],
        prepend: bool = False,
    ) -> None:
        """Registers a hook to be executed after the model is wrapped.

        The hook should be a callable that accepts a list of `FleetLayer` instances
        and returns a (potentially modified) list of `FleetLayer` instances.

        Args:
            hook: The hook to register.
            prepend: If True, the hook is inserted at the beginning of the execution
                chain. Otherwise, it is appended to the end.
        """
        if not hasattr(self, "_post_wrap_hooks"):
            self._post_wrap_hooks = []
        if prepend:
            self._post_wrap_hooks.insert(0, hook)
        else:
            self._post_wrap_hooks.append(hook)


def get_model(
    model_provider: ModelProviderMixin,
    ddp_config: DistributedDataParallelConfig,
    model_type=ModelType.encoder_or_decoder,
    overlap_param_gather_with_optimizer_step: bool = False,
    fp16: bool | None = None,
    bf16: bool | None = None,
    use_paddle_fsdp: bool = False,
    use_paddle_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = True,
    use_cpu_initialization: None | bool = False,
    init_model_with_meta_device: bool | None = None,
    pre_wrap_hook: Union[
        Callable[[list[FleetLayer]], list[FleetLayer]],
        list[Callable[[list[FleetLayer]], list[FleetLayer]]],
    ]
    | None = None,
) -> list[FleetLayer]:
    """Create and configure a model for distributed training.

    This function handles the complete model creation pipeline including:
    - Model instantiation with proper pipeline parallel configuration
    - GPU memory allocation
    - Mixed precision (FP16/BF16) wrapping
    - Float8 tensor correction
    - Distributed Data Parallel (DDP) wrapping

    Args:
        model_provider: ModelProviderMixin instance that creates the model.
            Uses the provide() method with optional pre_process(bool), post_process(bool),
            vp_stage(int) arguments for pipeline parallelism
        ddp_config: Configuration for distributed data parallel training
        model_type: Type of model (encoder, decoder, or encoder_and_decoder)
        overlap_param_gather_with_optimizer_step: Whether to overlap parameter
            gathering with optimizer step for performance optimization
        fp16: Enable FP16 mixed precision training. If None, uses model config
        bf16: Enable BF16 mixed precision training. If None, uses model config
        use_paddle_fsdp: Use Megatron's Fully Sharded Data Parallel
        use_paddle_fsdp2: Use PyPaddle's Fully Sharded Data Parallel v2
        wrap_with_ddp: Whether to wrap the model with DDP
        data_parallel_random_init: Whether to use random initialization for
            data parallel ranks (vs broadcasting from rank 0)
        use_cpu_initialization: Whether to initialize model on CPU to save GPU memory
        init_model_with_meta_device: Whether to initialize the model on the meta device
        pre_wrap_hook: A callable or list of callables that takes a list of `FleetLayer`
            and returns a modified list, or `None` to clear the hook. If a list is provided,
            hooks will be executed in order.

    Returns:
        list[FleetLayer]: List of model modules. Contains multiple modules
            when using virtual pipeline parallelism, otherwise a single module
    """
    if fp16:
        model_provider.fp16 = fp16
    if bf16:
        model_provider.bf16 = bf16

    model_provider.use_cpu_initialization = use_cpu_initialization if use_cpu_initialization else False
    if init_model_with_meta_device:
        model_provider.init_model_with_meta_device = True
        with paddle.device("meta"):
            model = _create_model(model_provider, model_type)
    else:
        model = _create_model(model_provider, model_type)

    if pre_wrap_hook:
        if isinstance(pre_wrap_hook, list):
            # Execute hooks in order
            for hook in pre_wrap_hook:
                if not callable(hook):
                    raise RuntimeError("All elements in pre_wrap_hook list must be callable")
                _model = hook(model)
                if _model is not None:
                    model = _model
        else:
            if not callable(pre_wrap_hook):
                raise RuntimeError("pre_wrap_hook must be a callable or a list of callables")
            _model = pre_wrap_hook(model)
            if _model is not None:
                model = _model

    # Set tensor model parallel attributes if not set
    # In case pre_wrap_hook augmented the model (e.g. adding PEFT adapters)
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    _print_num_params(model)

    model_config = get_model_config(model[0])

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if (
        not use_paddle_fsdp2
        and not model_config.use_cpu_initialization
        and not model_config.init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(paddle.cuda.current_device())

    # if (model_config.fp16 or model_config.bf16) and mixed_precision_wrapper is not None:
    #    model = [mixed_precision_wrapper(model_config, model_module) for model_module in model]

    if correct_amax_history_if_needed is not None:
        correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        raise NotImplementedError("DDP wrapping is not yet implemented")
        """
        model = _ddp_wrap(
            model,
            data_parallel_random_init,
            ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_paddle_fsdp=use_paddle_fsdp,
            use_paddle_fsdp2=use_paddle_fsdp2,
        )
        """

    return model


def _create_model(
    model_provider: ModelProviderMixin,
    model_type: ModelType,
) -> list[FleetLayer]:
    """Create model instances with appropriate pipeline parallel configuration.

    Handles virtual pipeline parallelism (VPP) by creating multiple model
    instances when needed. Sets pre_process and post_process flags based on
    pipeline parallel rank.

    Args:
        model_provider: ModelProviderMixin instance that creates the model
        model_type: ModelType enum indicating encoder, decoder, or both

    Returns:
        list: List of model instances. Multiple instances for VPP, otherwise single
    """

    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert (
            model_type != ModelType.encoder_and_decoder
        ), "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
            pre_process = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
            post_process = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
            this_model = model_provider.provide(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=i,
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = parallel_state.is_pipeline_first_stage()
        post_process = parallel_state.is_pipeline_last_stage()
        if model_type == ModelType.encoder_and_decoder:
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                rank = parallel_state.get_pipeline_model_parallel_rank()
                first_decoder_rank = parallel_state.get_pipeline_model_parallel_decoder_start()
                world_size = parallel_state.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == first_decoder_rank
                post_process = (rank == (first_decoder_rank - 1)) or (rank == (world_size - 1))
            model = model_provider.provide()
        else:
            model = model_provider.provide(
                pre_process=pre_process,
                post_process=post_process,
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    return model


def _print_num_params(model: list[FleetLayer]) -> None:
    """Print the number of parameters in the model on rank 0.

    Only prints on data parallel rank 0 to avoid duplicate output.
    Shows parameter count per (tensor parallel, pipeline parallel) rank.

    Args:
        model: List of model modules to count parameters from
    """
    if parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_context_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )
