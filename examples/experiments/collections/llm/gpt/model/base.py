import contextlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Literal, Union
import paddle
import paddle.distributed
from paddle import nn
import paddle.nn.functional as F

from paddlefleet import parallel_state as ps
from paddlefleet.transformer.enums import ModelType,AttnMaskType
from paddlefleet.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from paddlefleet.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from paddlefleet.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from paddlefleet.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel

from paddlefleet.tensor_parallel.layers import ColumnParallelLinear,RowParallelLinear
from paddlefleet.spec_utils import LayerSpec
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.mlp import MLP,MLPSublayersSpec


def gpt_forward_step(model, batch) -> paddle.Tensor:
    """Execute a forward step for the GPT model.

    This function prepares the arguments needed for the model's forward pass
    and handles both normal and packed sequence processing.

    Args:
        model: The GPT model
        batch: The input batch containing tokens, positions, and other required inputs

    Returns:
        torch.Tensor: Output tensor from the model forward pass
    """
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
    }

    if "attention_mask" not in batch:
        assert (
            HAVE_TE
        ), "The dataloader did not provide an attention mask, however Transformer Engine was not detected. \
            This requires Transformer Engine's implementation of fused or flash attention."
    else:
        forward_args["attention_mask"] = batch["attention_mask"]

    return model(**forward_args)

def default_layer_spec(config: "intermediate_size", vp_stage: Optional[int] = None) -> LayerSpec:
    """Determine the most appropriate layer specification based on availability.

    Uses Transformer Engine specs if available, otherwise falls back to local implementation.

    Args:
        config: GPT configuration object

    Returns:
        LayerSpec: The selected module specification
    """
    from paddlefleet.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_local_spec(
        num_experts=config.n_routed_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_qk_norm=config.use_qk_norm,
        normalization=config.normalization,
    )


def mtp_block_spec(config, vp_stage: Optional[int] = None) -> Optional[LayerSpec]:
    """Pass in the MTP block spec if model has MTP layers.

    Args:
        config: GPT configuration object

    Returns:
        LayerSpec: The MTP module specification
    """
    if getattr(config, "mtp_num_layers", None):
        from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

        if isinstance(config.transformer_layer_spec, Callable):
            if "vp_stage" in inspect.signature(config.transformer_layer_spec).parameters:
                spec = config.transformer_layer_spec(config, vp_stage=vp_stage)
            else:
                spec = config.transformer_layer_spec(config)
        else:
            spec = config.transformer_layer_spec
        if hasattr(spec, "layer_specs") and len(spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlocksublayers_spec) when using MoE.
            spec = local_layer_spec(config)
        return get_gpt_mtp_block_spec(config, spec, vp_stage=vp_stage)
    else:
        return None


@dataclass
class GPTConfig(TransformerConfig):
    """Configuration class for GPT models.

    Extends TransformerConfig with additional parameters specific to GPT models
    and provides utility methods for model configuration.
    """

    # From paddlefleet.models.gpt.gpt_model.GPTModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024
    attention_softmax_in_fp32: bool = False
    masked_softmax_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    gradient_accumulation_fusion: bool = False
    deallocate_pipeline_outputs: bool = True
    scatter_embedding_sequence_parallel: bool = False
    tp_only_amax_red: bool = False

    use_transformer_engine_full_layer_spec: bool = False
    transformer_layer_spec: Union[LayerSpec, Callable[["GPTConfig"], LayerSpec]] = default_layer_spec

    forward_step_fn: Callable = gpt_forward_step
    generation_config: Optional["GenerationConfig"] = None

    vocab_size: Optional[int] = None
    tp_comm_overlap_cfg = None

    def configure_model(self, tokenizer, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core GPT model based on this configuration.

        Args:
            tokenizer: Tokenizer used with the model
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """

        vp_size = self.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = getattr(self, "account_for_embedding_in_pipeline_split", False) or getattr(
            self, "account_for_loss_in_pipeline_split", False
        )
        is_pipeline_asymmetric |= (
            getattr(self, "num_layers_in_first_pipeline_stage", None)
            or getattr(self, "num_layers_in_last_pipeline_stage", None)
        ) is not None
        is_flexible_pp_layout = is_pipeline_asymmetric or (
            getattr(self, "pipeline_model_parallel_layout", None) is not None
        )
        if vp_size and not is_flexible_pp_layout:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_hidden_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        import inspect

        from paddlefleet import parallel_state

        # During fake lightning initialization, pass 0 to bypass the assertion that vp_stage must be
        # non-None when using virtual pipeline model parallelism
        vp_stage = vp_stage or 0

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, LayerSpec):
            # Check if the transformer_layer_spec function accepts vp_stage parameter
            if 'vp_stage' in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        if self.vocab_size is not None:
            vocab_size = self.vocab_size
            if tokenizer is not None:
                logging.info(
                    f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                    f" {vocab_size - tokenizer.vocab_size}."
                )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)
        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(paddle.device, device='meta')

        if 'mtp_block_spec' in inspect.signature(MCoreGPTModel.__init__).parameters:
            kwargs = {"mtp_block_spec": mtp_block_spec(self, vp_stage=vp_stage)}
        else:
            kwargs = {}

        # if self.attention_backend == AttnBackend.local:
        #     if hasattr(transformer_layer_spec, 'sublayers_spec'):
        #         transformer_layer_spec.sublayers_spec.self_attention.sublayers_spec.core_attention = MCoreDotProductAttention
        with model_init_device_context():
            model = MCoreGPTModel(
                self,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=vocab_size,
                max_sequence_length=self.seq_length,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process
                or parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
                post_process=post_process
                or parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
                vp_stage=vp_stage,
                **kwargs,
            )

        return model
