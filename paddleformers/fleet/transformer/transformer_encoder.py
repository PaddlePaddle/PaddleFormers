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

# Referred to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging

from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    ScheduleChunk,
    SharedLayerDesc,
)

from ..pipeline_parallel.pp_utils.utils import dict_to_tuple_helper
from .transformer_layer import (
    TransformerLayer,
    TransformerLayerNode,
    TransformerLayerOverlappedScheduleNode,
)

logger = logging.getLogger(__name__)


def build_overlapped_nodes(forward_chunk, backward_chunk):
    """Build overlapped nodes for TransformerLayer."""
    overlap_element_class = TransformerLayerNode
    forward_decoder_layer_num = 0
    backward_decoder_layer_num = 0

    assert isinstance(forward_chunk, ScheduleChunk) and isinstance(backward_chunk, ScheduleChunk)
    for n in forward_chunk.nodes:
        if isinstance(n, overlap_element_class):
            forward_decoder_layer_num += 1
    for n in reversed(backward_chunk.nodes):
        if isinstance(n, overlap_element_class):
            backward_decoder_layer_num += 1

    overlap_layers_num = min(forward_decoder_layer_num, backward_decoder_layer_num)

    # construct forward pre- and post-chunks
    forward_pre_layers = []
    forward_post_layers = []
    forward_overlap_layers = []
    is_pre = True
    for n in forward_chunk.nodes:
        if not isinstance(n, overlap_element_class):
            if is_pre:
                forward_pre_layers.append(n)
            else:
                forward_post_layers.append(n)
        else:
            is_pre = False
            if len(forward_overlap_layers) == overlap_layers_num:
                forward_post_layers.append(n)
            else:
                forward_overlap_layers.append(n)

    forward_pre_node = ScheduleChunk(forward_pre_layers)
    forward_post_node = ScheduleChunk(forward_post_layers)

    # construct backward pre- and post-chunks
    backward_pre_layers = []
    backward_post_layers = []
    backward_overlap_layers = []
    is_pre = True
    for n in reversed(backward_chunk.nodes):
        if not isinstance(n, overlap_element_class):
            if is_pre:
                backward_pre_layers.append(n)
            else:
                backward_post_layers.append(n)
        else:
            is_pre = False
            if len(backward_overlap_layers) == overlap_layers_num:
                backward_post_layers.append(n)
            else:
                backward_overlap_layers.append(n)

    backward_pre_node = ScheduleChunk(list(reversed(backward_pre_layers)))
    backward_post_node = ScheduleChunk(list(reversed(backward_post_layers)))

    # construct overlap chunk
    overlap_node = ScheduleChunk(
        [
            TransformerLayerOverlappedScheduleNode(forward_node, backward_node)
            for forward_node, backward_node in zip(forward_overlap_layers, backward_overlap_layers)
        ]
    )
    return (
        forward_pre_node,
        backward_pre_node,
        overlap_node,
        forward_post_node,
        backward_post_node,
    )


class TransformerEncoder(PipelineLayer):
    """Transformer Encoder Model.

    Args:
        gpt_layer_desc:
    """

    def __init__(
        self,
        sublayers_spec,
        **kwargs,
    ) -> None:
        self.config = kwargs["config"]
        self.modal = kwargs.pop("modal", None)
        self._pipeline_name_mapping = None
        self._pp_to_single_mapping = None
        self._sequential_layers = self.get_layer_desc_list(
            sublayers_spec,
        )
        self.layers = self.get_sequential_layers()
        del kwargs["config"]

        topology = (
            None if self.config.pipeline_model_parallel_size == 1 else fleet.get_hybrid_communicate_group().topology()
        )

        super().__init__(
            layers=self.layers,
            topology=topology,
            num_virtual_pipeline_stages=self.config.virtual_pipeline_model_parallel_size,
            **kwargs,
        )

    def get_layer_desc_list(self, spec):
        layers = []
        if self.modal:
            name_prefix = f"model.{self.modal}"
        else:
            name_prefix = "model"
        self.add_sequential_layer(layers, LayerDesc(spec.embedding), name_prefix)

        self.get_encoder_layer_desc_list(layers, spec, name_prefix)

        self.add_sequential_layer(layers, (spec.layer_norm), name_prefix)

        return layers

    def get_encoder_layer_desc_list(self, layers, spec, name_prefix):
        i = 0
        for head_empty_layer in spec.head_empty_layers:
            self.add_sequential_layer(layers, LayerDesc(head_empty_layer), f"{name_prefix}.layers.{i}")
            i += 1
        for transformer_layer_spec in spec.transformer_layers:
            self.add_sequential_layer(
                layers,
                LayerDesc(transformer_layer_spec),
                f"{name_prefix}.layers.{i}",
            )
            i += 1
        for tail_empty_layer in spec.tail_empty_layers:
            self.add_sequential_layer(layers, LayerDesc(tail_empty_layer), f"{name_prefix}.layers.{i}")
            i += 1

    def overlapped_forward_backward(
        self,
        forward_chunk,
        forward_inputs,
        forward_loss_fn_node,
        backward_chunk,
        backward_loss_fn_node,
        backward_input_grads,
        scaler,
        p2p_async_handle,
    ):
        if backward_loss_fn_node is not None:
            if scaler:
                backward_input_grads = backward_loss_fn_node.backward(scaler=scaler)
            else:
                backward_input_grads = backward_loss_fn_node.backward()

        (
            forward_pre_node,
            backward_pre_node,
            overlap_node,
            forward_post_node,
            backward_post_node,
        ) = build_overlapped_nodes(forward_chunk, backward_chunk)

        if len(overlap_node.nodes) > 0:
            assert not any(isinstance(node, TransformerLayerNode) for node in overlap_node.nodes)
            # origin assert, why ?
            # assert not any(
            #     isinstance(node, TransformerLayerNode)
            #     for node in forward_post_node.nodes
            # )
            # assert not any(
            #     isinstance(node, TransformerLayerNode)
            #     for node in backward_post_node.nodes
            # )

        if p2p_async_handle is not None:
            p2p_async_handle.forward_handle_wait()
            p2p_async_handle.backward_handle_wait()

        forward_inputs = forward_pre_node.forward(forward_inputs)
        backward_input_grads = backward_pre_node.backward(backward_input_grads)

        for i, node in enumerate(overlap_node.nodes):
            forward_inputs, backward_input_grads = node.forward_backward(
                forward_inputs,
                backward_input_grads,
                # split_bw=(i == len(overlap_node.nodes) - 1),
            )

        forward_inputs = forward_post_node.forward(forward_inputs)
        backward_input_grads = backward_post_node.backward(backward_input_grads)

        # forward_inputs = forward_chunk.forward(forward_inputs)

        if p2p_async_handle is not None:
            forward_inputs = dict_to_tuple_helper(forward_inputs)
            p2p_async_handle.forward_async_comm(forward_inputs)
            p2p_async_handle.backward_async_comm(backward_input_grads)

        # backward_input_grads = backward_chunk.backward(backward_input_grads)

        # used for bw split
        # if len(overlap_node.nodes) > 0:
        #     WeightGradStore.pop()
        #     assert WeightGradStore.funcs_queue.empty()

        if forward_loss_fn_node is not None:
            forward_loss = forward_loss_fn_node.forward(forward_inputs)
        else:
            forward_loss = None

        return forward_inputs, forward_loss, backward_input_grads

    def get_hardware_flops(self):
        return 989e3

    def add_sequential_layer(self, layers, layer_desc, name_prefix=""):
        """
        Add a sequential layer to the network with specified description and name prefix.

        Args:
            layers (list): List to store layer descriptions. Each element should be a dict
                with keys "layer" (LayerDesc) and "name_prefix" (str).
            layer_desc (LayerDesc|SharedLayerDesc): Layer description object containing
                layer self.configuration.
            name_prefix (str, optional): Prefix for layer names in the pipeline.
                Defaults to empty string.

        Returns:
            None: The layer description is appended to the input layers list.
        """
        layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layers(self):
        """
        Get all layers in the sequential network.

        Returns:
            List[paddle.nn.Layer]: List containing all layers.
        """
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixes(self):
        """
        Retrieve name prefixes for all parallel layers in the sequential network.

        Returns:
            Dict[str, str]: A dictionary mapping layer indices (as strings) to their
                corresponding name prefixes. The indices represent the position of
                each layer in the sequential order.
        """
        return {str(index): x["name_prefix"] for index, x in enumerate(self._sequential_layers)}

    def get_shardlayer_prefix(self, name_splited):
        """_summary_
            This function retrieves the prefix of a shared layer. The process involves:
            1. Identifying all key names of shared layers, like 'shared_weight01', 'shared_weight02', etc.
            2. For instance, given name_splited = ['shared_layers', 'shared_weight01', 'weight'],
                the 'shared_layer_key' would be name_splited[1], which is 'shared_weight01'.
            3. By traversing through all layers, the function checks if the specified
                shared_layer is present in the current stage. If found, it returns the corresponding prefix.

            Note: For retrieving all SharedLayer instances in Paddle, you can refer to the following Paddle code.
            https://github.com/PaddlePaddle/Paddle/blob/2cf724d055679a1a0e48766dfb1708b920273078/python/paddle/distributed/fleet/meta_parallel/parallel_layers/pp_layers.py#L460-L513
        Args:
            name_splited (_type_): _description_

        Returns:
            _type_: _description_
        """
        shared_layer_names = {s.layer_name for s in self.layers if isinstance(s, SharedLayerDesc)}
        assert name_splited[1] in shared_layer_names, f"The shared layer name {name_splited[1]} must be in prefixes!"
        shared_layer_key = name_splited[1]
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, SharedLayerDesc) and layer.layer_name == shared_layer_key:
                if self.get_stage_from_index(idx) == self._stage_id:
                    return self.get_sequential_name_prefixes()[str(idx)]

        # the prefix must be in the current stage, else raise error
        raise ValueError(f"The shared layer {shared_layer_key} must be in the current stage!")

    def _set_pipeline_name_mapping(self, mappings=None):
        """
        Set the name mapping for pipeline.

        Args:
            mappings (dict, optional): Dictionary storing name mapping relationships. Default is None, meaning no mapping operation.

        Returns:
            dict: Returns the updated or existing mapping relationship.

        """
        if mappings is not None:
            self._pipeline_name_mapping = mappings
        else:
            single_to_pp_mapping = {}
            pp_to_single_mapping = {}

            state_dict_keys = list(super().state_dict().keys())
            first_key = ""
            for k in state_dict_keys:
                if "shared_layers" not in k:
                    first_key = k
                    break
            first_key = first_key.split(".")
            # if use virtual pp_degree, the prefix is like 0.0.xxx
            # else it will be like 0.xxx
            use_virtual_pp_degree = first_key[0].isdigit() and first_key[1].isdigit()

            prefixes = self.get_sequential_name_prefixes()
            for k in state_dict_keys:
                name_splited = k.split(".")
                if use_virtual_pp_degree:
                    if name_splited[0].isdigit():
                        if name_splited[1].isdigit():
                            idx = str(int(name_splited[0]) + int(name_splited[1]))
                            single_name = [prefixes[idx]]
                            single_name.extend(name_splited[2:])
                        else:
                            single_name = [prefixes[str(len(prefixes) - 1)]]
                            single_name.extend(name_splited[2:])
                            logger.warning(
                                f"Please check! we treat this key as last layer, get {k}, \
                                        set origin name as {'.'.join(single_name)}"
                            )
                    elif name_splited[0] == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue
                else:
                    idx = name_splited[0]
                    # for normal pp layer
                    if idx.isdigit():
                        # allow empty prefix
                        single_name = [] if prefixes[idx] == "" else [prefixes[idx]]
                        single_name.extend(name_splited[1:])
                    elif idx == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue

                single_to_pp_mapping[".".join(single_name)] = k
                pp_to_single_mapping[k] = ".".join(single_name)

            self._pipeline_name_mapping = single_to_pp_mapping
            self._pp_to_single_mapping = pp_to_single_mapping

        return self._pipeline_name_mapping

    def state_dict(self, *args, **kwargs):
        """
        Return a dictionary with Pipeline Stage mapping.
        Args:
            *args (tuple): Variable argument list passed to parent method.
            **kwargs (dict): Optional keyword arguments passed to parent method.
        Returns:
            dict: Dictionary containing Pipeline Stage mapping.
        """
        state_dict = super().state_dict(*args, **kwargs)

        if "qwen3_vl" in getattr(self.config, "model_type", ""):
            name_prefix = "model.language_model."
        else:
            name_prefix = ""
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        # assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if name_prefix and k.startswith(name_prefix):
                k = k[len(name_prefix) :]
            if k not in self._pp_to_single_mapping:
                state_dict[k] = v
                continue
            v.key = self._pp_to_single_mapping[k]
            state_dict[self._pp_to_single_mapping[k]] = v
        return state_dict

    def set_state_dict(self, state_dict, *args, **kwargs):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k not in self._pipeline_name_mapping:
                continue
            state_dict[self._pipeline_name_mapping[k]] = v

        ret = super().set_state_dict(state_dict, *args, **kwargs)
        return ret

    def _check_shared_model_state(self):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        super_state_dict = super().state_dict()
        structure_name_to_tensor = {}
        for k, v in super_state_dict.items():
            k = self._pp_to_single_mapping[k]
            if k not in structure_name_to_tensor:
                structure_name_to_tensor[k] = v
            else:
                old_v = structure_name_to_tensor[k]
                assert old_v is v, f"Shared tensor with different structure name: {k}"

        missing_shared_keys = {}
        for k, v in self._pp_to_single_mapping.items():
            mapped_k = self._pipeline_name_mapping[v]
            if k != mapped_k:
                missing_shared_keys[k] = mapped_k
        return missing_shared_keys

    def sharded_state_dict(self, *args, **kwargs):
        """
        sharded_state_dict method for PipelinePretrainedModel.

        Remaps parameter keys according to the pipeline stage mapping, and converts expert indices from local to global.
        """
        sharded_state_dict = super().sharded_state_dict(*args, **kwargs)
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        if "qwen3_vl" in getattr(self.config, "model_type", ""):
            name_prefix = "model.language_model."
        else:
            name_prefix = ""

        for k in list(sharded_state_dict.keys()):
            v = sharded_state_dict.pop(k)
            # remove name_prefix
            if name_prefix and k.startswith(name_prefix):
                k = k[len(name_prefix) :]
            if k not in self._pp_to_single_mapping:
                sharded_state_dict[k] = v
                continue
            v.key = self._pp_to_single_mapping[k]
            sharded_state_dict[self._pp_to_single_mapping[k]] = v

        def increment_expert_number(s, increment):
            import re

            def replace(match):
                original_number = int(match.group(0))
                new_number = original_number + increment
                return str(new_number)

            return re.sub(r"(?<=experts\.)\d+", replace, s)

        renamed_sharded_state_dict = {}
        for k, v in sharded_state_dict.items():
            global_expert_id_offset = getattr(v, "global_expert_id_offset", None)
            layer_cnt = getattr(v, "layer_cnt", None)
            if global_expert_id_offset is not None:
                new_key = increment_expert_number(k, global_expert_id_offset)
                v.key = new_key
                delattr(v, "global_expert_id_offset")
                renamed_sharded_state_dict[new_key] = v
            elif layer_cnt is not None:
                new_key = k + "_layer_" + str(layer_cnt)
                v.key = new_key
                delattr(v, "layer_cnt")
                renamed_sharded_state_dict[new_key] = v
            else:
                renamed_sharded_state_dict[k] = v

        return renamed_sharded_state_dict

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        if self._num_virtual_pipeline_stages > 1:
            for idx, chunk in enumerate(self._model_chunks):
                for idx, layer in enumerate(chunk):
                    if isinstance(layer, TransformerLayer):
                        layer.fp8_quant_weight(
                            batch_mode=batch_mode,
                            quant_transpose=quant_transpose,
                        )
        else:
            for idx, layer in enumerate(self.run_function):
                if isinstance(layer, TransformerLayer):
                    layer.fp8_quant_weight(batch_mode=batch_mode, quant_transpose=quant_transpose)

    def use_fp8(self):
        if self._num_virtual_pipeline_stages > 1:
            for idx, chunk in enumerate(self._model_chunks):
                for idx, layer in enumerate(chunk):
                    if isinstance(layer, TransformerLayer) and layer.use_fp8():
                        return True
        else:
            for idx, layer in enumerate(self.run_function):
                if isinstance(layer, TransformerLayer) and layer.use_fp8():
                    return True
            return False
