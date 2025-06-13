# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

from paddle.distributed.fleet.model import PipelineParallel
from paddle.distributed.fleet.utils.log_util import logger

_GLOBAL_EXTRACT_LAYER_NAME_FUNC = None


def regitser_extract_layer_name_func(func):
    global _GLOBAL_EXTRACT_LAYER_NAME_FUNC
    _GLOBAL_EXTRACT_LAYER_NAME_FUNC = func


def get_extract_layer_name_func():
    global _GLOBAL_EXTRACT_LAYER_NAME_FUNC
    assert _GLOBAL_EXTRACT_LAYER_NAME_FUNC is not None, "extract layer func is not registered yet"
    return _GLOBAL_EXTRACT_LAYER_NAME_FUNC


_GLOBAL_INDEX_LAYER_FUNC = None


def register_index_layer_func(func):
    global _GLOBAL_INDEX_LAYER_FUNC
    _GLOBAL_INDEX_LAYER_FUNC = func


def get_index_layer_func():
    global _GLOBAL_INDEX_LAYER_FUNC
    assert _GLOBAL_INDEX_LAYER_FUNC is not None, "index layer func is not registered yet"
    return _GLOBAL_INDEX_LAYER_FUNC


_GLOBAL_SNAME_TO_TNAME_FUNC = None


def register_sname_to_tname_func(func):
    global _GLOBAL_SNAME_TO_TNAME_FUNC
    _GLOBAL_SNAME_TO_TNAME_FUNC = func


def has_register_sname_to_tname_func():
    global _GLOBAL_SNAME_TO_TNAME_FUNC
    return _GLOBAL_SNAME_TO_TNAME_FUNC is not None


def get_sname_to_tname_func():
    global _GLOBAL_SNAME_TO_TNAME_FUNC
    assert _GLOBAL_SNAME_TO_TNAME_FUNC is not None, "sname to tname func is not registered yet"
    return _GLOBAL_SNAME_TO_TNAME_FUNC


class LayerNameScope:
    """
    layer name scope for a layer, layer name of the same kind of layer will be named consecutively
    """

    registered_layers = []

    def __init__(self, prefix, template):
        self.prefix = prefix
        self.last_layer_id = ""
        self.last_old_layer_name = ""
        self.template = template
        self.index = -1
        self.sub_scopes = OrderedDict()

    @classmethod
    def get_layer_prefix(cls, old_layer_name):
        for k in cls.registered_layers:
            if old_layer_name.startswith(k):
                return k
        return None

    @classmethod
    def register_layer_prefix(cls, prefix):
        if prefix not in cls.registered_layers:
            cls.registered_layers.append(prefix)
            cls.registered_layers.sort(key=lambda x: len(x), reverse=True)

    def get_next_scope(self, layer_id, old_layer_name):
        if old_layer_name != self.last_old_layer_name or layer_id != self.last_layer_id:
            self.index = self.index + 1
            self.last_old_layer_name = old_layer_name
            self.last_layer_id = layer_id
            self.sub_scopes = OrderedDict()
        return self

    def get_layer_name(self):
        name = ""
        if self.template:
            name = self.template.format(self.index)
        if self.prefix:
            name = self.prefix + "_" + name
        return name

    def get_sub_scope(self, sub_layer_name):
        layer_prefix = self.get_layer_prefix(sub_layer_name)
        assert layer_prefix, f"{sub_layer_name} invalid, prefix {self.prefix}"
        if layer_prefix in self.sub_scopes:
            return self.sub_scopes[layer_prefix]
        layer_template = f"{layer_prefix}_{{}}"
        prefix = self.get_layer_name()
        scope = LayerNameScope(prefix, layer_template)
        self.sub_scopes[layer_prefix] = scope
        return scope


def register_layername_prefix(layer_name):
    LayerNameScope.register_layer_prefix(layer_name)


def extract_param_names_groupby_layer(
    meta,
    mp_rank=0,
):
    param_names_by_layer = OrderedDict()
    assert "parallel_config" in meta
    parallel_config = meta["parallel_config"]
    assert "pp_degree" in parallel_config
    pp_degree = int(parallel_config["pp_degree"])
    sharding_metas = meta["sharding_metas"]
    for pp_rank in range(pp_degree):
        suffix = f"tp{mp_rank:0>2d}_pp{pp_rank:0>2d}"
        assert suffix in sharding_metas
        assert "structure_name_mapping" in sharding_metas[suffix]
        name_mapping = sharding_metas[suffix]["structure_name_mapping"]
        for (k, v) in name_mapping.items():
            layer_name = get_extract_layer_name_func()(k)
            if layer_name not in param_names_by_layer:
                param_names_by_layer[layer_name] = []
            param_names_by_layer[layer_name].append((k, v))
    return param_names_by_layer


def build_pipeline_context(meta, pp_model):
    assert isinstance(pp_model, PipelineParallel), type(pp_model)
    layer_params = extract_param_names_groupby_layer(meta, 0)
    # 2、rename tensor names
    pipeline_context = PipeLineSegmentContext(
        pp_model,
        layer_params,
    )
    return pipeline_context


class LayerReNamingManager:
    def __init__(self):
        self.top_layer_name_scope = LayerNameScope(None, None)

    def get_new_layer_name(self, layer_id: str, old_name: str):
        name_scope = self.top_layer_name_scope.get_sub_scope(old_name).get_next_scope(layer_id, old_name)
        return name_scope.get_layer_name()

    def get_new_param_name(self, layer_id, old_name: str):
        names = old_name.split(".")
        layer_name = self.get_new_layer_name(layer_id, names[0])
        names[0] = layer_name
        return ".".join(names)


class PipeLinelayer:
    def __init__(self, layer_name, param_names):
        self._layer_name = layer_name

        # make sure name with the same sublayer type is ordered
        def sort_key(x):
            # assume param_name is of the type layer_type_{same_layer_index}.w_{weight_index}
            structure_name, param_name = x
            same_layer_index = param_name.split(".")[0].split("_")[-1]
            return int(same_layer_index)

        param_names = sorted(param_names, key=sort_key)
        self._params = OrderedDict()
        for (k, v) in param_names:
            self._params[k] = v

    @property
    def params(self):
        return self._params

    @property
    def name(self):
        return self._layer_name


class PipeLineSegment:
    def __init__(self, start_index, end_index):
        self._start_index = start_index
        self._end_index = end_index
        self._cur_index = start_index
        self._layers = OrderedDict()

    def add_layer(self, layer_name, param_names):
        assert self._cur_index < self._end_index
        layer = PipeLinelayer(layer_name, param_names)
        self._layers[layer_name] = layer
        self._cur_index = self._cur_index + 1

    @property
    def layers(self):
        assert self._cur_index <= self._end_index
        return self._layers


class PipeLineStage:
    def __init__(self):
        self._rename_mgr = LayerReNamingManager()
        # map segment start index to segment
        self._segments = OrderedDict()
        self._layer_to_segment = OrderedDict()
        self._param_to_tname = OrderedDict()
        self._wname_to_rname = OrderedDict()

    def add_segment(self, start_index, end_index):
        segment = PipeLineSegment(start_index, end_index)
        self._segments[start_index] = segment
        for i in range(start_index, end_index):
            self._layer_to_segment[i] = segment

    def add_layer(self, layer_index, layer_name, param_names):
        assert layer_index in self._layer_to_segment
        segment = self._layer_to_segment[layer_index]
        segment.add_layer(layer_name, param_names)

    def build_name_mapping(self, sname_to_tname=None):
        for (k, segment) in self._segments.items():
            for (i, layer) in segment.layers.items():
                for param in layer.params.items():
                    (param_name, tensor_name) = param
                    # map to a new name
                    n_name = self._rename_mgr.get_new_param_name(layer.name, tensor_name)
                    if sname_to_tname is not None:
                        if param_name in sname_to_tname.keys():
                            self._wname_to_rname[param_name] = sname_to_tname[param_name]
                    # logger.info(f"{param_name} {tensor_name}=>{n_name}")
                    self._param_to_tname[param_name] = (tensor_name, n_name)

    def map_name(self, param_name, t_name):
        assert param_name in self._param_to_tname
        tensor_name, n_name = self._param_to_tname[param_name]
        if param_name in self._wname_to_rname:
            n_name = self._wname_to_rname[param_name]
        assert tensor_name == t_name
        return n_name

    def print_name_mapping(self):
        for (name, mapping) in self._param_to_tname.items():
            logger.info(f"{name} mapping {mapping[0]} => {mapping[1]}\n")


# segment context for pp X sharding
class PipeLineSegmentContext:
    def __init__(
        self,
        pp_model,
        param_names_by_layer,
    ):
        self._pp_degree = pp_model._layers._num_stages
        self._vpp_degree = pp_model._layers._num_virtual_pipeline_stages
        self._segment_method = "layer"
        self._layers = list(param_names_by_layer.keys())
        self._pp_model = pp_model
        self._stages = []
        self._layer_index_to_stage = {}
        self._layer_name_to_index = {}
        self._layer_index_to_name = {}
        self._layer_name_to_stage = {}
        self._param_names_by_layer = param_names_by_layer

        self._index_layers()

        stage_segments = self._segment()
        if has_register_sname_to_tname_func():
            self._sname_to_tname = get_sname_to_tname_func()(pp_model)
        else:
            self._sname_to_tname = None

        for (i, stage_seg) in enumerate(stage_segments):
            pipe_stage = PipeLineStage()
            self._stages.append(pipe_stage)
            for seg in stage_seg:
                pipe_stage.add_segment(seg[0], seg[1])
                for j in range(*seg):
                    if j in self._layer_index_to_name:
                        layer_name = self._layer_index_to_name[j]
                        assert layer_name in self._param_names_by_layer
                        pipe_stage.add_layer(j, layer_name, self._param_names_by_layer[layer_name])
                    self._layer_index_to_stage[j] = i
                    self._layer_name_to_stage[layer_name] = i

        for stage in self._stages:
            stage.build_name_mapping(self._sname_to_tname)

    def _index_layers(self):
        for layer_name in self._param_names_by_layer.keys():
            index = get_index_layer_func()(layer_name)
            self._layer_name_to_index[layer_name] = index
            self._layer_index_to_name[index] = layer_name

    def _segment(self):
        index_segments = [[] for _ in range(self._pp_degree)]
        segment_parts = self._pp_model._layers.segment_parts
        for i in range(self._pp_model._layers._total_stages_with_virtual_stages):
            stage = i % self._pp_degree
            index_segments[stage].append((segment_parts[i], segment_parts[i + 1]))
        print(f"segment results {index_segments}")
        return index_segments

    def map_name(self, param_name, t_name):
        layer_name = get_extract_layer_name_func()(param_name)
        assert layer_name in self._layer_name_to_index
        layer_index = self._layer_name_to_index[layer_name]
        stage_index = self._layer_index_to_stage[layer_index]
        stage = self._stages[stage_index]
        return stage.map_name(param_name, t_name)

    def map_name_to_stage(self, name):
        layer_name = get_extract_layer_name_func()(name)
        assert layer_name in self._layer_name_to_index
        layer_index = self._layer_name_to_index[layer_name]
        stage_index = self._layer_index_to_stage[layer_index]
        return stage_index

    def print_name_mapping(self):
        for (i, stage) in enumerate(self._stages):
            print(f"{'='*30}stage {i} {'='*30}")
            stage.print_name_mapping()


def reshard(node_model_state, reshard_context, hcg):
    pp_degree = hcg.get_pipe_parallel_world_size()
    pp_rank = hcg.get_stage_id()
    group = hcg.get_pipe_parallel_group()

    # all gather
    def filter_func(name):
        names, rank = name
        stage_id = reshard_context.map_name_to_stage(names[0])
        assert stage_id < pp_degree
        return stage_id == pp_rank

    node_model_state.reshard(group, filter_func)

    def name_map_func(structure_name, p_name):
        map_name = reshard_context.map_name(structure_name, p_name)
        return map_name

    node_model_state.map_names(name_map_func)

    return node_model_state
