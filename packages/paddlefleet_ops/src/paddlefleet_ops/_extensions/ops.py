# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def filter_scores_grad(indices, topkscoresgrad):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("filter_scores_grad", indices, topkscoresgrad)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"Indices": indices, "TopkScoresGrad": topkscoresgrad}
        outs = {}
        outs_list = ["ProbsGrad"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("filter_scores_grad", **locals())

        outs["ProbsGrad"] = helper.create_variable(dtype="float32")
        helper.append_op(type="filter_scores_grad", inputs=ins, outputs=outs, attrs={})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def fused_swiglu_scale_bwd(x, scale, dout):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("fused_swiglu_scale_bwd", x, scale, dout)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"X": x, "Scale": scale, "DOut": dout}
        outs = {}
        outs_list = ["DX", "DScale"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fused_swiglu_scale_bwd", **locals())

        outs["DX"] = helper.create_variable(dtype="float32")
        outs["DScale"] = helper.create_variable(dtype="float32")
        helper.append_op(type="fused_swiglu_scale_bwd", inputs=ins, outputs=outs, attrs={})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def router_metadata(topkrouterindices, expertfrequencyoffset, k):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("router_metadata", topkrouterindices, expertfrequencyoffset, k)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"TopkRouterIndices": topkrouterindices, "ExpertFrequencyOffset": expertfrequencyoffset}
        outs = {}
        outs_list = [
            "PaddedExpertFrequencyOffset",
            "XGatherIdx",
            "SScatterIdxValid",
            "SReverseScatterIdxValid",
            "NumActivatedExpertPerTokenOffset",
        ]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("router_metadata", **locals())

        outs["PaddedExpertFrequencyOffset"] = helper.create_variable(dtype="float32")
        outs["XGatherIdx"] = helper.create_variable(dtype="float32")
        outs["SScatterIdxValid"] = helper.create_variable(dtype="float32")
        outs["SReverseScatterIdxValid"] = helper.create_variable(dtype="float32")
        outs["NumActivatedExpertPerTokenOffset"] = helper.create_variable(dtype="float32")
        helper.append_op(type="router_metadata", inputs=ins, outputs=outs, attrs={"K": k})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def tokens_unzip_stable(
    x,
    xscale,
    expert_routemap_topk,
    expert_prob_topk,
    topk,
    num_experts,
    tokens_per_expert,
    padding_multiplex,
    fill_output,
):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_unzip_stable",
            x,
            xscale,
            expert_routemap_topk,
            expert_prob_topk,
            topk,
            num_experts,
            tokens_per_expert,
            padding_multiplex,
            fill_output,
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {
            "X": x,
            "Xscale@OPTIONAL": xscale,
            "expert_routemap_topk": expert_routemap_topk,
            "expert_prob_topk": expert_prob_topk,
        }
        outs = {}
        outs_list = ["X_unzipped", "zipped_expertwise_rowmap", "token_prob_unzipped", "XScale_unzipped@OPTIONAL"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_unzip_stable", **locals())

        outs["X_unzipped"] = helper.create_variable(dtype="float32")
        outs["zipped_expertwise_rowmap"] = helper.create_variable(dtype="float32")
        outs["token_prob_unzipped"] = helper.create_variable(dtype="float32")
        outs["XScale_unzipped@OPTIONAL"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="tokens_unzip_stable",
            inputs=ins,
            outputs=outs,
            attrs={
                "topk": topk,
                "num_experts": num_experts,
                "tokens_per_expert": tokens_per_expert,
                "padding_multiplex": padding_multiplex,
                "fill_output": fill_output,
            },
        )
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def tokens_unzip_gather(x, x_scale, zipped_expertwise_rowmap, expert_id, tokens_per_expert, padding_multiplex):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_unzip_gather",
            x,
            x_scale,
            zipped_expertwise_rowmap,
            expert_id,
            tokens_per_expert,
            padding_multiplex,
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"x": x, "x_scale@OPTIONAL": x_scale, "zipped_expertwise_rowmap": zipped_expertwise_rowmap}
        outs = {}
        outs_list = ["x_unzipped", "x_scale_unzipped@OPTIONAL", "idx_unzipped"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_unzip_gather", **locals())

        outs["x_unzipped"] = helper.create_variable(dtype="float32")
        outs["x_scale_unzipped@OPTIONAL"] = helper.create_variable(dtype="float32")
        outs["idx_unzipped"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="tokens_unzip_gather",
            inputs=ins,
            outputs=outs,
            attrs={
                "expert_id": expert_id,
                "tokens_per_expert": tokens_per_expert,
                "padding_multiplex": padding_multiplex,
            },
        )
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def filter_scores(probs, indices):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("filter_scores", probs, indices)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"Probs": probs, "Indices": indices}
        outs = {}
        outs_list = ["TopkScores"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("filter_scores", **locals())

        outs["TopkScores"] = helper.create_variable(dtype="float32")
        helper.append_op(type="filter_scores", inputs=ins, outputs=outs, attrs={})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def tokens_zip_prob_seq_subbatch(unzipped_prob, zipped_expertwise_rowmap, dispatched_indices, subbatch_rows):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_zip_prob_seq_subbatch", unzipped_prob, zipped_expertwise_rowmap, dispatched_indices, subbatch_rows
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {
            "unzipped_prob@VECTOR": unzipped_prob,
            "zipped_expertwise_rowmap": zipped_expertwise_rowmap,
            "dispatched_indices": dispatched_indices,
        }
        outs = {}
        outs_list = ["zipped_prob"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_zip_prob_seq_subbatch", **locals())

        outs["zipped_prob"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="tokens_zip_prob_seq_subbatch", inputs=ins, outputs=outs, attrs={"subbatch_rows": subbatch_rows}
        )
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def fuse_weighted_swiglu_fp8_quant(expert_out_list, prob, using_pow2_scaling, use_ue8m0):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "fuse_weighted_swiglu_fp8_quant", expert_out_list, prob, using_pow2_scaling, use_ue8m0
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"expert_out_list": expert_out_list, "prob@OPTIONAL": prob}
        outs = {}
        outs_list = ["out", "scale"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fuse_weighted_swiglu_fp8_quant", **locals())

        outs["out"] = helper.create_variable(dtype="float32")
        outs["scale"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="fuse_weighted_swiglu_fp8_quant",
            inputs=ins,
            outputs=outs,
            attrs={"using_pow2_scaling": using_pow2_scaling, "use_ue8m0": use_ue8m0},
        )
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def tokens_unzip_slice(x, zipped_expertwise_rowmap, num_experts, total_unzipped_rows, start_idx, end_idx):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_unzip_slice", x, zipped_expertwise_rowmap, num_experts, total_unzipped_rows, start_idx, end_idx
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"x": x, "zipped_expertwise_rowmap": zipped_expertwise_rowmap}
        outs = {}
        outs_list = ["idx_unzipped"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_unzip_slice", **locals())

        outs["idx_unzipped"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="tokens_unzip_slice",
            inputs=ins,
            outputs=outs,
            attrs={
                "num_experts": num_experts,
                "total_unzipped_rows": total_unzipped_rows,
                "start_idx": start_idx,
                "end_idx": end_idx,
            },
        )
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def tokens_zip_unique_add(x_zipped, x_unzipped, idx_unzipped, zipped_rows):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("tokens_zip_unique_add", x_zipped, x_unzipped, idx_unzipped, zipped_rows)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"x_zipped": x_zipped, "x_unzipped": x_unzipped, "idx_unzipped": idx_unzipped}
        outs = {}
        outs_list = ["y_zipped"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_zip_unique_add", **locals())

        outs["y_zipped"] = helper.create_variable(dtype="float32")
        helper.append_op(type="tokens_zip_unique_add", inputs=ins, outputs=outs, attrs={"zipped_rows": zipped_rows})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def tokens_zip_prob(unzipped_prob, zipped_expertwise_rowmap, dispatched_indices):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("tokens_zip_prob", unzipped_prob, zipped_expertwise_rowmap, dispatched_indices)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {
            "unzipped_prob@VECTOR": unzipped_prob,
            "zipped_expertwise_rowmap": zipped_expertwise_rowmap,
            "dispatched_indices": dispatched_indices,
        }
        outs = {}
        outs_list = ["zipped_prob"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_zip_prob", **locals())

        outs["zipped_prob"] = helper.create_variable(dtype="float32")
        helper.append_op(type="tokens_zip_prob", inputs=ins, outputs=outs, attrs={})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def merge_subbatch_cast(x, dtype):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("merge_subbatch_cast", x, dtype)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"x@VECTOR": x}
        outs = {}
        outs_list = ["y"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("merge_subbatch_cast", **locals())

        outs["y"] = helper.create_variable(dtype="float32")
        helper.append_op(type="merge_subbatch_cast", inputs=ins, outputs=outs, attrs={"dtype": dtype})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def tokens_zip_unique_add_subbatch(x_zipped, x_unzipped, idx_unzipped, zipped_rows, subbatch_rows):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_zip_unique_add_subbatch", x_zipped, x_unzipped, idx_unzipped, zipped_rows, subbatch_rows
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx : start_idx + len(x_zipped)])
        start_idx += len(x_zipped)
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"x_zipped@VECTOR": x_zipped, "x_unzipped": x_unzipped, "idx_unzipped": idx_unzipped}
        outs = {}
        outs_list = ["y_zipped@VECTOR"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_zip_unique_add_subbatch", **locals())

        outs["y_zipped@VECTOR"] = x_zipped
        helper.append_op(
            type="tokens_zip_unique_add_subbatch",
            inputs=ins,
            outputs=outs,
            attrs={"zipped_rows": zipped_rows, "subbatch_rows": subbatch_rows},
        )
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def fused_apply_rotary_pos_emb_vision(tensor, freqs):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("fused_apply_rotary_pos_emb_vision", tensor, freqs)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"Tensor": tensor, "Freqs": freqs}
        outs = {}
        outs_list = ["Out"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fused_apply_rotary_pos_emb_vision", **locals())

        outs["Out"] = helper.create_variable(dtype="float32")
        helper.append_op(type="fused_apply_rotary_pos_emb_vision", inputs=ins, outputs=outs, attrs={})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def fused_swiglu_bwd(g, y):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("fused_swiglu_bwd", g, y)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"G": g, "Y": y}
        outs = {}
        outs_list = ["DX"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fused_swiglu_bwd", **locals())

        outs["DX"] = helper.create_variable(dtype="float32")
        helper.append_op(type="fused_swiglu_bwd", inputs=ins, outputs=outs, attrs={})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def fuse_transpose_split_fp8_quant(x, input_scales, outs, scales, tokens_per_expert, pow_2_scales, use_ue8m0):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "fuse_transpose_split_fp8_quant", x, input_scales, outs, scales, tokens_per_expert, pow_2_scales, use_ue8m0
        )
        res = []
        start_idx = 0
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"x": x, "input_scales@OPTIONAL": input_scales, "outs@VECTOR": outs, "scales@VECTOR": scales}
        outs = {}
        outs_list = []
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fuse_transpose_split_fp8_quant", **locals())

        helper.append_op(
            type="fuse_transpose_split_fp8_quant",
            inputs=ins,
            outputs=outs,
            attrs={"tokens_per_expert": tokens_per_expert, "pow_2_scales": pow_2_scales, "use_ue8m0": use_ue8m0},
        )
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def fused_swiglu_scale(x, scale):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("fused_swiglu_scale", x, scale)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"X": x, "Scale": scale}
        outs = {}
        outs_list = ["Out"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fused_swiglu_scale", **locals())

        outs["Out"] = helper.create_variable(dtype="float32")
        helper.append_op(type="fused_swiglu_scale", inputs=ins, outputs=outs, attrs={})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def fuse_stack_fp8_quant(x, using_pow2_scaling, using_ue8m0_scale, output_scale_transpose):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "fuse_stack_fp8_quant", x, using_pow2_scaling, using_ue8m0_scale, output_scale_transpose
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"X@VECTOR": x}
        outs = {}
        outs_list = ["output", "scale"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fuse_stack_fp8_quant", **locals())

        outs["output"] = helper.create_variable(dtype="float32")
        outs["scale"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="fuse_stack_fp8_quant",
            inputs=ins,
            outputs=outs,
            attrs={
                "using_pow2_scaling": using_pow2_scaling,
                "using_ue8m0_scale": using_ue8m0_scale,
                "output_scale_transpose": output_scale_transpose,
            },
        )
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def count_cumsum(x, e, do_cumsum):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("count_cumsum", x, e, do_cumsum)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"X": x}
        outs = {}
        outs_list = ["CountOutput", "CumsumOutput"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("count_cumsum", **locals())

        outs["CountOutput"] = helper.create_variable(dtype="float32")
        outs["CumsumOutput"] = helper.create_variable(dtype="float32")
        helper.append_op(type="count_cumsum", inputs=ins, outputs=outs, attrs={"E": e, "do_cumsum": do_cumsum})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


from paddle import _C_ops  # noqa: F811
from paddle.base.layer_helper import LayerHelper  # noqa: F811
from paddle.framework import in_dynamic_or_pir_mode  # noqa: F811
from paddle.jit.marker import unified


@unified
def fuse_stack_transpose_fp8_quant(x, using_pow2_scaling, using_ue8m0_scale, output_scale_transpose):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "fuse_stack_transpose_fp8_quant", x, using_pow2_scaling, using_ue8m0_scale, output_scale_transpose
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"X@VECTOR": x}
        outs = {}
        outs_list = ["output", "scale"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fuse_stack_transpose_fp8_quant", **locals())

        outs["output"] = helper.create_variable(dtype="float32")
        outs["scale"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="fuse_stack_transpose_fp8_quant",
            inputs=ins,
            outputs=outs,
            attrs={
                "using_pow2_scaling": using_pow2_scaling,
                "using_ue8m0_scale": using_ue8m0_scale,
                "output_scale_transpose": output_scale_transpose,
            },
        )
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res) == 1 else res


import importlib.abc
import importlib.util
import os
import sys
import types

import paddle

cur_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(cur_dir, "ops_pd_.so")


def __bootstrap__():
    assert os.path.exists(so_path), f"Compiled extension not found: {so_path}. " "Please build paddlefleet-ops first."
    # load custom op shared library with abs path
    custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

    if os.name == "nt" or sys.platform.startswith("darwin"):
        # Cpp Extension only support Linux now
        mod = types.ModuleType(__name__)
    else:
        try:
            spec = importlib.util.spec_from_file_location(__name__, so_path)
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(mod)
        except ImportError:
            mod = types.ModuleType(__name__)

    for custom_op in custom_ops:
        setattr(mod, custom_op, eval(custom_op))


__bootstrap__()
