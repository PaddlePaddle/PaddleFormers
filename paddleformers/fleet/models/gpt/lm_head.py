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

import paddle
from paddle.distributed.fleet.meta_parallel import ScheduleNode, build_spec_layer
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
)

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from paddleformers.fleet.transformer.identity_op import IdentityOp


class GPTLMHead(ColumnParallelLinear):
    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.skip_weight_param_allocation = kwargs["skip_weight_param_allocation"]
        self._dtype = self.config.params_dtype

        # Extract block_attn_res spec before passing kwargs to super
        block_attn_res_spec = kwargs.pop("block_attn_res", IdentityOp)

        kwargs["skip_weight_param_allocation"] = True
        if self.config.gpt_model_use_experimental_version:
            kwargs["bias"] = self.config.use_bias
        super().__init__(**kwargs)

        stride = kwargs["stride"] if "stride" in kwargs.keys() else 1
        init_method = kwargs["init_method"]
        keep_master_weight_for_test = (
            kwargs["keep_master_weight_for_test"] if "keep_master_weight_for_test" in kwargs.keys() else False
        )

        if not self.skip_weight_param_allocation:
            if self.config.use_cpu_initialization:
                self.weight = self.create_parameter(
                    shape=[self.output_size_per_partition, self.input_size],
                    dtype=self.config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                if self.config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=self.rank,
                        world_size=self.world_size,
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[self.output_size_per_partition, self.input_size],
                    dtype=self.config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )

                if self.config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )
            self.weight.is_distributed = True if self.world_size > 1 else False

        # Final Block Attention Residual (applied before LM head projection)
        self.block_attn_res = build_spec_layer(block_attn_res_spec, config=self.config)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="GPTLMHead")

    def _forward(self, hidden_states: paddle.Tensor):
        # Fused linear + cross-entropy path: skip materializing [B, S, V] logits
        # and delegate the linear projection into LanguageLoss, which will call
        # LigerFusedLinearCrossEntropyFunction.
        if getattr(self.config, "fused_linear_ce_loss_chunk", 0):
            if self.config.sequence_parallel:
                # [S, B, H] -> [B, S, H] to match the logits layout consumers expect.
                hidden_states = hidden_states.transpose([1, 0, 2]).contiguous()

            return (hidden_states, self.weight, self.bias)

        if self.config.recompute_modules is not None and "lm_head" in self.config.recompute_modules:
            recompute_func = super().forward

            def recompute_handler(hidden_states, weight):
                logits, _ = recompute_func(hidden_states, weight)
                return logits

            logits = recompute_handler(hidden_states, self.weight.T)
        else:
            logits, _ = super().forward(hidden_states, self.weight.T)
        if self.config.sequence_parallel:
            logits = logits.transpose([1, 0, 2]).contiguous()

        # Loss-path MD5 probe: lm_head weight and logits
        import os

        if os.environ.get("LOG_LAYER_MD5", "0") == "1" or os.environ.get("LOG_LOSS_MD5", "0") == "1":
            import hashlib

            rank = paddle.distributed.get_rank()
            w_md5 = hashlib.md5(self.weight.cast("float32").numpy().tobytes()).hexdigest()
            h_md5 = hashlib.md5(hidden_states.cast("float32").numpy().tobytes()).hexdigest()
            l_md5 = hashlib.md5(logits.cast("float32").numpy().tobytes()).hexdigest()
            print(
                f"[LOSS_PATH_MD5] rank={rank} lm_head_weight shape={list(self.weight.shape)} md5={w_md5}",
                flush=True,
            )
            print(
                f"[LOSS_PATH_MD5] rank={rank} lm_head_input shape={list(hidden_states.shape)} md5={h_md5}",
                flush=True,
            )
            print(
                f"[LOSS_PATH_MD5] rank={rank} lm_head_logits shape={list(logits.shape)} md5={l_md5}",
                flush=True,
            )

        return logits

    def forward(self, dict_args: dict):
        hidden_states = dict_args["hidden_states"]

        # Apply final Block Attention Residual if enabled
        if self.config.block_attention_residuals:
            blocks = dict_args.get("blocks", [])
            hidden_states = self.block_attn_res(hidden_states, blocks)

        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
        ):
            tensor_list = paddle.split(
                hidden_states,
                self.config.num_nextn_predict_layers + 1,
            )
            logits = [self._forward(tensor_list[0])]
            for i in range(self.config.num_nextn_predict_layers):
                logits.append(self._forward(tensor_list[i + 1]))
            return logits
        else:
            return self._forward(hidden_states)

    @property
    def embedding_weight(self):
        return self.weight

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(structured_name_prefix="")
        shard_rules = None if self.world_size == 1 else {"weight": 0, "bias": 0}
        return build_sharded_state_dict(state_dict, shard_rules, structured_name_prefix)


class GPTMainLMHead(GPTLMHead):
    """主干网 LM Head: 含 block_attn_res, 只做单次预测。"""

    def __init__(self, **kwargs):
        block_attn_res_spec = kwargs.pop("block_attn_res", IdentityOp)
        super().__init__(**kwargs)
        self.block_attn_res = build_spec_layer(block_attn_res_spec, config=self.config)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="GPTMainLMHead")

    def forward(self, dict_args: dict):
        hidden_states = dict_args["hidden_states"]
        mtp_loss = dict_args.get("mtp_loss", None)
        if self.config.block_attention_residuals:
            blocks = dict_args.get("blocks", [])
            hidden_states = self.block_attn_res(hidden_states, blocks)

        tensor_list = paddle.split(
            hidden_states,
            self.config.num_nextn_predict_layers + 1,
        )
        logits = self._forward(tensor_list[0])
        ret = {
            "logits": logits,
            "mtp_loss": mtp_loss,
        }
        # Filter out None values to avoid AttributeError in
        # convert_tensor_dict_to_tuple when pipeline stage boundary
        # separates GPTMainLMHead from MTPLanguageLoss
        for key in list(ret.keys()):
            if ret[key] is None:
                ret.pop(key)
        return ret

    @property
    def embedding_weight(self):
        return self.weight


class GPTMTPLMHead(GPTLMHead):
    """MTP LM Head: 将拼接的 hidden_states 拆分后逐MTP计算。"""

    def __init__(self, **kwargs):
        # MTP head 不需要 block_attn_res
        kwargs.pop("block_attn_res", None)
        super().__init__(**kwargs)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="GPTMTPLMHead")

    def forward(self, dict_args: dict):
        hidden_states = dict_args["hidden_states"]
        num_mtp = self.config.num_nextn_predict_layers
        tensor_list = paddle.split(hidden_states, num_mtp + 1)

        mtp_logits = []
        for i in range(num_mtp):
            mtp_logits.append(self._forward(tensor_list[i + 1]))

        dict_args["mtp_logits"] = mtp_logits
        return dict_args

    @property
    def embedding_weight(self):
        return self.weight
