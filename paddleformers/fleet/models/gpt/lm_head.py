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

# =============================================================================
# Multimax lm_head support
# -----------------------------------------------------------------------------
# When TransformerConfig.multimax_modules is a list containing "lm_head" (e.g.,
# ``multimax_modules: [lm_head]``), GPTLMHead adds two learnable [4]-shape
# parameters (multimax_ranges, multimax_ts) and applies a SegLU-style segmented
# modulation to logits before softmax/cross-entropy:
#
#     logits = SegLU(logits, multimax_ranges, multimax_ts)
#
# Both params are initialized to zero on cold start, which makes SegLU the
# identity at step 0 (safe for resume from checkpoints lacking these keys).
# Resumed training restores the trained values from the checkpoint -- the
# zero-init only applies to fresh runs.
#
# The "multimax" substring in the parameter names is matched by the
# pretraining trainers' no-weight-decay name filter, so these params are
# excluded from weight decay (timm-style convention).
#
# Sanity-check after a training launch:
#
#     grep -E "MULTIMAX-(CONFIG|LMHEAD-CONFIRM|LMHEAD-APPLIED)" <train.log>
#
# Expected output (per rank holding the LM head):
#   [MULTIMAX-CONFIG]            multimax_modules=['lm_head']
#   [MULTIMAX-LMHEAD-CONFIRM]    cls=... multimax_modules=['lm_head'] ranges.shape=[4] ...
#   [MULTIMAX-LMHEAD-APPLIED]    cls=... logits.shape=[...] ranges=[...] ts=[...]
#                                (or path=fused for the fused-CE branch)
#
# If [MULTIMAX-LMHEAD-APPLIED] is missing, the SegLU path did not execute
# (e.g., a different LM head class is used).
#
# Paths:
#   - fused_linear_ce_loss_chunk == 0: SegLU is applied here on the [B, S, V]
#     logits tensor right before the cross-entropy. Wrapped in `recompute` and
#     uses an in-place `+=` accumulator to bound peak activation memory.
#   - fused_linear_ce_loss_chunk  > 0: GPTLMHead emits a 5-tuple
#     (hidden, weight, bias, multimax_ranges, multimax_ts) and SegLU is
#     applied inside each CE chunk by LigerFusedLinearCrossEntropyFunction.
#     This avoids ever materializing the full [B*S, V] logits tensor and is
#     the recommended path at large vocab.
#
# Notes:
#   - SegLU is element-wise, so vocab-TP sharding is fine without collectives.
#   - Both paths share the same parameters (multimax_ranges/multimax_ts) and
#     produce identical math at step 0 (params init to zero -> SegLU = id).
# =============================================================================

import warnings

import paddle
from paddle.distributed.fleet.meta_parallel import (
    ScheduleNode,
    build_spec_layer,
)
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
)

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from paddleformers.fleet.transformer.identity_op import IdentityOp


def SegLU(x, ranges, ts):
    """Learnable segmented activation applied to logits before softmax.

    Named ``SegLU`` ("segmented learnable unit") to avoid confusion with the
    framework's built-in ``SELU`` activation.

    Reference: ViT-style modulation provided by user.
        x: [..., V] logits (last dim may be vocab-parallel sharded; SegLU is
           element-wise so applying it on the sharded logits is equivalent to
           applying it on gathered logits).
        ranges: [4] learnable thresholds.
        ts:     [4] learnable scales.

    With ranges/ts initialized to zero, SegLU is the identity (logits unchanged),
    so adding this op is a no-op at start of training and safe for resume.

    Memory-efficient eager form: instead of building one giant expression
    ``out = x + a + b + c + d`` (which materializes 4 sum-intermediates each
    the size of logits), we accumulate into a single owned buffer with ``+=``.
    Combined with ``recompute`` at the call site, peak activation memory for
    this op drops from ~10x the logits tensor to ~1x.

    NOTE on jit fusion: an earlier version was wrapped with
    ``@paddle.jit.to_static`` to fuse the element-wise chain. That backend
    (CINN) was observed to OOM on full-vocab logits because CINN allocates
    all traced intermediates inside a single ``CinnJitInstruction::Run`` call,
    which defeats both ``recompute`` and the in-place ``+=`` hints. We use
    the eager path here.
    """
    relu = paddle.nn.functional.relu
    # Clone once so we own the buffer and in-place += is safe under autograd
    # (the original `x` -- the linear's output -- must not be mutated, since
    # it may still be referenced by the autograd graph upstream).
    out = x.clone()
    out += ts[0] * relu(ranges[0] - x)
    out += ts[1] * relu(x - ranges[1])
    out += ts[2] * relu(ranges[2] - x) ** 2
    out += ts[3] * relu(x - ranges[3]) ** 2
    return out


class GPTLMHead(ColumnParallelLinear):
    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.skip_weight_param_allocation = kwargs[
            "skip_weight_param_allocation"
        ]
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
            kwargs["keep_master_weight_for_test"]
            if "keep_master_weight_for_test" in kwargs.keys()
            else False
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
        self.block_attn_res = build_spec_layer(
            block_attn_res_spec, config=self.config
        )

        # Multimax: learnable SegLU-style modulation on logits before softmax.
        # Names contain the "multimax" substring so the trainer's no-decay
        # filter excludes them from weight decay (mirrors timm's convention
        # of not decaying scalar/1-D learnable coefficients).
        multimax_mode = getattr(self.config, "multimax_modules", None) or []
        self.use_multimax_lmhead = "lm_head" in multimax_mode
        if self.use_multimax_lmhead:
            # Cold-start init to zero -> SegLU is identity at step 0, so the
            # untrained model produces bit-identical logits with/without the
            # feature flag and resuming from a checkpoint that lacks these
            # params (loaded non-strictly) is safe. Resumed training restores
            # the trained values from the checkpoint -- the zero init only
            # applies on cold start.
            self.multimax_ranges = self.create_parameter(
                shape=[4],
                dtype=self.config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            self.multimax_ts = self.create_parameter(
                shape=[4],
                dtype=self.config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            # Grep-friendly construction banner. See docstring at top of file
            # for the full sanity-check workflow.
            warnings.warn(
                f"[MULTIMAX-LMHEAD-CONFIRM] cls={type(self).__name__} "
                f"multimax_modules={multimax_mode} "
                f"ranges.shape={list(self.multimax_ranges.shape)} "
                f"ts.shape={list(self.multimax_ts.shape)} "
                f"dtype={self.config.params_dtype} "
                f"fused_linear_ce_loss_chunk="
                f"{getattr(self.config, 'fused_linear_ce_loss_chunk', 0)}"
            )
        # When the feature is disabled (default None / empty), stay silent --
        # no banner, no params -- so default training runs are not noisy.

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

            # Multimax lm_head + fused path: thread multimax_ranges/ts into
            # LanguageLoss so SegLU is applied inside each CE chunk and never
            # materializes a full [B*S, V] logits tensor. Also fire the same
            # one-shot [MULTIMAX-LMHEAD-APPLIED] grep banner used by the
            # unfused path so observability is consistent.
            if getattr(self, "use_multimax_lmhead", False):
                if not getattr(self, "_multimax_applied_logged", False):
                    warnings.warn(
                        f"[MULTIMAX-LMHEAD-APPLIED] cls={type(self).__name__} "
                        f"path=fused hidden.shape={list(hidden_states.shape)} "
                        f"ranges={self.multimax_ranges.detach().cast('float32').numpy().tolist()} "
                        f"ts={self.multimax_ts.detach().cast('float32').numpy().tolist()}"
                    )
                    self._multimax_applied_logged = True
                return (
                    hidden_states,
                    self.weight,
                    self.bias,
                    self.multimax_ranges,
                    self.multimax_ts,
                )

            return (hidden_states, self.weight, self.bias)

        if (
            self.config.recompute_modules is not None
            and "lm_head" in self.config.recompute_modules
        ):
            recompute_func = super().forward

            def recompute_handler(hidden_states, weight):
                logits, _ = recompute_func(hidden_states, weight)
                return logits

            logits = recompute_handler(hidden_states, self.weight.T)
        else:
            logits, _ = super().forward(hidden_states, self.weight.T)
        if (
            not self.config.gpt_model_use_experimental_version
            and self.config.sequence_parallel
        ):
            logits = logits.transpose([1, 0, 2]).contiguous()

        # Multimax lm_head (unfused path): apply learnable SegLU modulation on
        # logits before softmax/cross-entropy. SegLU is element-wise so it works
        # correctly on vocab-parallel sharded logits without extra collectives.
        # The fused path (fused_linear_ce_loss_chunk > 0) returns early above
        # with a 5-tuple and applies SegLU inside the chunked CE kernel.
        if getattr(self, "use_multimax_lmhead", False):
            # One-shot confirmation (per rank) that the SegLU path actually
            # executed at runtime. Grep tag: [MULTIMAX-LMHEAD-APPLIED].
            if not getattr(self, "_multimax_applied_logged", False):
                warnings.warn(
                    f"[MULTIMAX-LMHEAD-APPLIED] cls={type(self).__name__} "
                    f"logits.shape={list(logits.shape)} "
                    f"ranges={self.multimax_ranges.detach().cast('float32').numpy().tolist()} "
                    f"ts={self.multimax_ts.detach().cast('float32').numpy().tolist()}"
                )
                self._multimax_applied_logged = True
            # Wrap SegLU in `recompute` to drop saved intermediates: only the
            # input logits (already kept by upstream autograd) are saved; SegLU
            # is re-run during backward. Avoids OOM on full-vocab logits when
            # `fused_linear_ce_loss_chunk=0` (required by multimax).
            logits = recompute(
                SegLU, logits, self.multimax_ranges, self.multimax_ts
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
        """Sharding along axis 0, bias sharded.

        Multimax params (multimax_ranges, multimax_ts) are tiny [4]-shape
        replicated tensors; non-sharded parameters do not need to appear in
        ``shard_rules``, so we leave them out and let flex-checkpoint pick
        them up from the state dict directly.
        """
        state_dict = self.state_dict(structured_name_prefix="")
        if self.world_size == 1:
            shard_rules = None
        else:
            shard_rules = {"weight": 0, "bias": 0}
        return build_sharded_state_dict(
            state_dict, shard_rules, structured_name_prefix
        )


class GPTMainLMHead(GPTLMHead):
    """主干网 LM Head: 含 block_attn_res, 只做单次预测。"""

    def __init__(self, **kwargs):
        block_attn_res_spec = kwargs.pop("block_attn_res", IdentityOp)
        super().__init__(**kwargs)
        self.block_attn_res = build_spec_layer(
            block_attn_res_spec, config=self.config
        )

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
