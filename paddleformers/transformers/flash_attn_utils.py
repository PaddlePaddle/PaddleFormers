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

import paddle


# test if flash attention is available
def is_flash_attn_available():
    try:
        import os

        if "npu" in paddle.get_device():  # NOTE: flash attn has not been tested yet
            for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
                if lib.endswith(".so"):
                    paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(lib)
            from paddle.base import core

            def flash_attention_npu(
                query,
                key,
                value,
                dropout=0.0,
                causal=False,
                return_softmax=False,
                *,
                fixed_seed_offset=None,
                rng_name="",
                training=True,
                name=None,
                attn_mask=None,
                is_varlen=False,
                batch_size=None,
                seq_length=None,
            ):

                is_triangle_upper_mask = True

                if is_varlen:
                    if len(query.shape) == 4:
                        B, S, N, D = query.shape
                        query = query.reshape([B * S, N, D])
                        key = key.reshape([B * S, N, D])
                        value = value.reshape([B * S, N, D])
                    else:
                        assert batch_size is not None
                        assert seq_length is not None
                        B = batch_size
                        S = seq_length
                    actual_seq_q_len = actual_seq_kv_len = list(range(S, B * S + S, S))
                else:
                    actual_seq_q_len = actual_seq_kv_len = []

                out = core.eager._run_custom_op(
                    "flash_attention_npu",
                    query,
                    key,
                    value,
                    fixed_seed_offset,
                    attn_mask,
                    actual_seq_q_len,
                    actual_seq_kv_len,
                    dropout,
                    causal,
                    return_softmax,
                    not training,
                    is_triangle_upper_mask,
                    is_varlen,
                )[0]

                return out

            q = paddle.rand((1, 4, 2, 8)).astype("bfloat16")
            _ = flash_attention_npu(q, q, q, 0.9, False, False)
            paddle.nn.functional.flash_attention_npu = flash_attention_npu
            return True
        q = paddle.rand((1, 4, 2, 8)).astype("bfloat16")
        _ = paddle.nn.functional.flash_attention.flash_attention(q, q, q, 0.9, False, False)
        return True
    except:
        return False


HAS_FLASH_ATTN = is_flash_attn_available()


def has_flash_attn_func():
    if HAS_FLASH_ATTN:
        try:
            if "npu" in paddle.get_device():
                flash_attn_func_npu = paddle.nn.functional.flash_attention_npu
                return flash_attn_func_npu, flash_attn_func_npu
            else:
                from paddle.nn.functional.flash_attention import (
                    flash_attention as flash_attn_func,
                )
                from paddle.nn.functional.flash_attention import (
                    flash_attn_unpadded as flash_attn_varlen_func,
                )

                return flash_attn_func, flash_attn_varlen_func
        except:
            return None, None
    else:
        return None, None
