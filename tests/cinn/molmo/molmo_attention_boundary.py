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

"""Test-only no-cache Molmo attention boundary; no model formula/parameter changes."""
import types

import paddle

from paddleformers.transformers.molmo import modeling

_original_apply_rotary_pos_emb = modeling.apply_rotary_pos_emb


def rope_with_explicit_head_axis(q, k, cos, sin, rope_impl="interleave", unsqueeze_dim=1):
    # Keep ordinary callers equivalent; compiled core receives already-expanded inputs.
    if cos.ndim == 3:
        return _original_apply_rotary_pos_emb(q, k, cos, sin, rope_impl, unsqueeze_dim)
    rotate = modeling.rotate_every_two if rope_impl == "interleave" else modeling.rotate_half
    qt, kt = q.dtype, k.dtype
    q = q.astype(paddle.float32)
    k = k.astype(paddle.float32)
    return (q * cos + rotate(q) * sin).astype(qt), (k * cos + rotate(k) * sin).astype(kt)


def core(self, x, mask, cos, sin):
    return modeling.MolmoAttention.forward(
        self,
        hidden_states=x,
        attention_mask=mask,
        position_embeddings=(cos, sin),
        use_cache=False,
    )


def boundary(
    self,
    hidden_states,
    past_key_values=None,
    attention_mask=None,
    attn_mask_startend_row_indices=None,
    position_embeddings=None,
    use_cache=False,
):
    if use_cache or past_key_values is not None or attn_mask_startend_row_indices is not None:
        raise ValueError("This CINN test boundary only covers no-cache dense-mask attention")
    cos, sin = position_embeddings
    return self._cinn_test_core(hidden_states, attention_mask, cos.unsqueeze(1), sin.unsqueeze(1))


def install(model, compile_core):
    if modeling.apply_rotary_pos_emb is not _original_apply_rotary_pos_emb:
        raise RuntimeError("Molmo RoPE is already patched; restore the previous boundary first")
    original_forwards = [(layer.self_attn, layer.self_attn.forward) for layer in model.model.layers]
    modeling.apply_rotary_pos_emb = rope_with_explicit_head_axis

    def restore():
        modeling.apply_rotary_pos_emb = _original_apply_rotary_pos_emb
        for attn, original_forward in original_forwards:
            attn.forward = original_forward
            if hasattr(attn, "_cinn_test_core"):
                delattr(attn, "_cinn_test_core")

    try:
        for attn, _ in original_forwards:
            fn = types.MethodType(core, attn)
            attn._cinn_test_core = paddle.jit.to_static(fn, full_graph=True, backend="CINN") if compile_core else fn
            attn.forward = types.MethodType(boundary, attn)
    except Exception:
        restore()
        raise
    return restore
