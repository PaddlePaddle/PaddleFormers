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


from paddlefleet_ops import CUDNN_FRONTEND_HINT, is_cudnn_frontend_available


def _require_cudnn_frontend():
    if not is_cudnn_frontend_available():
        raise ImportError(CUDNN_FRONTEND_HINT)


def _patch_paddle_memory_format():
    """Patch paddle with torch.memory_format sentinel attributes for H-Card.

    The cudnn package's api_base.TensorDesc.is_contiguous() uses
    torch.contiguous_format / preserve_format / channels_last / channels_last_3d
    for memory layout checks. When Paddle's torch-proxy intercepts `import torch`,
    these attributes are missing. We inject them as unique sentinel objects so that
    identity/equality checks in cudnn work correctly.
    """
    import paddle

    class _MemoryFormat:
        """Minimal stand-in for torch.memory_format enum values."""

        def __init__(self, name):
            self._name = name

        def __repr__(self):
            return f"paddle.{self._name}"

        def __hash__(self):
            return hash(self._name)

        def __eq__(self, other):
            return self is other

    for name in (
        "contiguous_format",
        "preserve_format",
        "channels_last",
        "channels_last_3d",
    ):
        if not hasattr(paddle, name):
            setattr(paddle, name, _MemoryFormat(name))

    # Also need paddle.memory_format type for type annotations
    if not hasattr(paddle, "memory_format"):
        paddle.memory_format = type(_MemoryFormat("contiguous_format"))


_patch_paddle_memory_format()


def _patch_cutlass_nvgpu():
    """Patch cutlass.cute.nvgpu to expose OperandMajorMode at top level for H-Card.

    The cudnn SM90 backward kernel (dsa_bwd_sm90.py) references
    cute.nvgpu.OperandMajorMode, but in cutlass 4.4.x this symbol lives
    under cute.nvgpu.warpgroup. Promote it so the kernel can find it.
    """
    try:
        from cutlass.cute import nvgpu

        if not hasattr(nvgpu, "OperandMajorMode"):
            from cutlass.cute.nvgpu.warpgroup import OperandMajorMode

            nvgpu.OperandMajorMode = OperandMajorMode
    except (ImportError, AttributeError):
        pass


_patch_cutlass_nvgpu()


def _patch_paddle_nvtx_range():
    """Patch paddle nvtx classes with a ``range`` context-manager for H-Card.

    The cudnn SM90 backward kernel uses ``torch.cuda.nvtx.range(...)`` as a
    context manager. When Paddle's torch-proxy is active, ``torch.cuda.nvtx``
    resolves to ``paddle.cuda.nvtx`` (NOT paddle.device.nvtx). Both classes
    only expose ``range_push`` / ``range_pop``. We inject a compatible ``range``
    into both to be safe.
    """
    from contextlib import contextmanager

    import paddle

    def _inject_range(nvtx_cls):
        if nvtx_cls is None or hasattr(nvtx_cls, "range"):
            return

        @contextmanager
        def _nvtx_range(msg: str, *args, **kwargs):
            nvtx_cls.range_push(msg.format(*args, **kwargs))
            try:
                yield
            finally:
                nvtx_cls.range_pop()

        nvtx_cls.range = staticmethod(_nvtx_range)

    _inject_range(getattr(paddle.device, "nvtx", None))
    _inject_range(getattr(paddle.cuda, "nvtx", None))


_patch_paddle_nvtx_range()


def _patch_paddle_stream_cuda_stream():
    """Give paddle.device.Stream a torch-like ``cuda_stream`` property for H-Card.

    cudnn's resolve_stream() defaults to torch.cuda.current_stream().cuda_stream.
    Under Paddle's torch-proxy that returns a paddle Stream, which exposes the
    raw stream as stream_base.cuda_stream instead. Bridging the attribute makes
    resolve_stream() pick up Paddle's current stream on both SM90 and SM100.
    """
    import paddle
    from paddle.base import core

    S = paddle.device.Stream
    if not hasattr(S, "cuda_stream"):

        def _cuda_stream(self):
            assert isinstance(self.stream_base, core.CUDAStream), (
                "cuda_stream is only available for CUDA streams"
            )
            return self.stream_base.cuda_stream

        S.cuda_stream = property(_cuda_stream)


_patch_paddle_stream_cuda_stream()


def csa_sparse_attn_bwd_cudnn(
    q,  # (total_sq, H, D) bf16
    kv,  # (total_skv, D) bf16
    out,  # (total_sq, H, D) bf16
    dout,  # (total_sq, H, D) bf16
    lse,  # (total_sq, H) fp32
    attn_sink,  # (H,) fp32
    topk_idxs,  # (total_sq, topk) int32, global flat indices
    softmax_scale=None,
    topk_length=None,
):
    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.sparse_attention_backward.api import (
        sparse_attention_backward_wrapper,
    )

    # print(f"get here.....")
    result = sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    return result["dq"], result["dkv"], result["d_sink"]
