# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
from typing import List, Optional, Tuple

import paddle

from ..utils.log import logger
from .model_utils import PretrainedModel, unwrap_model

__all__ = ["export_model"]


def export_model(
    model: "PretrainedModel", input_spec=None, path: Optional[str] = None, model_format: Optional[str] = "paddle"
) -> Tuple[List[str], List[str]]:
    """
    Export paddle inference model or onnx model.

    Args:
        model ([`PretrainedModel`]:
            The model to export.
        input_spec (paddle.static.InputSpec, optional):
            Describes the input of the saved model’s forward method, which can be described
            by InputSpec or example Tensor.  Default None.
        path (Optional[str], optional):
            Output dir to save the exported model. Defaults to None.
        model_format (Optional[str], optional):
            Export model format. There are two options: paddle or onnx, defaults to paddle.

    """
    if path is None:
        path = "./"
        logger.info("Export path is missing, set default path to current dir.")

    if issubclass(type(model), PretrainedModel):
        model = unwrap_model(model)
    model.eval()

    model_format = model_format.lower()
    file_prefix = "model"
    if model_format == "paddle":
        # Convert to static graph with specific input description
        model = paddle.jit.to_static(model, input_spec=input_spec)
        # Save in static graph model.
        save_path = os.path.join(path, file_prefix)
        logger.info("Exporting inference model to %s" % save_path)
        paddle.jit.save(model, save_path)
        logger.info("Inference model exported.")
    elif model_format == "onnx":
        # Export ONNX model.
        save_path = os.path.join(path, file_prefix)
        logger.info("Exporting ONNX model to %s" % save_path)
        paddle.onnx.export(model, save_path, input_spec=input_spec)
        logger.info("ONNX model exported.")
    else:
        logger.info("This export format is not supported, please select paddle or onnx!")
