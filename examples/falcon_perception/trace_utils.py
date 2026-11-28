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

"""Framework-independent trace files for comparing two separate Python processes."""
import json
from pathlib import Path

import numpy as np


def as_numpy(value):
    value = value.detach().cpu()
    if "bfloat16" in str(value.dtype):
        value = value.float()
    return value.numpy().copy()


class InferenceTrace:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.current = {}
        self.manifest = []
        self.originals = []
        self.generated_tokens = []

    def replace(self, obj, name, replacement):
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def install(self, model):
        for name, layer in model.layers.items():
            forward = layer.forward

            def capture_layer(*args, _name=name, _forward=forward, **kwargs):
                output = _forward(*args, **kwargs)
                self.current[f"layer_{int(_name):02d}"] = as_numpy(output)
                return output

            self.replace(layer, "forward", capture_layer)
        forward = model.forward

        def capture_forward(*args, **kwargs):
            self.current = {}
            for key, value in kwargs.items():
                if hasattr(value, "numpy"):
                    self.current["input_" + key] = as_numpy(value)
            if self.step > 0:
                tokens = self.current.get("input_tokens")
                if tokens is None or tokens.shape != (1, 1):
                    raise ValueError("Generation tracing requires one actual decode token per call.")
                self.generated_tokens.extend(int(token) for token in tokens[0])
            logits, hidden = forward(*args, **kwargs)
            if self.step == 0:
                np.save(self.directory / "prefill_logits.npy", as_numpy(logits))
            self.current["logits"] = as_numpy(logits[:, -1])
            self.current["hidden"] = as_numpy(hidden)
            filename = f"step_{self.step:03d}.npz"
            np.savez(self.directory / filename, **self.current)
            self.manifest.append(filename)
            self.step += 1
            return logits, hidden

        self.replace(model, "forward", capture_forward)
        upsample = model.upsample_img_features

        def capture_upsample(*args, **kwargs):
            result = upsample(*args, **kwargs)
            sy, sx = max(1, result.shape[-2] // 32), max(1, result.shape[-1] // 32)
            np.save(self.directory / "upsampler_samples.npy", as_numpy(result[:, :, ::sy, ::sx]))
            return result

        self.replace(model, "upsample_img_features", capture_upsample)
        postprocess = model._postprocess_aux

        def capture_postprocess(aux_list, *args, **kwargs):
            masks = {
                f"mask_{i // 3:03d}": as_numpy(value) for i, value in enumerate(aux_list) if hasattr(value, "numpy")
            }
            np.savez(self.directory / "mask_logits.npz", **masks)
            return postprocess(aux_list, *args, **kwargs)

        self.replace(model, "_postprocess_aux", capture_postprocess)

    def finish(self, metadata, predictions):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)
        self.originals.clear()
        metadata = dict(
            metadata, trace_files=self.manifest, generated_tokens=self.generated_tokens, predictions=predictions
        )
        (self.directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
