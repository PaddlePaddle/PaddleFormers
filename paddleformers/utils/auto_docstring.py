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


def auto_docstring(obj=None, *, custom_intro=None, custom_args=None, checkpoint=None):
    r"""
    Automatically generates comprehensive docstrings for model classes and methods in the Transformers library.

    This decorator reduces boilerplate by automatically including standard argument descriptions while allowing
    overrides to add new or custom arguments. It inspects function signatures, retrieves predefined docstrings
    for common arguments (like `input_ids`, `attention_mask`, etc.), and generates complete documentation
    including examples and return value descriptions.

    For complete documentation and examples, read this [guide](https://huggingface.co/docs/transformers/auto_docstring).

    Examples of usage:

        Basic usage (no parameters):
        ```python
        @auto_docstring
        class MyAwesomeModel(PreTrainedModel):
            def __init__(self, config, custom_parameter: int = 10):
                r'''
                custom_parameter (`int`, *optional*, defaults to 10):
                    Description of the custom parameter for MyAwesomeModel.
                '''
                super().__init__(config)
                self.custom_parameter = custom_parameter
        ```

        Using `custom_intro` with a class:
        ```python
        @auto_docstring(
            custom_intro="This model implements a novel attention mechanism for improved performance."
        )
        class MySpecialModel(PreTrainedModel):
            def __init__(self, config, attention_type: str = "standard"):
                r'''
                attention_type (`str`, *optional*, defaults to "standard"):
                    Type of attention mechanism to use.
                '''
                super().__init__(config)
        ```

        Using `custom_intro` with a method, and specify custom arguments and example directly in the docstring:
        ```python
        @auto_docstring(
            custom_intro="Performs forward pass with enhanced attention computation."
        )
        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            r'''
            custom_parameter (`int`, *optional*, defaults to 10):
                Description of the custom parameter for MyAwesomeModel.

            Example:

            ```python
            >>> model = MyAwesomeModel(config)
            >>> model.forward(input_ids=torch.tensor([1, 2, 3]), attention_mask=torch.tensor([1, 1, 1]))
            ```
            '''
        ```

        Using `custom_args` to define reusable arguments:
        ```python
        VISION_ARGS = r'''
        pixel_values (`torch.FloatTensor`, *optional*):
            Pixel values of the input images.
        image_features (`torch.FloatTensor`, *optional*):
            Pre-computed image features for efficient processing.
        '''

        @auto_docstring(custom_args=VISION_ARGS)
        def encode_images(self, pixel_values=None, image_features=None):
            # ... method implementation
        ```

        Combining `custom_intro` and `custom_args`:
        ```python
        MULTIMODAL_ARGS = r'''
        vision_features (`torch.FloatTensor`, *optional*):
            Pre-extracted vision features from the vision encoder.
        fusion_strategy (`str`, *optional*, defaults to "concat"):
            Strategy for fusing text and vision modalities.
        '''

        @auto_docstring(
            custom_intro="Processes multimodal inputs combining text and vision.",
            custom_args=MULTIMODAL_ARGS
        )
        def forward(
            self,
            input_ids,
            attention_mask=None,
            vision_features=None,
            fusion_strategy="concat"
        ):
            # ... multimodal processing
        ```

        Using with ModelOutput classes:
        ```python
        @dataclass
        @auto_docstring(
            custom_intro="Custom model outputs with additional fields."
        )
        class MyModelOutput(ImageClassifierOutput):
            r'''
            loss (`torch.FloatTensor`, *optional*):
                The loss of the model.
            custom_field (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
                A custom output field specific to this model.
            '''

            # Standard fields like hidden_states, logits, attentions etc. can be automatically documented
            # However, given that the loss docstring is often different per model, you should document it above
            loss: Optional[torch.FloatTensor] = None
            logits: Optional[torch.FloatTensor] = None
            hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
            attentions: Optional[tuple[torch.FloatTensor, ...]] = None
            custom_field: Optional[torch.FloatTensor] = None
        ```

    Args:
        custom_intro (`str`, *optional*):
            Custom introduction text to add to the docstring. This replaces the default
            introduction text generated by the decorator before the Args section. Use this to describe what
            makes your model or method special.
        custom_args (`str`, *optional*):
            Custom argument documentation in docstring format. This allows you to define
            argument descriptions once and reuse them across multiple methods. The format should follow the
            standard docstring convention: `arg_name (`type`, *optional*, defaults to `value`): Description.`
        checkpoint (`str`, *optional*):
            Checkpoint name to use in examples within the docstring. This is typically
            automatically inferred from the model configuration class, but can be overridden if needed for
            custom examples.

    Note:
        - Standard arguments (`input_ids`, `attention_mask`, `pixel_values`, etc.) are automatically documented
          from predefined descriptions and should not be redefined unless their behavior differs in your model.
        - New or custom arguments should be documented in the method's docstring using the `r''' '''` block
          or passed via the `custom_args` parameter.
        - For model classes, the decorator derives parameter descriptions from the `__init__` method's signature
          and docstring.
        - Return value documentation is automatically generated for methods that return ModelOutput subclasses.
    """

    def auto_docstring_decorator(obj):
        if len(obj.__qualname__.split(".")) > 1:
            return auto_method_docstring(
                obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint
            )
        else:
            return auto_class_docstring(obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint)

    if obj:
        return auto_docstring_decorator(obj)

    return auto_docstring_decorator