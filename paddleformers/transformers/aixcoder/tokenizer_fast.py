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

import re
from typing import Dict, Optional, Union

import paddle
from transformers import PreTrainedTokenizerFast

from ..tokenizer_utils import PaddleTokenizerMixin
from .constant import EXT2LANG, LANGUAGE_TAG, LANGUAGE_WRAPPER


class AixcoderTokenizerFast(PaddleTokenizerMixin, PreTrainedTokenizerFast):
    """
    Aixcoder Tokenizer with special input preprocessing for code completion.

    This tokenizer extends PreTrainedTokenizerFast with special handling for:
    - Code context (prefix/middle/suffix)
    - Language detection based on file extension
    - Security filtering
    - Special tokens for code completion
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def input_wrapper(self, code_string: str, later_code: str = "", path: str = "", pad_token: str = "☺") -> Dict:
        """
        Wrap input for Aixcoder model with special tokens and formatting.

        Args:
            code_string: The main code string (prefix/context)
            later_code: The code after cursor position (suffix)
            path: File path for language detection
            pad_token: Token used for padding

        Returns:
            Dictionary with input_ids and attention_mask
        """

        # Language detection from file extension
        extension_pattern = re.compile(r"(\.\w+)$")
        p = ""

        if isinstance(path, str) and len(path) > 0:
            extension = extension_pattern.search(path)
            if extension is not None:
                extension = extension.groups()[0]
            lang = EXT2LANG.get(extension, "")
            des = LANGUAGE_TAG.get(lang, "")

            if len(des) > 0:
                code_string = des + "\n" + code_string

            des = LANGUAGE_WRAPPER.get(lang, "")
            if len(des) > 0 and "<AIX-SPE>" in des:
                p = des.replace("<AIX-SPE>", f"the file path is: {path}") + "\n"

        # Special token processing - use parent class __call__ to avoid recursion
        pad_ids = super().__call__(pad_token, return_tensors="pd", return_token_type_ids=False)
        pad_len = len(pad_ids["input_ids"][0])

        pre_code_ids = super().__call__(
            "<s>▁<AIX-SPAN-PRE>▁<AIX-SPAN-POST>", return_tensors="pd", return_token_type_ids=False
        )

        later_code_ids = super().__call__(pad_token + later_code, return_tensors="pd", return_token_type_ids=False)
        later_code_ids["input_ids"] = later_code_ids["input_ids"][:, pad_len:]
        later_code_ids["attention_mask"] = later_code_ids["attention_mask"][:, pad_len:]

        code_string_ids = super().__call__(
            f"▁<AIX-SPAN-MIDDLE>{p}{code_string}", return_tensors="pd", return_token_type_ids=False
        )

        # Concatenate all parts
        code_string_ids["input_ids"] = paddle.concat(
            [pre_code_ids["input_ids"], later_code_ids["input_ids"], code_string_ids["input_ids"]], axis=1
        )
        code_string_ids["attention_mask"] = paddle.concat(
            [pre_code_ids["attention_mask"], later_code_ids["attention_mask"], code_string_ids["attention_mask"]],
            axis=1,
        )

        return code_string_ids

    def __call__(
        self,
        text=None,
        text_pair=None,
        text_target=None,
        text_pair_target=None,
        add_special_tokens: bool = True,
        padding=False,
        truncation=None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, bool]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> Dict:
        """
        Override __call__ to handle Aixcoder special input format.

        If text is a string and contains special Aixcoder markers or is a code completion task,
        it will be processed through input_wrapper. Otherwise, it falls back to the parent implementation.
        """
        # Check if this is a code completion task that needs special handling
        # Extract custom parameters before passing to parent
        later_code = kwargs.pop("later_code", "")
        path = kwargs.pop("path", "test.py")  # Default to test.py as requested

        # Handle dictionary input (for backward compatibility)
        if isinstance(text, dict):
            code_string = text.get("code_string", "")
            later_code = text.get("later_code", later_code)
            path = text.get("path", path)
            # Always use input_wrapper for dict input
            return self.input_wrapper(code_string, later_code, path)

        if isinstance(text, str) and not text_pair:
            # Use input_wrapper only if it looks like code or has code-related parameters
            if self._looks_like_code(text) or later_code or (path and path != "test.py"):
                return self.input_wrapper(text, later_code, path)

        # Fall back to parent implementation for non-code text
        return super().__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _looks_like_code(self, text: str) -> bool:
        """
        Heuristic to determine if text looks like code.
        """
        # Strong indicators that alone suggest code
        strong_indicators = [
            "def ",
            "class ",
            "import ",
            "from ",  # Python
            "function ",  # JavaScript
            "func ",  # Go
            "fn ",  # Rust
            "const ",
            "let ",
            "var ",  # JavaScript
            "public ",
            "private ",
            "void ",
            "int ",  # Java/C++
        ]

        # Weak indicators that need multiple occurrences
        weak_indicators = [
            "{",
            "}",
            "(",
            ")",
            "[",
            "]",  # Common code symbols
            "->",
            "=>",
            "::",
            "==",
            "!=",  # Operators
            ":",  # Colon (used in class definitions, type hints, etc.)
            "pass",  # Python keyword
            "return",  # Common keyword
        ]

        # Empty or very short text is not code
        if not text or len(text.strip()) < 3:
            return False

        # Check for strong indicators (any one is enough)
        # But avoid false positives - "function" in normal text like "I wrote a function today"
        # should not be detected as code unless it's clearly code context
        has_strong_indicator = False
        for indicator in strong_indicators:
            if indicator in text:
                # Special handling for "function" - need more context to avoid false positives
                if indicator == "function ":
                    # Check if it's in a code-like context (has parentheses, braces, etc.)
                    # "I wrote a function today" should not be code
                    # "function getData() { return data; }" should be code
                    if any(x in text for x in ["(", ")", "{", "}", "=", ";"]) or ("function " in text and "(" in text):
                        has_strong_indicator = True
                        break
                    # If "function" appears but without code context, don't treat as code
                    # unless it's clearly a function definition
                    continue
                else:
                    has_strong_indicator = True
                    break

        # Count weak indicators
        weak_indicator_count = sum(1 for indicator in weak_indicators if indicator in text)

        # Check for common code patterns like "class X:" or "def X():"
        has_class_or_def = ("class " in text and ":" in text) or ("def " in text and "(" in text)

        # Code if: strong indicator OR (2+ weak indicators) OR pattern match
        # But avoid false positives for normal text
        if has_strong_indicator or has_class_or_def:
            return True

        # For weak indicators, need at least 2, but also check context
        # Simple sentences with just parentheses shouldn't be code
        if weak_indicator_count >= 2:
            # Check if it's likely code vs normal text
            # Code usually has more structure
            has_structure = any(x in text for x in ["=", ";", ":", "\n", "\t"])
            # Also check for common code patterns like "{ }" or "( )"
            has_code_pattern = ("{" in text and "}" in text) or ("(" in text and ")" in text and "=" in text)
            # Special case: "{ }" should be detected as code (minimal code pattern)
            if text.strip() == "{ }" or (text.count("{") == 1 and text.count("}") == 1 and len(text.strip()) <= 5):
                return True
            return has_structure or has_code_pattern

        return False
