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

"""Tests for datasets_v2/schema.py.

Run with: python -m pytest tests/datasets_v2/test_schema.py -v
"""

import pytest

from paddleformers.datasets_v2.schema import (
    PAIR_KEYS,
    PREFIXED_PAIR_KEYS,
    ROLES,
    STANDARD_KEYS,
    StandardRow,
    cast_images,
    cast_media_list,
    check_messages,
    remove_non_standard_keys,
)

# ============================================================
# check_messages
# ============================================================


class TestCheckMessages:
    def test_valid_single_turn(self):
        """Valid user/assistant pair should pass."""
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        check_messages(msgs)  # should not raise

    def test_valid_with_system(self):
        """System + user + assistant should pass."""
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        check_messages(msgs)

    def test_valid_with_loss_field(self):
        """Messages with 'loss' field should pass."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "loss": False},
        ]
        check_messages(msgs)

    def test_valid_tool_roles(self):
        """Tool roles should be valid."""
        msgs = [
            {"role": "user", "content": "Call weather API"},
            {"role": "tool_call", "content": "get_weather()"},
            {"role": "tool_response", "content": "sunny"},
            {"role": "assistant", "content": "It's sunny!"},
        ]
        check_messages(msgs)

    def test_empty_messages_raises(self):
        """Empty messages list should fail."""
        with pytest.raises(AssertionError, match="empty messages"):
            check_messages([])

    def test_missing_role_raises(self):
        """Message without 'role' should fail."""
        with pytest.raises(AssertionError, match='missing "role"'):
            check_messages([{"content": "hello"}])

    def test_missing_content_raises(self):
        """Message without 'content' should fail."""
        with pytest.raises(AssertionError, match='missing "content"'):
            check_messages([{"role": "user"}])

    def test_invalid_role_raises(self):
        """Invalid role name should fail."""
        with pytest.raises(AssertionError, match="invalid role"):
            check_messages([{"role": "invalid", "content": "hi"}])

    def test_none_content_raises(self):
        """None content should fail."""
        with pytest.raises(AssertionError, match="content is None"):
            check_messages([{"role": "user", "content": None}])

    def test_extra_keys_raises(self):
        """Unexpected keys in message should fail."""
        with pytest.raises(AssertionError, match="unexpected keys"):
            check_messages([{"role": "user", "content": "hi", "extra": "bad"}])


# ============================================================
# cast_images
# ============================================================


class TestCastImages:
    def test_string_input(self):
        """Single path string -> List[ImageMedia]."""
        result = cast_images("/path/to/img.jpg")
        assert result == [{"bytes": None, "path": "/path/to/img.jpg"}]

    def test_dict_input(self):
        """Single dict -> List[dict]."""
        img = {"bytes": b"data", "path": "img.png"}
        result = cast_images(img)
        assert result == [img]

    def test_list_of_strings(self):
        """List of path strings."""
        result = cast_images(["a.jpg", "b.png"])
        assert result == [
            {"bytes": None, "path": "a.jpg"},
            {"bytes": None, "path": "b.png"},
        ]

    def test_list_of_dicts(self):
        """List of dicts passed through."""
        imgs = [{"bytes": None, "path": "a.jpg"}, {"bytes": None, "path": "b.jpg"}]
        result = cast_images(imgs)
        assert result == imgs

    def test_mixed_list(self):
        """List with strings and dicts."""
        result = cast_images(["a.jpg", {"bytes": None, "path": "b.jpg"}])
        assert result == [
            {"bytes": None, "path": "a.jpg"},
            {"bytes": None, "path": "b.jpg"},
        ]

    def test_invalid_type_raises(self):
        """Non-string/dict/list should raise TypeError."""
        with pytest.raises(TypeError, match="unsupported images type"):
            cast_images(123)


# ============================================================
# cast_media_list
# ============================================================


class TestCastMediaList:
    def test_string_input(self):
        """Single string -> list."""
        assert cast_media_list("video.mp4") == ["video.mp4"]

    def test_list_input(self):
        """List passed through."""
        assert cast_media_list(["a.mp4", "b.mp4"]) == ["a.mp4", "b.mp4"]


# ============================================================
# remove_non_standard_keys
# ============================================================


class TestRemoveNonStandardKeys:
    def test_keeps_standard(self):
        """Standard keys are preserved."""
        row = {"messages": [{"role": "user", "content": "hi"}], "images": None}
        result = remove_non_standard_keys(row)
        assert "messages" in result
        assert "images" in result

    def test_removes_extra(self):
        """Non-standard keys are stripped."""
        row = {"messages": [], "custom_col": "value", "id": 123}
        result = remove_non_standard_keys(row)
        assert "messages" in result
        assert "custom_col" not in result
        assert "id" not in result

    def test_empty_row(self):
        """Empty row returns empty dict."""
        assert remove_non_standard_keys({}) == {}


# ============================================================
# Constants
# ============================================================


class TestConstants:
    def test_roles_tuple(self):
        assert "user" in ROLES
        assert "assistant" in ROLES
        assert "system" in ROLES
        assert "tool_call" in ROLES
        assert "tool_response" in ROLES

    def test_pair_keys(self):
        assert "messages" in PAIR_KEYS
        assert "images" in PAIR_KEYS
        assert "videos" in PAIR_KEYS

    def test_prefixed_pair_keys(self):
        assert "rejected_messages" in PREFIXED_PAIR_KEYS
        assert "positive_images" in PREFIXED_PAIR_KEYS
        assert "negative_videos" in PREFIXED_PAIR_KEYS

    def test_standard_keys_includes_all(self):
        for k in PAIR_KEYS:
            assert k in STANDARD_KEYS
        for k in PREFIXED_PAIR_KEYS:
            assert k in STANDARD_KEYS

    def test_standard_row_defaults(self):
        row = StandardRow()
        assert row.messages == []
        assert row.images is None
        assert row.label is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
