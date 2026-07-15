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
"""Small helper for enforcing strict config validation.

Used by PaddleFormers trainer modules to reject silent or auto-rewrites
of user-provided configuration fields.
"""


def _raise_config_conflict(name, current, expected, reason, extra=""):
    """Raise a ValueError with a uniform ConfigConflict message."""
    raise ValueError(
        f"[ConfigConflict] `{name}` was set to {current!r}, but {reason} "
        f"requires {expected!r}. "
        f"Please set `{name}`={expected!r} explicitly"
        f"{(' or ' + extra) if extra else ''}."
    )
