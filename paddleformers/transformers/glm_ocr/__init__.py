"""Package"""
import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "image_processor": ["Glm46VImageProcessor"],
    "processor": ["GlmOcrProcessor"],
    "configuration": ["GlmOcrConfig"],
    "modeling": ["GlmOcrForConditionalGeneration"],
}

if TYPE_CHECKING:
    from .configuration import *
    from .image_processor import *
    from .modeling import *
    from .processor import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
