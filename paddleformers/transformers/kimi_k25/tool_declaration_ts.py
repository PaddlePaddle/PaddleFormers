# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The Moonshot AI Inc. team. All rights reserved.
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

"""
Encode structured tool declaration to typescript style string.
"""
import dataclasses
import json
from collections.abc import Sequence
from typing import Any

from ...utils.log import logger

_TS_INDENT = "  "
_TS_FIELD_DELIMITER = ",\n"


class _SchemaRegistry:
    """Registry for schema definitions to handle $ref resolution"""

    def __init__(self):
        self.definitions = {}
        self.has_self_ref = False

    def register_definitions(self, defs: dict[str, Any]):
        """Register schema definitions from $defs section"""
        if not defs:
            return
        for def_name, def_schema in defs.items():
            self.definitions[def_name] = def_schema

    def resolve_ref(self, ref: str) -> dict[str, Any]:
        """Resolve a reference to its schema definition"""
        if ref == "#":
            self.has_self_ref = True
            return {"$self_ref": True}
        elif ref.startswith("#/$defs/"):
            def_name = ref.split("/")[-1]
            if def_name not in self.definitions:
                raise ValueError(f"Reference not found: {ref}")
            return self.definitions[def_name]
        else:
            raise ValueError(f"Unsupported reference format: {ref}")


def _format_description(description: str, indent: str = "") -> str:
    return "\n".join([f"{indent}// {line}" if line else "" for line in description.split("\n")])


class _BaseType:
    description: str
    constraints: dict[str, Any]

    def __init__(
        self,
        extra_props: dict[str, Any],
        *,
        allowed_constraint_keys: Sequence[str] = (),
    ):
        self.description = extra_props.get("description", "")
        self.constraints = {k: v for k, v in extra_props.items() if k in allowed_constraint_keys}

    def to_typescript_style(self, indent: str = "") -> str:
        raise NotImplementedError

    def format_docstring(self, indent: str) -> str:
        lines = []
        if self.description:
            lines.append(_format_description(self.description, indent))
        if self.constraints:
            constraints_str = ", ".join(f"{k}: {v}" for k, v in sorted(self.constraints.items(), key=lambda kv: kv[0]))
            lines.append(f"{indent}// {constraints_str}")

        return "".join(x + "\n" for x in lines)


class _ParameterTypeScalar(_BaseType):
    type: str

    def __init__(self, type: str, extra_props: dict[str, Any] | None = None):
        self.type = type

        allowed_constraint_keys: list[str] = []
        if self.type == "string":
            allowed_constraint_keys = ["maxLength", "minLength", "pattern"]
        elif self.type in ("number", "integer"):
            allowed_constraint_keys = ["maximum", "minimum"]

        super().__init__(extra_props or {}, allowed_constraint_keys=allowed_constraint_keys)

    def to_typescript_style(self, indent: str = "") -> str:
        # Map integer to number in TypeScript
        if self.type == "integer":
            return "number"
        return self.type


class _ParameterTypeObject(_BaseType):
    properties: list["_Parameter"]
    additional_properties: Any | None = None

    def __init__(self, json_schema_object: dict[str, Any], registry: _SchemaRegistry | None = None):
        super().__init__(json_schema_object)

        self.properties = []
        self.additional_properties = None

        if not json_schema_object:
            return

        if "$defs" in json_schema_object and registry:
            registry.register_definitions(json_schema_object["$defs"])

        self.additional_properties = json_schema_object.get("additionalProperties")
        if isinstance(self.additional_properties, dict):
            self.additional_properties = _parse_parameter_type(self.additional_properties, registry)

        if "properties" not in json_schema_object:
            return

        required_parameters = json_schema_object.get("required", [])
        optional_parameters = set(json_schema_object["properties"].keys()) - set(required_parameters)

        self.properties = [
            _Parameter(
                name=name,
                type=_parse_parameter_type(prop, registry),
                optional=name in optional_parameters,
                default=prop.get("default") if isinstance(prop, dict) else None,
            )
            for name, prop in json_schema_object["properties"].items()
        ]

    def to_typescript_style(self, indent: str = "") -> str:
        # sort by optional, make the required parameters first
        parameters = [p for p in self.properties if not p.optional]
        opt_params = [p for p in self.properties if p.optional]

        parameters = sorted(parameters, key=lambda p: p.name)
        parameters.extend(sorted(opt_params, key=lambda p: p.name))

        param_strs = []
        for p in parameters:
            one = p.to_typescript_style(indent=indent + _TS_INDENT)
            param_strs.append(one)

        if self.additional_properties is not None:
            ap_type_str = "any"
            if self.additional_properties is True:
                ap_type_str = "any"
            elif self.additional_properties is False:
                ap_type_str = "never"
            elif isinstance(self.additional_properties, _ParameterType):
                ap_type_str = self.additional_properties.to_typescript_style(indent=indent + _TS_INDENT)
            else:
                raise ValueError(f"Unknown additionalProperties: {self.additional_properties}")
            param_strs.append(f"{indent + _TS_INDENT}[k: string]: {ap_type_str}")

        if not param_strs:
            return "{}"

        params_str = _TS_FIELD_DELIMITER.join(param_strs)
        if params_str:
            # add new line before and after
            params_str = f"\n{params_str}\n"
        # always wrap with object
        return f"{{{params_str}{indent}}}"


class _ParameterTypeArray(_BaseType):
    item: "_ParameterType"

    def __init__(self, json_schema_object: dict[str, Any], registry: _SchemaRegistry | None = None):
        super().__init__(json_schema_object, allowed_constraint_keys=("minItems", "maxItems"))
        if json_schema_object.get("items"):
            self.item = _parse_parameter_type(json_schema_object["items"], registry)
        else:
            self.item = _ParameterTypeScalar(type="any")

    def to_typescript_style(self, indent: str = "") -> str:
        item_docstring = self.item.format_docstring(indent + _TS_INDENT)
        if item_docstring:
            return (
                "Array<\n"
                + item_docstring
                + indent
                + _TS_INDENT
                + self.item.to_typescript_style(indent=indent + _TS_INDENT)
                + "\n"
                + indent
                + ">"
            )
        else:
            return f"Array<{self.item.to_typescript_style(indent=indent)}>"


class _ParameterTypeEnum(_BaseType):
    # support scalar types only
    enum: list[str | int | float | bool | None]

    def __init__(self, json_schema_object: dict[str, Any]):
        super().__init__(json_schema_object)
        self.enum = json_schema_object["enum"]

        # Validate enum values against declared type if present
        if "type" in json_schema_object:
            typ = json_schema_object["type"]
            if isinstance(typ, list):
                if len(typ) == 1:
                    typ = typ[0]
                elif len(typ) == 2:
                    if "null" not in typ:
                        raise ValueError(f"Enum type {typ} is not supported")
                    else:
                        typ = typ[0] if typ[0] != "null" else typ[1]
                else:
                    raise ValueError(f"Enum type {typ} is not supported")
            for val in self.enum:
                if val is None:
                    continue
                if typ == "string" and not isinstance(val, str):
                    raise ValueError(f"Enum value {val} is not a string")
                elif typ == "number" and not isinstance(val, (int, float)):
                    raise ValueError(f"Enum value {val} is not a number")
                elif typ == "integer" and not isinstance(val, int):
                    raise ValueError(f"Enum value {val} is not an integer")
                elif typ == "boolean" and not isinstance(val, bool):
                    raise ValueError(f"Enum value {val} is not a boolean")

    def to_typescript_style(self, indent: str = "") -> str:
        return " | ".join([f'"{e}"' if isinstance(e, str) else str(e) for e in self.enum])


class _ParameterTypeAnyOf(_BaseType):
    types: list["_ParameterType"]

    def __init__(
        self,
        json_schema_object: dict[str, Any],
        registry: _SchemaRegistry | None = None,
    ):
        super().__init__(json_schema_object)
        self.types = [_parse_parameter_type(t, registry) for t in json_schema_object["anyOf"]]

    def to_typescript_style(self, indent: str = "") -> str:
        return " | ".join([t.to_typescript_style(indent=indent) for t in self.types])


class _ParameterTypeUnion(_BaseType):
    types: list[str]

    def __init__(self, json_schema_object: dict[str, Any]):
        super().__init__(json_schema_object)

        mapping = {
            "string": "string",
            "number": "number",
            "integer": "number",
            "boolean": "boolean",
            "null": "null",
            "object": "{}",
            "array": "Array<any>",
        }
        self.types = [mapping[t] for t in json_schema_object["type"]]

    def to_typescript_style(self, indent: str = "") -> str:
        return " | ".join(self.types)


class _ParameterTypeRef(_BaseType):
    ref_name: str
    is_self_ref: bool = False

    def __init__(self, json_schema_object: dict[str, Any], registry: _SchemaRegistry):
        super().__init__(json_schema_object)

        ref = json_schema_object["$ref"]
        resolved_schema = registry.resolve_ref(ref)

        if resolved_schema.get("$self_ref", False):
            self.ref_name = "parameters"
            self.is_self_ref = True
        else:
            self.ref_name = ref.split("/")[-1]

    def to_typescript_style(self, indent: str = "") -> str:
        return self.ref_name


_ParameterType = (
    _ParameterTypeScalar
    | _ParameterTypeObject
    | _ParameterTypeArray
    | _ParameterTypeEnum
    | _ParameterTypeAnyOf
    | _ParameterTypeUnion
    | _ParameterTypeRef
)


@dataclasses.dataclass
class _Parameter:
    """
    A parameter in a function, or a field in a object.
    It consists of the type as well as the name.
    """

    type: _ParameterType
    name: str = "_"
    optional: bool = True
    default: Any | None = None

    @classmethod
    def parse_extended(cls, attributes: dict[str, Any]) -> "_Parameter":
        if not attributes:
            raise ValueError("attributes is empty")

        return cls(
            name=attributes.get("name", "_"),
            type=_parse_parameter_type(attributes),
            optional=attributes.get("optional", False),
            default=attributes.get("default"),
        )

    def to_typescript_style(self, indent: str = "") -> str:
        comments = self.type.format_docstring(indent)

        if self.default is not None:
            default_repr = (
                json.dumps(self.default, ensure_ascii=False)
                if not isinstance(self.default, (int, float, bool))
                else repr(self.default)
            )
            comments += f"{indent}// Default: {default_repr}\n"

        return (
            comments
            + f"{indent}{self.name}{'?' if self.optional else ''}: {self.type.to_typescript_style(indent=indent)}"
        )


def _parse_parameter_type(
    json_schema_object: dict[str, Any] | bool, registry: _SchemaRegistry | None = None
) -> _ParameterType:
    if isinstance(json_schema_object, bool):
        if json_schema_object:
            return _ParameterTypeScalar(type="any")
        else:
            logger.warning(f"Warning: Boolean value {json_schema_object} is not supported, use null instead.")
            return _ParameterTypeScalar(type="null")

    if "$ref" in json_schema_object and registry:
        return _ParameterTypeRef(json_schema_object, registry)

    if "anyOf" in json_schema_object:
        return _ParameterTypeAnyOf(json_schema_object, registry)
    elif "enum" in json_schema_object:
        return _ParameterTypeEnum(json_schema_object)
    elif "type" in json_schema_object:
        typ = json_schema_object["type"]
        if isinstance(typ, list):
            return _ParameterTypeUnion(json_schema_object)
        elif typ == "object":
            return _ParameterTypeObject(json_schema_object, registry)
        elif typ == "array":
            return _ParameterTypeArray(json_schema_object, registry)
        else:
            return _ParameterTypeScalar(typ, json_schema_object)
    elif json_schema_object == {}:
        return _ParameterTypeScalar(type="any")
    else:
        raise ValueError(f"Invalid JSON Schema object: {json_schema_object}")


def _openai_function_to_typescript_style(
    function: dict[str, Any],
) -> str:
    """Convert OpenAI function definition (dict) to TypeScript style string."""
    registry = _SchemaRegistry()
    parameters = function.get("parameters") or {}
    parsed = _ParameterTypeObject(parameters, registry)

    interfaces = []
    root_interface_name = None
    if registry.has_self_ref:
        root_interface_name = "parameters"
        params_str = _TS_FIELD_DELIMITER.join([p.to_typescript_style(indent=_TS_INDENT) for p in parsed.properties])
        params_str = f"\n{params_str}\n" if params_str else ""
        interface_def = f"interface {root_interface_name} {{{params_str}}}"
        interfaces.append(interface_def)

    definitions_copy = dict(registry.definitions)
    for def_name, def_schema in definitions_copy.items():
        obj_type = _parse_parameter_type(def_schema, registry)
        params_str = obj_type.to_typescript_style()

        description_part = ""
        if obj_description := def_schema.get("description", ""):
            description_part = _format_description(obj_description) + "\n"

        interface_def = f"{description_part}interface {def_name} {params_str}"
        interfaces.append(interface_def)

    interface_str = "\n".join(interfaces)
    function_name = function.get("name", "function")
    if root_interface_name:
        type_def = f"type {function_name} = (_: {root_interface_name}) => any;"
    else:
        params_str = parsed.to_typescript_style()
        type_def = f"type {function_name} = (_: {params_str}) => any;"

    description = function.get("description")
    return "\n".join(
        filter(
            bool,
            [
                interface_str,
                ((description and _format_description(description)) or ""),
                type_def,
            ],
        )
    )


def encode_tools_to_typescript_style(
    tools: list[dict[str, Any]],
) -> str:
    """
    Convert tools (list of dict) to TypeScript style string.

    Supports OpenAI format: {"type": "function", "function": {...}}

    Args:
        tools: List of tool definitions in dict format

    Returns:
        TypeScript style string representation of the tools
    """
    if not tools:
        return ""

    functions = []

    for tool in tools:
        tool_type = tool.get("type")
        if tool_type == "function":
            func_def = tool.get("function", {})
            if func_def:
                functions.append(_openai_function_to_typescript_style(func_def))
        else:
            # Skip unsupported tool types (like "_plugin")
            continue

    if not functions:
        return ""

    functions_str = "\n".join(functions)
    result = "# Tools\n\n"

    if functions_str:
        result += "## functions\nnamespace functions {\n"
        result += functions_str + "\n"
        result += "}\n"

    return result
