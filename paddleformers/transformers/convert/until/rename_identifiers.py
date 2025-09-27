import libcst as cst
from libcst import CSTTransformer
import re
import os
from typing import Set, List, Union

class GenericRenamerTransformer(CSTTransformer):
    """
    一个通用的CST转换器，用于安全地将代码中的标识符从多个源名称替换为同一个目标名称，
    并能智能地保留原始名称的大小写风格。
    """
    def __init__(self, from_names: Union[Set[str], List[str]], to_name: str):
        """
        Args:
            from_names: 要被替换的源名称集合或列表 (例如 {'t5', 'llama', 'utils'})。
            to_name:    用于替换的目标名称 (例如 'qwen2')。
        """
        self.to_name = to_name
        
        # 1. 构建一个包含所有源名称的正则表达式 | (OR 逻辑)
        #    - 使用 re.escape() 确保特殊字符被正确处理。
        #    - 使用 | 符号连接所有名称，实现多选一匹配。
        #    - 确保列表非空
        if not from_names:
             raise ValueError("from_names 列表不能为空。")

        escaped_names = [re.escape(name) for name in from_names]
        pattern = "|".join(escaped_names)
        
        # 2. 编译一个不区分大小写 (re.IGNORECASE) 的正则表达式
        self.regex = re.compile(pattern, re.IGNORECASE)

    def _case_preserving_replace(self, match: re.Match) -> str:
        """
        这是一个自定义的替换函数，它根据匹配到的字符串的大小写风格，
        来决定 to_name 应该使用哪种大小写形式。
        """
        found_str = match.group(0)
        # 如果找到的是全大写 (e.g., LLAMA)
        if found_str.isupper():
            return self.to_name.upper()
        # 如果找到的是首字母大写 (e.g., Llama)
        if found_str.istitle():
            return self.to_name.title()
        # 默认情况，包括全小写 (e.g., llama)，返回全小写
        return self.to_name.lower()

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.Name:
        """
        当访问离开一个名称节点时，使用正则表达式和自定义替换函数执行重命名。
        """
        # 使用 regex.sub() 和我们的自定义函数来进行替换
        new_name_str = self.regex.sub(self._case_preserving_replace, updated_node.value)
        
        # 仅在名称确实发生改变时才创建一个新节点
        if new_name_str != updated_node.value:
            if not new_name_str.isidentifier():
                original_name = original_node.value
                # 警告，而不是跳过，因为这在依赖于上下文的重命名中可能是允许的。
                # 但对于 cst.Name 节点，它必须是有效标识符。
                print(f"警告：尝试将 '{original_name}' 重命名为无效标识符 '{new_name_str}'。跳过此重命名。")
                return updated_node
            return updated_node.with_changes(value=new_name_str)
        
        return updated_node

def rename_identifiers(source_code: str, from_names: Union[Set[str], List[str]], to_name: str) -> str:
    """
    接收一段Python源代码，将其中的所有 from_names 相关标识符安全地重命名为 to_name。

    Args:
        source_code: 包含Python代码的字符串。
        from_names:  要被替换的源名称集合或列表 (例如 {"t5", "llama"})。
        to_name:     用于替换的目标名称 (例如 "qwen2")。

    Returns:
        重构后的Python代码字符串。
    """
    try:
        module = cst.parse_module(source_code)
        transformer = GenericRenamerTransformer(from_names, to_name)
        modified_module = module.visit(transformer)
        return modified_module.code
    except cst.ParserSyntaxError as e:
        print(f"Error: Failed to parse the source code. {e}")
        return source_code
    except ValueError as e:
        print(f"Error in rename process: {e}")
        return source_code

# --- 示例用法 ---
# source_code = """
# class LlamaModel(T5Model):
#     def forward(self, input_ids):
#         return self.llama_layer(input_ids)
# LLAMA_CONFIG = 1
# """
# from_list = ['llama', 't5']
# to_name = 'qwen2'

# new_code = rename_identifiers(source_code, from_list, to_name)
# print(new_code)
# # 预期输出:
# # class Qwen2Model(Qwen2Model):
# #     def forward(self, input_ids):
# #         return self.qwen2_layer(input_ids)
# # QWEN2_CONFIG = 1