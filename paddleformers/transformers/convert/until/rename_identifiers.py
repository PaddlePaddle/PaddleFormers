import libcst as cst
from libcst import CSTTransformer
import re
import os

class GenericRenamerTransformer(CSTTransformer):
    """
    一个通用的CST转换器，用于安全地将代码中的标识符从一个名称替换为另一个，
    并能智能地保留原始名称的大小写风格。
    """
    def __init__(self, from_name: str, to_name: str):
        self.from_name = from_name
        self.to_name = to_name
        # 编译一个不区分大小写的正则表达式，用于查找 from_name
        # re.escape() 用于安全处理 from_name 中可能包含的特殊字符
        self.regex = re.compile(re.escape(from_name), re.IGNORECASE)

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
            return updated_node.with_changes(value=new_name_str)
        
        return updated_node

def rename_identifiers(source_code: str, from_name: str, to_name: str) -> str:
    """
    接收一段Python源代码，将其中的 from_name 相关标识符安全地重命名为 to_name。

    Args:
        source_code: 包含Python代码的字符串。
        from_name:   要被替换的源名称 (例如 "llama")。
        to_name:     用于替换的目标名称 (例如 "qwen2")。

    Returns:
        重构后的Python代码字符串。
    """
    try:
        module = cst.parse_module(source_code)
        transformer = GenericRenamerTransformer(from_name, to_name)
        modified_module = module.visit(transformer)
        return modified_module.code
    except cst.ParserSyntaxError as e:
        print(f"Error: Failed to parse the source code. {e}")
        return source_code

# --- 示例用法 ---

if __name__ == "__main__":
    
    # 1. 根据您的示例，我们从文件名中推断出目标名称
    target_filename = "mudular_qwen2.py"
    # 假设源名称是 'llama'
    from_name = "llama"
    # 从文件名提取 'qwen2'
    to_name = os.path.basename(target_filename).split('_')[-1].split('.')[0]

    print(f"执行重构任务: 将 '{from_name}' 重命名为 '{to_name}' (智能大小写)\n")

    # 2. 准备一个包含各种大小写风格的输入代码
    input_code = """
import torch
from torch import nn

# 类定义和继承 (首字母大写)
class LlamaRotaryEmbedding(nn.Module):
    pass

class MyModel(LlamaRotaryEmbedding):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # 变量赋值 (驼峰式/小写)
        self.llama_mlp = LlamaMLP(config)
        # 全大写常量
        self.LLAMA_VERSION = "2.0"

    def forward(self, hidden_states):
        # 局部变量 (全小写)
        llama_output = self.llama_mlp(hidden_states)
        return llama_output

def some_utility_for_Llama():
    print(f"This is a {LLAMA_VERSION} utility.")
"""

    print("--- 原始代码 ---")
    print(input_code)
    print("\n" + "="*50 + "\n")

    # 3. 执行通用的重命名重构
    output_code = rename_identifiers(
        source_code=input_code,
        from_name=from_name,
        to_name=to_name
    )

    print("--- 重构后的代码 ---")
    print(output_code)