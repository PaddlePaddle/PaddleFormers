二、分步解析与函数作用
步骤 1：解析代码为抽象语法树（AST）
作用：将源代码转换为机器可操作的语法树结构，便于后续分析和修改。
核心函数：parse_code_to_ast(code: str) -> AST

调用 Python 内置 ast 模块，将字符串形式的代码解析为 AST 节点。

示例：
父类 BaseEncoder 的代码被解析为如下 AST 结构（简化）：

python
运行
# 伪代码表示 AST 结构
ClassDef(
    name='BaseEncoder',
    bases=[Name(id='nn.Module')],  # 基类
    body=[
        FunctionDef(name='__init__', ...),  # __init__ 方法
        FunctionDef(name='post_init', ...)  # post_init 方法
    ]
)
步骤 2：提取类结构信息
作用：从 AST 中提取父类和子类的关键信息（基类、属性、方法等），为合并做准备。
核心函数：extract_class_info(ast_node: AST) -> ClassInfo

返回一个数据结构 ClassInfo，包含：
bases：基类列表（如 [nn.Module]）
attrs：类属性（如 _version = "1.0"）
methods：方法字典（键为方法名，值为方法 AST 节点）
docstring：文档字符串

示例：
从 BaseEncoder 提取的信息：

python
运行
ClassInfo(
    bases=[nn.Module],
    methods={
        '__init__': FunctionDef(...),  # 包含 input_proj 等初始化逻辑
        'post_init': FunctionDef(...)
    },
    docstring="基础编码器：处理输入嵌入和基础层堆叠"
)
步骤 3：合并基类列表
作用：将子类和父类的基类合并，去除中间父类，只保留顶层基类。
核心函数：merge_bases(child_bases: list, parent_bases: list) -> list

逻辑：若子类继承父类（如 TransformerEncoder → BaseEncoder），则用父类的基类（BaseEncoder → nn.Module）替换子类中的父类。

示例：

子类原始基类：[BaseEncoder]
父类基类：[nn.Module]
合并后基类：[nn.Module]（去除中间父类 BaseEncoder）
步骤 4：合并类属性与文档字符串
作用：整合父类和子类的类属性（如 _supports_cuda = True）和文档字符串。
核心函数：merge_class_attrs(child_attrs: dict, parent_attrs: dict) -> dict

规则：子类属性覆盖父类同名属性，非同名属性全部保留。

示例：

父类属性：{'dropout_rate': 0.1}
子类属性：{'use_attention': True}
合并后属性：{'dropout_rate': 0.1, 'use_attention': True}
步骤 5：合并方法（核心步骤）
作用：整合父类和子类的方法，处理方法重写和 super() 调用展开。
核心函数：merge_methods(child_methods: dict, parent_methods: dict) -> dict

分两种情况处理：
情况 1：子类未重写的方法
直接继承父类方法。
示例：
父类有 post_init 方法，子类未定义，则合并后保留父类的 post_init。
情况 2：子类重写的方法（如 __init__）
通过 expand_super_calls 函数展开 super() 调用，将父类方法逻辑合并到子类中。
核心函数：expand_super_calls(child_method: AST, parent_method: AST) -> AST

步骤：
在子类方法中找到 super().method(...) 调用位置。
提取父类方法中除 super() 自身调用外的逻辑。
将父类逻辑插入到子类 super() 调用处，替换原 super() 语句。

示例：

子类 __init__ 中的 super().__init__(...) 被替换为父类 __init__ 的逻辑（input_proj、layer_norm 等初始化代码）。
步骤 6：修正代码规范
作用：修复合并过程中产生的代码不规范问题（如 super() 位置错误、重复语句）。
核心函数：fix_code_style(merged_ast: AST) -> AST

包含多个子步骤：
fix_super_position：确保 super().__init__ 在 __init__ 方法的文档字符串之后、子类逻辑之前。
fix_post_init_position：确保 post_init() 在 __init__ 末尾调用。
remove_duplicates：移除重复的属性赋值（如子类和父类都定义的 self.config）。

示例：

子类原始 __init__ 中 super() 调用前的代码（self.num_heads = ...）被移至 super() 之后。
重复的 self.dropout = ... 语句只保留子类的定义。
步骤 7：生成合并后的代码
作用：将处理后的 AST 转换回字符串形式的代码，并格式化。
核心函数：ast_to_code(merged_ast: AST) -> str

调用 ast.unparse() 将 AST 转换为代码字符串，再通过 ruff 工具格式化。

示例：
最终生成的 TransformerEncoder 代码包含：

直接继承 nn.Module
合并后的 __init__（包含父类的 input_proj 和子类的 attention）
继承的 post_init 方法
三、关键函数调用链
整个流程的函数调用关系如下：

python
运行
# 简化的主函数逻辑
def merge_classes(child_code: str, parent_code: str) -> str:
    # 步骤1：解析AST
    child_ast = parse_code_to_ast(child_code)
    parent_ast = parse_code_to_ast(parent_code)
    
    # 步骤2：提取信息
    child_info = extract_class_info(child_ast)
    parent_info = extract_class_info(parent_ast)
    
    # 步骤3-6：合并与修正
    merged_bases = merge_bases(child_info.bases, parent_info.bases)
    merged_attrs = merge_class_attrs(child_info.attrs, parent_info.attrs)
    merged_methods = merge_methods(child_info.methods, parent_info.methods)
    merged_ast = build_merged_ast(
        name=child_info.name,
        bases=merged_bases,
        attrs=merged_attrs,
        methods=merged_methods,
        docstring=child_info.docstring or parent_info.docstring
    )
    fixed_ast = fix_code_style(merged_ast)
    
    # 步骤7：生成代码
    return ast_to_code(fixed_ast)

四、总结
代码合并工具通过 AST 解析→信息提取→结构合并→逻辑整合→规范修正 的流程，将子类和父类的代码无缝融合。核心难点在于处理 __init__ 方法中的 super() 调用展开和参数合并，确保合并后的代码既保留子类定制逻辑，又继承父类基础功能，同时符合 Python 语法规范。

这个过程完全自动化，避免了手动复制粘贴可能导致的错误，特别适合大型项目中模块化代码的整合。