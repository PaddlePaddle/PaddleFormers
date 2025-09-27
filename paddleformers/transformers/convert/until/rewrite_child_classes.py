import libcst as cst
from typing import Dict, Optional, List, Set, Union
from libcst import matchers as m
import builtins
import os

# ==============================================================================
# SECTION 1: 智能类合并引擎
# ==============================================================================


def get_node_code(node: cst.CSTNode) -> str:
    """辅助函数，用于获取CST节点的代码字符串，以便比较。"""
    return cst.Module(body=[node]).code.strip()

def merge_parameters(
    child_params: cst.Parameters, parent_params: cst.Parameters
) -> cst.Parameters:
    """智能合并两个方法的参数列表。"""
    child_param_map = {p.name.value: p for p in child_params.params}
    
    insertion_point = len(child_params.params)
    for i, p in enumerate(child_params.params):
        if p.star:
            insertion_point = i
            break

    new_params_from_parent = []
    for p in parent_params.params:
        if p.name.value not in child_param_map and p.default is not None:
            new_params_from_parent.append(p)

    final_params_list = list(child_params.params)
    final_params_list[insertion_point:insertion_point] = new_params_from_parent
    
    return child_params.with_changes(params=tuple(final_params_list))

def _get_class_var_names(class_body: list) -> set:
    """从类的 body 中提取所有类变量的名称。"""
    var_names = set()
    for stmt in class_body:
        if m.matches(stmt, m.SimpleStatementLine(body=[m.Assign()])):
            assign_node = stmt.body[0]
            for target in assign_node.targets:
                if isinstance(target.target, cst.Name):
                    var_names.add(target.target.value)
    return var_names

def merge_parent_class_final(
    child_class: cst.ClassDef, parent_class: cst.ClassDef
) -> cst.ClassDef:
    """
    类合并主函数（最终智能版）：
    - 智能展开super()调用，避免代码冗余。
    - 智能合并方法的参数列表，防止运行时错误。
    - 正确处理类变量和未覆盖方法的继承。
    """
    child_body_list = list(child_class.body.body)
    parent_body_map = {
        stmt.name.value: stmt
        for stmt in parent_class.body.body
        if hasattr(stmt, 'name') and isinstance(stmt.name, cst.Name)
    }

    final_body = list(child_body_list)

    # 1. 处理被子类覆盖的方法 (包括 __init__)
    for i, child_stmt in enumerate(child_body_list):
        if not isinstance(child_stmt, cst.FunctionDef):
            continue

        method_name = child_stmt.name.value
        parent_method = parent_body_map.get(method_name)

        if not parent_method or not isinstance(parent_method, cst.FunctionDef):
            continue
        
        # 1a. 智能展开 super()
        child_method_body = list(child_stmt.body.body)
        parent_method_body = list(parent_method.body.body)
        
        super_call_index = -1
        for j, stmt in enumerate(child_method_body):
            if m.matches(stmt, m.SimpleStatementLine(body=[m.Expr(value=m.Call(func=m.Attribute(value=m.Call(func=m.Name("super")))))]) ) \
            or m.matches(stmt, m.Return(value=m.Call(func=m.Attribute(value=m.Call(func=m.Name("super")))))):
                super_call_index = j
                break
        
        new_method_body_stmts = child_method_body
        if super_call_index != -1:
            child_prefix_stmts = child_method_body[:super_call_index]
            child_suffix_stmts = child_method_body[super_call_index + 1:]
            child_prefix_codes = [get_node_code(s) for s in child_prefix_stmts]
            
            divergence_index = 0
            for k, parent_stmt in enumerate(parent_method_body):
                if k < len(child_prefix_codes) and get_node_code(parent_stmt) == child_prefix_codes[k]:
                    divergence_index += 1
                else:
                    break
            
            parent_suffix_stmts = parent_method_body[divergence_index:]
            new_method_body_stmts = child_prefix_stmts + parent_suffix_stmts + child_suffix_stmts

        # 1b. 合并参数列表
        new_params = merge_parameters(child_stmt.params, parent_method.params)

        # 1c. 创建最终的方法节点
        new_body_block = child_stmt.body.with_changes(body=tuple(new_method_body_stmts))
        final_method = child_stmt.with_changes(body=new_body_block, params=new_params)
        
        final_body[i] = final_method

    # 2. 添加父类中未被覆盖的成员
    child_member_names = {stmt.name.value for stmt in final_body if hasattr(stmt, 'name')}
    child_class_var_names = _get_class_var_names(final_body)
    
    for parent_stmt in parent_class.body.body:
        if hasattr(parent_stmt, 'name') and parent_stmt.name.value in child_member_names:
            continue
            
        if m.matches(parent_stmt, m.SimpleStatementLine(body=[m.Assign()])):
            parent_var_names = _get_class_var_names([parent_stmt])
            if not parent_var_names.isdisjoint(child_class_var_names):
                continue

        final_body.append(parent_stmt)
    
    # 3. 清理 pass 语句
    pass_matcher = m.SimpleStatementLine(body=[m.Pass()])
    non_pass_statements = [stmt for stmt in final_body if not m.matches(stmt, pass_matcher)]
    
    if not non_pass_statements:
        cleaned_body = (cst.SimpleStatementLine(body=(cst.Pass(),)),)
    else:
        cleaned_body = tuple(non_pass_statements)
        
    # 4. 返回最终结果
    return child_class.with_changes(
        bases=parent_class.bases,
        body=child_class.body.with_changes(body=cleaned_body)
    )

# ==============================================================================
# SECTION 2:代码重构工具框架 (已集成新逻辑)
# ==============================================================================

class ComprehensiveRenamer(cst.CSTTransformer):
    """智能、大小写敏感地重命名所有匹配的名称。"""
    def __init__(self, rename_map: Dict[str, str]):
        self.rename_pairs = []
        for from_sub, to_sub in rename_map.items():
            self.rename_pairs.append((from_sub.lower(), to_sub.lower()))
            self.rename_pairs.append((from_sub.capitalize(), to_sub.capitalize()))
            self.rename_pairs.append((from_sub.upper(), to_sub.upper()))
        self.rename_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        for from_name, to_name in self.rename_pairs:
            if from_name in original_node.value:
                new_value = original_node.value.replace(from_name, to_name)
                return updated_node.with_changes(value=new_value)
        return updated_node

def get_base_class_name(base: cst.BaseExpression) -> Optional[str]:
    """提取基类名称。"""
    if isinstance(base, cst.Name):
        return base.value
    elif isinstance(base, cst.Attribute):
        parts = []
        node = base
        while isinstance(node, cst.Attribute):
            parts.append(node.attr.value)
            node = node.value
        if isinstance(node, cst.Name):
            parts.append(node.value)
            return ".".join(reversed(parts))
    return None

def find_class_in_source(module_node: cst.Module) -> Optional[cst.ClassDef]:
    """从模块节点中提取第一个类定义。"""
    for node in module_node.body:
        if isinstance(node, cst.ClassDef):
            return node
    return None

class DependencyVisitor(cst.CSTVisitor):
    """扫描代码以查找所有潜在的外部引用。"""
    def __init__(self):
        self.scopes: List[Set[str]] = [set()]
        self.dependencies: Set[str] = set()
        self.builtins = set(dir(builtins))

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        param_names = {p.name.value for p in node.params.params}
        self.scopes.append(param_names)

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.scopes.pop()

    def visit_Assign(self, node: cst.Assign) -> None:
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                self.scopes[-1].add(target.target.value)

    def visit_Name(self, node: cst.Name) -> None:
        is_local = any(node.value in scope for scope in self.scopes)
        if not is_local and node.value not in self.builtins:
            self.dependencies.add(node.value)

def find_usage_dependencies(node: Union[cst.ClassDef, cst.FunctionDef], expanded: Dict[str, str]) -> Set[str]:
    """分析节点的CST，找出其使用到的其他实体。"""
    visitor = DependencyVisitor()
    node.visit(visitor)
    return {dep for dep in visitor.dependencies if dep in expanded}

def get_full_name(node: Union[cst.Name, cst.Attribute, cst.ImportFrom]) -> str:
    """
    从CST节点递归获取完整名称，如 a.b.c 或 ..a.b
    """
    if isinstance(node, cst.Name):
        return node.value
    elif isinstance(node, cst.Attribute):
        # 递归获取基础部分 (a.b)
        base_name = get_full_name(node.value)
        # 拼接当前属性 (.c)
        return f"{base_name}.{node.attr.value}" if base_name else node.attr.value
    elif isinstance(node, cst.ImportFrom):
        # 处理 from ... import ... 语句的模块路径
        module_parts = []
        if node.relative:
            module_parts.append("." * len(node.relative))
        if node.module:
            module_parts.append(get_full_name(node.module))
        return "".join(module_parts)
    return ""

def filter_specific_modeling_imports(
    import_nodes: Union[Dict[str, cst.BaseSmallStatement], List[cst.BaseSmallStatement]]
) -> Dict[str, cst.BaseSmallStatement]:
    """
    【修正版】只移除严格符合 `from ..***.modeling import ...` 模式的导入。
    
    这个版本可以智能处理输入是字典或列表的情况，并且总是返回一个字典。
    """
    kept_imports_dict: Dict[str, cst.BaseSmallStatement] = {}
    
    # 【核心修正】: 检查输入类型，并确保我们总是遍历 CST 节点
    nodes_to_iterate = []
    if isinstance(import_nodes, dict):
        # 如果输入是字典，我们只关心它的值（CST 节点）
        nodes_to_iterate = list(import_nodes.values())
    elif isinstance(import_nodes, list):
        # 如果输入已经是列表，直接使用
        nodes_to_iterate = import_nodes

    for node in nodes_to_iterate:
        should_keep = True
        
        if isinstance(node, cst.ImportFrom):
            is_two_dots_relative = node.relative and len(node.relative) == 2
            
            if is_two_dots_relative:
                module_path = get_full_name(node.module) if node.module else ""
                
                if module_path.endswith(".modeling"):
                    should_keep = False

        if should_keep:
            kept_imports_dict[get_node_code(node)] = node
            
    return kept_imports_dict

class EntityFinder(cst.CSTVisitor):
    """
    A visitor to find the first ClassDef or FunctionDef node in a CST.
    """
    def __init__(self):
        self.found_node = None

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Found a class, store it and stop searching
        if self.found_node is None:
            self.found_node = node
        return False  # Return False to stop traversing deeper

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # Found a function, store it and stop searching
        if self.found_node is None:
            self.found_node = node
        return False # Return False to stop traversing deeper

def find_entity_in_source(source_cst_node: cst.Module) -> Optional[cst.CSTNode]:
    """
    Parses a CST module to find the first class or function definition.

    Args:
        source_cst_node: The parsed Concrete Syntax Tree of the source file.

    Returns:
        The found ClassDef or FunctionDef node, or None if not found.
    """
    if not isinstance(source_cst_node, cst.Module):
        # Ensure we have a valid CST to visit
        return None

    finder = EntityFinder()
    source_cst_node.visit(finder)
    return finder.found_node

def rewrite_child_classes(
    expanded: Dict[str, str],
    target_file: str,
    template_comment: str,
    output_file: str,
    rename_map: Optional[Dict[str, str]] = None
):
    """完整的类重写工具 (已集成VFinal版合并引擎)。"""
    if rename_map is None: rename_map = {}
        
    # --- 阶段一 & 二：解析代码 ---
    print("阶段一：正在预解析所有父类代码...")
    parsed_expanded: Dict[str, cst.Module] = {}
    imports_to_inject: Dict[str, cst.BaseSmallStatement] = {}
    for name, source in expanded.items():
        try:
            module_node = cst.parse_module(source)
            parsed_expanded[name] = module_node
            for node in module_node.body:
                if m.matches(node, m.SimpleStatementLine(body=[m.Import() | m.ImportFrom()])):
                    imports_to_inject[module_node.code_for_node(node)] = node
        except Exception as e:
            print(f"警告：预解析 {name} 失败: {e}")

    print("\n阶段二：正在分析目标文件...")
    with open(target_file, "r", encoding="utf-8") as f:
        module = cst.parse_module(f.read())

    imports_from_target: Dict[str, cst.SimpleStatementLine] = {}
    body_statements: List[cst.BaseStatement] = []
    for stmt in module.body:
        # 匹配导入语句
        if m.matches(stmt, m.SimpleStatementLine(body=[m.Import() | m.ImportFrom()])):
            imports_from_target[module.code_for_node(stmt)] = stmt
        
        # 匹配 try-except 块（通常用于可选导入）
        elif isinstance(stmt, cst.Try):
            imports_from_target[module.code_for_node(stmt)] = stmt
        
        # 匹配 __all__ 定义
        elif m.matches(stmt, m.SimpleStatementLine(body=[m.Assign(targets=[m.AssignTarget(target=m.Name("__all__"))])])):
            imports_from_target[module.code_for_node(stmt)] = stmt
        
        # 其他语句放入主体
        else:
            body_statements.append(stmt)
    imports_from_target=filter_specific_modeling_imports(imports_from_target)    
    # --- 阶段三 & 四：依赖分析与合并 ---
    nodes_to_inject: Dict[str, Union[cst.ClassDef, cst.FunctionDef]] = {}
    existing_names: Set[str] = {stmt.name.value for stmt in body_statements if hasattr(stmt, 'name')}
    visiting: Set[str] = set()

    def collect_dependencies(name: str):
        # 1. 边界检查 (完全不变)
        # 无论是类还是函数，这些检查（是否已解析、已收集、已存在、正在访问）都同样适用。
        if name not in parsed_expanded or name in nodes_to_inject or name in existing_names or name in visiting:
            return

        # 2. 查找实体节点 (需要泛化)
        # find_entity_in_source 现在可以返回 ClassDef 或 FunctionDef 节点。
        entity_node = find_entity_in_source(parsed_expanded[name])
        if not entity_node:
            return

        # 3. 标记正在访问 (完全不变)
        visiting.add(name)

        # 4. 处理类特有的依赖：继承 (只对类执行)
        # 如果实体是类，才处理其父类依赖。函数没有继承，会自然跳过此块。
        if isinstance(entity_node, cst.ClassDef):
            for base in entity_node.bases:
                if base_name := get_base_class_name(base.value):
                    collect_dependencies(base_name)

        # 5. 处理通用依赖：使用关系 (对类和函数都执行)
        # 这里的 `find_usage_dependencies` 函数也必须是通用的，
        # 它需要能解析类和函数体内的依赖。
        # - 对于类: 查找成员变量的类型注解等。
        # - 对于函数: 查找参数的类型注解、返回值的类型注解、函数体内调用的其他函数、实例化的类等。
        for dep_name in find_usage_dependencies(entity_node, expanded):
            collect_dependencies(dep_name)

        # 6. 完成处理，加入结果集 (完全不变)
        # 无论是类还是函数，都在其所有依赖项被处理完毕后，才将自身加入结果集。
        visiting.remove(name)
        nodes_to_inject[name] = entity_node
    print("\n阶段三：正在进行全局依赖扫描...")
    for stmt in body_statements:
        if isinstance(stmt, cst.ClassDef):
            for base in stmt.bases:
                if base_name := get_base_class_name(base.value):
                    collect_dependencies(base_name)
            for dep_name in find_usage_dependencies(stmt, expanded):
                collect_dependencies(dep_name)
    
    print("\n阶段四：正在执行类合并操作...")
    processed_body_statements = []
    merged_parents: Set[str] = set()
    for stmt in body_statements:
        if isinstance(stmt, cst.ClassDef) and stmt.bases:
            if base_name := get_base_class_name(stmt.bases[0].value):
                if base_name in parsed_expanded:
                    parent_module = parsed_expanded[base_name]
                    if parent_class_node := find_class_in_source(parent_module):
                        print(f"  > 正在合并 {base_name} -> {stmt.name.value}...")
                        # <<<--- ★★★核心修改点：调用新的合并函数★★★
                        stmt = merge_parent_class_final(stmt, parent_class_node)
                        merged_parents.add(base_name)
        processed_body_statements.append(stmt)
        
    # --- 阶段五：按正确顺序重新组装文件 ---
    print("\n阶段五：正在生成最终文件...")
    
    nodes_to_inject_after_merge = {k: v for k, v in nodes_to_inject.items() if k not in merged_parents}
    main_defined_names = {stmt.name.value for stmt in processed_body_statements if hasattr(stmt, 'name')}
    
    print("  > 正在应用智能重命名规则并检测冲突...")
    final_nodes_to_inject = {}
    renamer = ComprehensiveRenamer(rename_map)

    for original_name, node in nodes_to_inject_after_merge.items():
        renamed_node = node.visit(renamer)
        new_name = renamed_node.name.value
        if new_name in main_defined_names:
            print(f"    - 检测到主代码中已存在 '{new_name}'，将跳过注入 '{original_name}'")
            continue
        print(f"    - 正在处理依赖 '{original_name}'...")
        final_nodes_to_inject[new_name] = renamed_node
            
    final_imports = {**imports_from_target, **imports_to_inject}
    new_body = []
    new_header = []
    #加转换注释
    for line in template_comment.splitlines():
        stripped_line = line.strip()
        if stripped_line:
            comment_node = cst.Comment(stripped_line)
            new_header.append(cst.EmptyLine(
                comment=comment_node,
                indent=True,
                whitespace=cst.SimpleWhitespace(value="")
            ))
    for item in module.header:
        if isinstance(item, cst.EmptyLine) and item.comment:
            new_header.append(item)
        elif isinstance(item, cst.TrailingWhitespace) and item.comment:
            new_header.append(item)

    if final_imports:
        unique_imports = {module.code_for_node(n): n for n in final_imports.values()}
        new_body.extend(unique_imports.values())

    injected_items = sorted(final_nodes_to_inject.values(), key=lambda n: n.name.value)
     # 2. 分类依赖项：方法和类
    methods_to_inject = []
    classes_to_inject = []
    for node in injected_items:
        if isinstance(node, cst.FunctionDef):
            print(node.name.value)
            methods_to_inject.append(node)
        elif isinstance(node, cst.ClassDef):
            classes_to_inject.append(node)
        else:
            print(f"警告：遇到未知类型的节点，无法分类: {type(node.name.value)}")
    # 3. 注入方法（放在 imports 之后，主逻辑之前）
    if methods_to_inject:
        new_body.extend([cst.EmptyLine(), cst.EmptyLine(comment=cst.Comment("# --- Injected Methods ---"))])
        new_body.extend(methods_to_inject)
     # 4. 处理类的注入顺序
     # 分组：有父类在主逻辑中的类 vs 没有的
    classes_with_parent_in_main = []
    classes_without_parent_in_main = []
    if classes_to_inject:
        # 获取主逻辑中的所有类名
        main_classes = {stmt.name.value for stmt in processed_body_statements if isinstance(stmt, cst.ClassDef)}
        
        
        
        for cls_node in classes_to_inject:
            has_parent_in_main = False
            if isinstance(cls_node, cst.ClassDef) and cls_node.bases:
                for base in cls_node.bases:
                    if base_name := get_base_class_name(base.value):
                        if base_name in main_classes:
                            has_parent_in_main = True
                            break
            
            if has_parent_in_main:
                classes_with_parent_in_main.append(cls_node)
            else:
                classes_without_parent_in_main.append(cls_node)

        # 4.1 先注入没有父类依赖的类（放在 imports 之后）
        if classes_without_parent_in_main:
            new_body.extend([cst.EmptyLine(), cst.EmptyLine(comment=cst.Comment("# --- Injected Classes ---"))])
            new_body.extend(classes_without_parent_in_main)

    
    # 4. 动态遍历主逻辑，在父类定义后插入其子类 
    if processed_body_statements:
        # 4.1 收集所有主逻辑的类名
        classes_with_parent_in_main = {
            cls for cls in classes_with_parent_in_main
            if isinstance(cls, cst.ClassDef)
        }
        
        # 4.2 按顺序处理主逻辑的语句
        for stmt in processed_body_statements:
            new_body.append(stmt)
            
            # 如果是类定义，检查是否有子类需要注入
            if isinstance(stmt, cst.ClassDef):
                parent_name = stmt.name.value
                # 查找依赖此父类的子类
                child_classes = [
                    cls for cls in classes_with_parent_in_main
                    if any(
                        get_base_class_name(base.value) == parent_name
                        for base in cls.bases
                    )
                ]
                # 注入子类
                if child_classes:
                    new_body.extend([
                        cst.EmptyLine(),
                        cst.EmptyLine(comment=cst.Comment(f"# --- Children of {parent_name} ---")),
                        *child_classes
                    ])
                    # 从待注入列表中移除已处理的子类
                    classes_with_parent_in_main = [
                        cls for cls in classes_with_parent_in_main
                        if cls not in child_classes
                    ]

# 5. 注入剩余未处理的依赖主逻辑的类（可能是跨文件的依赖）   
    if classes_with_parent_in_main:
        new_body.extend([cst.EmptyLine(), cst.EmptyLine(comment=cst.Comment("# --- Remaining Injected Child Classes ---"))])
        new_body.extend(classes_with_parent_in_main)
    
    """
    if injected_items:
        new_body.extend([cst.EmptyLine(), cst.EmptyLine(comment=cst.Comment("# --- Injected Dependencies ---"))])
        new_body.extend(injected_items)
    
    if processed_body_statements:
        new_body.extend([cst.EmptyLine(), cst.EmptyLine(comment=cst.Comment("# --- Main Application Logic ---"))])
        new_body.extend(processed_body_statements)
    """
    new_module = module.with_changes(
    header=tuple(new_header),  # 使用新的头部注释
    body=tuple(new_body)       # 使用新的主体内容
)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(new_module.code)
    
    print(f"\n成功生成合并后的文件: {output_file}")
    
# ==============================================================================
# SECTION 3: 演示
# ==============================================================================
if __name__ == "__main__":
    
    # --- 步骤1: 准备演示环境 ---
    # 创建一个虚拟的 child_class.py 文件供脚本读取
    child_class_content = """
class MyChildClass(ParentClass):
    def __init__(self, config, child_param):
        # 与父类重复的语句
        if config.flag:
            self.param1 = config.param1
        else:
            self.param1 = config.default_param1
        
        # 调用super
        super().__init__(config)
        
        # 新增的属性和逻辑
        self.child_param = child_param
        print("Child class logic executed.")

    def child_method(self):
        return "子类方法"
"""
    with open("child_class.py", "w", encoding="utf-8") as f:
        f.write(child_class_content)
    
    # --- 步骤2: 定义父类和祖父类源代码 ---
    expanded_parents = {
        "ParentClass": '''
class ParentClass(GrandParentClass):
    def __init__(self, config):
        # 条件语句
        if config.flag:
            self.param1 = config.param1
        else:
            self.param1 = config.default_param1
            
        # 循环语句
        for i in range(5):
            self.param2 = i
            
        # 方法调用
        self.initialize(config)
        
        # super调用（指向祖父类）
        super().__init__()
    
    def initialize(self, config):
        self.param3 = config.param3
        
    def parent_method(self):
        return "父类方法"
''',
        "GrandParentClass": '''
class GrandParentClass:
    def __init__(self):
        self.grand_param = "祖父参数"
        
    def grand_method(self):
        return "祖父方法"
'''
    }
    
    # --- 步骤3: 运行重写工具 ---
    print("--- 开始运行代码重写工具 ---")
    rewrite_child_classes(
        expanded=expanded_parents,
        target_file="child_class.py",
        output_file="merged_class.py"
    )
    
    # --- 步骤4: 打印结果 ---
    print("\n--- 查看生成的 merged_class.py 文件 ---")
    with open("merged_class.py", "r", encoding="utf-8") as f:
        print(f.read())
        
    # --- 步骤5: 清理 ---
    os.remove("child_class.py")
    os.remove("merged_class.py")