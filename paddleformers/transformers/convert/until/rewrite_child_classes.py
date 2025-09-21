import libcst as cst
from typing import Dict, Optional, List, Set, Tuple,Union
from libcst import matchers as m
import builtins
def extract_modified_params(child_init: cst.FunctionDef) -> Set[str]:
    """
    提取子类中修改过的父类参数名
    改进版：处理更多赋值情况
    """
    modified_params = set()
    
    class AssignmentVisitor(cst.CSTVisitor):
        def __init__(self):
            self.modified_attrs = set()
        
        def visit_Assign(self, node: cst.Assign) -> None:
            for target in node.targets:
                if isinstance(target.target, cst.Attribute):
                    if isinstance(target.target.value, cst.Name) and target.target.value.value == "self":
                        self.modified_attrs.add(target.target.attr.value)
        
        def visit_AugAssign(self, node: cst.AugAssign) -> None:
            if isinstance(node.target, cst.Attribute):
                if isinstance(node.target.value, cst.Name) and node.target.value.value == "self":
                    self.modified_attrs.add(node.target.attr.value)
    
    visitor = AssignmentVisitor()
    child_init.visit(visitor)
    return visitor.modified_attrs

def is_super_call(node: cst.Call) -> bool:
    """
    检查是否是 super() 调用
    """
    if isinstance(node.func, cst.Attribute) and node.func.attr.value == "__init__":
        if isinstance(node.func.value, cst.Call):
            if isinstance(node.func.value.func, cst.Name) and node.func.value.func.value == "super":
                return True
    return False

def merge_init_methods(
    child_init: cst.FunctionDef, 
    parent_init: cst.FunctionDef
) -> cst.FunctionDef:
    """
    合并__init__方法：
    1. 保留父类所有代码（包括super调用）
    2. 去除子类的super调用
    3. 子类修改的父类属性直接在父类位置替换
    4. 子类新增的属性放在最后
    5. 保持父类所有语句的原始顺序
    """
    # 获取子类修改过的参数名
    modified_params = extract_modified_params(child_init)
    
    # 构建父类语句列表（保持原始顺序）
    parent_stmts = []
    # 父类所有属性名集合
    parent_attrs = set()
    
    for stmt in parent_init.body.body:
        # 检查是否是 super 调用
        is_super_stmt = False
        if isinstance(stmt, cst.SimpleStatementLine):
            for expr in stmt.body:
                if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                    if is_super_call(expr.value):
                        is_super_stmt = True
                        break
        
        if is_super_stmt:
            parent_stmts.append(("super", stmt))
            continue
            
        # 收集属性赋值语句
        attr_names = set()
        if isinstance(stmt, cst.SimpleStatementLine):
            for expr in stmt.body:
                if isinstance(expr, cst.Assign):
                    for target in expr.targets:
                        if isinstance(target.target, cst.Attribute):
                            attr_name = target.target.attr.value
                            attr_names.add(attr_name)
                            parent_attrs.add(attr_name)
                elif isinstance(expr, cst.AugAssign):
                    if isinstance(expr.target, cst.Attribute):
                        attr_name = expr.target.attr.value
                        attr_names.add(attr_name)
                        parent_attrs.add(attr_name)
        
        if attr_names:
            # 标记为属性语句
            parent_stmts.append(("attr", stmt, attr_names))
        else:
            # 其他语句
            parent_stmts.append(("other", stmt))
    
    # 处理子类代码：替换修改的属性，保留新增属性
    child_attr_assignments = {}
    child_new_stmts = []
    
    for stmt in child_init.body.body:
        # 跳过super调用
        is_super_stmt = False
        if isinstance(stmt, cst.SimpleStatementLine):
            for expr in stmt.body:
                if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                    if is_super_call(expr.value):
                        is_super_stmt = True
                        break
        
        if is_super_stmt:
            continue
        
        # 收集属性赋值语句
        attr_names = set()
        if isinstance(stmt, cst.SimpleStatementLine):
            for expr in stmt.body:
                if isinstance(expr, cst.Assign):
                    for target in expr.targets:
                        if isinstance(target.target, cst.Attribute):
                            attr_name = target.target.attr.value
                            attr_names.add(attr_name)
                elif isinstance(expr, cst.AugAssign):
                    if isinstance(expr.target, cst.Attribute):
                        attr_name = expr.target.attr.value
                        attr_names.add(attr_name)
        
        if attr_names:
            # 记录子类属性赋值语句
            for attr_name in attr_names:
                child_attr_assignments[attr_name] = stmt
        else:
            # 非属性语句作为新增语句
            child_new_stmts.append(stmt)
    
    # 构建合并后的函数体：
    # 1. 按原始顺序处理父类所有语句
    # 2. 子类新增的语句放在最后
    new_body = []
    
    # 步骤1: 按顺序处理父类所有语句
    for stmt_info in parent_stmts:
        stmt_type = stmt_info[0]
        stmt = stmt_info[1]
        
        if stmt_type == "super":
            # 保留super调用（指向祖父类）
            new_body.append(stmt)
        elif stmt_type == "other":
            # 保留其他语句
            new_body.append(stmt)
        elif stmt_type == "attr":
            # 处理属性语句
            attr_names = stmt_info[2]
            
            # 检查是否有属性被子类修改
            modified = False
            for attr_name in attr_names:
                if attr_name in modified_params and attr_name in child_attr_assignments:
                    # 使用子类的实现替换
                    stmt = child_attr_assignments[attr_name]
                    modified = True
                    # 只替换一次（一个语句可能包含多个属性）
                    break
            
            new_body.append(stmt)
    
    # 步骤2: 添加子类新增的语句（包括新增属性和其他语句）
    # 首先，收集所有已经被处理的语句（在父类位置替换的）
    processed_stmts = set(new_body)
    
    # 然后，添加真正新增的语句
    for stmt in child_init.body.body:
        # 跳过super调用
        is_super_stmt = False
        if isinstance(stmt, cst.SimpleStatementLine):
            for expr in stmt.body:
                if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                    if is_super_call(expr.value):
                        is_super_stmt = True
                        break
        
        if is_super_stmt:
            continue
        
        # 检查是否是新增属性（不在父类中的属性）
        is_new_attr = False
        if stmt not in processed_stmts:
            # 检查这个语句是否包含新增属性
            attr_names = set()
            if isinstance(stmt, cst.SimpleStatementLine):
                for expr in stmt.body:
                    if isinstance(expr, cst.Assign):
                        for target in expr.targets:
                            if isinstance(target.target, cst.Attribute):
                                attr_name = target.target.attr.value
                                attr_names.add(attr_name)
                    elif isinstance(expr, cst.AugAssign):
                        if isinstance(expr.target, cst.Attribute):
                            attr_name = expr.target.attr.value
                            attr_names.add(attr_name)
            
            # 如果包含至少一个不在父类中的属性，或者是非属性语句
            if not attr_names or any(attr_name not in parent_attrs for attr_name in attr_names):
                new_body.append(stmt)
                processed_stmts.add(stmt)
    
    return child_init.with_changes(
        body=child_init.body.with_changes(body=new_body)
    )

def get_base_class_name(base: cst.BaseExpression) -> Optional[str]:
    """
    提取基类名称，处理更多情况
    """
    if isinstance(base, cst.Name):
        return base.value
    elif isinstance(base, cst.Attribute):
        # 处理如 parent.Child 的情况
        parts = []
        node = base
        while isinstance(node, cst.Attribute):
            parts.append(node.attr.value)
            node = node.value
        if isinstance(node, cst.Name):
            parts.append(node.value)
            return ".".join(reversed(parts))
    return None

def _get_class_var_names(class_body: list) -> set:
    """一个辅助函数，用于从类的 body 中提取所有类变量的名称。"""
    var_names = set()
    for stmt in class_body:
        # 类变量通常是一个简单语句行，内部包含一个 Assign 节点
        if isinstance(stmt, cst.SimpleStatementLine) and stmt.body and isinstance(stmt.body[0], cst.Assign):
            assign_node = stmt.body[0]
            for target in assign_node.targets:
                # 只处理简单的名称赋值，如 a = 1，忽略 a.b = 1
                if isinstance(target.target, cst.Name):
                    var_names.add(target.target.value)
    return var_names

def merge_parent_class(
    child_class: cst.ClassDef, 
    parent_class: cst.ClassDef
) -> cst.ClassDef:
    """
    类合并主函数（修正版）：
    能够正确处理子类对父类公共变量的覆盖。
    """
    child_body = list(child_class.body.body)
    
    # ... (定位和合并 __init__ 方法的逻辑保持不变) ...
    # 定位子类的__init__
    child_init = None
    init_index = -1
    for i, stmt in enumerate(child_body):
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value == "__init__":
            child_init = stmt
            init_index = i
            break
    
    # 定位父类的__init__
    parent_init = None
    for stmt in parent_class.body.body:
        if isinstance(stmt, cst.FunctionDef) and stmt.name.value == "__init__":
            parent_init = stmt
            break
    
    # 合并__init__方法
    if child_init and parent_init:
        merged_init = merge_init_methods(child_init, parent_init)
        new_body = child_body[:init_index] + [merged_init] + child_body[init_index+1:]
    elif parent_init and not child_init:
        new_body = [parent_init] + child_body
    else:
        new_body = child_body
    
    # --- 核心修正：改进“合并其他成员”的逻辑 ---

    # 1. 预先收集子类中已有的成员名（用于方法等）和类变量名
    existing_member_names = {
        stmt.name.value
        for stmt in child_body 
        if hasattr(stmt, 'name') and isinstance(stmt.name, cst.Name)
    }
    child_class_var_names = _get_class_var_names(child_body)
    
    # 2. 遍历父类，只添加子类中不存在的成员和变量
    for stmt in parent_class.body.body:
        # a. 处理带 name 的成员（方法、内部类等）
        if hasattr(stmt, 'name') and isinstance(stmt.name, cst.Name):
            if stmt.name.value == "__init__":
                continue
            if stmt.name.value not in existing_member_names:
                new_body.append(stmt)
        
        # b. 专门处理类变量赋值语句
        elif isinstance(stmt, cst.SimpleStatementLine) and stmt.body and isinstance(stmt.body[0], cst.Assign):
            parent_var_names = _get_class_var_names([stmt])
            # 只有当这个变量没有被子类覆盖时，才添加
            if not any(name in child_class_var_names for name in parent_var_names):
                new_body.append(stmt)
        
        # c. 处理其他类型的语句（比如父类的文档字符串，如果子类没有的话）
        else:
             # 为避免重复添加父类的文档字符串，可以加一个检查
            is_parent_docstring = (isinstance(stmt, cst.SimpleStatementLine) and stmt.body and 
                                   isinstance(stmt.body[0], cst.Expr) and isinstance(stmt.body[0].value, cst.SimpleString))
            if not is_parent_docstring: # 简单地忽略父类的文档字符串
                 new_body.append(stmt)

     # +++ 新增：清理冗余的 pass 语句 +++
    # 使用 LibCST Matchers 来精确匹配 pass 语句
    pass_matcher = m.SimpleStatementLine(body=[m.Pass()])
    
    # 过滤掉所有 pass 语句
    cleaned_body = [stmt for stmt in new_body if not m.matches(stmt, pass_matcher)]
    
    # 如果清理后 body 为空，为了保持语法正确，再把 pass 加回来
    if not cleaned_body:
        cleaned_body.append(cst.SimpleStatementLine(body=[cst.Pass()]))
        
    # --- 返回最终结果 ---
    return child_class.with_changes(
        body=child_class.body.with_changes(body=cleaned_body),
        bases=parent_class.bases  # 继承父类的基类（祖父类）
    )

def find_class_in_source(module_node: cst.Module) -> Optional[cst.ClassDef]:
    """从一个已解析的模块节点中安全地提取第一个类定义。"""
    for node in module_node.body:
        if isinstance(node, cst.ClassDef):
            return node
    return None

class DependencyVisitor(cst.CSTVisitor):
    """一个CST访问器，用于扫描代码以查找所有潜在的外部引用。"""
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
    """分析一个节点的CST，找出其使用到的、且存在于expanded中的其他实体。"""
    visitor = DependencyVisitor()
    node.visit(visitor)
    return {dep for dep in visitor.dependencies if dep in expanded}

def rewrite_child_classes(
    expanded: Dict[str, str],
    target_file: str,
    output_file: str
):
    """
    完整的类重写工具。
    终极版V5：采用两阶段分析，正确处理“既是合并源又是依赖项”的复杂场景。
    """
    # --- 阶段一：预解析所有外部源代码 (不变) ---
    print("阶段一：正在预解析所有父类代码...")
    parsed_expanded: Dict[str, cst.Module] = {}
    imports_to_inject: Dict[str, cst.BaseSmallStatement] = {}
    # ... (此部分代码不变, 省略)
    for name, source in expanded.items():
        try:
            module_node = cst.parse_module(source)
            parsed_expanded[name] = module_node
            for node in module_node.body:
                 if isinstance(node, cst.SimpleStatementLine) and node.body and isinstance(node.body[0], (cst.Import, cst.ImportFrom)):
                    import_code = module_node.code_for_node(node)
                    imports_to_inject[import_code] = node
        except Exception as e:
            print(f"警告：预解析 {name} 失败: {e}")

    # --- 阶段二：读取并解析目标文件 (不变) ---
    print("\n阶段二：正在分析目标文件...")
    with open(target_file, "r", encoding="utf-8") as f:
        code = f.read()
    module = cst.parse_module(code)

    imports_from_target: Dict[str, cst.SimpleStatementLine] = {}
    body_statements: List[cst.BaseStatement] = []
    # ... (分离 import 和 body 的代码不变, 省略)
    for stmt in module.body:
        is_import = isinstance(stmt, cst.SimpleStatementLine) and stmt.body and isinstance(stmt.body[0], (cst.Import, cst.ImportFrom))
        if is_import:
            imports_from_target[module.code_for_node(stmt)] = stmt
        else:
            body_statements.append(stmt)
            
    # --- 核心修正：两阶段分析 ---

    nodes_to_inject: Dict[str, Union[cst.ClassDef, cst.FunctionDef]] = {}
    existing_names: Set[str] = {stmt.name.value for stmt in body_statements if hasattr(stmt, 'name')}
    visiting: Set[str] = set()

    def collect_dependencies(name: str):
        # ... (此内部函数本身逻辑不变, 省略)
        if name not in parsed_expanded or name in nodes_to_inject or name in existing_names: return
        if name in visiting: return
        
        entity_node = None
        module_node = parsed_expanded[name]
        for node in module_node.body:
            if isinstance(node, (cst.ClassDef, cst.FunctionDef)) and node.name.value == name:
                entity_node = node
                break
        
        if not entity_node: return
        visiting.add(name)
        if isinstance(entity_node, cst.ClassDef):
            for base in entity_node.bases:
                base_name = get_base_class_name(base.value)
                if base_name: collect_dependencies(base_name)
        usage_deps = find_usage_dependencies(entity_node, expanded)
        for dep_name in usage_deps:
            collect_dependencies(dep_name)
        visiting.remove(name)
        nodes_to_inject[name] = entity_node

    # +++ 阶段 3.1: 全局依赖扫描 +++
    print("\n阶段三：正在进行全局依赖扫描...")
    for stmt in body_statements:
        if isinstance(stmt, cst.ClassDef):
            # 扫描继承依赖
            for base in stmt.bases:
                base_name = get_base_class_name(base.value)
                if base_name:
                    collect_dependencies(base_name)
            # 扫描使用依赖
            usage_deps = find_usage_dependencies(stmt, expanded)
            for dep_name in usage_deps:
                collect_dependencies(dep_name)
        elif isinstance(stmt, cst.FunctionDef):
             # 函数也可能有使用依赖
            usage_deps = find_usage_dependencies(stmt, expanded)
            for dep_name in usage_deps:
                collect_dependencies(dep_name)
    
    # 在这个阶段结束后, nodes_to_inject 包含了所有潜在需要的依赖项

    # +++ 阶段 3.2: 执行合并并记录被合并的父类 +++
    print("\n阶段四：正在执行类合并操作...")
    processed_body_statements = []
    merged_parents: Set[str] = set()

    for stmt in body_statements:
        if isinstance(stmt, cst.ClassDef) and stmt.bases:
            base_name = get_base_class_name(stmt.bases[0].value)
            if base_name and base_name in parsed_expanded:
                parent_module = parsed_expanded[base_name]
                parent_class_node = find_class_in_source(parent_module)
                if parent_class_node:
                    print(f"  > 正在合并 {base_name} -> {stmt.name.value}...")
                    stmt = merge_parent_class(stmt, parent_class_node)
                    # 记录下这个父类已经被合并掉了
                    merged_parents.add(base_name)
        
        processed_body_statements.append(stmt)
    
    # --- 阶段五：按正确顺序重新组装文件 ---
    print("\n阶段五：正在生成最终文件...")
    
    # +++ 最终过滤：从注入列表中移除被合并掉的父类 +++
    final_nodes_to_inject = {
        name: node for name, node in nodes_to_inject.items()
        if name not in merged_parents
    }

    # ... (后面的文件组装逻辑使用 final_nodes_to_inject 即可, 代码不变)
    final_imports = {**imports_from_target, **imports_to_inject}
    new_body = []
    
    if final_imports:
        sorted_imports = sorted(final_imports.values(), key=lambda node: module.code_for_node(node))
        new_body.extend(sorted_imports)

    injected_funcs = {k: v for k, v in final_nodes_to_inject.items() if isinstance(v, cst.FunctionDef)}
    injected_classes = {k: v for k, v in final_nodes_to_inject.items() if isinstance(v, cst.ClassDef)}

    if injected_funcs:
        new_body.append(cst.EmptyLine())
        new_body.append(cst.EmptyLine(comment=cst.Comment("# --- Injected Dependency Functions ---")))
        new_body.extend(injected_funcs.values())

    if injected_classes:
        new_body.append(cst.EmptyLine())
        new_body.append(cst.EmptyLine(comment=cst.Comment("# --- Injected Dependency Classes ---")))
        new_body.extend(injected_classes.values())

    if processed_body_statements:
        new_body.append(cst.EmptyLine())
        new_body.append(cst.EmptyLine(comment=cst.Comment("# --- Main Application Logic ---")))
        new_body.extend(processed_body_statements)

    new_module = module.with_changes(body=new_body)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(new_module.code)
    
    print(f"\n成功生成合并后的文件: {output_file}")
# 示例用法
if __name__ == "__main__":
    # 示例配置
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
        
        # 祖父类不需要展开
        "GrandParentClass": '''
class GrandParentClass:
    def __init__(self):
        self.grand_param = "祖父参数"
        
    def grand_method(self):
        return "祖父方法"
        '''
    }
    
    # 实际使用
    rewrite_child_classes(
        expanded=expanded_parents,
        target_file="child_class.py",
        output_file="merged_class.py"
    )