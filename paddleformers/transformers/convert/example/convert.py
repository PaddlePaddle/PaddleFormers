import libcst as cst
import libcst.matchers as m
from libcst.metadata import MetadataWrapper
from libcst.metadata import (
    PositionProvider,
    ParentNodeProvider,
    ScopeProvider,
    QualifiedNameProvider
)
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import importlib.util
import sys
import os
import inspect
from dataclasses import dataclass

@dataclass
class ClassInfo:
    """类信息数据结构"""
    node: cst.ClassDef
    bases: List[str]
    methods: Dict[str, cst.FunctionDef]
    attributes: List[Tuple[str, cst.BaseExpression]]
    dependencies: Set[str]

class ImportCollector(cst.CSTVisitor):
    """收集模块中的所有导入语句"""
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider, ScopeProvider)
    
    def __init__(self):
        super().__init__()
        self.imports: List[cst.Import] = []
        self.from_imports: List[cst.ImportFrom] = []
        self.imported_names: Set[str] = set()
    
    def visit_Import(self, node: cst.Import) -> None:
        self.imports.append(node)
        for name in node.names:
            self.imported_names.add(name.evaluated_name)
    
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        self.from_imports.append(node)
        module = node.module.evaluated_name if node.module else ""
        for name in node.names:
            full_name = f"{module}.{name.evaluated_name}" if module else name.evaluated_name
            self.imported_names.add(full_name)

class ClassCollector(cst.CSTVisitor):
    """收集模块中的所有类定义及其依赖"""
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
        ScopeProvider,
        QualifiedNameProvider
    )
    
    def __init__(self):
        super().__init__()
        self.classes: Dict[str, ClassInfo] = {}
        self.current_class: Optional[str] = None
    
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """收集类定义及其基类"""
        class_name = node.name.value
        bases = []
        
        # 收集基类名称
        for base in node.bases:
            if isinstance(base.value, cst.Name):
                bases.append(base.value.value)
            elif isinstance(base.value, cst.Attribute):
                # 处理形如BaseModel.Config的情况
                bases.append(self._get_full_attr_name(base.value))
        
        # 初始化类信息
        self.classes[class_name] = ClassInfo(
            node=node,
            bases=bases,
            methods={},
            attributes=[],
            dependencies=set(bases)
        )
        self.current_class = class_name
    
    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.current_class = None
    
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """收集类方法及其依赖"""
        if not self.current_class:
            return
            
        # 添加到类方法
        self.classes[self.current_class].methods[node.name.value] = node
        
        # 分析方法体中的依赖
        analyzer = DependencyAnalyzer()
        node.visit(analyzer)
        self.classes[self.current_class].dependencies.update(analyzer.dependencies)
    
    def visit_Assign(self, node: cst.Assign) -> None:
        """收集类属性及其依赖"""
        if not self.current_class:
            return
            
        # 处理类属性赋值
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                # 类级别变量
                self.classes[self.current_class].attributes.append(
                    (target.target.value, node.value)
                )
            elif (isinstance(target.target, cst.Attribute) and 
                  isinstance(target.target.value, cst.Name) and
                  target.target.value.value == "self"):
                # 实例属性
                self.classes[self.current_class].attributes.append(
                    (target.target.attr.value, node.value)
                )
        
        # 分析赋值表达式中的依赖
        analyzer = DependencyAnalyzer()
        node.visit(analyzer)
        self.classes[self.current_class].dependencies.update(analyzer.dependencies)
    
    def _get_full_attr_name(self, node: cst.Attribute) -> str:
        """获取完整的属性名"""
        if isinstance(node.value, cst.Name):
            return f"{node.value.value}.{node.attr.value}"
        elif isinstance(node.value, cst.Attribute):
            return f"{self._get_full_attr_name(node.value)}.{node.attr.value}"
        return node.attr.value

class DependencyAnalyzer(cst.CSTVisitor):
    """分析代码中的依赖关系"""
    METADATA_DEPENDENCIES = (QualifiedNameProvider,)
    
    def __init__(self):
        super().__init__()
        self.dependencies: Set[str] = set()
    
    def visit_Name(self, node: cst.Name) -> None:
        """收集名称依赖"""
        qnames = self.get_metadata(QualifiedNameProvider, node)
        for qname in qnames:
            if "." in qname.name:
                # 只取最外层的模块名
                module = qname.name.split(".", 1)[0]
                if module not in {"builtins", "typing"}:
                    self.dependencies.add(module)

class SuperCallTransformer(cst.CSTTransformer):
    """转换super().__init__()调用为展开的父类初始化代码"""
    
    def __init__(self, parent_class_init: cst.FunctionDef, class_name: str):
        self.parent_init = parent_class_init
        self.class_name = class_name
        self.super_call_found = False
        self.super_call_node = None
    
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        """处理super().__init__()调用"""
        if (m.matches(original_node.func, m.Attribute(
                value=m.Call(func=m.Name("super")),
                attr=m.Name("__init__")
            ))):
            self.super_call_found = True
            self.super_call_node = original_node
            
            # 创建展开后的父类初始化代码
            parent_body = self.parent_init.body
            
            # 添加注释说明
            comment = cst.EmptyLine(
                comment=cst.Comment(f"# 以下代码从父类 {self.parent_init.name.value} 展开")
            )
            
            # 替换super调用为父类代码
            return cst.FlattenSentinel([
                cst.SimpleStatementLine(body=[comment]),
                *parent_body.body
            ])
        
        return updated_node

class ModularModelConverter:
    """模块化模型转换器主类"""
    
    def __init__(
        self,
        base_model: str,
        new_model: str,
        modular_file: str,
        output_dir: str,
        base_model_dir: Optional[str] = None
    ):
        """
        初始化转换器
        :param base_model: 基础模型名称，如"Llama"
        :param new_model: 新模型名称，如"Qwen2"
        :param modular_file: 模块化定义文件路径
        :param output_dir: 输出目录
        :param base_model_dir: 基础模型代码目录(可选)
        """
        self.base_model = base_model
        self.new_model = new_model
        self.modular_file = os.path.abspath(modular_file)
        self.modular_dir = os.path.dirname(self.modular_file)
        self.output_dir = os.path.abspath(output_dir)
        self.base_model_dir = base_model_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 解析模块化文件
        with open(self.modular_file, "r", encoding="utf-8") as f:
            self.modular_content = f.read()
        
        # 创建CST模块并解析
        self.modular_cst = cst.parse_module(self.modular_content)
        self.modular_wrapper = MetadataWrapper(self.modular_cst)
        
        # 收集导入和类定义
        self._collect_imports_and_classes()
        
        # 初始化依赖分析器
        self.dependency_resolver = DependencyResolver(
            base_model=base_model,
            new_model=new_model,
            base_model_dir=base_model_dir,
            modular_dir=self.modular_dir
        )
    
    def _collect_imports_and_classes(self):
        """收集导入语句和类定义"""
        # 收集导入
        import_collector = ImportCollector()
        self.modular_wrapper.visit(import_collector)
        self.imports = import_collector.imports
        self.from_imports = import_collector.from_imports
        self.imported_names = import_collector.imported_names
        
        # 收集类定义
        class_collector = ClassCollector()
        self.modular_wrapper.visit(class_collector)
        self.classes = class_collector.classes
    
    def _resolve_dependencies(self):
        """解析所有依赖项"""
        # 解析模块依赖
        self.dependency_resolver.resolve(
            imports=self.imports,
            from_imports=self.from_imports,
            classes=self.classes
        )
        
        # 获取解析后的依赖代码
        self.dependency_code = self.dependency_resolver.get_dependency_code()
    
    def _generate_modeling_file(self) -> str:
        """生成modeling_<new_model>.py文件内容"""
        # 处理每个类定义
        updated_classes = []
        for class_name, class_info in self.classes.items():
            updated_class = self._process_class(class_info)
            updated_classes.append(updated_class)
        
        # 构建新的模块体
        new_body = []
        
        # 添加文件头注释
        new_body.append(cst.SimpleStatementLine(
            body=[cst.Comment(f"# Auto-generated modeling file for {self.new_model}")]
        ))
        
        # 添加必要的导入
        new_body.extend(self._generate_required_imports())
        
        # 添加依赖代码
        for code_block in self.dependency_code.values():
            new_body.append(code_block)
        
        # 添加处理后的类定义
        new_body.extend(updated_classes)
        
        # 创建完整模块
        new_module = cst.Module(body=new_body)
        
        # 重命名类
        new_module = self._rename_classes(new_module)
        
        # 生成代码字符串
        return new_module.code
    
    def _process_class(self, class_info: ClassInfo) -> cst.ClassDef:
        """处理单个类定义"""
        # 展开super().__init__调用
        class_def = self._expand_super_init(class_info.node)
        
        # 处理其他方法
        updated_methods = []
        for method_name, method_node in class_info.methods.items():
            if method_name != "__init__":
                updated_methods.append(method_node)
        
        # 更新类体
        return class_def.with_changes(
            body=class_def.body.with_changes(
                body=[*class_def.body.body[:1], *updated_methods]
            )
        )
    
    def _expand_super_init(self, class_def: cst.ClassDef) -> cst.ClassDef:
        """展开类中的super().__init__调用"""
        class_name = class_def.name.value
        
        # 检查类是否有__init__方法
        init_method = None
        for method in class_def.body.body:
            if isinstance(method, cst.FunctionDef) and method.name.value == "__init__":
                init_method = method
                break
        
        if not init_method:
            return class_def
        
        # 获取父类的__init__方法
        parent_init = self._get_parent_class_init(class_name)
        if not parent_init:
            return class_def
        
        # 转换super().__init__调用
        transformer = SuperCallTransformer(parent_init, class_name)
        updated_init = init_method.visit(transformer)
        
        if not transformer.super_call_found:
            return class_def
        
        # 替换类定义中的__init__方法
        updated_body = []
        for item in class_def.body.body:
            if isinstance(item, cst.FunctionDef) and item.name.value == "__init__":
                updated_body.append(updated_init)
            else:
                updated_body.append(item)
        
        return class_def.with_changes(
            body=class_def.body.with_changes(
                body=updated_body
            )
        )
    
    def _get_parent_class_init(self, class_name: str) -> Optional[cst.FunctionDef]:
        """获取父类的__init__方法"""
        if class_name not in self.classes:
            return None
        
        class_info = self.classes[class_name]
        if not class_info.bases:
            return None
        
        parent_class = class_info.bases[0]
        
        # 检查是否在依赖解析器中能找到父类
        if parent_class in self.dependency_resolver.classes:
            parent_methods = self.dependency_resolver.classes[parent_class].methods
            if "__init__" in parent_methods:
                return parent_methods["__init__"]
        
        return None
    
    def _rename_classes(self, module: cst.Module) -> cst.Module:
        """将类名中的基础模型名替换为新模型名"""
        class RenameTransformer(cst.CSTTransformer):
            def __init__(self, base_model: str, new_model: str):
                self.base_model = base_model
                self.new_model = new_model
            
            def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
                """重命名类定义"""
                class_name = original_node.name.value
                if self.base_model in class_name:
                    new_class_name = class_name.replace(self.base_model, self.new_model)
                    return updated_node.with_changes(
                        name=cst.Name(value=new_class_name)
                    )
                return updated_node
            
            def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
                """重命名类型注解和变量引用中的类名"""
                if self.base_model in original_node.value:
                    new_name = original_node.value.replace(self.base_model, self.new_model)
                    return cst.Name(value=new_name)
                return updated_node
            
            def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.Attribute:
                """重命名属性访问中的类名"""
                if isinstance(original_node.value, cst.Name) and self.base_model in original_node.value.value:
                    new_value = original_node.value.value.replace(self.base_model, self.new_model)
                    return updated_node.with_changes(
                        value=cst.Name(value=new_value)
                    )
                return updated_node
        
        transformer = RenameTransformer(self.base_model, self.new_model)
        return module.visit(transformer)
    
    def _generate_required_imports(self) -> List[cst.SimpleStatementLine]:
        """生成必要的导入语句"""
        required_imports = []
        
        # 添加标准库导入
        std_imports = ["typing", "os", "math", "dataclasses"]
        for imp in std_imports:
            required_imports.append(
                cst.Import(names=[cst.ImportAlias(name=cst.Name(imp))])
            )
        
        # 添加框架相关导入
        framework_imports = [
            "paddle", "paddle.nn", "paddle.nn.functional",
            "paddlenlp", "paddlenlp.transformers"
        ]
        for imp in framework_imports:
            required_imports.append(
                cst.Import(names=[cst.ImportAlias(name=cst.Name(imp))])
            )
        
        return required_imports
    
    def convert(self) -> None:
        """执行转换过程，生成标准模型文件"""
        # 解析依赖
        self._resolve_dependencies()
        
        # 生成modeling文件
        modeling_content = self._generate_modeling_file()
        modeling_file = os.path.join(
            self.output_dir,
            f"modeling_{self.new_model.lower()}.py"
        )
        
        with open(modeling_file, "w", encoding="utf-8") as f:
            f.write(modeling_content)
        
        print(f"转换完成! 生成的文件保存在: {self.output_dir}")

class DependencyResolver:
    """依赖解析器，负责解析和收集依赖代码"""
    
    def __init__(
        self,
        base_model: str,
        new_model: str,
        base_model_dir: Optional[str],
        modular_dir: str
    ):
        self.base_model = base_model
        self.new_model = new_model
        self.base_model_dir = base_model_dir
        self.modular_dir = modular_dir
        self.classes: Dict[str, ClassInfo] = {}
        self.dependency_code: Dict[str, cst.CSTNode] = {}
        self.processed_files: Set[str] = set()
    
    def resolve(
        self,
        imports: List[cst.Import],
        from_imports: List[cst.ImportFrom],
        classes: Dict[str, ClassInfo]
    ) -> None:
        """解析所有依赖项"""
        self.classes = classes
        
        # 解析直接导入
        for imp in imports:
            for name in imp.names:
                self._resolve_import(name.evaluated_name)
        
        # 解析from...import
        for from_imp in from_imports:
            module = from_imp.module.evaluated_name if from_imp.module else ""
            for name in from_imp.names:
                full_name = f"{module}.{name.evaluated_name}" if module else name.evaluated_name
                self._resolve_import(full_name)
        
        # 解析类依赖
        for class_info in classes.values():
            for dep in class_info.dependencies:
                self._resolve_import(dep)
    
    def _resolve_import(self, import_name: str) -> None:
        """解析单个导入项"""
        # 跳过已处理或内置模块
        if (import_name in self.processed_files or
            import_name.split(".")[0] in {"builtins", "typing"}):
            return
        
        # 标记为已处理
        self.processed_files.add(import_name)
        
        # 尝试从本地文件解析
        if self._resolve_local_file(import_name):
            return
        
        # 尝试从Python模块解析
        self._resolve_python_module(import_name)
    
    def _resolve_local_file(self, import_name: str) -> bool:
        """尝试从本地文件解析依赖"""
        # 构造可能的文件路径
        possible_paths = [
            os.path.join(self.modular_dir, f"{import_name.replace('.', '/')}.py"),
            os.path.join(self.modular_dir, import_name.split(".")[-1] + ".py")
        ]
        
        if self.base_model_dir:
            possible_paths.extend([
                os.path.join(self.base_model_dir, f"{import_name.replace('.', '/')}.py"),
                os.path.join(self.base_model_dir, import_name.split(".")[-1] + ".py")
            ])
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                self._process_file_dependencies(file_path)
                return True
        return False
    
    def _resolve_python_module(self, module_name: str) -> bool:
        """尝试从Python模块解析依赖"""
        try:
            module = importlib.import_module(module_name)
            self._process_module_dependencies(module)
            return True
        except ImportError:
            return False
    
    def _process_file_dependencies(self, file_path: str) -> None:
        """处理文件依赖"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        module = cst.parse_module(content)
        wrapper = MetadataWrapper(module)
        
        # 收集类和导入
        class_collector = ClassCollector()
        wrapper.visit(class_collector)
        
        # 更新类信息
        for class_name, class_info in class_collector.classes.items():
            if class_name not in self.classes:
                self.classes[class_name] = class_info
        
        # 添加到依赖代码
        self.dependency_code[os.path.basename(file_path)] = module
    
    def _process_module_dependencies(self, module: Any) -> None:
        """处理模块依赖"""
        try:
            source = inspect.getsource(module)
            module_node = cst.parse_module(source)
            
            # 添加到依赖代码
            self.dependency_code[module.__name__] = module_node
            
            # 收集类和导入
            wrapper = MetadataWrapper(module_node)
            class_collector = ClassCollector()
            wrapper.visit(class_collector)
            
            # 更新类信息
            for class_name, class_info in class_collector.classes.items():
                if class_name not in self.classes:
                    self.classes[class_name] = class_info
        except (OSError, TypeError):
            pass
    
    def get_dependency_code(self) -> Dict[str, cst.CSTNode]:
        """获取解析后的依赖代码"""
        return self.dependency_code

if __name__ == "__main__":
    # 使用示例
    converter = ModularModelConverter(
        base_model="Llama",
        new_model="Qwen2",
        modular_file="./modular_qwen2.py",
        output_dir="output",
        #base_model_dir="path/to/llama/source"
    )
    converter.convert()