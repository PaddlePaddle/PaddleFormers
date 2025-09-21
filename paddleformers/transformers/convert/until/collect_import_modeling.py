import libcst as cst
import os
from pathlib import Path
from typing import Dict, Set, Union, List, Tuple

def get_full_name(node: Union[cst.Name, cst.Attribute, cst.ImportFrom]) -> str:
    if isinstance(node, cst.Name):
        return node.value
    elif isinstance(node, cst.Attribute):
        return get_full_name(node.value) + "." + node.attr.value
    elif isinstance(node, cst.ImportFrom):
        module_parts = []
        if node.relative:
            module_parts.append("." * len(node.relative))
        if node.module:
            module_parts.append(get_full_name(node.module))
        return "".join(module_parts)
    else:
        return ""

class ModelingImportCollector(cst.CSTVisitor):
    def __init__(self):
        self.imports: Dict[str, str] = {}  # name -> module_path

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        modname = get_full_name(node)
        if "modeling" in modname:
            for alias in node.names:
                self.imports[alias.evaluated_name] = modname

class DependencyCollector(cst.CSTVisitor):
    def __init__(self):
        self.names: Set[str] = set()

    def visit_Name(self, node: cst.Name) -> None:
        self.names.add(node.value)

# NEW: A more powerful collector for definitions and imports
class ModuleInfoCollector(cst.CSTVisitor):
    def __init__(self):
        self.defs: Dict[str, Union[cst.ClassDef, cst.FunctionDef, cst.Assign]] = {}
        self.imports: Dict[str, Union[cst.Import, cst.ImportFrom]] = {}
        self.class_stack: List[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.defs[node.name.value] = node
        self.class_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.class_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if not self.class_stack:
            self.defs[node.name.value] = node
        else:
            fullname = ".".join(self.class_stack + [node.name.value])
            self.defs[fullname] = node

    def visit_Assign(self, node: cst.Assign) -> None:
        if not self.class_stack:
            for target_wrapper in node.targets:
                if isinstance(target_wrapper.target, cst.Name):
                    self.defs[target_wrapper.target.value] = node

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            name_in_scope = alias.asname.name.value if alias.asname else alias.name.value
            self.imports[name_in_scope] = node

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        for alias in node.names:
            name_in_scope = alias.asname.name.value if alias.asname else alias.name.value
            self.imports[name_in_scope] = node

# UPDATED: parse_file now uses the new collector
def parse_file(file_path: str) -> Tuple[Dict, Dict, cst.Module]:
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    module = cst.parse_module(code)
    collector = ModuleInfoCollector()
    module.visit(collector)
    return collector.defs, collector.imports, module

# UPDATED: collect_recursive now finds and returns imports
def collect_recursive(
    name: str,
    defs: Dict[str, cst.CSTNode],
    imports: Dict[str, cst.CSTNode],
    seen: Set[str],
    module: cst.Module,
) -> Tuple[Dict[str, str], Set[str]]:
    if name in seen or name not in defs:
        return {}, set()

    seen.add(name)
    node = defs[name]
    dep_collector = DependencyCollector()
    node.visit(dep_collector)

    results = {name: module.code_for_node(node)}
    collected_imports = set()

    for dep in dep_collector.names:
        if dep in defs and dep not in seen:
            dep_results, dep_imports = collect_recursive(dep, defs, imports, seen, module)
            results.update(dep_results)
            collected_imports.update(dep_imports)
        elif dep in imports:
            import_node = imports[dep]
            import_code = module.code_for_node(import_node)
            collected_imports.add(import_code)
            
    return results, collected_imports

def resolve_file_path(current_file: str, modpath: str) -> Path:
    dots = len(modpath) - len(modpath.lstrip("."))
    parts = modpath.lstrip(".").split(".")
    cur_dir = Path(current_file).parent
    for _ in range(dots - 1):
        cur_dir = cur_dir.parent
    file_path = cur_dir.joinpath(*parts).with_suffix(".py")
    return file_path if file_path.exists() else None

# UPDATED: The main function now handles both definitions and imports
def expand_modeling_imports(file_path: str) -> Dict[str, str]:
    """
    MODIFIED: Collects all definitions and their required imports, returning
    them as a single dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    module = cst.parse_module(code)
    imp_collector = ModelingImportCollector()
    module.visit(imp_collector)

    expanded_defs = {}
    all_imports = set()
    seen = set()

    for name, modpath in imp_collector.imports.items():
        target_file = resolve_file_path(file_path, modpath)
        if not target_file:
            continue
        
        defs, imports, parsed_module = parse_file(str(target_file))
        
        if name in defs:
            new_defs, new_imports = collect_recursive(name, defs, imports, seen, parsed_module)
            expanded_defs.update(new_defs)
            all_imports.update(new_imports)
    
    # --- NEW: Combine imports and definitions into a single dictionary ---
    expanded = {}
    
    # Add imports first, using special keys to identify them
    for i, import_code in enumerate(sorted(list(all_imports))):
        expanded[f"__import_{i}__"] = import_code
        
    # Add the code definitions
    expanded.update(expanded_defs)
            
    return expanded

def save_results_to_txt(result: Dict[str, str], output_file: str):
    """
    MODIFIED: Accepts a single dictionary and separates imports from
    definitions before writing to the file.
    """
    # --- NEW: Separate the combined dictionary back into imports and definitions ---
    imports_to_write = []
    defs_to_write = {}

    for key, value in result.items():
        # Use the special key prefix to identify imports
        if key.startswith("__import_"):
            imports_to_write.append(value)
        else:
            defs_to_write[key] = value

    # --- Writing logic remains similar, but uses the separated items ---
    with open(output_file, "w", encoding="utf-8") as f:
        if imports_to_write:
            f.write("### === Imports === ###\n")
            # Imports are already sorted from the creation step
            for imp in imports_to_write:
                f.write(f"{imp}\n")
            f.write("\n" + "="*50 + "\n\n")

        if defs_to_write:
            f.write("### === Definitions === ###\n")
            for k, v in sorted(defs_to_write.items()):
                f.write(f"=== {k} ===\n")
                f.write(f"{v}\n\n")

if __name__ == "__main__":
    file_to_parse = "/home/hsz/PaddleFormers/PaddleFormers/paddleformers/transformers/convert/example/test_model.py"
    output_filename = "modeling_imports_results.txt"
    
    result_defs, result_imports = expand_modeling_imports(file_to_parse)
    
    save_results_to_txt(result_defs, result_imports, output_filename)
    print(f"Code extraction complete. Results saved to {output_filename}")