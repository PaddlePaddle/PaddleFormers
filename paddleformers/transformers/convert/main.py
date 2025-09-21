from until.collect_import_modeling import expand_modeling_imports,save_results_to_txt
from until.rewrite_child_classes import rewrite_child_classes
from until.rename_identifiers import rename_identifiers
from pathlib import Path

def main():
    from_name = "llama"
    to_name = "qwen2"
    file_to_parse = Path("/home/hsz/PaddleFormers/PaddleFormers/paddleformers/transformers/convert/test_modular.py")
    output_file = Path("test_model_expanded.py")
    temp_merged_file = Path(output_file.stem + "_temp_merged.py")
    print(f"--- 开始转换 ---")
    print(f"源模型: '{from_name}', 目标模型: '{to_name}'")
    print(f"输入文件: '{file_to_parse}'")
    print(f"最终输出: '{output_file}'")
    print(">>> Step 1: Collect modeling imports...")
    expanded = expand_modeling_imports(file_to_parse)
    save_results_to_txt(expanded, "modeling_imports_results.txt" )

    print(">>> Step 2: Rewrite child classes...")
    rewrite_child_classes(expanded, file_to_parse, temp_merged_file)

     # --- 步骤 3: 全局重命名 (外观重构) ---
    # 读取上一步生成的中间文件，对其进行一次完整的、全局的重命名
    print("\n>>> 步骤 3: 全局重命名标识符...")
    try:
        merged_code = temp_merged_file.read_text(encoding="utf-8")
        final_code = rename_identifiers(merged_code, from_name, to_name)
        output_file.write_text(final_code, encoding="utf-8")
        print(f"    重命名完成，最终代码已写入 '{output_file}'。")
    except FileNotFoundError:
        print(f"    [错误] 找不到中间文件 '{temp_merged_file}'，无法进行重命名。")
    
    # --- 清理 ---
    if temp_merged_file.exists():
        temp_merged_file.unlink()
        print("\n>>> 清理临时文件完毕。")

    print("\n--- 转换流程结束 ---")


if __name__ == "__main__":
    main()

