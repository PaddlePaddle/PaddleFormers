import sys
import logging
from paddle.utils.download import get_path_from_url
from types import ModuleType

class AistudioSdkPatcher:
    """动态修补aistudio_sdk导入问题的类"""
    def __init__(self):
        self.original_modules = {}
        
    def __enter__(self):
        # 保存原始模块状态
        self.original_modules = sys.modules.copy()
        
        # 拦截问题模块导入
        sys.modules['aistudio_sdk.hub'] = self.create_mock_module()
        sys.modules['paddlenlp.transformers.aistudio_utils'] = self.patch_aistudio_utils()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始模块状态
        for mod in list(sys.modules.keys()):
            if mod not in self.original_modules:
                del sys.modules[mod]
        
    def create_mock_module(self):
        """创建模拟的aistudio_sdk.hub模块"""
        mock = ModuleType('aistudio_sdk.hub')
        
        # 添加必要的占位函数
        mock.download = self.mock_download
        return mock
    
    def mock_download(self, *args, **kwargs):
        """模拟下载函数（实际不执行任何操作）"""
        logging.warning("aistudio_sdk download() function is disabled by patch")
        return None
    
    def patch_aistudio_utils(self):
        """修补paddlenlp.transformers.aistudio_utils模块"""
        patched = ModuleType('paddlenlp.transformers.aistudio_utils')
        
        # 重写问题函数
        def aistudio_download(repo_id, filename, cache_dir):
            """完全替代原始下载功能"""
            base_url = f"https://bj.bcebos.com/paddlenlp/models/{repo_id}/{filename}"
            return get_path_from_url(base_url, cache_dir)
            
        patched.aistudio_download = aistudio_download
        return patched

# 应用全局补丁（必须在任何PaddleNLP导入前执行）
patcher = AistudioSdkPatcher()
patcher.__enter__()

# ======= 测试区域 ========
if __name__ == "__main__":
    # 在修补后安全导入PaddleNLP
    import paddlenlp
    from paddlenlp.transformers import AutoTokenizer, AutoModel
    
    print(f"成功导入PaddleNLP版本：{paddlenlp.__version__}")
    
    # 测试模型加载
    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-mini-zh")
    model = AutoModel.from_pretrained("ernie-3.0-mini-zh")
    print("测试通过：ERNIE模型加载成功")
