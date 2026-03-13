# 自动加载 .env（若存在），以便从文件注入环境变量
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # 若未安装或加载失败，继续使用系统环境变量
    pass

from .cogmem import CogMem
from .config import CogMemConfig, DEFAULT_CONFIG, ModelConfig

__all__ = ["CogMem", "CogMemConfig", "DEFAULT_CONFIG", "ModelConfig"]
__version__ = "0.1.0"
