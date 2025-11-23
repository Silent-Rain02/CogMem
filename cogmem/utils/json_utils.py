"""JSON工具函数 - 使用 demjson3 提供更宽松的 JSON 解析"""

from __future__ import annotations

import json
from typing import Any

# 尝试导入 demjson3，如果失败则回退到标准 json
try:
    import demjson3
    HAS_DEMJSON3 = True
except ImportError:
    HAS_DEMJSON3 = False
    demjson3 = None


def safe_json_loads(text: str, default: Any = None) -> Any:
    """安全解析JSON字符串，失败时返回默认值
    
    优先使用 demjson3 进行更宽松的解析，支持非标准 JSON 格式。
    如果 demjson3 不可用，回退到标准 json 库。
    
    支持从包含额外文本的字符串中提取JSON对象或数组（如LLM返回的带说明的JSON）
    
    Args:
        text: JSON字符串，可能包含额外的文本或非标准格式
        default: 解析失败时的默认返回值，默认为None，如果为None则尝试提取对象/数组
        
    Returns:
        解析后的JSON对象，或默认值
    """
    # 首先尝试使用 demjson3（如果可用）
    if HAS_DEMJSON3:
        try:
            # demjson3 可以处理更宽松的 JSON 格式，包括尾随逗号、注释等
            return demjson3.decode(text, strict=False)
        except Exception:
            # 如果 demjson3 失败，尝试提取 JSON 对象/数组
            pass
    
    # 回退到标准 json 解析
    try:
        return json.loads(text)
    except Exception:
        # 尝试提取JSON对象（优先）
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            try:
                if HAS_DEMJSON3:
                    return demjson3.decode(text[obj_start : obj_end + 1], strict=False)
                else:
                    return json.loads(text[obj_start : obj_end + 1])
            except Exception:
                pass
        
        # 尝试提取JSON数组
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            try:
                if HAS_DEMJSON3:
                    return demjson3.decode(text[arr_start : arr_end + 1], strict=False)
                else:
                    return json.loads(text[arr_start : arr_end + 1])
            except Exception:
                pass
        
        return default if default is not None else []

