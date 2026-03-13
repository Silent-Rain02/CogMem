"""Text utilities for name handling and answer extraction."""

import re
import json
from typing import Dict, Any

def unwrap_person_name(name: str) -> str:
    """Normalize person name from possible ID-like strings."""
    if not name:
        return ""
    n = name.strip()
    # Remove common prefixes
    for prefix in ("p_", "person_", "person-", "person "):
        if n.lower().startswith(prefix):
            n = n[len(prefix):]
            break
    return n.strip() or ""

def normalize_to_id(text: str) -> str:
    """Convert a name to a safe ID fragment."""
    if not text:
        return "unknown"
    t = text.lower().strip()
    # replace non-alphanumeric with underscore
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = t.strip("_")
    return t or "unknown"

def extract_json_from_markdown(text: str) -> Dict[str, Any]:
    """从 Markdown 文本中提取 JSON 对象"""
    if not text:
        return {}
    
    # 尝试匹配 ```json ... ``` 块
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
            
    # 尝试匹配第一个 { 和最后一个 } 之间的内容
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        pass
        
    return {}

def extract_clean_answer(answer: str, question: str = None) -> str:
    """提取清理后的答案：优先解析 <answer> 标签，处理JSON格式错误。
    
    Args:
        answer: 原始答案字符串
        question: 问题字符串（可选，保留以兼容现有调用）
    """
    if not answer:
        return ""

    # 如果是字典，尝试提取文本内容
    if isinstance(answer, dict):
        for key in ['final_answer', 'answer', 'response', 'message', 'text', 'content']:
            if key in answer:
                value = answer[key]
                if isinstance(value, str) and value.strip():
                    answer = value
                    break
        else:
            try:
                import json
                answer = json.dumps(answer, ensure_ascii=False)
            except:
                answer = str(answer)

    if not isinstance(answer, str):
        answer = str(answer)

    # 尝试提取 <answer> 标签
    answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()

    # 提取 JSON 字段
    json_answer_match = re.search(r'["\'](?:answer|final_answer)["\']\s*:\s*["\']([^"\']+)["\']', answer, re.IGNORECASE)
    if json_answer_match:
        answer = json_answer_match.group(1).strip()

    # 清理前缀
    cleaned = answer.strip()
    # 移除常见的 JSON 包装符号，如果 LLM 返回了纯 JSON 字符串
    cleaned = re.sub(r'^\{.*?"final_answer"\s*:\s*"', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'"\s*,\s*"confidence".*?\}$', '', cleaned, flags=re.DOTALL)
    
    prefixes_to_remove = [
        r"^Based on the evidence,\s*",
        r"^According to the information,\s*",
        r"^The answer is\s*",
        r"^Final Answer:\s*",
        r"^Answer:\s*"
    ]
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()
