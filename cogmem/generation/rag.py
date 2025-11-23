from __future__ import annotations

from typing import List

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client
from ..prompt import RAG_ANSWER_SYSTEM, RAG_ANSWER_USER
from ..storage.schema import RetrievalResult
from .adaptive_generator import AdaptiveGenerator


# _format_entities 和 _format_evidence 已不再需要（Agentic系统使用自己的格式化）


def normalize_answer(s: str) -> str:
    """标准化答案文本，参考官方LoCoMo评测代码（用于评测）"""
    s = s.replace(',', "")
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|and)\b', ' ', text, flags=re.IGNORECASE)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def generate_answer(cfg: CogMemConfig, ret: RetrievalResult, extra_evidence: List[str] | None = None) -> str:
    """答案生成（使用自适应生成器）
    
    使用AdaptiveGenerator根据证据质量和置信度自适应调整生成策略。
    """
    # 合并所有证据
    evidence = list(ret.evidence or [])
    if extra_evidence:
        evidence.extend(extra_evidence)

    # 获取实体名称列表
    entity_names = ret.extracted_entity_names or [e.name for e in ret.entities or []]

    # 使用自适应生成器
    generator = AdaptiveGenerator(cfg)
    return generator.generate(ret.query, evidence, entity_names)


def _format_evidence_for_qa(evidence: List[str]) -> str:
    """格式化证据用于问答"""
    formatted_parts = []
    for ev in evidence[:10]:
        ev_clean = ev.strip()
        if ev_clean:
            formatted_parts.append(ev_clean)
    return "\n\n".join(formatted_parts) if formatted_parts else ""


def _extract_clean_answer(raw_answer: str) -> str:
    """简洁的答案提取：只做最基本的清理"""
    if not raw_answer:
        return "无法确定"
    
    answer = raw_answer.strip()
    
    # 移除明显的LLM前缀（简单处理）
    prefixes = ["答案：", "答案是：", "Answer:", "答案:", "The answer is"]
    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    # 取第一句话
    if "." in answer:
        answer = answer.split(".")[0].strip()
    elif "。" in answer:
        answer = answer.split("。")[0].strip()
    
    return answer if answer else "无法确定"
