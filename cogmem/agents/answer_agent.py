"""智能回答Agent：生成答案（第4阶段优化 - 敢于推测）

第4阶段优化：AnswerAgent 升级，敢于基于常识推测，但要诚实标注不确定性。
策略：
1. 删除质量评估和多路径策略判断
2. 直接把所有证据喂给 LLM
3. 提示词中加入指令：如果证据相关性低，可基于常识推测，但需标注"不确定"
4. 只使用单一、简洁的生成路径
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cogmem import CogMem
    from ..storage.schema import RetrievalResult

from ..models.llm_client import create_llm_client
from ..prompt import (
    RAG_ANSWER_PHASE4_SYSTEM,  # 第4阶段新系统提示词
    RAG_ANSWER_PHASE4_USER     # 第4阶段新用户提示词
)
import re


class AnswerAgent:
    """回答Agent：生成答案，敢于推测但要诚实标注不确定性"""

    def __init__(self, cogmem: CogMem):
        """
        Args:
            cogmem: CogMem实例
        """
        self.cogmem = cogmem
        self.llm = create_llm_client(cogmem.cfg)
        self.cfg = cogmem.cfg

    def generate_answer(self, query: str, retrieval_result: RetrievalResult) -> str:
        """生成答案（第4阶段：敢于推测，诚实标注不确定性）

        新的策略：
        - 不评估质量，直接生成
        - LLM提示词中加入敢于推测的指令
        - 只使用单一简洁的生成路径
        - 让模型自己决定是否要推测

        Args:
            query: 用户查询
            retrieval_result: 检索结果

        Returns:
            生成的答案（可能包含"[不确定]"标注）
        """
        # 格式化证据
        formatted_evidence = self._format_evidence_for_qa(retrieval_result.evidence)
        # 构建证据统计信息（帮助模型判断）
        evidence_stats = {
            "num_entities": len(retrieval_result.entities),
            "num_evidences": len(retrieval_result.evidence),
            "entity_names": [e.name for e in retrieval_result.entities[:5]],
            "evidence_types": list(set(re.sub(r'^([^:]*):.*$', r'\1', ev).strip() for ev in retrieval_result.evidence[:10] if ':' in ev))
        }

        # 使用第4阶段的新提示词（鼓励推测但要求诚实）
        response = self.llm.chat(
            RAG_ANSWER_PHASE4_SYSTEM,
            RAG_ANSWER_PHASE4_USER.format(
                query=query,
                evidence=formatted_evidence,
                evidence_stats=str(evidence_stats)
            )
        )

        # 抽取核心答案
        answer = self._extract_clean_answer(response)

        # 如果答案为空，尝试基于常识推测（但要标注）
        if not answer or answer.strip() == "":
            answer = "[不确定] 基于现有证据无法确定答案。"

        return answer

    def _format_evidence_for_qa(self, evidence: list[str]) -> str:
        """格式化证据用于问答"""
        formatted_parts = []

        for ev in evidence[:15]:  # 限制数量，避免上下文过长
            ev_clean = ev.strip()
            if ev_clean:
                formatted_parts.append(f"• {ev_clean}")

        return "\n".join(formatted_parts) if formatted_parts else "（无相关证据）"

    def _extract_clean_answer(self, raw_answer: str) -> str:
        """简洁的答案提取：只做最基本的清理"""
        if not raw_answer:
            return ""

        answer = raw_answer.strip()

        # 移除可能的代码块标记
        answer = re.sub(r'```[\w]*\n?', '', answer)
        answer = answer.strip()

        # 提取第一个段落（如果有多段）
        paragraphs = answer.split('\n\n')
        if paragraphs:
            answer = paragraphs[0].strip()

        # 如果答案过长，限制长度
        if len(answer) > 200:
            # 优先保留包含[不确定]或[推测]标记的部分
            if "[不确定]" in answer or "[推测]" in answer:
                return answer  # 保留完整，因为包含重要标记
            answer = answer[:150] + "..."

        return answer
