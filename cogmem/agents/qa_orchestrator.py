"""端到端问答协调器：整合检索和回答流程（第4阶段优化 - 简化逻辑）

第4阶段目标：移除过度设计的防御性代码，让模型敢于回答。
策略：
1. 始终相信 RetrievalAgent（经过前几步优化后）尽力找回了最好的内容
2. 将所有检索内容（带相关性分数）全部喂给 AnswerAgent
3. 删除硬性回退和质量阈值判断
4. AnswerAgent 提示词中加入推测指令
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cogmem import CogMem

from .retrieval_agent import RetrievalAgent
from .answer_agent import AnswerAgent


class QAOrchestrator:
    """端到端问答协调器：极简流水线，相信检索和回答Agent"""

    def __init__(self, cogmem: CogMem):
        """
        Args:
            cogmem: CogMem实例
        """
        self.cogmem = cogmem
        self.retrieval_agent = RetrievalAgent(cogmem)
        self.answer_agent = AnswerAgent(cogmem)

    def answer(self, query: str, top_k: int = 10) -> str:
        """极简问答流程：检索 -> 回答

        Args:
            query: 用户查询
            top_k: 返回证据数量

        Returns:
            生成的答案
        """
        # 1. 执行检索（相信 RetrievalAgent 会尽力）
        retrieval_result = self.retrieval_agent.retrieve(query, top_k)

        # 2. 生成答案（让 AnswerAgent 决定是否要推测）
        # 移除所有硬性阈值和回退逻辑
        answer = self.answer_agent.generate_answer(query, retrieval_result)

        return answer
