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

    def answer(self, query: str, top_k: int = 10, return_retrieval_result: bool = False):
        """极简问答流程：检索 -> 回答

        Args:
            query: 用户查询
            top_k: 返回证据数量
            return_retrieval_result: 是否同时返回检索结果

        Returns:
            如果 return_retrieval_result=True，返回 (answer, retrieval_result)
            否则只返回 answer
        """
        # 1. 执行检索（相信 RetrievalAgent 会尽力）
        retrieval_result = self.retrieval_agent.retrieve(query, top_k)
        
        # 打印最终检索到的 evidence（用于检查检索情况）
        print("\n" + "="*80)
        print(f"[QAOrchestrator] 最终检索结果统计:")
        print(f"  查询: {query}")
        print(f"  检索到的实体数: {len(retrieval_result.entities)}")
        print(f"  检索到的证据数: {len(retrieval_result.evidence)}")
        print(f"  检索路径: {retrieval_result.retrieval_path}")
        
        if retrieval_result.entities:
            print(f"\n  检索到的实体 (前10个):")
            for i, entity in enumerate(retrieval_result.entities[:10], 1):
                print(f"    [{i}] {entity.name} (type: {getattr(entity, 'type', 'unknown')}, id: {entity.id})")
        
        if retrieval_result.evidence:
            print(f"\n  最终检索到的证据 (共 {len(retrieval_result.evidence)} 条):")
            for i, ev in enumerate(retrieval_result.evidence, 1):
                # 截断过长的证据
                ev_display = ev[:200] + "..." if len(ev) > 200 else ev
                print(f"    [{i}] {ev_display}")
        else:
            print(f"\n  ⚠️  未检索到任何证据！")
        
        print("="*80 + "\n")
        
        # 2. 生成答案（让 AnswerAgent 决定是否要推测）
        # 移除所有硬性阈值和回退逻辑
        answer = self.answer_agent.generate_answer(query, retrieval_result)

        if return_retrieval_result:
            return answer, retrieval_result
        return answer
