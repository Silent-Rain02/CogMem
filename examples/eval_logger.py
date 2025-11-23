"""
评测日志记录器：记录实体提取、检索路径、问答结果等，用于分析bad case
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class EvalLogEntry:
    """单条评测日志条目"""
    question: str
    gold_answer: str
    category: str
    database: str  # dialog/observation/session_summary
    
    # 实体提取
    extracted_entities: List[str]  # 从查询中提取的实体名
    
    # 检索结果
    retrieval_path: List[str]  # 检索路径（exact-match/community-fusion/vector-search/chunk-search等）
    retrieved_entities: List[Dict]  # 检索到的实体（name, description, type）
    retrieved_chunks: List[str]  # 检索到的文档片段（前3个）
    num_entities: int
    num_chunks: int
    
    # 答案生成
    predicted_answer: str
    
    # 评测结果
    em_score: float
    f1_score: float
    evidence_hit: float
    
    # 元数据
    timestamp: str
    question_id: Optional[str] = None


class EvalLogger:
    """评测日志记录器"""
    
    def __init__(self, log_dir: str = "logs/eval") -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.entries: List[EvalLogEntry] = []
    
    def log(
        self,
        question: str,
        gold_answer: str,
        category: str,
        database: str,
        extracted_entities: List[str],
        retrieval_result,
        predicted_answer: str,
        em_score: float,
        f1_score: float,
        evidence_hit: float,
        question_id: Optional[str] = None,
    ) -> None:
        """记录一条评测日志"""
        # 解析检索结果
        retrieval_path = []
        retrieved_chunks = []
        
        for ev in retrieval_result.evidence or []:
            if ev in ["exact-match", "community-fusion", "vector-search", "chunk-search", "gap-iter", "empty"]:
                retrieval_path.append(ev)
            elif len(ev) > 20:  # 可能是文档片段
                retrieved_chunks.append(ev[:200])  # 截断以便日志
        
        retrieved_entities = [
            {"name": e.name, "description": e.description[:50], "type": e.type}
            for e in retrieval_result.entities[:5]
        ]
        
        entry = EvalLogEntry(
            question=question,
            gold_answer=gold_answer,
            category=category,
            database=database,
            extracted_entities=extracted_entities,
            retrieval_path=retrieval_path,
            retrieved_entities=retrieved_entities,
            retrieved_chunks=retrieved_chunks[:3],
            num_entities=len(retrieval_result.entities),
            num_chunks=len(retrieved_chunks),
            predicted_answer=predicted_answer,
            em_score=em_score,
            f1_score=f1_score,
            evidence_hit=evidence_hit,
            timestamp=datetime.now().isoformat(),
            question_id=question_id,
        )
        self.entries.append(entry)
    
    def save(self, filename: Optional[str] = None) -> str:
        """保存日志到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_log_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # 转换为字典列表
        data = [asdict(entry) for entry in self.entries]
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def analyze_bad_cases(self, threshold_f1: float = 0.5) -> Dict:
        """分析bad case统计"""
        bad_cases = [e for e in self.entries if e.f1_score < threshold_f1]
        
        if not bad_cases:
            return {"total": len(self.entries), "bad_cases": 0, "bad_rate": 0.0}
        
        # 按类别统计
        by_category: Dict[str, int] = {}
        by_retrieval_path: Dict[str, int] = {}
        no_entities: int = 0
        no_chunks: int = 0
        
        for case in bad_cases:
            by_category[case.category] = by_category.get(case.category, 0) + 1
            for path in case.retrieval_path:
                by_retrieval_path[path] = by_retrieval_path.get(path, 0) + 1
            if case.num_entities == 0:
                no_entities += 1
            if case.num_chunks == 0:
                no_chunks += 1
        
        return {
            "total": len(self.entries),
            "bad_cases": len(bad_cases),
            "bad_rate": len(bad_cases) / len(self.entries),
            "by_category": by_category,
            "by_retrieval_path": by_retrieval_path,
            "no_entities_count": no_entities,
            "no_chunks_count": no_chunks,
        }

