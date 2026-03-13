"""LongMemEval 数据集加载器

根据 LongMemEval 数据集格式，将历史会话转换为 Doc 对象，
并提取 QA 样本用于评测。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from cogmem.storage.schema import Doc

@dataclass
class QASample:
    """QA 样本"""
    question: str
    answer: str
    question_id: str
    question_type: str
    question_date: str
    answer_session_ids: List[str]  # 证据会话ID列表
    category: Optional[str] = None

@dataclass
class LongMemEvalBundle:
    """LongMemEval 数据包"""
    docs: List[Doc]
    qa: List[QASample]

def map_question_type_to_category(question_type: str) -> str:
    """将问题类型映射到类别名称"""
    type_mapping = {
        "single-session-user": "Single-Session-User",
        "single-session-assistant": "Single-Session-Assistant",
        "single-session-preference": "Single-Session-Preference",
        "temporal-reasoning": "Temporal-Reasoning",
        "knowledge-update": "Knowledge-Update",
        "multi-session": "Multi-Session",
    }
    
    # 检查是否是 abstention 问题
    if question_type.endswith("_abs") or "abstention" in question_type.lower():
        return "Abstention"
    
    return type_mapping.get(question_type, question_type)

class LongMemEvalLoader:
    """LongMemEval 数据加载器"""
    
    def __init__(self, path: str) -> None:
        """初始化加载器
        
        Args:
            path: JSON 文件路径（longmemeval_s_cleaned.json, longmemeval_m_cleaned.json, 或 longmemeval_oracle.json）
        """
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    
    def _build_session_docs(self, sample: dict) -> List[Doc]:
        """构建会话文档 (Session-level)
        
        将历史会话转换为 Doc 对象。每个 Session 作为一个独立的文档，
        并显式注入 [Session Date: ...] 标签以及每行的时间前缀。
        
        Args:
            sample: LongMemEval 数据样本
        
        Returns:
            List[Doc]: 文档列表（每个 session 一个文档）
        """
        docs: List[Doc] = []
        
        haystack_sessions = sample.get("haystack_sessions", [])
        haystack_session_ids = sample.get("haystack_session_ids", [])
        haystack_dates = sample.get("haystack_dates", [])
        
        n_sessions = len(haystack_sessions)
        
        for session_idx in range(n_sessions):
            session = haystack_sessions[session_idx]
            session_id = haystack_session_ids[session_idx] if session_idx < len(haystack_session_ids) else f"session_{session_idx}"
            session_date = haystack_dates[session_idx] if session_idx < len(haystack_dates) else ""
            
            if not session or not isinstance(session, list):
                continue
            
            # 确保从 User 开始处理
            current_session = list(session)
            while current_session and current_session[0].get("role") != "user":
                current_session.pop(0)
            
            if not current_session:
                continue
            
            # 构造文档内容
            formatted_msgs = []
            if session_date:
                formatted_msgs.append(f"[Session Date: {session_date}]")
            
            for msg in current_session:
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
                if content:
                    # 每一行注入时间前缀辅助提取
                    formatted_msgs.append(f"[{session_date}] {role}: {content}")
            
            if not formatted_msgs:
                continue
                
            combined_text = "\n".join(formatted_msgs)
            doc_id = f"lm-{session_id}"
            docs.append(Doc(id=doc_id, text=combined_text))
        
        return docs
    
    def _build_qa(self, sample: dict) -> List[QASample]:
        """构建 QA 样本列表
        
        Args:
            sample: LongMemEval 数据样本
        
        Returns:
            List[QASample]: QA 样本列表
        """
        question_id = sample.get("question_id", "")
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()  # 确保转换为字符串
        question_type = sample.get("question_type", "")
        question_date = str(sample.get("question_date", "")).strip()  # 日期也可能是其他类型
        answer_session_ids = sample.get("answer_session_ids", [])
        
        if not question:
            return []
        
        # 映射类别
        category = map_question_type_to_category(question_type)
        
        qa_sample = QASample(
            question=question,
            answer=answer,
            question_id=question_id,
            question_type=question_type,
            question_date=question_date,
            answer_session_ids=answer_session_ids if isinstance(answer_session_ids, list) else [],
            category=category
        )
        
        return [qa_sample]
    
    def load(self, max_samples: Optional[int] = None) -> LongMemEvalBundle:
        """加载数据集
        
        Args:
            max_samples: 最大样本数（None 表示加载全部）
        
        Returns:
            LongMemEvalBundle: 包含文档和 QA 样本的数据包
        """
        docs: List[Doc] = []
        qa_all: List[QASample] = []
        
        count = 0
        for sample in self.data:
            if max_samples is not None and count >= max_samples:
                break
            
            # 构建文档（从历史会话）
            session_docs = self._build_session_docs(sample)
            docs.extend(session_docs)
            
            # 构建 QA 样本
            qa_samples = self._build_qa(sample)
            qa_all.extend(qa_samples)
            
            count += 1
        
        return LongMemEvalBundle(docs=docs, qa=qa_all)

def load_longmemeval_dataset(
    data_dir: str, 
    dataset_name: str = "longmemeval_s_cleaned",
    max_samples: Optional[int] = None
) -> List[Dict]:
    """加载 LongMemEval 数据集，返回适合评测的格式
    
    Args:
        data_dir: 数据目录路径或直接的文件路径
        dataset_name: 数据集名称（longmemeval_s_cleaned, longmemeval_m_cleaned, 或 longmemeval_oracle）
        max_samples: 最大样本数
    
    Returns:
        List[Dict]: 每个字典包含 id, question, answer, question_type, category 等字段
    """
    # 如果直接指定了文件路径，直接使用
    if os.path.isfile(data_dir):
        json_path = data_dir
    else:
        # 查找 JSON 文件
        possible_names = [
            f"{dataset_name}.json",
            f"longmemeval_{dataset_name}.json",
            dataset_name
        ]
        
        json_path = None
        for name in possible_names:
            candidate = os.path.join(data_dir, name)
            if os.path.exists(candidate):
                json_path = candidate
                break
        
        if not json_path:
            # 尝试在 data 子目录中查找
            for name in possible_names:
                candidate = os.path.join(data_dir, "data", name)
                if os.path.exists(candidate):
                    json_path = candidate
                    break
        
        if not json_path:
            raise FileNotFoundError(
                f"找不到 LongMemEval 数据文件。尝试路径: {data_dir}, 数据集: {dataset_name}"
            )
    
    loader = LongMemEvalLoader(json_path)
    
    # 需要从原始数据中提取 haystack_session_ids
    # 重新加载原始数据以获取完整信息
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    bundle = loader.load(max_samples=max_samples)
    
    # 转换为评测格式，同时包含原始数据中的 haystack_session_ids
    dataset = []
    qa_index = 0
    for raw_sample in raw_data:
        if max_samples is not None and qa_index >= max_samples:
            break
        
        # 找到对应的 QA 样本
        if qa_index < len(bundle.qa):
            qa = bundle.qa[qa_index]
            
            # 从原始样本中提取 haystack_session_ids
            haystack_session_ids = raw_sample.get("haystack_session_ids", [])
            
            dataset.append({
                "id": qa.question_id,
                "question": qa.question,
                "answer": qa.answer,
                "question_type": qa.question_type,
                "question_date": qa.question_date,
                "answer_session_ids": qa.answer_session_ids,
                "haystack_session_ids": haystack_session_ids,  # 添加这个字段用于证据命中率计算
                "category": qa.category
            })
            qa_index += 1
    
    return dataset
