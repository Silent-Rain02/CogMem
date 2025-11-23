from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from cogmem.storage.schema import Doc

DBType = Literal["conversation", "dialog", "observation", "session_summary"]


def map_category(cat: int | str | None) -> str:
    """将LoCoMo数字类别映射到标准名称
    
    根据LoCoMo论文：
    1 = Single Hop
    2 = Multi-Hop  
    3 = Temporal
    4 = Open Domain
    5 = Other/Uncategorized
    """
    if cat is None:
        return "Unknown"
    cat_str = str(cat).strip()
    mapping = {
        "1": "Single Hop",
        "2": "Multi-Hop",
        "3": "Temporal",
        "4": "Open Domain",
        "5": "Other",
    }
    return mapping.get(cat_str, "Unknown")


@dataclass
class QASample:
    question: str
    answer: str
    evidence_turn_ids: List[str]
    category: Optional[str] = None


@dataclass
class LocomoBundle:
    docs: List[Doc]
    qa: List[QASample]


class LocomoLoader:
    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def _build_dialog_docs(self, sample: dict, turns_per_doc: int = 5) -> List[Doc]:
        """构建对话文档，合并多个轮次以提供更丰富的上下文
        
        Args:
            sample: LoCoMo数据样本
            turns_per_doc: 每个文档包含的对话轮次数（默认5，平衡上下文和效率）
        """
        docs: List[Doc] = []
        conv = sample.get("conversation", {})
        
        # 提取所有会话时间信息
        session_times = {}
        for k, v in conv.items():
            if k.endswith("_date_time") and isinstance(v, str):
                # 提取session编号（如"session_1_date_time" -> "1"）
                session_num = k.replace("_date_time", "").replace("session_", "")
                session_times[session_num] = v
        
        for k, v in conv.items():
            if k.startswith("session_") and isinstance(v, list):
                # 将同一个session的turns合并成更大的文档
                session_id = k  # e.g., "session_1"
                session_num = k.replace("session_", "")
                session_time = session_times.get(session_num, "")
                
                turns = []
                turn_ids = []
                for turn in v:
                    txt = turn.get("text")
                    if not txt:
                        continue
                    dia_id = str(turn.get("dia_id")) if turn.get("dia_id") is not None else None
                    turns.append(str(txt))
                    if dia_id:
                        turn_ids.append(dia_id)
                
                # 按 turns_per_doc 合并turns
                for i in range(0, len(turns), turns_per_doc):
                    chunk_turns = turns[i:i + turns_per_doc]
                    chunk_ids = turn_ids[i:i + turns_per_doc]
                    
                    # 合并多个turns，用换行分隔以保持可读性
                    combined_text = "\n".join(chunk_turns)
                    
                    # 在文档开头添加会话时间信息（重要：用于时序理解）
                    if session_time:
                        combined_text = f"[Session Date: {session_time}]\n{combined_text}"
                    
                    # 生成文档ID：session_id + 起始turn_id
                    if chunk_ids:
                        doc_id = f"dialog-{session_id}-{chunk_ids[0]}"
                    else:
                        doc_id = f"dialog-{session_id}-chunk{i}"
                    
                    docs.append(Doc(id=doc_id, text=combined_text))
        
        return docs

    def _build_observation_docs(self, sample: dict) -> List[Doc]:
        docs: List[Doc] = []
        obs = sample.get("observation", {})
        for k, v in obs.items():
            if k.endswith("_observation") and isinstance(v, list):
                for i, item in enumerate(v):
                    # 可能是 [text, "D1:3"] 或 纯字符串
                    if isinstance(item, list) and item:
                        txt = item[0]
                    else:
                        txt = item
                    if not txt:
                        continue
                    docs.append(Doc(id=f"obs-{k}-{i}", text=str(txt)))
        return docs

    def _build_session_summary_docs(self, sample: dict) -> List[Doc]:
        docs: List[Doc] = []
        summ = sample.get("session_summary", {})
        for k, v in summ.items():
            if k.endswith("_summary") and isinstance(v, str) and v.strip():
                docs.append(Doc(id=f"sum-{k}", text=v.strip()))
        return docs

    def _build_qa(self, sample: dict) -> List[QASample]:
        """构建QA样本列表
        
        注意：会过滤掉答案为空的QA（主要是Adversarial类别，答案本身就是None）
        原始数据集有1986个QA，过滤后为1542个有效QA
        """
        qa = sample.get("qa", [])
        out: List[QASample] = []
        for item in qa:
            q = str(item.get("question") or "").strip()
            a = str(item.get("answer") or "").strip()
            ev = item.get("evidence") or []
            cat = item.get("category")
            # 过滤：只保留问题和答案都不为空的QA
            # 注意：Adversarial类别（category=5）的问题答案通常为None，会被过滤
            if q and a:
                # 映射类别到标准名称
                cat_mapped = map_category(cat)
                out.append(QASample(question=q, answer=a, evidence_turn_ids=[str(x) for x in ev], category=cat_mapped))
        return out

    def load(self, db: DBType, max_samples: Optional[int] = None) -> LocomoBundle:
        docs: List[Doc] = []
        qa_all: List[QASample] = []
        count = 0
        for sample in self.data:
            if max_samples is not None and count >= max_samples:
                break
            if db == "conversation" or db == "dialog":
                d = self._build_dialog_docs(sample)
            elif db == "observation":
                d = self._build_observation_docs(sample)
            else:
                d = self._build_session_summary_docs(sample)
            if not d:
                continue
            docs.extend(d)
            qa_all.extend(self._build_qa(sample))
            count += 1
        return LocomoBundle(docs=docs, qa=qa_all)


def load_locomo_dataset(data_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    """加载LoCoMo数据集，返回适合评测的格式
    
    Args:
        data_dir: 数据目录路径（应包含locomo10.json）
        max_samples: 最大样本数
    
    Returns:
        List[Dict]: 每个字典包含 id, question, answer, evidence, category 等字段
    """
    import os
    
    # 查找locomo10.json文件
    json_path = os.path.join(data_dir, "data", "locomo10.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(data_dir, "locomo10.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到locomo10.json文件，尝试路径: {json_path}")
    
    loader = LocomoLoader(json_path)
    bundle = loader.load(db="conversation", max_samples=max_samples)
    
    # 转换为评测格式
    dataset = []
    for i, qa in enumerate(bundle.qa):
        dataset.append({
            "id": f"qa_{i}",
            "question": qa.question,
            "answer": qa.answer,
            "evidence": qa.evidence_turn_ids,
            "category": qa.category or "Unknown"
        })
    
    return dataset
