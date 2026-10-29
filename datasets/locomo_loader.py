from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from cogmem.storage.schema import Doc

DBType = Literal["conversation", "dialog", "observation", "session_summary"]

def map_category(cat: int | str | None) -> str:
    """将LoCoMo数字类别映射到标准名称

    根据LoCoMo论文：
    1 = Multi-Hop
    2 = Temporal
    3 = Open Domain
    4 = Single Hop
    5 = Adversarial
    """
    if cat is None:
        return "Unknown"
    cat_str = str(cat).strip()
    mapping = {
        "1": "Multi-Hop",
        "2": "Temporal",
        "3": "Open Domain",
        "4": "Single Hop",
        "5": "Adversarial",
    }
    return mapping.get(cat_str, "Unknown")

@dataclass
class QASample:
    question: str
    answer: str
    evidence_turn_ids: List[str]
    category: Optional[str] = None
    category_id: Optional[int] = None

@dataclass
class LocomoBundle:
    docs: List[Doc]
    qa: List[QASample]

class LocomoLoader:
    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def _build_dialog_docs(self, sample: dict) -> List[Doc]:
        """构建对话文档，按完整session处理（不再切分turns）

        Args:
            sample: LoCoMo数据样本
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
                # 处理整个session，不再切分
                session_id = k  # e.g., "session_1"
                session_num = k.replace("session_", "")
                session_time = session_times.get(session_num, "")

                turns = []
                turn_ids = []
                speakers = []
                for turn in v:
                    # 安全地获取text字段
                    if isinstance(turn, dict):
                        txt = turn.get("text")
                        speaker = turn.get("speaker", "Unknown")
                    elif isinstance(turn, str):
                        txt = turn
                        speaker = "Unknown"
                    else:
                        print(f"⚠️ 跳过非法turn格式: {type(turn)}")
                        continue

                    # Image captions are source content supplied by the dataset.
                    caption = turn.get("blip_caption") if isinstance(turn, dict) else None
                    if caption:
                        txt = (str(txt or "").strip() + "\n[Image caption] " + str(caption)).strip()

                    # 验证text不为空
                    if not txt or not str(txt).strip():
                        continue

                    dia_id = str(turn.get("dia_id")) if isinstance(turn, dict) and turn.get("dia_id") is not None else None
                    turns.append(str(txt))
                    speakers.append(speaker)
                    turn_ids.append(dia_id)

                # 过滤空文本
                if not turns:
                    continue

                # 合并所有turns，用发言者标记分隔以保持可读性
                formatted_turns = []
                for i, (turn, speaker) in enumerate(zip(turns, speakers)):
                    reference = f"[{turn_ids[i]}] " if turn_ids[i] else ""
                    turn_prefix = reference + f"[{speaker}]"
                    formatted_turns.append(f"{turn_prefix} {turn}")

                combined_text = "\n".join(formatted_turns)

                # 在文档开头添加会话时间信息（重要：用于时序理解）
                if session_time:
                    combined_text = f"[Session Date: {session_time}]\n{combined_text}"

                # 最终验证：确保combined_text不为空
                if not combined_text or not combined_text.strip():
                    print(f"⚠️ 跳过空文档: {session_id}")
                    continue

                # 生成文档ID：确保包含session_，这样chunker能识别为session文档
                # CRITICAL: doc.id必须包含"session_"，这样chunker才不会切分
                doc_id = f"dialog-{session_id}"

                # CRITICAL FIX: 创建Doc对象，在metadata中设置doc_type
                # ChunkerFactory使用getattr(doc, "doc_type", None)来获取doc_type
                # 我们通过metadata.doc_type存储，并在Doc对象上设置属性以便getattr能获取
                from cogmem.storage.schema import Metadata
                metadata = Metadata(
                    source=f"locomo-{session_id}",
                    doc_type="dialog"  # 设置文档类型为dialog
                )

                doc = Doc(id=doc_id, text=combined_text, metadata=metadata)
                # CRITICAL: 使用object.__setattr__绕过Pydantic验证，设置doc_type属性
                # 这样ChunkerFactory.create()可以通过getattr(doc, "doc_type", None)获取
                object.__setattr__(doc, "doc_type", "dialog")

                docs.append(doc)

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

                    # 验证text不为空且是字符串
                    if not txt or not isinstance(txt, str) or not txt.strip():
                        continue

                    docs.append(Doc(id=f"obs-{k}-{i}", text=str(txt).strip()))
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

        过滤None和空白答案，保留0、False等明确给出的标量值。
        有效题数以输入文件为准，不使用历史版本的统计常数。
        """
        qa = sample.get("qa", [])
        out: List[QASample] = []
        for item in qa:
            q = str(item.get("question") or "").strip()
            raw_answer = item.get("answer")
            a = "" if raw_answer is None else str(raw_answer).strip()
            ev = item.get("evidence") or []
            cat = item.get("category")
            # 过滤：只保留问题和答案都不为空的QA
            # 注意：Adversarial类别（category=5）的问题答案通常为None，会被过滤
            if q and a:
                # 映射类别到标准名称
                cat_mapped = map_category(cat)
                out.append(QASample(question=q, answer=a, evidence_turn_ids=[str(x) for x in ev], category=cat_mapped, category_id=int(cat) if str(cat).isdigit() else None))
        return out

    def load(self, db: DBType, max_samples: Optional[int] = None,
             subset_ratio: Optional[float] = None, subset_seed: int = 42) -> LocomoBundle:
        """加载数据

        Args:
            db: 数据库类型
            max_samples: 最大样本数（conversation数量）
            subset_ratio: 采样比例（0.0-1.0），例如 0.1 表示10%
            subset_seed: 采样随机种子（用于可复现性）
        """
        # 如果指定了采样比例，先对conversation进行采样
        data_to_process = self.data
        if subset_ratio is not None:
            if not 0.0 < subset_ratio <= 1.0:
                raise ValueError(f"subset_ratio 必须在 (0.0, 1.0] 范围内，当前值: {subset_ratio}")

            # 设置随机种子
            random.seed(subset_seed)

            # 计算采样数量
            original_size = len(self.data)
            subset_size = max(1, int(original_size * subset_ratio))

            # 随机采样conversation
            data_to_process = random.sample(self.data, min(subset_size, original_size))

            print(f"  采样对话：从 {original_size} 个对话采样到 {len(data_to_process)} 个对话（比例: {subset_ratio*100:.1f}%）")

        docs: List[Doc] = []
        qa_all: List[QASample] = []
        count = 0
        for sample in data_to_process:
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

def load_locomo_dataset(
    data_dir: str,
    max_samples: Optional[int] = None,
    subset_ratio: Optional[float] = None,
    subset_seed: int = 42
) -> List[Dict]:
    """加载LoCoMo数据集，返回适合评测的格式

    Args:
        data_dir: 数据目录路径（应包含locomo10.json）或直接的文件路径
        max_samples: 最大样本数
        subset_ratio: 采样比例（0.0-1.0），例如 0.1 表示10%。如果指定，会先采样conversation，然后从这些conversation中提取QA
        subset_seed: 采样随机种子（用于可复现性）

    Returns:
        List[Dict]: 每个字典包含 id, question, answer, evidence, category 等字段
    """
    import os

    # 如果直接指定了文件路径，直接使用
    if os.path.isfile(data_dir):
        json_path = data_dir
    else:
        # 查找locomo10.json文件
        json_path = os.path.join(data_dir, "data", "locomo10.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(data_dir, "locomo10.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到locomo10.json文件，尝试路径: {json_path}")

    loader = LocomoLoader(json_path)
    # 使用与索引时相同的采样逻辑：先采样conversation，然后从这些conversation中提取QA
    bundle = loader.load(
        db="conversation",
        max_samples=max_samples,
        subset_ratio=subset_ratio,
        subset_seed=subset_seed
    )

    # 转换为评测格式
    dataset = []
    for i, qa in enumerate(bundle.qa):
        dataset.append({
            "id": f"qa_{i}",
            "question": qa.question,
            "answer": qa.answer,
            "evidence": qa.evidence_turn_ids,
            "category": qa.category_id if qa.category_id is not None else (qa.category or "Unknown")
        })

    return dataset
