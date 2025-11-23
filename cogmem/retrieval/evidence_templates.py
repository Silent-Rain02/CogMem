"""证据模板系统：根据查询类型生成结构化的、可直接用于答案的证据

第二阶段优化：生成结构化三元组证据，自动带出上下文信息。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class StructuredEvidence:
    """结构化证据：三元组形式 (Head) -[Relation]-> (Tail)

    优势：
    - 保留完整的语义结构，便于后续推理
    - 可以附加时间、置信度等元信息
    - 更容易与其他证据进行关联和聚合
    """
    head: str  # 头实体（人或事物）
    relation: str  # 关系/谓词
    tail: Optional[str] = None  # 尾实体（值、对象、属性）
    evidence_type: str = "fact"  # 证据类型（fact, time, location, identity, relationship）
    confidence: float = 1.0  # 置信度
    valid_from: Optional[str] = None  # 生效时间
    valid_to: Optional[str] = None  # 失效时间
    context_nodes: List[Dict[str, Any]] = None  # 上下文节点（如时间、地点）

    def __post_init__(self):
        if self.context_nodes is None:
            self.context_nodes = []

    def to_string(self) -> str:
        """转换为文本格式，兼容现有接口"""
        if self.relation in ["date", "time", "location"]:
            # 特殊元信息
            return f"{self.relation.title()}: {self.tail}"

        if self.relation == "identity":
            # 身份信息
            return f"Identity: {self.tail}"

        if self.relation == "status":
            # 状态信息
            return f"Status: {self.tail}"

        # 通用事实格式
        head_name = self._clean_id(self.head)
        tail_str = self._format_tail(self.tail)

        return f"Fact: {head_name} {self.relation} {tail_str}"

    def to_triple(self) -> tuple:
        """返回纯三元组"""
        return (self.head, self.relation, self.tail)

    def _clean_id(self, entity_id: str) -> str:
        """清理实体ID，提取可读名称"""
        if not entity_id or not isinstance(entity_id, str):
            return str(entity_id)

        # 移除前缀
        for prefix in ["person_", "entity_", "event_", "fact_"]:
            if entity_id.startswith(prefix):
                name = entity_id[len(prefix):]
                return name.replace("_", " ").title()

        return entity_id.replace("_", " ").title()

    def _format_tail(self, tail: Optional[str]) -> str:
        """格式化尾实体"""
        if not tail:
            return ""

        if isinstance(tail, str):
            return self._clean_id(tail)

        return str(tail)


class EvidenceTemplate:
    """证据模板基类"""

    @staticmethod
    def format_date(iso_date: str) -> str:
        """将ISO日期格式化为可读的日期字符串

        Args:
            iso_date: ISO格式日期字符串，如 "2023-05-08T13:56:00"

        Returns:
            可读日期，如 "8 May 2023"
        """
        if not iso_date:
            return ""

        try:
            # 提取日期部分（忽略时间）
            date_part = iso_date.split("T")[0]
            dt = datetime.strptime(date_part, "%Y-%m-%d")
            return dt.strftime("%d %B %Y")  # "08 May 2023"
        except ValueError:
            # 如果已经是可读格式，直接返回
            return iso_date

    @staticmethod
    def format_list(items: List[str]) -> str:
        """将列表格式化为逗号分隔的字符串

        Args:
            items: 项目列表

        Returns:
            逗号分隔的字符串
        """
        if not items:
            return ""

        # 去重并保持顺序
        seen = set()
        unique_items = []
        for item in items:
            if item and item.lower() not in seen:
                seen.add(item.lower())
                unique_items.append(item)

        if len(unique_items) == 1:
            return unique_items[0]

        return ", ".join(unique_items[:5])  # 最多5个


class TemporalEvidence(EvidenceTemplate):
    """时序证据模板：从Event节点提取时间信息

    根本性修改：显式提取并格式化 timestamp 属性。
    对于 Event 节点，格式为：[Date: {timestamp}] Event: {description}
    """

    @classmethod
    def generate(cls, event: Any, session_date: Optional[str] = None) -> Optional[str]:
        """从Event节点生成时序证据（字符串格式）

        Args:
            event: Event节点对象
            session_date: Session日期（用于相对时间计算）

        Returns:
            格式化的时序证据字符串，如果没有时间信息则返回None
            格式：[Date: {timestamp}] Event: {description}
        """
        if not event:
            return None

        # EXPLICITLY extract timestamp property (priority: time_start > time_end > timestamp)
        timestamp = None
        if hasattr(event, "time_start") and event.time_start:
            timestamp = event.time_start
        elif hasattr(event, "time_end") and event.time_end:
            timestamp = event.time_end
        elif hasattr(event, "timestamp") and event.timestamp:
            timestamp = event.timestamp

        if not timestamp:
            return None

        # Get title and location
        title = getattr(event, "title", "Event")
        location = getattr(event, "location", None)
        participants = getattr(event, "participants", None)

        # Build description
        description_parts = [str(title)]

        # Add location if available
        if location:
            description_parts.append(f"location {location}")

        # Add participants if available
        if participants:
            if isinstance(participants, list):
                participants_str = ", ".join(participants[:3])
                if len(participants) > 3:
                    participants_str += f" and {len(participants) - 3} others"
                description_parts.append(f"participants {participants_str}")
            else:
                description_parts.append(f"participants {participants}")

        # Join description parts
        description = " | ".join(description_parts)

        # Format with explicit timestamp extraction: [Date: {timestamp}] Event: {description}
        return f"[Date: {timestamp}] Event: {description}"

    @classmethod
    def generate_structured(cls, event: Any, session_date: Optional[str] = None) -> Optional[StructuredEvidence]:
        """从Event节点生成结构化时序证据

        Args:
            event: Event节点对象
            session_date: Session日期（用于相对时间计算）

        Returns:
            StructuredEvidence对象，如果没有时间信息则返回None
        """
        if not event:
            return None

        # 提取时间信息（优先级：time_start > 其他时间字段）
        time_value = None

        if hasattr(event, "time_start") and event.time_start:
            time_value = event.time_start
        elif hasattr(event, "time_end") and event.time_end:
            time_value = event.time_end
        elif hasattr(event, "time") and event.time:
            time_value = event.time

        if not time_value:
            return None

        # 格式化日期
        formatted_date = cls.format_date(time_value)

        if not formatted_date:
            return None

        # 获取标题和位置
        title = getattr(event, "title", "Event")
        location = getattr(event, "location", None)

        # 构建上下文节点列表
        context_nodes = []
        if location:
            context_nodes.append({
                "node_type": "Location",
                "value": location,
                "relation": "located_at"
            })

        # 创建结构化证据（Event本身作为head，时间作为tail）
        return StructuredEvidence(
            head=str(title),
            relation="date",
            tail=formatted_date,
            evidence_type="time",
            confidence=event.time_confidence if hasattr(event, 'time_confidence') else 1.0,
            valid_from=time_value,
            context_nodes=context_nodes
        )

    @classmethod
    def generate_from_fact(cls, fact: Any) -> Optional[str]:
        """从Fact节点生成时序证据（有些Fact也包含时间）"""
        if not fact:
            return None

        # 检查predicate是否包含时间信息
        if hasattr(fact, "predicate") and "time" in fact.predicate.lower():
            value = getattr(fact, "value", None) or getattr(fact, "object", "")
            if value:
                subject_name = cls._extract_name_from_id(fact.subject)
                return f"Time: {value}"

        return None

    @staticmethod
    def _extract_name_from_id(entity_id: str) -> str:
        """从实体ID中提取名称

        Args:
            entity_id: 实体ID，如 "person_caroline"

        Returns:
            可读名称，如 "Caroline"
        """
        if not entity_id:
            return "Unknown"

        # 移除前缀
        name = entity_id
        for prefix in ["person_", "entity_", "event_", "fact_"]:
            if entity_id.startswith(prefix):
                name = entity_id[len(prefix):]
                break

        # 转换为标题格式
        return name.replace("_", " ").title()


class IdentityEvidence(EvidenceTemplate):
    """身份证据模板：从Fact节点提取身份信息"""

    @classmethod
    def generate(cls, fact: Any) -> Optional[str]:
        """从Fact节点生成身份证据（字符串格式，兼容旧接口）

        Args:
            fact: Fact节点对象

        Returns:
            格式化的身份证据字符串
        """
        structured = cls.generate_structured(fact)
        return structured.to_string() if structured else None

    @classmethod
    def generate_structured(cls, fact: Any) -> Optional[StructuredEvidence]:
        """从Fact节点生成结构化身份证据

        Args:
            fact: Fact节点对象

        Returns:
            StructuredEvidence对象
        """
        if not fact or not hasattr(fact, "predicate"):
            return None

        predicate = fact.predicate.lower()

        # 检查是否是身份信息
        identity_indicators = ["identity", "gender", "orientation", "is"]

        if not any(ind in predicate for ind in identity_indicators):
            return None

        # 提取值
        value = getattr(fact, "value", None) or getattr(fact, "object", "")

        if not value:
            return None

        # 格式化
        subject_name = cls._extract_name_from_id(fact.subject)

        # 清理值（移除实体ID前缀）
        clean_value = cls._clean_entity_value(value)

        # 没有清理到有效值？尝试从predicate推断
        if not clean_value or clean_value.lower() in ["true", "false", "unknown", "none"]:
            # 检查谓词是否包含身份相关信息
            if any(ind in predicate for ind in identity_indicators):
                # 从谓词本身提取信息
                for ind in identity_indicators:
                    if ind in predicate:
                        # 例如 predicate = "has_identity" -> 提取 "identity"
                        identity_info = predicate.replace(ind, "").replace("_", " ").strip()
                        if identity_info:
                            clean_value = identity_info.title()
                        break

        if not clean_value:
            return None

        # 特殊处理：如果值包含transgender相关，直接标记为Identity
        value_lower = str(value).lower()
        if any(term in value_lower for term in ["transgender", "trans"]):
            clean_value = "Transgender"

        # 创建结构化证据
        return StructuredEvidence(
            head=fact.subject,
            relation="identity",
            tail=clean_value,
            evidence_type="identity",
            confidence=fact.confidence if hasattr(fact, 'confidence') else 1.0,
            valid_from=fact.valid_from if hasattr(fact, 'valid_from') else None,
            valid_to=fact.valid_to if hasattr(fact, 'valid_to') else None
        )

    @classmethod
    def _clean_entity_value(cls, value: str) -> str:
        """清理实体值，移除ID前缀"""
        if not value or not isinstance(value, str):
            return str(value)

        # 如果是实体ID，提取名称
        if value.startswith(("person_", "entity_")):
            return cls._extract_name_from_id(value)

        return value

    @staticmethod
    def _extract_name_from_id(entity_id: str) -> str:
        """从实体ID中提取名称"""
        if not entity_id:
            return "Unknown"

        name = entity_id
        for prefix in ["person_", "entity_", "event_", "fact_"]:
            if entity_id.startswith(prefix):
                name = entity_id[len(prefix):]
                break

        return name.replace("_", " ").title()


class RelationshipEvidence(EvidenceTemplate):
    """关系证据模板：从Fact节点提取关系状态"""

    @classmethod
    def generate(cls, fact: Any) -> Optional[str]:
        """从Fact节点生成关系证据

        Args:
            fact: Fact节点对象

        Returns:
            格式化的关系证据字符串
        """
        if not fact or not hasattr(fact, "predicate"):
            return None

        predicate = fact.predicate.lower()

        # 检查是否是关系/状态信息
        relationship_indicators = [
            "relationship",
            "status",
            "married",
            "single",
            "divorced",
            "dating",
            "engaged",
            "spouse",
            "partner",
        ]

        if not any(ind in predicate for ind in relationship_indicators):
            return None

        # 提取值
        value = getattr(fact, "value", None) or getattr(fact, "object", "")

        if not value:
            return None

        # 格式化
        clean_value = cls._clean_entity_value(value)

        return f"Status: {clean_value}"

    @classmethod
    def _clean_entity_value(cls, value: str) -> str:
        """清理实体值"""
        if not value or not isinstance(value, str):
            return str(value)

        if value.startswith(("person_", "entity_")):
            return cls._extract_name_from_id(value)

        return value

    @staticmethod
    def _extract_name_from_id(entity_id: str) -> str:
        """从实体ID中提取名称"""
        if not entity_id:
            return "Unknown"

        name = entity_id
        for prefix in ["person_", "entity_", "event_", "fact_"]:
            if entity_id.startswith(prefix):
                name = entity_id[len(prefix):]
                break

        return name.replace("_", " ").title()


class LocationEvidence(EvidenceTemplate):
    """位置证据模板：从Event/Fact节点提取位置信息"""

    @classmethod
    def generate_from_event(cls, event: Any) -> Optional[str]:
        """从Event节点生成位置证据（字符串格式）"""
        structured = cls.generate_structured_from_event(event)
        return structured.to_string() if structured else None

    @classmethod
    def generate_structured_from_event(cls, event: Any) -> Optional[StructuredEvidence]:
        """从Event节点生成结构化位置证据"""
        if not event:
            return None

        location = getattr(event, "location", None)

        if not location:
            return None

        title = getattr(event, "title", "Event")

        return StructuredEvidence(
            head=str(title),
            relation="location",
            tail=location,
            evidence_type="location"
        )

    @classmethod
    def generate_from_fact(cls, fact: Any) -> Optional[str]:
        """从Fact节点生成位置证据（字符串格式）"""
        structured = cls.generate_structured_from_fact(fact)
        return structured.to_string() if structured else None

    @classmethod
    def generate_structured_from_fact(cls, fact: Any) -> Optional[StructuredEvidence]:
        """从Fact节点生成结构化位置证据"""
        if not fact or not hasattr(fact, "predicate"):
            return None

        predicate = fact.predicate.lower()

        # 检查是否是位置信息
        location_indicators = ["location", "live", "work", "at", "in", "from"]

        if not any(ind in predicate for ind in location_indicators):
            return None

        value = getattr(fact, "value", None) or getattr(fact, "object", "")

        if not value:
            return None

        clean_value = cls._clean_entity_value(value)

        return StructuredEvidence(
            head=fact.subject,
            relation="location",
            tail=clean_value,
            evidence_type="location",
            confidence=fact.confidence if hasattr(fact, 'confidence') else 1.0
        )

    @staticmethod
    def _clean_entity_value(value: str) -> str:
        """清理实体值"""
        if not value or not isinstance(value, str):
            return str(value)

        if value.startswith(("person_", "entity_")):
            # 提取名称
            name = value
            for prefix in ["person_", "entity_", "event_", "fact_"]:
                if value.startswith(prefix):
                    name = value[len(prefix):]
                    break
            return name.replace("_", " ").title()

        return value


class MultiFactEvidence(EvidenceTemplate):
    """多事实证据模板：聚合多个相关Fact"""

    @classmethod
    def generate(cls, facts: List[Any]) -> List[str]:
        """从多个Fact节点生成证据列表

        Args:
            facts: Fact节点列表

        Returns:
            证据字符串列表
        """
        if not facts:
            return []

        evidence_list = []

        # 按predicate分组
        fact_groups = {}
        for fact in facts:
            if not hasattr(fact, "predicate"):
                continue

            pred = fact.predicate
            if pred not in fact_groups:
                fact_groups[pred] = []
            fact_groups[pred].append(fact)

        # 为每个组生成证据
        for predicate, group_facts in fact_groups.items():
            # 提取所有值
            values = []
            subjects = set()

            for fact in group_facts:
                value = getattr(fact, "value", None) or getattr(fact, "object", "")
                if value:
                    clean_value = cls._clean_value(value)
                    values.append(clean_value)

                if hasattr(fact, "subject"):
                    subjects.add(fact.subject)

            if values:
                # 格式化
                formatted_values = cls.format_list(values)

                if len(subjects) == 1:
                    # 同一主体的多个值
                    subject_name = cls._extract_name_from_id(list(subjects)[0])
                    evidence_list.append(f"{predicate.title()}: {formatted_values}")
                else:
                    # 不同主体
                    evidence_list.append(f"{predicate.title()}: {formatted_values}")

        return evidence_list[:5]  # 最多5条

    @classmethod
    def _clean_value(cls, value: str) -> str:
        """清理值"""
        if not value or not isinstance(value, str):
            return str(value)

        if value.startswith(("person_", "entity_")):
            return cls._extract_name_from_id(value)

        return value

    @staticmethod
    def _extract_name_from_id(entity_id: str) -> str:
        """从实体ID中提取名称"""
        if not entity_id:
            return "Unknown"

        name = entity_id
        for prefix in ["person_", "entity_", "event_", "fact_"]:
            if entity_id.startswith(prefix):
                name = entity_id[len(prefix):]
                break

        return name.replace("_", " ").title()
