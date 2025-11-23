"""查询类型分析器：识别查询的语义类型和期望的答案格式"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ..models.llm_client import create_llm_client
from ..utils.json_utils import safe_json_loads


@dataclass
class QueryAnalysis:
    """查询分析结果"""

    query_type: str  # Temporal, Identity, Relationship, Location, Multi-hop, Single-fact, List
    entities: List[str]  # 提取的实体名称
    expected_format: str  # Date, Name, Status, Location, List, Yes/No
    temporal_context: Optional[Dict] = None  # 时序上下文
    keywords: Set[str] = None  # 关键词集合


class QueryTypeAnalyzer:
    """分析查询类型和意图"""

    # 查询类型关键词映射
    TEMPORAL_KEYWORDS = {
        "when",
        "time",
        "date",
        "day",
        "month",
        "year",
        "ago",
        "yesterday",
        "today",
        "tomorrow",
        "last",
        "next",
        "before",
        "after",
    }

    IDENTITY_KEYWORDS = {
        "identity",
        "gender",
        "orientation",
        "who is",
        "what is",
    }

    RELATIONSHIP_KEYWORDS = {
        "relationship",
        "status",
        "married",
        "single",
        "divorced",
        "engaged",
        "dating",
        "spouse",
        "partner",
    }

    LOCATION_KEYWORDS = {
        "where",
        "location",
        "place",
        "city",
        "country",
        "live",
        "work",
    }

    MULTI_HOP_INDICATORS = {
        ",",
        "and",
        "or",
        "activities",
        "things",
        "events",
    }

    # 相对时间表达式模式
    RELATIVE_TIME_PATTERNS = [
        r"the \w+ before ([\d\w ]+)",  # "the sunday before 25 May"
        r"(\d+) (days?|weeks?|months?|years?) ago",  # "2 weeks ago"
        r"last (week|month|year|weekend|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
        r"next (week|month|year|weekend|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
        r"this (week|month|year|weekend)",
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = create_llm_client(cfg)

    def analyze(self, query: str) -> QueryAnalysis:
        """分析查询类型

        Args:
            query: 用户查询字符串

        Returns:
            查询分析结果
        """
        query_lower = query.lower().strip()
        words = set(query_lower.split())

        # 提取实体（简单提取，主要名词）
        entities = self._extract_entities(query)

        # 检测查询类型
        query_type = self._detect_query_type(query_lower, words)

        # 检测期望的答案格式
        expected_format = self._detect_expected_format(query_type, query_lower)

        # 检测时序上下文
        temporal_context = None
        if query_type == "Temporal":
            temporal_context = self._detect_temporal_context(query_lower)

        return QueryAnalysis(
            query_type=query_type,
            entities=entities,
            expected_format=expected_format,
            temporal_context=temporal_context,
            keywords=words,
        )

    def _extract_entities(self, query: str) -> List[str]:
        """从查询中提取实体名称

        简单实现：提取大写开头的词（人名）
        或者引号内的词
        """
        entities = []

        # 提取引号内的内容
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)

        # 提取大写开头的词（可能为人名）
        capitalized = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b", query)
        entities.extend(capitalized)

        # 去重并保持顺序
        seen = set()
        unique_entities = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique_entities.append(e)

        return unique_entities[:3]  # 最多3个实体

    def _detect_query_type(self, query_lower: str, words: Set[str]) -> str:
        """检测查询类型"""
        # 时序查询
        if any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS):
            return "Temporal"

        # 身份查询
        if any(kw in query_lower for kw in self.IDENTITY_KEYWORDS):
            return "Identity"

        # 关系状态查询
        if any(kw in query_lower for kw in self.RELATIONSHIP_KEYWORDS):
            return "Relationship"

        # 位置查询
        if any(kw in query_lower for kw in self.LOCATION_KEYWORDS):
            return "Location"

        # 多跳查询（检查是否有列表指示词）
        if "," in query_lower:
            return "Multi-Hop"

        # 检查是否询问多个事物
        multi_indicators = {"activities", "things", "events", "hobbies", "preferences"}
        if any(ind in query_lower for ind in multi_indicators):
            return "Multi-Hop"

        # 单事实查询（默认）
        return "Single-Fact"

    def _detect_expected_format(self, query_type: str, query_lower: str) -> str:
        """检测期望的答案格式"""
        if query_type == "Temporal":
            return "Date"

        if query_type in ["Identity", "Location", "Relationship"]:
            return "Name"

        # 检查是否询问列表
        if any(word in query_lower for word in ["what", "which", "list"]):
            return "List"

        # 检查是否是否/是问题
        if query_lower.startswith("is ") or query_lower.startswith("does "):
            return "Yes/No"

        return "Name"  # 默认

    def _detect_temporal_context(self, query_lower: str) -> Dict:
        """检测时序上下文"""
        context = {
            "requires_session_date": False,
            "contains_relative_expr": False,
            "relative_expressions": [],
            "absolute_dates": [],
        }

        # 查找相对时间表达式
        for pattern in self.RELATIVE_TIME_PATTERNS:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                context["contains_relative_expr"] = True
                context["relative_expressions"].extend(matches)

        # 查找绝对日期（可选）
        date_patterns = [
            r"\b(\d{1,2} \w+ \d{4})\b",  # "8 May 2023"
            r"\b(\w+ \d{1,2},? \d{4})\b",  # "May 8 2023"
            r"\b(\d{4})\b",  # "2023"
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                context["absolute_dates"].extend(matches)

        # 如果需要相对时间表达式，则需要Session Date作为基准
        context["requires_session_date"] = context["contains_relative_expr"]

        return context

    def get_relevant_node_types(self, query_type: str) -> List[str]:
        """获取与查询类型相关的节点类型

        Args:
            query_type: 查询类型

        Returns:
            相关节点类型列表
        """
        relevance_map = {
            "Temporal": ["Event"],  # Event节点包含时间
            "Identity": ["Fact"],  # Fact节点包含身份信息
            "Relationship": ["Fact"],  # Fact节点包含关系信息
            "Location": ["Event", "Fact"],  # Event包含地点
            "Multi-Hop": ["Fact", "Preference"],  # 多个Fact/Preference
            "Single-Fact": ["Fact", "Preference"],  # 单个Fact
        }

        return relevance_map.get(query_type, ["Event", "Fact", "Preference"])
