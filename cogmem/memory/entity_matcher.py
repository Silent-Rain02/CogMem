"""EntityMatcher: 实体匹配与智能合并组件

这个模块实现了基于名称的实体匹配和智能属性合并，解决了Person节点重复的问题。

核心优势：
- 稳定ID生成：基于canonical_name，避免随机ID
- 智能匹配：canonical_name + aliases双重查找
- 属性合并：列表去重、字符串智能选择、背景追加
- O(1)查找：基于内存索引快速匹配

Author: CogMem Team
Date: 2025-12
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Set
from collections import defaultdict

from ..storage.graph_store import GraphStore
from ..storage.schema import Person

class EntityMatcher:
    """实体匹配与合并组件：基于名称的稳定ID生成和智能属性合并
    
    设计理念：
    - 生成稳定ID：基于canonical_name而非随机uuid
    - 名称匹配：优先canonical_name，其次aliases
    - 智能合并：列表去重、字符串择优、背景累积
    - 快速查找：内存索引O(1)匹配
    
    使用示例：
        matcher = EntityMatcher(graph_store)
        
        # 查找或创建Person
        actual_id, merged_person = matcher.find_or_create_person(new_person)
        
        # 重置索引（新批次处理时）
        matcher.reset()
    """
    
    def __init__(self, graph_store: GraphStore):
        """初始化实体匹配器
        
        Args:
            graph_store: 图存储实例
        """
        self._graph = graph_store
        
        # 快速查找索引：name -> person_id
        self._name_to_person_id: Dict[str, str] = {}
        self._alias_to_person_id: Dict[str, str] = {}
        
        # 缓存已处理的Person（避免重复从图中读取）
        self._person_cache: Dict[str, Person] = {}
        
        # 统计信息
        self._stats = {
            "matched_by_name": 0,
            "matched_by_alias": 0,
            "created_new": 0,
            "merged_count": 0
        }
    
    def find_or_create_person(self, person: Person) -> Tuple[str, Person]:
        """查找或创建Person，返回实际使用的ID和合并后的节点
        
        匹配策略：
        1. 过滤代词等通用名称（避免合并为全局的 \"he\" 节点）
        2. 基于canonical_name查找现有Person
        3. 基于aliases查找现有Person
        4. 如果未找到，生成稳定ID创建新Person
        5. 如果找到，智能合并属性
        
        Args:
            person: 待匹配的Person节点
        
        Returns:
            Tuple[actual_id, merged_person]
        """
        # 规范化名称（去除多余空格）
        canonical_name = person.canonical_name.strip()
        if not canonical_name:
            # 如果canonical_name为空，使用原始ID
            self._stats["created_new"] += 1
            return person.id, person
            
        # [NEW] 黑名单检查：禁止将通用代词合并为全局实体
        # 这防止了 HotpotQA 中多个文档的 "he" 被合并成同一个节点
        generic_blacklist = {
            "he", "she", "it", "they", "them", "his", "her", "him",
            "the film", "the series", "the book", "the person", "the group", 
            "the band", "the team", "the city", "the country", "the state",
            "the director", "the writer", "the actor", "the singer", "the artist"
        }
        if canonical_name.lower() in generic_blacklist:
            # 如果是通用名称，不进行全局匹配，直接返回带有时空背景的原始ID
            # 这样 "he" 会被视为该文档私有的实体，而不会污染全局
            self._stats["created_new"] += 1
            return person.id, person
        
        # ROOT FIX: Enhanced matching strategy
        # 1. 按canonical_name匹配
        existing_id = self._name_to_person_id.get(canonical_name)
        if existing_id:
            self._stats["matched_by_name"] += 1
            merged = self._merge_person(existing_id, person)
            return existing_id, merged

        # 2. 检查canonical_name是否匹配现有Person的别名（双向匹配）
        # This handles cases like:
        # - Existing: canonical_name="Jon", aliases=["Jonathan"]
        # - New: canonical_name="Jonathan", aliases=[]
        if canonical_name in self._alias_to_person_id:
            existing_id = self._alias_to_person_id[canonical_name]
            self._stats["matched_by_alias"] += 1
            merged = self._merge_person(existing_id, person)
            return existing_id, merged

        # 3. 按aliases匹配（检查新Person的别名是否匹配现有Person的canonical_name或别名）
        for alias in person.aliases:
            alias = alias.strip()
            if not alias:
                continue
            
            # Check if alias matches existing Person's canonical_name
            if alias in self._name_to_person_id:
                existing_id = self._name_to_person_id[alias]
                self._stats["matched_by_alias"] += 1
                merged = self._merge_person(existing_id, person)
                return existing_id, merged
            
            # Check if alias matches existing Person's aliases
            if alias in self._alias_to_person_id:
                existing_id = self._alias_to_person_id[alias]
                self._stats["matched_by_alias"] += 1
                merged = self._merge_person(existing_id, person)
                return existing_id, merged
        
        # The graph builder has already assigned this identity to relations and
        # source pointers. Replacing it here (formerly truncating names at 50
        # characters) leaves those addresses dangling and can collide identities.
        stable_id = person.id
        
        # 注册到索引
        self._register_person(stable_id, person)
        self._stats["created_new"] += 1
        
        return stable_id, person
    
    def _register_person(self, person_id: str, person: Person) -> None:
        """注册Person到索引
        
        Args:
            person_id: Person ID
            person: Person对象
        """
        # 注册canonical_name
        if person.canonical_name:
            self._name_to_person_id[person.canonical_name] = person_id
        
        # 注册aliases
        for alias in person.aliases:
            alias = alias.strip()
            if alias:
                self._alias_to_person_id[alias] = person_id
        
        # 缓存Person对象
        self._person_cache[person_id] = person
    
    def _merge_person(self, existing_id: str, new_person: Person) -> Person:
        """智能合并Person属性
        
        合并策略：
        - canonical_name: 保持现有
        - aliases: 去重合并
        - persona_traits: 去重合并
        - profession: 选择更详细或置信度更高的
        - background: 追加新信息
        - identity: 优先新值
        - confidence: 取最大值
        
        Args:
            existing_id: 现有Person的ID
            new_person: 新的Person数据
        
        Returns:
            合并后的Person对象
        """
        # 从缓存或图中获取现有Person
        existing = self._person_cache.get(existing_id)
        if not existing:
            existing = self._graph.get_person(existing_id)
        
        if not existing:
            # 如果现有Person不存在，直接返回新Person
            new_person.id = existing_id
            self._register_person(existing_id, new_person)
            return new_person
        
        self._stats["merged_count"] += 1
        
        # 合并策略
        merged = Person(
            id=existing_id,
            canonical_name=existing.canonical_name,  # 保持现有
            
            # 列表合并：去重
            aliases=self._merge_list_field(existing.aliases, new_person.aliases),
            persona_traits=self._merge_list_field(
                existing.persona_traits, new_person.persona_traits
            ),
            
            # 字符串合并：选择更详细或追加
            profession=self._merge_string_field(
                existing.profession, new_person.profession,
                existing.confidence, new_person.confidence
            ),
            background=self._merge_background(
                existing.background, new_person.background
            ),
            
            # 其他字段
            identity=new_person.identity or existing.identity,
            confidence=max(existing.confidence, new_person.confidence),
            
            # 保留现有的元数据和关系
            metadata=existing.metadata,
        )
        
        # 更新缓存
        self._person_cache[existing_id] = merged
        
        return merged
    
    def _merge_list_field(self, old: list, new: list) -> list:
        """合并列表字段：去重
        
        Args:
            old: 现有列表
            new: 新列表
        
        Returns:
            合并去重后的列表
        """
        if not new:
            return old
        if not old:
            return new
        
        # 去重合并（保持顺序）
        seen = set()
        merged = []
        
        for item in old + new:
            # 规范化字符串（去除多余空格）
            normalized = item.strip() if isinstance(item, str) else item
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
        
        return merged
    
    def _merge_string_field(
        self, 
        old: Optional[str], 
        new: Optional[str],
        old_conf: float, 
        new_conf: float
    ) -> Optional[str]:
        """合并字符串字段：选择更详细或置信度更高的
        
        Args:
            old: 现有值
            new: 新值
            old_conf: 现有置信度
            new_conf: 新置信度
        
        Returns:
            合并后的值
        """
        if not new:
            return old
        if not old:
            return new
        
        # 选择更长的（更详细）或置信度更高的
        if len(new) > len(old):
            return new
        elif len(new) == len(old) and new_conf > old_conf:
            return new
        else:
            return old
    
    def _merge_background(self, old: Optional[str], new: Optional[str]) -> Optional[str]:
        """合并背景信息：追加新信息（避免重复）
        
        Args:
            old: 现有背景
            new: 新背景
        
        Returns:
            合并后的背景
        """
        if not new:
            return old
        if not old:
            return new
        
        # 检查是否重复（简单子串检查）
        if new in old:
            return old
        if old in new:
            return new
        
        # 追加新信息
        return f"{old}. {new}"
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息
        
        Returns:
            统计数据字典
        """
        return self._stats.copy()
    
    def reset(self) -> None:
        """重置匹配器（清空索引和缓存）
        
        用于新批次处理时
        """
        self._name_to_person_id.clear()
        self._alias_to_person_id.clear()
        self._person_cache.clear()
        self._stats = {
            "matched_by_name": 0,
            "matched_by_alias": 0,
            "created_new": 0,
            "merged_count": 0
        }
