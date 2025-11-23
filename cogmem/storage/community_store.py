from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np


@dataclass
class SemanticCommunity:
    id: str
    member_ids: List[str]
    centroid: List[float]


@dataclass
class StructureCommunity:
    id: str
    member_ids: List[str]


class CommunityStore:
    def __init__(self) -> None:
        self.semantic: List[SemanticCommunity] = []
        self.structure: List[StructureCommunity] = []

    def set_semantic(self, communities: List[SemanticCommunity]) -> None:
        self.semantic = communities

    def get_semantic(self) -> List[SemanticCommunity]:
        return self.semantic

    def set_structure(self, communities: List[StructureCommunity]) -> None:
        self.structure = communities

    def get_structure(self) -> List[StructureCommunity]:
        return self.structure
    
    def get_entity_communities(self, entity_id: str) -> List[SemanticCommunity | StructureCommunity]:
        """获取实体所属的所有社区（语义社区 + 结构社区）
        
        支持Person、Event、Fact、Entity等所有节点类型
        """
        communities = []
        # 语义社区
        for comm in self.semantic:
            if entity_id in comm.member_ids:
                communities.append(comm)
        # 结构社区
        for comm in self.structure:
            if entity_id in comm.member_ids:
                communities.append(comm)
        return communities
    
    def get_community_members(self, entity_ids: List[str]) -> List[str]:
        """获取与给定实体在同一社区的所有其他实体ID
        
        Args:
            entity_ids: 初始实体ID列表
            
        Returns:
            所有在同一社区的其他实体ID列表（去重，不包括输入实体）
        """
        all_member_ids = set()
        
        # 找到所有相关社区
        for entity_id in entity_ids:
            communities = self.get_entity_communities(entity_id)
            for comm in communities:
                all_member_ids.update(comm.member_ids)
        
        # 移除输入实体，返回其他成员
        return [mid for mid in all_member_ids if mid not in entity_ids]
    
    def get_community_entities(self, community_id: str) -> List[str]:
        """获取社区中的所有实体ID"""
        # 先查找语义社区
        for comm in self.semantic:
            if comm.id == community_id:
                return comm.member_ids.copy()
        # 再查找结构社区
        for comm in self.structure:
            if comm.id == community_id:
                return comm.member_ids.copy()
        return []

    def save(self, json_path: str, centroid_path: str) -> None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        meta = {
            "semantic": [{"id": c.id, "member_ids": c.member_ids} for c in self.semantic],
            "structure": [{"id": c.id, "member_ids": c.member_ids} for c in self.structure],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        # 质心单独保存
        centroids = [c.centroid for c in self.semantic]
        with open(centroid_path, "wb") as f:
            np.save(f, np.array(centroids, dtype=np.float32))

    def load(self, json_path: str, centroid_path: str) -> None:
        if not (os.path.exists(json_path) and os.path.exists(centroid_path)):
            self.semantic = []
            self.structure = []
            return
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(centroid_path, "rb") as f:
            centroids = list(np.load(f))
        sem = []
        for meta, cen in zip(data.get("semantic", []), centroids):
            sem.append(SemanticCommunity(id=meta["id"], member_ids=list(meta["member_ids"]), centroid=list(map(float, cen))))
        self.semantic = sem
        self.structure = [StructureCommunity(id=meta["id"], member_ids=list(meta["member_ids"])) for meta in data.get("structure", [])]
