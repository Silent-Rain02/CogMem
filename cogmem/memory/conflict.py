from __future__ import annotations

import json
from typing import List, Dict

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client
from ..prompt import CONFLICT_SUGGEST_SYSTEM, CONFLICT_SUGGEST_USER
from .schema import Entity


class ConflictResolver:
    def __init__(self, cfg: CogMemConfig) -> None:
        self._cfg = cfg
        self._llm = create_llm_client(cfg)

    def detect_conflicts_rule(self, entities: List[Entity]) -> List[str]:
        # 规则1：重复名称但描述显著不同（长度>0且相似度低）
        issues: List[str] = []
        seen: Dict[str, str] = {}
        for e in entities:
            name = e.name.strip().lower()
            desc = (e.description or "").strip()
            if name in seen and desc and seen[name] and desc[:10] != seen[name][:10]:
                issues.append(f"duplicate-name-diff-desc:{e.name}")
            else:
                if desc:
                    seen[name] = desc
        return issues

    def suggest_update_llm(self, entity: Entity, evidence: List[str], issues: List[str]) -> Dict:
        user = CONFLICT_SUGGEST_USER.format(
            entity=json.dumps(entity.model_dump(), ensure_ascii=False),
            evidence=json.dumps(evidence, ensure_ascii=False),
            issues=json.dumps(issues, ensure_ascii=False),
        )
        resp = self._llm.chat(CONFLICT_SUGGEST_SYSTEM, user)
        try:
            data = json.loads(resp)
            if isinstance(data, dict):
                return data
        except Exception:
            return {"conflict": False, "patch": {}, "reason": "parse-failed"}
        return {"conflict": False, "patch": {}, "reason": "invalid-format"}
