"""
LongMemEval 评测脚本：与 LoCoMo 主评测共用 skill 消融开关与结果格式。

使用方式：
  python scripts/eval_longmemeval.py --data_dir datasets/longmemeval_s_cleaned.json --workspace .cogmem_longmem
  python scripts/eval_longmemeval.py --no-temporal-skill --workspace .cogmem_longmem
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Prefer the colocated LoCoMo runner when this script is executed directly.
if SCRIPTS_DIR in sys.path:
    sys.path.remove(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import dotenv
dotenv.load_dotenv()

from cogmem import CogMem
from cogmem.config import DEFAULT_CONFIG
from cogmem.storage.schema import NodeTypeEnum

# 复用 LoCoMo 的指标与清理函数
from eval_locomo_complete import (
    _clean_answer_for_evaluation,
    compute_exact_match,
    compute_f1_score,
    compute_bleu_score,
)

def load_bundle(data_path: str, max_samples: int = None):
    """加载 LongMemEval 数据包（docs + qa）。"""
    from datasets.longmemeval_loader import LongMemEvalLoader
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"LongMemEval 数据文件不存在: {data_path}")
    loader = LongMemEvalLoader(data_path)
    return loader.load(max_samples=max_samples)

def main():
    parser = argparse.ArgumentParser(description="LongMemEval 评测（与 LoCoMo 共用 skill 开关与输出格式）")
    parser.add_argument("--data_dir", type=str, default="datasets/longmemeval_s_cleaned.json", help="LongMemEval JSON 文件路径")
    parser.add_argument("--workspace", type=str, default=".cogmem_longmemeval", help="CogMem 工作空间")
    parser.add_argument("--output", type=str, default="longmemeval_eval_results.json", help="输出 JSON 路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评测样本数")
    parser.add_argument("--skip_index", action="store_true", help="跳过索引，使用已有图谱")
    parser.add_argument("--skip_consolidation", action="store_true", help="跳过记忆巩固")
    parser.add_argument("--no-temporal-skill", action="store_true", dest="no_temporal_skill", help="消融：关闭 temporal skill")
    parser.add_argument("--no-evidence-skill", action="store_true", dest="no_evidence_skill", help="消融：关闭 evidence skill")
    parser.add_argument("--no-routing-skill", action="store_true", dest="no_routing_skill", help="消融：关闭 routing skill")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    config.extraction.use_enhanced_extraction = True
    config.chunk.enable_chunk_embedding = False
    config.chunk.keep_raw_text = True

    if getattr(args, "no_temporal_skill", False):
        config.agent.enable_temporal_skill = False
        print("  消融: 已关闭 temporal_skill")
    if getattr(args, "no_evidence_skill", False):
        config.agent.enable_evidence_skill = False
        print("  消融: 已关闭 evidence_skill")
    if getattr(args, "no_routing_skill", False):
        config.agent.enable_routing_skill = False
        print("  消融: 已关闭 routing_skill")

    print("加载 LongMemEval 数据...")
    bundle = load_bundle(args.data_dir, max_samples=args.max_samples)
    if not bundle.qa:
        print("未找到 QA 样本")
        return

    cogmem = CogMem(config=config, workspace_dir=args.workspace, auto_load=True)
    total_nodes = cogmem.graph.node_count
    print(f"图节点数: {total_nodes}")

    if not args.skip_index and total_nodes == 0:
        print("构建索引...")
        cogmem.add_documents(bundle.docs)
        cogmem.save()
        print("索引完成")
    elif args.skip_index and total_nodes == 0:
        print("错误: skip_index 但图谱为空")
        return

    if not args.skip_consolidation:
        print("执行记忆巩固...")
        cogmem.consolidate_memory()
        cogmem.save()

    results = []
    for idx, qa in enumerate(bundle.qa):
        question = qa.question
        gold = qa.answer
        qid = getattr(qa, "question_id", f"lm_{idx}")
        print(f"[{idx+1}/{len(bundle.qa)}] {question[:60]}...")
        try:
            full = cogmem.answer(question, return_full_result=True, verbose=False)
            pred = full.get("answer", "")
            reasoning_path = full.get("reasoning_path", [])
            cleaned_pred = _clean_answer_for_evaluation(pred)
            em = compute_exact_match(cleaned_pred, gold)
            f1 = compute_f1_score(cleaned_pred, gold)
            bleu = compute_bleu_score(cleaned_pred, gold)
            result = {
                "question_id": qid,
                "question": question,
                "gold_answer": gold,
                "predicted_answer": cleaned_pred,
                "category": getattr(qa, "category", None) or getattr(qa, "question_type", ""),
                "em_score": em,
                "f1_score": f1,
                "bleu_score": bleu,
                "reasoning_path": [s if isinstance(s, dict) else {"raw": str(s)} for s in reasoning_path],
            }
            if isinstance(full, dict):
                if "question_type" in full:
                    result["question_type"] = full["question_type"]
                if "skill_flags" in full:
                    result["skill_flags"] = full["skill_flags"]
                if "skill_usage" in full:
                    result["skill_usage"] = full["skill_usage"]
                result["effective_agent_config"] = full.get("effective_agent_config") or {}
            results.append(result)
        except Exception as e:
            results.append({
                "question_id": qid,
                "question": question,
                "gold_answer": gold,
                "predicted_answer": f"[ERROR] {e}",
                "em_score": 0.0,
                "f1_score": 0.0,
                "bleu_score": 0.0,
                "error": str(e),
            })

    n = len(results)
    avg_em = sum(r["em_score"] for r in results) / n if n else 0
    avg_f1 = sum(r["f1_score"] for r in results) / n if n else 0
    avg_bleu = sum(r["bleu_score"] for r in results) / n if n else 0

    skill_flags = {}
    try:
        if hasattr(cogmem, "qa_orchestrator") and getattr(cogmem.qa_orchestrator, "skill_runtime", None):
            skill_flags = cogmem.qa_orchestrator.skill_runtime.skill_flags()
    except Exception:
        pass
    parts = []
    if not skill_flags.get("enable_temporal_skill", True):
        parts.append("no_temporal")
    if not skill_flags.get("enable_evidence_skill", True):
        parts.append("no_evidence")
    if not skill_flags.get("enable_routing_skill", True):
        parts.append("no_routing")
    experiment_name = "full" if not parts else "_".join(parts)
    agent_base_config = {}
    try:
        if hasattr(cogmem, "cfg") and getattr(cogmem.cfg, "agent", None):
            agent_base_config = asdict(cogmem.cfg.agent)
    except Exception:
        pass

    output_data = {
        "experiment_name": experiment_name,
        "skill_flags": skill_flags,
        "agent_base_config": agent_base_config,
        "summary": {
            "total_samples": n,
            "avg_em_score": avg_em,
            "avg_f1_score": avg_f1,
            "avg_bleu_score": avg_bleu,
            "timestamp": datetime.now().isoformat(),
        },
        "detailed_results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {args.output}")
    print(f"  EM: {avg_em:.3f}, F1: {avg_f1:.3f}, BLEU: {avg_bleu:.3f}")

if __name__ == "__main__":
    main()
