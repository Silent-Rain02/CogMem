import os
import sys
import argparse
import csv
from datetime import datetime
from typing import List, Dict, Optional

from cogmem import CogMem
from cogmem.storage.schema import Doc

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets.locomo_loader import LocomoLoader, QASample
from examples.eval_simple import average_precision, ndcg_at_k
from examples.eval_logger import EvalLogger
from tqdm import tqdm
from collections import defaultdict


def normalize_text(s: str) -> str:
    return "".join(s.lower().strip().split())


def f1_score(pred: str, gold: str) -> float:
    """使用官方LoCoMo评测的F1计算方法（使用词干提取）"""
    from cogmem.generation.rag import normalize_answer
    from collections import Counter
    
    try:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
    except ImportError:
        print("警告：nltk未安装，F1计算将不使用词干提取")
        ps = None
    
    # 标准化答案文本
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    # 词干提取（关键步骤）
    if ps:
        pred_tokens = [ps.stem(w) for w in pred_norm.split()]
        gold_tokens = [ps.stem(w) for w in gold_norm.split()]
    else:
        pred_tokens = pred_norm.split()
        gold_tokens = gold_norm.split()
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # 使用Counter计算词频交集（考虑重复词）
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def em_score(pred: str, gold: str) -> float:
    """使用官方LoCoMo评测的EM计算方法"""
    from cogmem.generation.rag import normalize_answer
    
    # 标准化后比较词集合是否相等
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    return 1.0 if set(pred_norm.split()) == set(gold_norm.split()) else 0.0


def multi_answer_f1(predictions: List[str], ground_truths: List[str]) -> float:
    """处理多答案情况的F1计算，参考官方LoCoMo评测代码"""
    if not predictions or not ground_truths:
        return 0.0
    
    # 计算每个ground truth与所有predictions的最佳F1，然后取平均
    f1_scores = []
    for gt in ground_truths:
        best_f1 = max([f1_score(pred, gt) for pred in predictions])
        f1_scores.append(best_f1)
    
    return sum(f1_scores) / len(f1_scores)


def evidence_hit_rate(gold_answer: str, evidence_chunks: List[str]) -> float:
    """计算证据命中率：检索到的片段是否包含答案关键词"""
    if not evidence_chunks or not gold_answer:
        return 0.0
    gold_lower = gold_answer.lower()
    # 提取答案中的关键词（简单版：去除停用词）
    gold_words = set(w for w in gold_lower.split() if len(w) > 2)
    if not gold_words:
        return 0.0
    # 检查是否有片段包含答案关键词
    for chunk in evidence_chunks[:5]:  # 只看前5个片段
        chunk_lower = chunk.lower()
        matched_words = sum(1 for w in gold_words if w in chunk_lower)
        if matched_words >= len(gold_words) * 0.5:  # 至少50%关键词匹配
            return 1.0
    return 0.0


def evaluate_bundle(cm: CogMem, qa_samples: List[QASample], top_k: int = 10, database: str = "", logger: Optional[EvalLogger] = None):
    """修复后的评测：只使用RAG答案的EM/F1，不比较答案与实体名"""
    total = len(qa_samples)
    EM = 0.0
    F1 = 0.0
    EvidenceHit = 0.0
    
    # 按类别分组统计
    by_category: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # category -> [(q, gold, ans)]

    for i, qa in enumerate(tqdm(qa_samples, desc="Evaluating QAs", total=len(qa_samples))):
        q = qa.question
        gold = qa.answer
        category = qa.category or "Unknown"
        
        # 检索（获取证据片段）
        ret = cm.retrieve(q, top_k=top_k)
        
        # 提取从查询中解析的实体（通过retriever内部逻辑提取，这里简化处理）
        # 注意：由于retrieve是黑盒，我们通过检索结果推断提取的实体
        extracted_entities = []
        if ret.entities:
            # 优先取前几个实体的名称作为提取结果
            extracted_entities = [e.name for e in ret.entities[:5]]
        
        # 提取文本片段作为证据（过滤掉metadata字符串）
        evidence_chunks = [e for e in ret.evidence if len(e) > 20 and not e.startswith(("exact-match", "community-fusion", "vector-search", "gap-iter", "chunk-search"))]
        
        # RAG答案生成
        ans = cm.answer(q, top_k=top_k)
        
        # 计算指标
        # 多答案处理：分割逗号分隔的答案，取最佳匹配
        if qa.category == "Multi-Hop":
            # 多跳问题：分割答案并计算部分F1
            pred_parts = [p.strip() for p in ans.split(',')]
            gold_parts = [g.strip() for g in gold.split(',')]
            f1 = multi_answer_f1(pred_parts, gold_parts)
            em = 1.0 if f1 > 0.9 else 0.0  # 对多跳问题，F1>0.9视为EM
        else:
            # 单跳/时序/开放域问题：直接计算
            em = em_score(ans, gold)
            f1 = f1_score(ans, gold)
        
        ev_hit = evidence_hit_rate(gold, evidence_chunks)
        
        EM += em
        F1 += f1
        EvidenceHit += ev_hit
        
        # 记录日志
        if logger:
            logger.log(
                question=q,
                gold_answer=gold,
                category=category,
                database=database,
                extracted_entities=extracted_entities,
                retrieval_result=ret,
                predicted_answer=ans,
                em_score=em,
                f1_score=f1,
                evidence_hit=ev_hit,
                question_id=f"{database}_{i}",
            )
        
        # 按类别记录
        by_category[category].append((q, gold, ans))

    denom = max(1, total)
    # 按类别计算指标
    category_metrics = {}
    for cat, samples in by_category.items():
        cat_em = sum(em_score(a, g) for _, g, a in samples) / max(1, len(samples))
        cat_f1 = sum(f1_score(a, g) for _, g, a in samples) / max(1, len(samples))
        category_metrics[cat] = {
            "count": len(samples),
            "EM": cat_em,
            "F1": cat_f1,
        }
    
    return {
        "EM": EM / denom,
        "F1": F1 / denom,
        "EvidenceHit": EvidenceHit / denom,
        "by_category": category_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", type=str, default="locomo/data/locomo10.json", help="path to locomo10.json")
    ap.add_argument("--max_samples", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--workspace", type=str, default="test_locomo", help="workspace directory for index persistence (default: per-db)")
    ap.add_argument("--force_rebuild", action="store_true", help="force rebuild index even if exists")
    ap.add_argument("--output_csv", type=str, default=None, help="output CSV file path (default: results/eval_{timestamp}.csv)")
    ap.add_argument("--log_dir", type=str, default="logs/eval", help="log directory for detailed evaluation logs")
    ap.add_argument("--save_logs", default=True, action="store_true", help="save detailed evaluation logs for bad case analysis")
    args = ap.parse_args()

    loader = LocomoLoader(args.locomo)

    # 准备CSV输出
    csv_path = args.output_csv
    if csv_path is None:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"results/eval_{timestamp}.csv"
    
    # 初始化日志记录器
    logger = EvalLogger(log_dir=args.log_dir) if args.save_logs else None
    
    all_results: List[Dict] = []

    # 根据官方评测，只使用 conversation 数据，不分 dialog/observation/session_summary
    for db in ["conversation"]:
        bundle = loader.load(db=db, max_samples=args.max_samples)
        if not bundle.docs or not bundle.qa:
            print(f"[{db}] no data, skipped")
            continue
        # 工作空间目录：若指定则统一，否则按db分开
        ws_dir = args.workspace or f".cogmem-{db}"
        print(f"\n[{db}] workspace: {ws_dir}")
        print(f"[{db}] indexing documents ({len(bundle.docs)})...")
        cm = CogMem(workspace_dir=ws_dir, auto_load=True)
        cm.reset_metrics()
        
        # 增量索引或全量重建
        if cm.is_indexed() and not args.force_rebuild:
            print(f"[{db}] Found existing index, using incremental indexing...")
            cm.add_documents_incremental(bundle.docs, force_rebuild=False)
        else:
            print(f"[{db}] Building index from scratch...")
            cm.add_documents(bundle.docs)
        cm.discover_communities(semantic=True)
        cm.save()  # 保存索引以便下次复用
        metrics = evaluate_bundle(cm, bundle.qa, top_k=args.top_k, database=db, logger=logger)
        usage = cm.get_metrics()
        
        # 汇总结果
        chat_usage = usage.get("chat", {})
        embed_usage = usage.get("embed", {})
        # 提取总体指标和类别指标
        overall = {k: v for k, v in metrics.items() if k != "by_category"}
        result_row = {
            "database": db,
            "workspace": ws_dir,
            "num_docs": len(bundle.docs),
            "num_qa": len(bundle.qa),
            "top_k": args.top_k,
            **overall,
            "chat_calls": chat_usage.get("calls", 0),
            "chat_cache_hits": chat_usage.get("cache_hits", 0),
            "chat_total_tokens": chat_usage.get("total_tokens", 0),
            "chat_latency_ms": chat_usage.get("total_latency_ms", 0.0),
            "embed_calls": embed_usage.get("calls", 0),
            "embed_cache_hits": embed_usage.get("cache_hits", 0),
            "embed_latency_ms": embed_usage.get("total_latency_ms", 0.0),
        }
        all_results.append(result_row)
        
        print(f"\n=== {db} ===")
        print("Overall Metrics:")
        for k, v in overall.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.3f}")
        print("\nBy Category:")
        for cat, cat_metrics in metrics.get("by_category", {}).items():
            print(f"  {cat} (n={cat_metrics['count']}):")
            print(f"    EM: {cat_metrics['EM']:.3f}, F1: {cat_metrics['F1']:.3f}")
        print("[Usage] chat:", chat_usage)
        print("[Usage] embed:", embed_usage)

    # 写入CSV
    if all_results:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[Results saved to] {csv_path}")
    
    # 保存日志并分析bad cases
    if logger:
        log_path = logger.save()
        print(f"\n[Evaluation logs saved to] {log_path}")
        bad_analysis = logger.analyze_bad_cases(threshold_f1=0.5)
        print("\n=== Bad Case Analysis (F1 < 0.5) ===")
        print(f"Total: {bad_analysis['total']}, Bad Cases: {bad_analysis['bad_cases']} ({bad_analysis['bad_rate']:.1%})")
        if bad_analysis['bad_cases'] > 0:
            print("\nBy Category:")
            for cat, count in bad_analysis['by_category'].items():
                print(f"  {cat}: {count}")
            print("\nBy Retrieval Path:")
            for path, count in bad_analysis['by_retrieval_path'].items():
                print(f"  {path}: {count}")
            print(f"\nNo Entities: {bad_analysis['no_entities_count']}")
            print(f"No Chunks: {bad_analysis['no_chunks_count']}")


if __name__ == "__main__":
    main()
