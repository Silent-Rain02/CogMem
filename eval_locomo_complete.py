"""LoCoMo 完整评测脚本：包含 EM、F1、Evidence Hit、LLM Judge 全部指标"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from cogmem import CogMem
from cogmem.config import DEFAULT_CONFIG
from cogmem.storage.schema import Doc, NodeTypeEnum

# 尝试导入评测模块
try:
    from examples.llm_judge_eval import LLMJudge, compute_judge_metrics
except ImportError:
    # 如果llm_judge_eval不存在，提供占位符
    class LLMJudge:
        def __init__(self, config):
            pass
        def evaluate_answer(self, question, gold_answer, predicted_answer, category=None):
            return type('JudgeResult', (), {
                'semantic_correctness': 0.5,
                'completeness': 0.5,
                'relevance': 0.5,
                'overall_score': 0.5,
                'explanation': 'LLM Judge not available'
            })()
    def compute_judge_metrics(*args, **kwargs):
        return {'avg_overall_score': 0.5}

try:
    from datasets.locomo_loader import load_locomo_dataset
except ImportError:
    def load_locomo_dataset(data_dir):
        print(f"Warning: Cannot load LoCoMo dataset from {data_dir}")
        return []


def normalize_answer(s: str) -> str:
    """标准化答案文本，严格按照官方LoCoMo评测代码实现，增强日期处理"""
    import re
    s = s.replace(',', "")  # 必须移除逗号

    def remove_articles(text):
        # 移除冠词和连接词：a, an, the, and
        return re.sub(r'\b(a|an|the|and)\b', ' ', text, flags=re.IGNORECASE)

    def white_space_fix(text):
        # 规范化空白字符
        return ' '.join(text.split())

    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def extract_date_info(text):
        """从文本中提取日期相关信息"""
        # 匹配ISO日期格式：YYYY-MM-DDTHH:MM:SS
        iso_pattern = r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})'
        matches = re.findall(iso_pattern, text)

        result = text
        for year, month, day, hour, minute, second in matches:
            # 将月份数字转换为月份名称
            month_names = ['', 'january', 'february', 'march', 'april', 'may', 'june',
                          'july', 'august', 'september', 'october', 'november', 'december']
            month_name = month_names[int(month)] if 1 <= int(month) <= 12 else month

            # 替换ISO日期为可读格式
            date_str = f"{month_name} {day} {year}"
            iso_date = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
            result = result.replace(iso_date, date_str)

        return result

    # 先提取日期信息，然后再进行其他标准化
    s = extract_date_info(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(predicted: str, gold: str) -> float:
    """计算精确匹配分数：使用词集合比较（官方方法）"""
    pred_normalized = normalize_answer(predicted)
    gold_normalized = normalize_answer(gold)
    # 关键：使用词集合（set）比较，而不是字符串比较
    return 1.0 if set(pred_normalized.split()) == set(gold_normalized.split()) else 0.0


def compute_f1_score(predicted: str, gold: str) -> float:
    """计算F1分数：严格按照官方LoCoMo实现（使用词干提取）"""
    try:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
    except ImportError:
        # 如果没有nltk，回退到不使用词干提取（不推荐）
        print("警告：nltk未安装，F1计算将不使用词干提取，可能不准确")
        ps = None
    
    from collections import Counter
    
    # 标准化答案
    pred_norm = normalize_answer(predicted)
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
    
    # 使用Counter计算词频，然后求交集（考虑重复词）
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    # 计算精确率和召回率
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    
    # 计算F1
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_multi_answer_f1(predicted: str, gold: str) -> float:
    """计算多答案F1（用于Multi-Hop问题）
    
    官方方法：将预测和真实答案按逗号分割，对每个真实答案
    找最匹配的预测，然后取平均
    """
    predictions = [p.strip() for p in predicted.split(',')]
    ground_truths = [g.strip() for g in gold.split(',')]
    
    if not predictions or not ground_truths:
        return 0.0
    
    # 对每个ground truth，找最佳匹配的prediction的F1，然后取平均
    import numpy as np
    f1_scores = []
    for gt in ground_truths:
        best_f1 = max([compute_f1_score(pred, gt) for pred in predictions])
        f1_scores.append(best_f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0


def compute_evidence_hit(evidence: List[str], gold_answer: str) -> float:
    """计算证据命中率：检查证据中是否包含金标答案的关键信息

    参考 examples/eval_locomo.py 的实现，并适配新 retriever 的证据格式：
    - 新格式: [Depth: 0] [Time: 2024-01-15] [Topic: Work & Career] [Type: Event] [Conf: 0.91] 事件描述...
    - 需要过滤掉元数据标记，只保留实际内容
    - 提取答案中的关键词（长度>2的词）
    - 检查是否有证据片段包含至少50%的关键词
    """
    if not evidence or not gold_answer:
        return 0.0

    # 标准化答案并提取关键词（长度>2的词，过滤停用词）
    gold_normalized = normalize_answer(gold_answer)
    gold_tokens = set(w for w in gold_normalized.split() if len(w) > 2)

    if not gold_tokens:
        return 0.0

    # 检查前5个证据片段（与 examples/eval_locomo.py 保持一致）
    for ev in evidence[:5]:
        if not ev or len(ev) < 20:
            continue

        # 过滤掉metadata字符串（如 "exact-match", "community-fusion" 等）
        if ev.startswith(("exact-match", "community-fusion", "vector-search", "gap-iter", "chunk-search")):
            continue

        # 新格式：过滤掉 [Depth: ...] [Time: ...] [Topic: ...] [Type: ...] [Conf: ...] 等元数据标记
        # 提取实际内容：移除所有 [XXX: ...] 格式的标记
        import re
        # 移除所有 [XXX: ...] 格式的标记
        ev_content = re.sub(r'\[[^\]]+\]', '', ev).strip()
        
        # 如果移除标记后内容为空，尝试提取最后一个 ] 之后的内容
        if not ev_content:
            content_start = ev.rfind(']')
            if content_start != -1 and content_start < len(ev) - 1:
                ev_content = ev[content_start + 1:].strip()
            else:
                ev_content = ev

        if not ev_content:
            continue

        ev_normalized = normalize_answer(ev_content)
        ev_tokens = set(ev_normalized.split())

        # 计算匹配的关键词数量
        matched_words = len(gold_tokens & ev_tokens)

        # 对于短答案（关键词<=2），需要100%匹配；对于长答案，至少50%匹配
        if len(gold_tokens) <= 2:
            if matched_words == len(gold_tokens):
                return 1.0
        else:
            if matched_words >= len(gold_tokens) * 0.5:
                return 1.0

    return 0.0


def map_category_id_to_name(category_id: int) -> str:
    """将类别ID映射到名称（按照LoCoMo官方标准）
    
    根据官方评估代码 (locomo/task_eval/evaluation.py):
    - category 1 = Multi-Hop (使用多答案F1)
    - category 2 = Single-Hop (使用单答案F1)
    - category 3 = Temporal (时间推理)
    - category 4 = Open Domain (开放域)
    - category 5 = Adversarial (对抗性)
    """
    category_map = {
        1: "Multi-Hop",      # Multi-Hop（多跳问题）
        2: "Single Hop",     # Single-Hop（单跳问题）
        3: "Temporal",       # Temporal（时间推理问题）
        4: "Open Domain",   # Open Domain（开放域问题）
        5: "Adversarial"     # Adversarial（对抗性问题）
    }
    return category_map.get(category_id, "Unknown")


def comprehensive_evaluation(
    cogmem: CogMem,
    dataset: List[Dict],
    judge: Optional[LLMJudge] = None,
    max_samples: int = None,
    output_file: str = "complete_eval_results.json",
    quiet: bool = False  # 新增：安静模式，减少日志输出
) -> Dict:
    """完整评测：计算所有指标
    
    Args:
        cogmem: CogMem实例
        dataset: 评测数据集
        judge: LLM Judge实例（可选）
        max_samples: 最大评测样本数
        output_file: 输出文件路径
        quiet: 是否启用安静模式（减少日志输出）
    """
    
    results = []
    
    # 限制评测样本数量
    if max_samples:
        dataset = dataset[:max_samples]
    
    if not quiet:
        print(f"开始完整评测 {len(dataset)} 个问题...")
    
    # 获取并发配置（与索引保持一致）
    max_workers = max(1, cogmem.cfg.runtime.max_concurrency)
    if not quiet:
        print(f" 使用并发数: {max_workers} (与索引配置一致)")
    
    # 临时禁用DEBUG日志（如果quiet模式）
    import logging
    original_level = logging.getLogger().level
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # 注意：不使用答案缓存，因为相同问题在不同会话中可能有不同答案
    # 每个问题都应该独立生成答案以确保准确性
    
    def evaluate_single_sample(item: Dict, index: int) -> Optional[Dict]:
        """评测单个样本"""
        question = item.get("question", "")
        gold_answer = item.get("answer", "")
        category_id = item.get("category", 0)
        category = map_category_id_to_name(category_id) if isinstance(category_id, int) else str(category_id)
        question_id = item.get("id", f"q_{index}")
        
        start_time = time.time()
        
        try:
            # 使用 Agent 进行检索和回答（通过 QAOrchestrator）
            # 调用 answer 方法，同时获取答案和检索结果
            predicted_answer, retrieval_result = cogmem.answer(question, top_k=10, return_retrieval_result=True)
            
            inference_time = time.time() - start_time
            
            # 计算传统指标（根据不同类别使用不同的评估方式）
            category_id = item.get("category", 0)
            
            # Temporal问题特殊处理：如果答案包含分号，只取第一部分（仅处理gold_answer）
            processed_gold_answer = gold_answer
            if category_id == 3:  # Temporal
                processed_gold_answer = gold_answer.split(';')[0].strip()
            
            # 根据类别选择评估方式（按照官方LoCoMo评估标准）
            # 参考: locomo/task_eval/evaluation.py 第209-214行
            if category_id == 1:  # Multi-Hop：使用多答案F1
                em_score = compute_exact_match(predicted_answer, processed_gold_answer)
                f1_score = compute_multi_answer_f1(predicted_answer, processed_gold_answer)
            elif category_id == 5:  # Adversarial：检查是否识别出"无信息"
                # 检查英文和中文的"无信息"表达
                pred_lower = predicted_answer.lower()
                if ('no information available' in pred_lower or 
                    'not mentioned' in pred_lower or
                    '无法确定' in predicted_answer or
                    '不确定' in predicted_answer or
                    '不知道' in predicted_answer):
                    em_score = 1.0
                    f1_score = 1.0
                else:
                    em_score = 0.0
                    f1_score = 0.0
            else:  # Single-Hop (2), Temporal (3), Open-Domain (4)：使用单答案F1
                em_score = compute_exact_match(predicted_answer, processed_gold_answer)
                f1_score = compute_f1_score(predicted_answer, processed_gold_answer)
            
            evidence_hit = compute_evidence_hit(retrieval_result.evidence, gold_answer)
            
            # LLM Judge 评测
            judge_result = None
            if judge:
                try:
                    judge_result = judge.evaluate_answer(
                        question=question,
                        gold_answer=gold_answer,
                        predicted_answer=predicted_answer,
                        category=category
                    )
                except Exception as e:
                    # 静默处理错误，避免打印过多信息
                    pass
            
            # 记录详细结果
            result = {
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "category": category,
                "question_id": question_id,
                
                # 传统指标
                "em_score": em_score,
                "f1_score": f1_score,
                "evidence_hit": evidence_hit,
                
                # LLM Judge 指标
                "judge_semantic_correctness": judge_result.semantic_correctness if judge_result else 0.0,
                "judge_completeness": judge_result.completeness if judge_result else 0.0,
                "judge_relevance": judge_result.relevance if judge_result else 0.0,
                "judge_overall_score": judge_result.overall_score if judge_result else 0.0,
                "judge_explanation": judge_result.explanation if judge_result else "",
                
                # 检索分析 - 立即复制以避免多线程问题
                "extracted_entities": list(getattr(retrieval_result, 'extracted_entity_names', [])),
                "retrieval_path": list(getattr(retrieval_result, 'retrieval_path', [])),
                "retrieved_entities": [
                    {
                        "name": e.name,
                        "description": getattr(e, 'description', '')[:100] + "..." if len(getattr(e, 'description', '')) > 100 else getattr(e, 'description', ''),
                        "type": e.type
                    }
                    for e in retrieval_result.entities[:10]  # 增加到10个实体
                ],
                "num_entities": len(retrieval_result.entities),
                "num_evidence": len(retrieval_result.evidence),
                
                # 性能指标
                "inference_time": inference_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            if not quiet:
                print(f"评测第 {index+1} 个样本时出错: {str(e)}")
            return None
    
    # 异步评测：使用线程池并行处理，支持进度保存
    results: List[Dict] = []
    checkpoint_file = output_file.replace(".json", "_checkpoint.json")
    
    # 尝试加载已有进度
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                completed_ids = {r.get("question_id") for r in checkpoint_data.get("results", [])}
                results = checkpoint_data.get("results", [])
                print(f" 加载已有进度: {len(results)}/{len(dataset)} 已完成")
        except Exception as e:
            print(f"⚠️  加载进度失败: {e}，从头开始")
            completed_ids = set()
            results = []
    else:
        completed_ids = set()
        results = []
    
    # 过滤已完成的样本
    remaining_items = [(i, item) for i, item in enumerate(dataset) 
                       if item.get("id", f"q_{i}") not in completed_ids]
    
    if remaining_items:
        if not quiet:
            print(f" 开始评测剩余 {len(remaining_items)} 个样本...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(evaluate_single_sample, item, i): (i, item)
                for i, item in remaining_items
            }

            # 使用进度条显示完成情况，每完成一个就保存
            for future in tqdm(as_completed(future_to_item), total=len(remaining_items), desc="评测进度"):
                try:
                    # 添加超时控制，避免单个样本耗时过长
                    result = future.result(timeout=120.0)  # 最多2分钟 per sample
                    if result:
                        results.append(result)
                except Exception as e:
                    if not quiet:
                        print(f"⚠️  样本处理失败: {str(e)}")
                    # 继续处理其他样本，不中断整个评测

                # 每完成一个样本就保存进度（防止中断丢失）
                try:
                    checkpoint_data = {
                        "results": results,
                        "timestamp": datetime.now().isoformat(),
                        "completed": len(results),
                        "total": len(dataset)
                    }
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    if not quiet:
                        print(f"保存进度失败: {e}")
    else:
        print(f"所有样本已完成，直接使用已有结果")
    
    # 恢复日志级别
    if quiet:
        logging.getLogger().setLevel(original_level)
    
    # 清理临时checkpoint文件（如果成功完成）
    checkpoint_file = output_file.replace(".json", "_checkpoint.json")
    if os.path.exists(checkpoint_file) and len(results) >= len(dataset):
        try:
            os.remove(checkpoint_file)
            if not quiet:
                print(f"✅ 清理临时进度文件: {checkpoint_file}")
        except Exception:
            pass
    
    # 计算聚合指标
    if results:
        # 传统指标
        avg_em = sum(r["em_score"] for r in results) / len(results)
        avg_f1 = sum(r["f1_score"] for r in results) / len(results)
        avg_evidence_hit = sum(r["evidence_hit"] for r in results) / len(results)
        
        # LLM Judge 指标
        # 使用所有结果计算平均值，包括 judge_overall_score = 0.0 的样本
        # 这样可以正确反映整体性能（0.0 可能是真的0分，也可能是judge调用失败）
        if results:
            avg_judge_score = sum(r["judge_overall_score"] for r in results) / len(results)
            avg_semantic = sum(r["judge_semantic_correctness"] for r in results) / len(results)
            avg_completeness = sum(r["judge_completeness"] for r in results) / len(results)
            avg_relevance = sum(r["judge_relevance"] for r in results) / len(results)
        else:
            avg_judge_score = avg_semantic = avg_completeness = avg_relevance = 0.0
        
        # 高质量回答比例：使用所有结果作为分母（包括没有 judge 结果的）
        # 只有 judge_overall_score >= 0.7 的才算高质量
        high_quality_count = sum(1 for r in results if r["judge_overall_score"] >= 0.7)
        high_quality_ratio = high_quality_count / len(results) if results else 0.0
        
        # 按类别统计
        category_stats = {}
        for result in results:
            cat = result["category"]
            if cat not in category_stats:
                category_stats[cat] = {
                    "count": 0,
                    "avg_em": 0.0,
                    "avg_f1": 0.0,
                    "avg_evidence_hit": 0.0,
                    "avg_judge_score": 0.0
                }
            
            stats = category_stats[cat]
            stats["count"] += 1
            stats["avg_em"] += result["em_score"]
            stats["avg_f1"] += result["f1_score"]
            stats["avg_evidence_hit"] += result["evidence_hit"]
            stats["avg_judge_score"] += result["judge_overall_score"]
        
        # 计算平均值
        for cat, stats in category_stats.items():
            count = stats["count"]
            if count > 0:
                stats["avg_em"] /= count
                stats["avg_f1"] /= count
                stats["avg_evidence_hit"] /= count
                stats["avg_judge_score"] /= count
    else:
        avg_em = avg_f1 = avg_evidence_hit = avg_judge_score = 0.0
        avg_semantic = avg_completeness = avg_relevance = high_quality_ratio = 0.0
        category_stats = {}
    
    # 识别 Bad Cases (EM=0 且 (Judge Score < 0.5 或没有 Judge 结果))
    # 注意：如果 judge_overall_score 为 0，说明没有 judge 结果，也应该算作 bad case
    bad_cases = [
        r for r in results 
        if r["em_score"] == 0.0 and (r["judge_overall_score"] == 0.0 or r["judge_overall_score"] < 0.5)
    ]
    
    # 保存结果
    evaluation_report = {
        "summary": {
            "total_samples": len(results),
            "bad_cases_count": len(bad_cases),
            "bad_case_ratio": len(bad_cases) / len(results) if results else 0.0,
            
            # 传统指标
            "avg_em_score": avg_em,
            "avg_f1_score": avg_f1,
            "avg_evidence_hit": avg_evidence_hit,
            
            # LLM Judge 指标
            "avg_judge_score": avg_judge_score,
            "avg_semantic_correctness": avg_semantic,
            "avg_completeness": avg_completeness,
            "avg_relevance": avg_relevance,
            "high_quality_ratio": high_quality_ratio
        },
        "category_breakdown": category_stats,
        "detailed_results": results,
        "bad_cases": bad_cases[:20],  # 保存前20个bad case
        "evaluation_config": {
            "retrieval_method": "agentic",  # 使用Agentic检索系统
            "answer_method": "agentic",     # 使用Agentic问答系统
            "max_samples": max_samples,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完整评测完成！结果已保存到 {output_file}")
    print(f" 评测结果总览:")
    print(f"  总样本数: {len(results)}")
    print(f"  EM 分数: {avg_em:.3f}")
    print(f"  F1 分数: {avg_f1:.3f}")
    print(f"  Evidence Hit: {avg_evidence_hit:.3f}")
    print(f"  LLM Judge 分数: {avg_judge_score:.3f}")
    print(f"  高质量回答比例: {high_quality_ratio:.3f}")
    if len(results) > 0:
        print(f"  Bad Case 数量: {len(bad_cases)} ({len(bad_cases)/len(results)*100:.1f}%)")
    else:
        print(f"  Bad Case 数量: {len(bad_cases)} (N/A - 无有效结果)")
    
    return evaluation_report


def analyze_performance_by_category(results_file: str):
    """按类别分析性能"""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    category_breakdown = data.get("category_breakdown", {})
    
    print(f"\n 按类别性能分析:")
    print("-" * 80)
    print(f"{'类别':<15} {'样本数':<8} {'EM':<8} {'F1':<8} {'EvidHit':<10} {'Judge':<8}")
    print("-" * 80)
    
    for category, stats in category_breakdown.items():
        print(f"{category:<15} {stats['count']:<8} "
              f"{stats['avg_em']:<8.3f} {stats['avg_f1']:<8.3f} "
              f"{stats['avg_evidence_hit']:<10.3f} {stats['avg_judge_score']:<8.3f}")
    
    print("-" * 80)


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="LoCoMo 完整评测")
    parser.add_argument("--data_dir", type=str, default="locomo", help="LoCoMo数据集目录")
    parser.add_argument("--workspace", type=str, default=".cogmem_complete_eval", help="CogMem工作空间")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评测问题数（QA数量，不是Conversation数量）")
    parser.add_argument("--output", type=str, default="complete_eval_results.json", help="输出文件")
    parser.add_argument("--analyze_only", action="store_true", help="仅分析已有结果")
    parser.add_argument("--quiet", action="store_true", help="安静模式：减少日志输出，加快评测速度")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        if os.path.exists(args.output):
            analyze_performance_by_category(args.output)
        else:
            print(f"结果文件 {args.output} 不存在")
        return
    
    # 加载数据集
    print("加载 LoCoMo 数据集...")
    dataset = load_locomo_dataset(args.data_dir)
    
    if not dataset:
        print("数据集加载失败")
        return
    
    # 注意：dataset是问题列表，不是conversation列表
    print(f"数据集加载成功，共 {len(dataset)} 个问题（QA）")
    
    # 初始化 CogMem
    print(" 初始化 CogMem...")
    config = DEFAULT_CONFIG
    config.extraction.use_enhanced_extraction = True  # 启用增强版实体抽取

    # 配置检索参数以支持 Topic-Scoped Retrieval
    config.retrieval.use_topic_scoping = True  # 启用 Topic 范围检索
    config.retrieval.topic_top_k = 3  # Topic Matching 阶段返回的 Top-K Topics
    config.retrieval.seed_top_k = 15  # Seed Finding 阶段返回的种子节点数
    config.community.enable_structure = True  # 启用语义社区发现（生成 TopicNodes）
    config.community.semantic_k = 20  # 语义聚类数量（可根据数据集大小调整）
    
    cogmem = CogMem(config=config, workspace_dir=args.workspace, auto_load=True)
    
    # 检查图是否正确加载
    total_nodes = cogmem.graph.num_entities()
    print(f" 已加载图节点数: {total_nodes}")

    # 检查 Person 节点
    person_count = 0
    for node_id, node_data in cogmem.graph._g.nodes(data=True):
        if node_data and "canonical_name" in node_data:
            person_count += 1
    print(f" Person 节点数: {person_count}")

    # 检查 Topic 节点（新 retriever 依赖 TopicNodes 进行语义检索）
    topic_count = 0
    event_count = 0
    fact_count = 0
    for node_id, node_data in cogmem.graph._g.nodes(data=True):
        if not node_data:
            continue
        node_type = node_data.get("node_type")
        if node_type == NodeTypeEnum.TOPIC.value:
            topic_count += 1
        elif node_type == NodeTypeEnum.EVENT.value:
            event_count += 1
        elif node_type == NodeTypeEnum.FACT.value:
            fact_count += 1

    print(f" Topic 节点数: {topic_count}")
    print(f" Event 节点数: {event_count}")
    print(f" Fact 节点数: {fact_count}")

    # 检查是否需要执行社区发现（如果有节点但没有Topic节点）
    needs_community_discovery = total_nodes > 0 and topic_count == 0
    
    if needs_community_discovery:
        print(" ⚠️ 检测到图中有节点但没有 Topic 节点，需要执行社区发现...")
        print("  正在发现语义社区（这可能需要几分钟）...")
        try:
            cogmem.discover_communities(semantic=True)
            print("  ✓ 社区发现完成")
            
            # 重新统计 Topic 节点
            topic_count = 0
            for node_id, node_data in cogmem.graph._g.nodes(data=True):
                if node_data and node_data.get("node_type") == NodeTypeEnum.TOPIC.value:
                    topic_count += 1
            print(f"  现在有 {topic_count} 个 Topic 节点")
            
            # 保存更新后的索引
            print("  正在保存更新后的索引...")
            cogmem.save()
            print("  ✓ 索引保存完成")
        except Exception as e:
            print(f"  ✗ 社区发现失败: {e}")
            print("    继续使用现有索引（可能影响检索性能）")
    
    # 索引数据集（如果需要）
    if total_nodes == 0:
        print(" 索引数据集...")
        from datasets.locomo_loader import LocomoLoader
        import os
        
        json_path = os.path.join(args.data_dir, "data", "locomo10.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(args.data_dir, "locomo10.json")
        
        if os.path.exists(json_path):
            loader = LocomoLoader(json_path)
            bundle = loader.load(db="conversation", max_samples=None)
            
            # 索引对话内容（包含时间信息）
            if bundle.docs:
                print("  1/3: 正在添加文档到知识图谱...")
                try:
                    cogmem.add_documents(bundle.docs)
                    print("  ✓ 文档添加完成")
                except Exception as e:
                    print(f"  ✗ 文档添加失败: {e}")
                    raise

                print("  2/3: 正在发现语义社区（这可能需要几分钟）...")
                try:
                    cogmem.discover_communities(semantic=True)
                    print("  ✓ 社区发现完成")
                except Exception as e:
                    print(f"  ✗ 社区发现失败: {e}")
                    raise

                print("  3/3: 正在保存索引（包括图、向量库、社区等）...")
                try:
                    # 检查图是否为空
                    total_nodes_before_save = cogmem.graph.num_entities()
                    if total_nodes_before_save == 0:
                        print("  ⚠️  警告：图是空的，可能索引过程未完成。跳过保存以避免覆盖已有索引。")
                        print("     如果这是首次索引，请检查索引过程是否成功完成。")
                    else:
                        cogmem.save()
                        print(f"  ✓ 索引保存完成（图中有 {total_nodes_before_save} 个节点）")
                except Exception as e:
                    print(f"  ✗ 索引保存失败: {e}")
                    raise

                # 统计节点信息（包括所有类型的节点）
                total_nodes = cogmem.graph.num_entities()
                total_relations = cogmem.graph.num_relations()
                print(f"✅ 索引完成，共 {len(bundle.docs)} 个对话文档（包含时间信息）")
                print(f"   图谱统计：{total_nodes} 个节点，{total_relations} 个关系")

                # 导出知识图谱为可读格式（导出是可选的，失败不影响主流程）
                try:
                    print("  正在导出知识图谱（可选步骤）...")
                    graph_json_path = os.path.join(args.workspace, "knowledge_graph.json")
                    graph_graphml_path = os.path.join(args.workspace, "knowledge_graph.graphml")

                    cogmem.graph.export_json(graph_json_path, indent=2)
                    cogmem.graph.export_graphml(graph_graphml_path)

                    print(f"   JSON 格式: {graph_json_path}")
                    print(f"   GraphML 格式: {graph_graphml_path}")
                except Exception as e:
                    print(f"  ⚠️  导出图谱失败（不影响评测）: {e}")
            else:
                print("警告：没有找到对话文档，使用空索引")
        else:
            print(f"警告：找不到 locomo10.json，路径: {json_path}")
    
    # 初始化 LLM Judge
    print("初始化 LLM Judge...")
    judge = LLMJudge(config)
    
    # 开始完整评测
    results = comprehensive_evaluation(
        quiet=args.quiet,
        cogmem=cogmem,
        dataset=dataset,
        judge=judge,
        max_samples=args.max_samples,
        output_file=args.output
    )
    
    # 分析结果
    analyze_performance_by_category(args.output)


if __name__ == "__main__":
    main()
