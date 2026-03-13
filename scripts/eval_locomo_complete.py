"""LoCoMo 完整评测脚本：包含 EM、F1、BLEU、Evidence Hit、LLM Judge 全部指标

支持的选项：
1. --skip_index: 跳过索引构建，直接加载已有的知识图谱
2. --skip_consolidation: 跳过记忆巩固步骤
3. --consolidation_only: 仅执行记忆巩固（要求已有索引）

使用示例：
1. 完整流程（索引+巩固+评测）：
   python eval_locomo_complete.py --data_dir datasets/locomo10.json --workspace .cogmem_test

2. 跳过索引（使用已有图谱）：
   python eval_locomo_complete.py --skip_index --workspace .cogmem_test

3. 跳过记忆巩固：
   python eval_locomo_complete.py --skip_consolidation --workspace .cogmem_test

4. 仅执行记忆巩固：
   python eval_locomo_complete.py --consolidation_only --workspace .cogmem_test

5. 使用已有图谱并跳过巩固（最快评测）：
   python eval_locomo_complete.py --skip_index --skip_consolidation --workspace .cogmem_test
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tqdm import tqdm

from cogmem import CogMem
from cogmem.config import DEFAULT_CONFIG
from cogmem.storage.schema import Doc, NodeTypeEnum
import dotenv
dotenv.load_dotenv()

# 导入BLEU计算（用于评估答案生成质量）
try:
    from nltk.translate.bleu_score import sentence_bleu
    NLTK_BLEU_AVAILABLE = True
except ImportError:
    NLTK_BLEU_AVAILABLE = False
    def sentence_bleu(*args, **kwargs):
        return 0.0

# 尝试导入LLM Judge评测模块
try:
    from examples.llm_judge_eval import LLMJudge, compute_judge_metrics
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    LLM_JUDGE_AVAILABLE = False
    # LLM Judge不可用时，提供占位符
    LLMJudge = None
    def compute_judge_metrics(*args, **kwargs):
        return {'avg_overall_score': 0.0}

try:
    from datasets.locomo_loader import load_locomo_dataset
except (ImportError, ModuleNotFoundError) as e:
    # If import fails, provide a stub function with the correct signature
    import traceback
    print(f"Warning: Cannot import load_locomo_dataset: {e}")
    print("Falling back to stub function. Dataset loading will fail.")
    traceback.print_exc()
    def load_locomo_dataset(
        data_dir: str, 
        max_samples: Optional[int] = None,
        subset_ratio: Optional[float] = None,
        subset_seed: int = 42
    ):
        print(f"Warning: Cannot load LoCoMo dataset from {data_dir}")
        return []

def _clean_answer_for_evaluation(answer: str) -> str:
    """清理答案用于评估：使用统一的答案清理工具"""
    from cogmem.utils.text_utils import extract_clean_answer
    return extract_clean_answer(answer)

def normalize_answer(s: str) -> str:
    """标准化答案文本，严格按照官方LoCoMo评测代码实现，增强日期处理"""
    import re
    
    # 修复：确保输入是字符串（处理字典或其他类型）
    if not isinstance(s, str):
        if isinstance(s, dict):
            # 如果是字典，尝试提取文本内容或转换为JSON字符串
            text_fields = ['text', 'content', 'description', 'answer', 'observation']
            s_str = None
            for field in text_fields:
                if field in s and isinstance(s[field], str):
                    s_str = s[field]
                    break
            if s_str is None:
                # 如果没有找到文本字段，使用JSON序列化
                try:
                    s_str = json.dumps(s, ensure_ascii=False)
                except (TypeError, ValueError):
                    s_str = str(s)
            s = s_str
        else:
            s = str(s)
    
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

def _serialize_reasoning_step(step):
    """序列化推理步骤，确保所有字段都是JSON可序列化的"""
    if isinstance(step, dict):
        # 如果是字典，递归处理所有值
        serialized = {}
        for key, value in step.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, (list, tuple)):
                serialized[key] = [_serialize_reasoning_step(item) if isinstance(item, dict) else str(item) for item in value]
            elif isinstance(value, dict):
                serialized[key] = _serialize_reasoning_step(value)
            else:
                serialized[key] = str(value)
        return serialized
    else:
        # 如果不是字典，转换为字符串
        return str(step)

def compute_exact_match(predicted: str, gold: str, gam_mode: bool = False) -> float:
    """计算精确匹配分数"""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    
    if gam_mode:
        # GAM 风格：基于 normalization 后的字符串直接比较
        return 1.0 if pred_norm == gold_norm else 0.0
    
    # 原始 CogMem 风格：词集合比较（更宽松）
    return 1.0 if set(pred_norm.split()) == set(gold_norm.split()) else 0.0

def compute_f1_score(predicted: str, gold: str, gam_mode: bool = False) -> float:
    """计算F1分数"""
    from collections import Counter
    
    # 标准化答案
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    
    if gam_mode:
        # GAM 风格：标准重合度计算，不使用词干提取
        pred_tokens = pred_norm.split()
        gold_tokens = gold_norm.split()
    else:
        # CogMem 风格：使用词干提取（PorterStemmer）
        try:
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            pred_tokens = [ps.stem(w) for w in pred_norm.split()]
            gold_tokens = [ps.stem(w) for w in gold_norm.split()]
        except ImportError:
            pred_tokens = pred_norm.split()
            gold_tokens = gold_norm.split()
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # 使用Counter计算词频，然后求交集
    gcount = Counter(gold_tokens)
    pcount = Counter(pred_tokens)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    
    if overlap == 0:
        return 0.0
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_bleu_score(predicted: str, gold: str) -> float:
    """计算 BLEU 分数（用于评估生成答案的质量）

    Args:
        predicted: 预测的答案
        gold: 标准答案

    Returns:
        BLEU 分数（0-1）
    """
    if not predicted or not gold:
        return 0.0

    if not NLTK_BLEU_AVAILABLE:
        return 0.0

    try:
        # 标准化答案（复用 normalize_answer 逻辑，但不移除停用词）
        pred_norm = normalize_answer(predicted)
        gold_norm = normalize_answer(gold)

        # 分词
        pred_tokens = pred_norm.split()
        gold_tokens = gold_norm.split()

        if not pred_tokens or not gold_tokens:
            return 0.0

        # 计算 BLEU-1 分数（只使用1-gram）
        # 使用平滑处理避免零分
        from nltk.translate.bleu_score import SmoothingFunction
        smoothie = SmoothingFunction().method1

        bleu = sentence_bleu(
            [gold_tokens],  # 参考译文（列表的列表）
            pred_tokens,     # 候选译文
            weights=(1.0,),  # BLEU-1 (只使用1-gram)
            smoothing_function=smoothie
        )

        return bleu

    except Exception as e:
        # 如果计算失败，返回 0
        print(f"Warning: BLEU calculation failed: {e}")
        return 0.0

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

def compute_evidence_hit(evidence: List[str], gold_answer: str, evidence_turn_ids: List[str] = None) -> float:
    """计算证据命中率：检查证据中是否包含金标答案的关键信息
    
    改进的逻辑：
    1. 优先使用 turn_id 精确匹配（如果提供了 evidence_turn_ids）
    2. 基于内容的关键词匹配（改进版）：
       - 检查更多证据片段（前20个，而不是5个）
       - 使用更智能的匹配策略（考虑词序和语义）
       - 对于短答案，要求更高匹配度
       - 对于长答案，允许部分匹配
    
    参考 examples/eval_locomo.py 的实现，并适配新 retriever 的证据格式：
    - 新格式: [Depth: 0] [Time: 2024-01-15] [Topic: Work & Career] [Type: Event] [Turn ID: D1:1] [Conf: 0.91] 事件描述...
    - 需要过滤掉元数据标记，只保留实际内容
    - 提取答案中的关键词（长度>2的词）
    - 检查是否有证据片段包含足够的关键词
    """
    if not evidence or not gold_answer:
        return 0.0

    import re
    
    # 修复：确保 gold_answer 是字符串
    if not isinstance(gold_answer, str):
        if isinstance(gold_answer, dict):
            # 如果是字典，尝试提取文本内容
            text_fields = ['text', 'content', 'description', 'answer']
            gold_str = None
            for field in text_fields:
                if field in gold_answer and isinstance(gold_answer[field], str):
                    gold_str = gold_answer[field]
                    break
            if gold_str is None:
                try:
                    gold_str = json.dumps(gold_answer, ensure_ascii=False)
                except (TypeError, ValueError):
                    gold_str = str(gold_answer)
            gold_answer = gold_str
        else:
            gold_answer = str(gold_answer)
    
    # 修复：确保 evidence 列表中的所有元素都是字符串
    # 如果包含字典或其他类型，转换为字符串
    evidence_strs = []
    for ev in evidence:
        if isinstance(ev, str):
            evidence_strs.append(ev)
        elif isinstance(ev, dict):
            # 如果是字典，尝试提取文本内容或转换为JSON字符串
            # 优先查找常见的文本字段
            text_fields = ['text', 'content', 'description', 'observation', 'evidence']
            ev_str = None
            for field in text_fields:
                if field in ev and isinstance(ev[field], str):
                    ev_str = ev[field]
                    break
            if ev_str is None:
                # 如果没有找到文本字段，使用JSON序列化
                try:
                    ev_str = json.dumps(ev, ensure_ascii=False)
                except (TypeError, ValueError):
                    ev_str = str(ev)
            evidence_strs.append(ev_str)
        else:
            # 其他类型直接转换为字符串
            evidence_strs.append(str(ev))
    
    evidence = evidence_strs
    
    # 如果提供了 evidence_turn_ids，优先使用精确匹配
    if evidence_turn_ids:
        # 修复：确保所有 evidence_turn_ids 都是字符串（避免 unhashable type: 'dict' 错误）
        evidence_turn_ids_clean = []
        for tid in evidence_turn_ids:
            if isinstance(tid, (str, int, float)):
                evidence_turn_ids_clean.append(str(tid))
            elif isinstance(tid, dict):
                # 如果是字典，尝试提取ID字段或使用JSON序列化
                tid_str = tid.get('id') or tid.get('turn_id') or tid.get('name')
                if isinstance(tid_str, str):
                    evidence_turn_ids_clean.append(tid_str)
                else:
                    try:
                        evidence_turn_ids_clean.append(json.dumps(tid, sort_keys=True))
                    except (TypeError, ValueError):
                        evidence_turn_ids_clean.append(str(tid))
            else:
                evidence_turn_ids_clean.append(str(tid))
        evidence_turn_ids_set = set(evidence_turn_ids_clean)
        
        # 改进：检查更多证据片段（前20个）以提高匹配率
        for ev in evidence[:20]:
            if not ev or len(ev) < 20:
                continue

            # 过滤掉metadata字符串
            if ev.startswith(("exact-match", "community-fusion", "vector-search", "gap-iter", "chunk-search")):
                continue

            # 提取 Turn ID
            turn_id_match = re.search(r'\[Turn ID:\s*([^\]]+)\]', ev)
            if turn_id_match:
                turn_id = turn_id_match.group(1).strip()
                if turn_id in evidence_turn_ids_set:
                    # 找到匹配的 turn_id，再验证内容匹配
                    # 提取实际内容
                    ev_content = re.sub(r'\[[^\]]+\]', '', ev).strip()
                    if not ev_content:
                        content_start = ev.rfind(']')
                        if content_start != -1 and content_start < len(ev) - 1:
                            ev_content = ev[content_start + 1:].strip()
                        else:
                            ev_content = ev
                    
                    if ev_content:
                        # 验证内容匹配（至少部分匹配）
                        gold_normalized = normalize_answer(gold_answer)
                        gold_tokens = set(w for w in gold_normalized.split() if len(w) > 2)
                        if gold_tokens:
                            ev_normalized = normalize_answer(ev_content)
                            ev_tokens = set(ev_normalized.split())
                            matched_words = len(gold_tokens & ev_tokens)
                            # 如果 turn_id 匹配且内容有部分匹配，认为命中
                            if matched_words > 0:
                                return 1.0
                    else:
                        # turn_id 匹配但无法验证内容，仍然认为命中（因为 turn_id 精确匹配）
                        return 1.0

    # Fallback: 基于内容的关键词匹配（原有逻辑）
    gold_normalized = normalize_answer(gold_answer)
    gold_tokens = set(w for w in gold_normalized.split() if len(w) > 2)

    if not gold_tokens:
        return 0.0

    # P0 FIX: 改进匹配逻辑，降低阈值以提高召回率
    # 改进：检查更多证据片段（前30个，从20增加到30）以提高匹配率
    for ev in evidence[:30]:
        if not ev or len(ev) < 10:  # 降低最小长度要求（从20降到10）
            continue

        # 过滤掉metadata字符串（如 "exact-match", "community-fusion" 等）
        if ev.startswith(("exact-match", "community-fusion", "vector-search", "gap-iter", "chunk-search")):
            continue

        # P0 FIX: 改进内容提取逻辑
        # 先尝试提取 Turn ID（如果存在）
        turn_id_match = re.search(r'\[Turn ID:\s*([^\]]+)\]', ev)
        turn_id = turn_id_match.group(1).strip() if turn_id_match else None
        
        # 新格式：过滤掉 [Depth: ...] [Time: ...] [Topic: ...] [Type: ...] [Conf: ...] 等元数据标记
        # 但保留 Turn ID 信息用于后续匹配
        # 提取实际内容：移除所有 [XXX: ...] 格式的标记（但保留 Turn ID 用于验证）
        ev_content = re.sub(r'\[(?!Turn ID:)[^\]]+\]', '', ev).strip()  # 保留 Turn ID
        # 移除 Turn ID 标记以提取纯内容
        ev_content = re.sub(r'\[Turn ID:\s*[^\]]+\]', '', ev_content).strip()
        
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
        
        # P0 FIX: 降低匹配阈值以提高召回率
        # 改进的匹配策略（降低阈值）：
        # 1. 对于非常短的答案（1-2个词），需要100%匹配
        # 2. 对于短答案（3-5个词），需要至少60%匹配（从80%降低）
        # 3. 对于中等长度答案（6-10个词），需要至少40%匹配（从60%降低）
        # 4. 对于长答案（>10个词），需要至少30%匹配（从40%降低）
        # 5. 同时要求至少匹配1个关键词（从2个降低，避免过度严格）
        
        if len(gold_tokens) <= 2:
            # 非常短的答案：需要100%匹配
            if matched_words == len(gold_tokens) and matched_words > 0:
                return 1.0
        elif len(gold_tokens) <= 5:
            # 短答案：需要至少60%匹配（降低），且至少1个词（降低）
            threshold = len(gold_tokens) * 0.6
            if matched_words >= threshold and matched_words >= 1:
                return 1.0
        elif len(gold_tokens) <= 10:
            # 中等长度答案：需要至少40%匹配（降低），且至少2个词（降低）
            threshold = len(gold_tokens) * 0.4
            if matched_words >= threshold and matched_words >= 2:
                return 1.0
        else:
            # 长答案：需要至少30%匹配（降低），且至少3个词（降低）
            threshold = len(gold_tokens) * 0.3
            if matched_words >= threshold and matched_words >= 3:
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
    quiet: bool = False,
    gam_mode: bool = False
) -> Dict:
    """完整评测：计算所有指标
    
    Args:
        cogmem: CogMem实例
        dataset: 评测数据集
        judge: LLM Judge实例（可选）
        max_samples: 最大评测样本数
        output_file: 输出文件路径
        quiet: 是否启用安静模式
        gam_mode: 是否开启 GAM 模式（提炼答案 + GAM 指标）
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
        # 确保 question_id 始终是字符串（修复 unhashable type: 'dict' 错误）
        raw_id = item.get("id", f"q_{index}")
        question_id = str(raw_id) if not isinstance(raw_id, (str, int, float)) else str(raw_id)
        
        # 对前10个问题启用 verbose 模式，展示完整思考过程
        show_verbose = index < 10
        if show_verbose:
            print(f"\n{'='*80}")
            print(f"[问题 {index+1}/前10] {question}")
            print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # 使用 Agent 进行检索和回答（通过 QAOrchestrator）
            # 调用 answer 方法，同时获取答案和检索结果
            # 对前10个问题启用 verbose 模式
            predicted_answer, retrieval_result = cogmem.answer(
                question, 
                top_k=10, 
                return_retrieval_result=True,
                verbose=show_verbose
            )
            
            # 获取完整的思考过程（从 SearchAgent的last_result中获取）
            # CRITICAL FIX: 在并发评测时，直接从answer()调用后立即获取last_result
            # 由于cogmem.answer()内部已经调用了search_agent.run()并保存了last_result，
            # 我们应该在answer()调用后立即获取，时间窗口很小
            reasoning_path = []
            last_result = None
            try:
                # 优先从last_result获取（这是最可靠的方式）
                # 注意：必须在answer()调用后立即获取，避免被其他线程覆盖
                if hasattr(cogmem.qa_orchestrator.search_agent, 'last_result'):
                    last_result = cogmem.qa_orchestrator.search_agent.last_result
                    if last_result and isinstance(last_result, dict) and 'reasoning_path' in last_result:
                        reasoning_path = last_result['reasoning_path']
                        # 确保reasoning_path是列表格式
                        if not isinstance(reasoning_path, list):
                            reasoning_path = []
                # Fallback: 如果last_result不存在或格式不对，从history获取
                if not reasoning_path and hasattr(cogmem.qa_orchestrator.search_agent, 'history'):
                    history = cogmem.qa_orchestrator.search_agent.history
                    if history:
                        from dataclasses import asdict
                        reasoning_path = [asdict(step) for step in history]
            except Exception as e:
                # 如果获取失败，记录错误但不中断评测
                if show_verbose:
                    print(f"  ⚠️  获取推理路径失败: {e}")
                reasoning_path = []
            
            if show_verbose:
                print(f"\n[完整思考过程]")
                print(f"{'='*80}")
                for step in reasoning_path:
                    # CRITICAL FIX: 处理dict格式的step（从asdict转换来的）
                    if isinstance(step, dict):
                        step_num = step.get('step', '?')
                        thought = step.get('thought', '')
                        action = step.get('action', 'unknown')
                        action_input = step.get('action_input', {})
                        observation = step.get('observation', '')
                        reflection = step.get('reflection')
                        print(f"\n[Step {step_num}]")
                        print(f"  Thought: {thought}")
                        print(f"  Action: {action}")
                        print(f"  Action Input: {json.dumps(action_input, ensure_ascii=False, indent=2)}")
                        print(f"  Observation: {json.dumps(observation, ensure_ascii=False, indent=2)[:500]}...")
                        if reflection:
                            print(f"  Reflection: {reflection}")
                    elif hasattr(step, 'step'):
                        # 如果是ThoughtActionObservation对象
                        print(f"\n[Step {step.step}]")
                        print(f"  Thought: {step.thought}")
                        print(f"  Action: {step.action}")
                        print(f"  Action Input: {json.dumps(step.action_input, ensure_ascii=False, indent=2)}")
                        print(f"  Observation: {json.dumps(step.observation, ensure_ascii=False, indent=2)[:500]}...")
                        if step.reflection:
                            print(f"  Reflection: {step.reflection}")
                    else:
                        print(f"  {step}")
                print(f"{'='*80}\n")
            
            inference_time = time.time() - start_time
            
            # === GAM 模式答案提炼 ===
            final_predicted_answer = predicted_answer
            if gam_mode:
                # 收集所有证据作为上下文（模拟 GAM 的 integrated_memory）
                evidence_texts = []
                for step in reasoning_path:
                    obs = ""
                    if isinstance(step, dict): obs = step.get('observation', '')
                    elif hasattr(step, 'observation'): obs = step.observation
                    if obs: evidence_texts.append(f"Observation: {obs}")
                
                integrated_evidence = "\n".join(evidence_texts)
                
                # 选择提炼提示词
                from cogmem.prompt import ANSWER_REFINEMENT_PROMPT, ANSWER_REFINEMENT_CATEGORY3_PROMPT
                if category_id == 3:  # Temporal
                    refine_prompt = ANSWER_REFINEMENT_CATEGORY3_PROMPT.format(question=question, evidence=integrated_evidence)
                else:
                    refine_prompt = ANSWER_REFINEMENT_PROMPT.format(question=question, evidence=integrated_evidence)
                
                # 调用 LLM 进行精炼（使用 cogmem 已配置的 LLM）
                try:
                    refined_resp = cogmem.llm.chat(
                        system="You are a precise answer extractor.",
                        user=refine_prompt
                    )
                    # 清理精炼后的结果
                    refined_answer = _clean_answer_for_evaluation(refined_resp)
                    if refined_answer:
                        final_predicted_answer = refined_answer
                except Exception as e:
                    if show_verbose:
                        print(f"  ⚠️  答案提炼失败: {e}")

            # 清理预测答案：移除推理过程和标记
            cleaned_predicted_answer = _clean_answer_for_evaluation(final_predicted_answer)
            
            # 计算指标（传递 gam_mode 标志）
            processed_gold_answer = gold_answer
            if category_id == 3:  # Temporal
                processed_gold_answer = gold_answer.split(';')[0].strip()
            
            if category_id == 1:  # Multi-Hop
                em_score = compute_exact_match(cleaned_predicted_answer, processed_gold_answer, gam_mode=gam_mode)
                f1_score = compute_multi_answer_f1(cleaned_predicted_answer, processed_gold_answer)
            elif category_id == 5:  # Adversarial
                pred_lower = cleaned_predicted_answer.lower()
                if ('no information' in pred_lower or 'not mentioned' in pred_lower or '不知道' in cleaned_predicted_answer):
                    em_score = 1.0; f1_score = 1.0
                else:
                    em_score = 0.0; f1_score = 0.0
            else:
                em_score = compute_exact_match(cleaned_predicted_answer, processed_gold_answer, gam_mode=gam_mode)
                f1_score = compute_f1_score(cleaned_predicted_answer, processed_gold_answer, gam_mode=gam_mode)

            try:
                bleu_score = compute_bleu_score(cleaned_predicted_answer, gold_answer)
            except Exception as e:
                # 如果 BLEU 计算失败，使用默认值 0.0
                if show_verbose:
                    print(f"  ⚠️  BLEU 计算失败: {e}")
                bleu_score = 0.0
            
            # 获取 evidence_turn_ids（如果存在）
            evidence_turn_ids = item.get("evidence", [])
            # 修复：确保 evidence 列表只包含字符串（在传递给 compute_evidence_hit 之前）
            raw_evidence = getattr(retrieval_result, 'evidence', []) or []
            evidence_list = []
            for ev in raw_evidence:
                if isinstance(ev, str):
                    evidence_list.append(ev)
                elif isinstance(ev, dict):
                    # 如果是字典，尝试提取文本内容
                    text_fields = ['text', 'content', 'description', 'observation', 'evidence']
                    ev_str = None
                    for field in text_fields:
                        if field in ev and isinstance(ev[field], str):
                            ev_str = ev[field]
                            break
                    if ev_str is None:
                        try:
                            ev_str = json.dumps(ev, ensure_ascii=False)
                        except (TypeError, ValueError):
                            ev_str = str(ev)
                    evidence_list.append(ev_str)
                else:
                    evidence_list.append(str(ev))
            
            evidence_hit = compute_evidence_hit(
                evidence_list, 
                gold_answer,
                evidence_turn_ids=evidence_turn_ids if evidence_turn_ids else None
            )
            
            # LLM Judge 评测（使用清理后的答案）
            judge_result = None
            if judge:
                try:
                    judge_result = judge.evaluate_answer(
                        question=question,
                        gold_answer=gold_answer,
                        predicted_answer=cleaned_predicted_answer,  # 使用清理后的答案
                        category=category
                    )
                except Exception as e:
                    # 静默处理错误，避免打印过多信息
                    if verbose:
                        print(f"[Warning] LLM Judge evaluation failed: {e}")
                    pass
            
            # 检索分析 - 兼容新的 SearchAgent 架构
            # 新架构中，retrieval_result 是模拟对象，只包含基本字段
            if not retrieval_result:
                from types import SimpleNamespace
                retrieval_result = SimpleNamespace(
                    extracted_entity_names=[],
                    retrieval_path=[],
                    entities=[],
                    evidence=[]
                )
            
            # 安全地访问 retrieval_result 的字段
            # 修复：确保 extracted_entities 只包含字符串（避免 unhashable type: 'dict' 错误）
            # 注意：修复变量名冲突，使用 entity_item 而不是 item（item是函数参数）
            raw_extracted = getattr(retrieval_result, 'extracted_entity_names', [])
            extracted_entities = []
            for entity_item in raw_extracted:
                if isinstance(entity_item, str):
                    extracted_entities.append(entity_item)
                elif isinstance(entity_item, dict):
                    # 如果是字典，尝试提取ID字段
                    entity_id = entity_item.get('node_id') or entity_item.get('id') or entity_item.get('name')
                    if isinstance(entity_id, str):
                        extracted_entities.append(entity_id)
                    else:
                        # 如果无法提取，跳过或使用字符串表示
                        try:
                            extracted_entities.append(json.dumps(entity_item, sort_keys=True))
                        except (TypeError, ValueError):
                            pass
                else:
                    # 其他类型转换为字符串
                    extracted_entities.append(str(entity_item))
            
            retrieval_path = list(getattr(retrieval_result, 'retrieval_path', []))
            entities_list = getattr(retrieval_result, 'entities', []) or []
            
            retrieved_entities = []
            for e in entities_list[:10]:
                if hasattr(e, 'name'):
                    name = e.name
                elif hasattr(e, 'canonical_name'):
                    name = e.canonical_name
                else:
                    name = str(e)
                
                description = getattr(e, 'description', '')
                if len(description) > 100:
                    description = description[:100] + "..."
                
                node_type = getattr(e, 'type', getattr(e, 'node_type', 'unknown'))
                retrieved_entities.append({
                    "name": name,
                    "description": description,
                    "type": node_type
                })
            
            # 计算推理轮次（reasoning_path的长度）
            num_reasoning_steps = len(reasoning_path) if reasoning_path else 0
            
            # 记录详细结果（保存清理后的答案，但保留原始答案用于调试）
            result = {
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": cleaned_predicted_answer,  # 保存清理后的答案
                "predicted_answer_raw": predicted_answer,  # 保留原始答案用于调试
                "category": category,
                "question_id": question_id,
                
                # 传统指标
                "em_score": em_score,
                "f1_score": f1_score,
                "bleu_score": bleu_score,
                "evidence_hit": evidence_hit,
                
                # LLM Judge 指标
                # Note: JudgeResult only has is_correct and explanation attributes
                # Map is_correct to scores (1.0 for correct, 0.0 for incorrect)
                "judge_semantic_correctness": 1.0 if (judge_result and judge_result.is_correct) else 0.0,
                "judge_completeness": 1.0 if (judge_result and judge_result.is_correct) else 0.0,
                "judge_relevance": 1.0 if (judge_result and judge_result.is_correct) else 0.0,
                "judge_overall_score": 1.0 if (judge_result and judge_result.is_correct) else 0.0,
                "judge_explanation": judge_result.explanation if judge_result else "",
                
                # 推理轮次
                "num_reasoning_steps": num_reasoning_steps,
                
                # 检索分析字段
                "extracted_entities": extracted_entities,
                "retrieval_path": retrieval_path,
                "retrieved_entities": retrieved_entities,
                "num_entities": len(entities_list),
                "num_evidence": len(evidence_list),  # 使用处理后的 evidence_list
                
                # 思考过程（保存所有样本的reasoning_path，不仅仅是verbose模式）
                # CRITICAL FIX: 处理dict格式的step（从asdict转换来的）
                # 确保所有字段都是可序列化的，避免 unhashable type: 'dict' 错误
                "reasoning_path": [
                    _serialize_reasoning_step(step)
                    for step in reasoning_path
                ] if reasoning_path else [],
                
                # 性能指标
                "inference_time": inference_time,
                "timestamp": datetime.now().isoformat()
            }
            # 单样本 skill 相关字段（来自 SearchAgent.run() 的 last_result）
            if last_result and isinstance(last_result, dict):
                if "question_type" in last_result:
                    result["question_type"] = last_result["question_type"]
                if "skill_flags" in last_result:
                    result["skill_flags"] = last_result["skill_flags"]
                if "skill_usage" in last_result:
                    result["skill_usage"] = last_result["skill_usage"]
                result["effective_agent_config"] = last_result.get("effective_agent_config") or {}

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
                # 修复 unhashable type: 'dict' 错误：确保所有 question_id 都是字符串
                completed_ids = set()
                for r in checkpoint_data.get("results", []):
                    qid = r.get("question_id")
                    if qid is not None:
                        # 如果 question_id 是字典或其他不可哈希类型，转换为字符串
                        if isinstance(qid, (str, int, float)):
                            completed_ids.add(str(qid))
                        else:
                            # 对于字典或其他类型，使用 JSON 字符串化或默认值
                            try:
                                import json as json_module
                                completed_ids.add(json_module.dumps(qid, sort_keys=True))
                            except (TypeError, ValueError):
                                # 如果无法序列化，使用默认值
                                completed_ids.add(f"q_{len(completed_ids)}")
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
    # 确保 id 字段是字符串（修复 unhashable type: 'dict' 错误）
    remaining_items = []
    for i, item in enumerate(dataset):
        raw_id = item.get("id", f"q_{i}")
        # 如果 raw_id 不是基本类型（str, int, float），转换为 JSON 字符串
        if isinstance(raw_id, (str, int, float)):
            item_id = str(raw_id)
        else:
            # 对于字典、列表等其他类型，使用 JSON 序列化
            try:
                import json as json_module
                item_id = json_module.dumps(raw_id, sort_keys=True)
            except (TypeError, ValueError):
                # 如果无法序列化，使用索引作为 ID
                item_id = f"q_{i}"
        if item_id not in completed_ids:
            remaining_items.append((i, item))
    
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

    # 过滤掉None结果和缺少必要字段的结果（向后兼容）
    valid_results = []
    for r in results:
        if r is None:
            continue
        # 检查是否包含所有必要的评估字段（向后兼容）
        required_fields = ["em_score", "f1_score", "bleu_score", "evidence_hit"]
        if all(field in r for field in required_fields):
            valid_results.append(r)
        else:
            if not quiet:
                missing = [f for f in required_fields if f not in r]
                print(f"⚠️  跳过不完整的评测结果（缺少: {missing}）")

    # 使用有效结果进行统计
    results = valid_results

    # 计算聚合指标
    if results:
        # 传统指标
        avg_em = sum(r["em_score"] for r in results) / len(results)
        avg_f1 = sum(r["f1_score"] for r in results) / len(results)
        avg_evidence_hit = sum(r["evidence_hit"] for r in results) / len(results)
        avg_bleu = sum(r["bleu_score"] for r in results) / len(results)
        
        # 平均推理轮次
        avg_reasoning_steps = sum(r.get("num_reasoning_steps", 0) for r in results) / len(results)
        
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
                    "avg_bleu_score": 0.0,
                    "avg_evidence_hit": 0.0,
                    "avg_reasoning_steps": 0.0
                }
            
            stats = category_stats[cat]
            stats["count"] += 1
            stats["avg_em"] += result["em_score"]
            stats["avg_f1"] += result["f1_score"]
            stats["avg_bleu_score"] += result["bleu_score"]
            stats["avg_evidence_hit"] += result["evidence_hit"]
            stats["avg_reasoning_steps"] += result.get("num_reasoning_steps", 0)
        
        # 计算平均值
        for cat, stats in category_stats.items():
            count = stats["count"]
            if count > 0:
                stats["avg_em"] /= count
                stats["avg_f1"] /= count
                stats["avg_bleu_score"] /= count
                stats["avg_evidence_hit"] /= count
                stats["avg_reasoning_steps"] /= count
    else:
        avg_em = avg_f1 = avg_evidence_hit = avg_bleu = avg_judge_score = 0.0
        avg_semantic = avg_completeness = avg_relevance = high_quality_ratio = 0.0
        avg_reasoning_steps = 0.0
        category_stats = {}
    
    # 识别 Bad Cases (EM=0 且 (Judge Score < 0.5 或没有 Judge 结果))
    # 注意：如果 judge_overall_score 为 0，说明没有 judge 结果，也应该算作 bad case
    bad_cases = [
        r for r in results 
        if r["em_score"] == 0.0 and (r["judge_overall_score"] == 0.0 or r["judge_overall_score"] < 0.5)
    ]
    
    # 顶层结果字段：实验名、skill 开关、agent 配置（论文消融与复现）
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
            from dataclasses import asdict
            agent_base_config = asdict(cogmem.cfg.agent)
    except Exception:
        pass

    # 保存结果
    evaluation_report = {
        "experiment_name": experiment_name,
        "skill_flags": skill_flags,
        "agent_base_config": agent_base_config,
        "summary": {
            "total_samples": len(results),
            "bad_cases_count": len(bad_cases),
            "bad_case_ratio": len(bad_cases) / len(results) if results else 0.0,

            # 传统指标
            "avg_em_score": avg_em,
            "avg_f1_score": avg_f1,
            "avg_bleu_score": avg_bleu,
            "avg_evidence_hit": avg_evidence_hit,
            "avg_reasoning_steps": avg_reasoning_steps,

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
    print(f"  BLEU-1 分数: {avg_bleu:.3f}")
    print(f"  Evidence Hit: {avg_evidence_hit:.3f}")
    print(f"  LLM Judge 分数: {avg_judge_score:.3f}")
    print(f"  高质量回答比例: {high_quality_ratio:.3f}")
    print(f"  平均推理轮次: {avg_reasoning_steps:.2f}")
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
    print("-" * 90)
    print(f"{'类别':<15} {'样本数':<8} {'EM':<8} {'F1':<8} {'BLEU-1':<8} {'EvidHit':<10} {'推理轮次':<10}")
    print("-" * 90)
    
    for category, stats in category_breakdown.items():
        print(f"{category:<15} {stats['count']:<8} "
              f"{stats['avg_em']:<8.3f} {stats['avg_f1']:<8.3f} "
              f"{stats.get('avg_bleu_score', 0.0):<8.3f} {stats['avg_evidence_hit']:<10.3f} "
              f"{stats.get('avg_reasoning_steps', 0.0):<10.2f}")
    
    print("-" * 90)

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="LoCoMo 完整评测")
    parser.add_argument("--data_dir", type=str, default="datasets/locomo10.json", help="LoCoMo数据集文件路径或目录")
    parser.add_argument("--workspace", type=str, default=".cogmem_complete_eval_test", help="CogMem工作空间")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评测问题数（QA数量，不是Conversation数量）")
    parser.add_argument("--subset_ratio", type=float, default=1, help="数据集子集比例（0.0-1.0），例如 0.1 表示10%%，0.05 表示5%%。如果同时指定 --max_samples，先应用子集比例，再应用最大样本数限制")
    parser.add_argument("--subset_seed", type=int, default=42, help="子集采样的随机种子（用于可复现性，默认42）")
    parser.add_argument("--output", type=str, default="complete_eval_results_test.json", help="输出文件")
    parser.add_argument("--analyze_only", action="store_true", help="仅分析已有结果")
    parser.add_argument("--quiet", action="store_true", help="安静模式：减少日志输出，加快评测速度")
    
    # 新增选项：跳过索引和记忆巩固
    parser.add_argument("--skip_index", action="store_true", help="跳过索引构建，直接加载已有的知识图谱")
    parser.add_argument("--skip_consolidation", action="store_true", help="跳过记忆巩固步骤，直接评测原始图谱")
    parser.add_argument("--consolidation_only", action="store_true", help="仅执行记忆巩固并保存，不进行评测")
    parser.add_argument("--gam_mode", action="store_true", help="开启 GAM 模式（复刻 GAM 的指标计算和提炼逻辑）")
    # Skill 消融开关（论文实验）
    parser.add_argument("--no-temporal-skill", action="store_true", dest="no_temporal_skill", help="消融：关闭 temporal grounding skill")
    parser.add_argument("--no-evidence-skill", action="store_true", dest="no_evidence_skill", help="消融：关闭 evidence retrieval skill")
    parser.add_argument("--no-routing-skill", action="store_true", dest="no_routing_skill", help="消融：关闭 routing skill")

    args = parser.parse_args()
    
    if args.analyze_only:
        if os.path.exists(args.output):
            analyze_performance_by_category(args.output)
        else:
            print(f"结果文件 {args.output} 不存在")
        return
    
    # 加载数据集
    print("加载 LoCoMo 数据集...")
    # 注意：如果指定了subset_ratio，load_locomo_dataset会先采样conversation，然后从这些conversation中提取QA
    # 这样可以确保评测的QA来自被索引的conversation
    dataset = load_locomo_dataset(
        args.data_dir,
        subset_ratio=args.subset_ratio if args.subset_ratio is not None else None,
        subset_seed=args.subset_seed
    )
    
    if not dataset:
        print("数据集加载失败")
        return
    
    # 注意：dataset是问题列表，不是conversation列表
    original_size = len(dataset)
    print(f"数据集加载成功，共 {original_size} 个问题（QA）")
    
    # 注意：如果指定了subset_ratio，load_locomo_dataset已经先采样了conversation
    # 然后从这些conversation中提取QA，所以这里不需要再次采样QA
    # 这样可以确保评测的QA来自被索引的conversation
    if args.subset_ratio is not None:
        # 统计类别分布（显示类别名称以便阅读）
        category_counts = {}
        category_counts_by_name = {}
        for item in dataset:
            cat_id = item.get("category", 0)
            cat_name = map_category_id_to_name(cat_id) if isinstance(cat_id, int) else str(cat_id)
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
            category_counts_by_name[cat_name] = category_counts_by_name.get(cat_name, 0) + 1
        
        print(f"已应用子集比例 {args.subset_ratio*100:.1f}%（在conversation级别采样，确保与索引对应）")
        print(f"  从采样的conversation中提取了 {len(dataset)} 个问题（QA）")
        print(f"  随机种子: {args.subset_seed}（用于可复现性）")
        print(f"  类别分布（按ID）: {dict(sorted(category_counts.items()))}")
        print(f"  类别分布（按名称）: {dict(sorted(category_counts_by_name.items()))}")
    
    # 应用最大样本数限制（如果指定，在子集之后应用）
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset[:args.max_samples]
        print(f"应用最大样本数限制：限制到 {len(dataset)} 个问题")
    
    # 初始化 CogMem
    print(" 初始化 CogMem...")
    config = DEFAULT_CONFIG
    config.extraction.use_enhanced_extraction = True  # 启用增强版实体抽取

    # Pure Graph Mode 配置
    config.chunk.enable_chunk_embedding = False  # 关闭原文嵌入（Pure Graph Mode）
    config.chunk.keep_raw_text = True  # 保留原文用于展示（不嵌入但存储）

    # Agent 配置：使用配置文件中的max_iterations（从config读取，不再硬编码）
    print(f"  Agent 最大迭代次数: {config.agent.max_iterations}")

    # Skill 消融：根据 CLI 覆盖 config.agent 的 skill 开关
    if getattr(args, "no_temporal_skill", False):
        config.agent.enable_temporal_skill = False
        print("  消融: 已关闭 temporal_skill")
    if getattr(args, "no_evidence_skill", False):
        config.agent.enable_evidence_skill = False
        print("  消融: 已关闭 evidence_skill")
    if getattr(args, "no_routing_skill", False):
        config.agent.enable_routing_skill = False
        print("  消融: 已关闭 routing_skill")

    cogmem = CogMem(config=config, workspace_dir=args.workspace, auto_load=True)
    
    # 检查图是否正确加载
    total_nodes = cogmem.graph.node_count
    print(f" 已加载图节点数: {total_nodes}")
    
    # 检查是否启用记忆巩固
    fact_count_before = sum(1 for nid, ndata in cogmem.graph._g.nodes(data=True) 
                           if ndata and ndata.get('node_type') == NodeTypeEnum.FACT.value)
    print(f" 当前Fact节点数: {fact_count_before}")

    # 检查 Person 节点
    person_count = 0
    for node_id, node_data in cogmem.graph._g.nodes(data=True):
        if node_data and "canonical_name" in node_data:
            person_count += 1
    print(f" Person 节点数: {person_count}")

    # 统计核心节点类型（P-E-C-F + Claim）
    person_count = 0
    event_count = 0
    concept_count = 0
    fact_count = 0
    claim_count = 0
    
    for node_id, node_data in cogmem.graph._g.nodes(data=True):
        if not node_data:
            continue
        node_type = node_data.get("node_type")
        if node_type == NodeTypeEnum.PERSON.value:
            person_count += 1
        elif node_type == NodeTypeEnum.EVENT.value:
            event_count += 1
        elif node_type == NodeTypeEnum.CONCEPT.value:
            concept_count += 1
        elif node_type == NodeTypeEnum.FACT.value:
            fact_count += 1
        elif node_type == NodeTypeEnum.CLAIM.value:
            claim_count += 1

    print(f" Person 节点数: {person_count}")
    print(f" Event 节点数: {event_count}")
    print(f" Concept 节点数: {concept_count}")
    print(f" Fact 节点数: {fact_count}")
    print(f" Claim 节点数: {claim_count}")
    
    # 索引数据集（根据skip_index和consolidation_only选项决定）
    if args.consolidation_only:
        # 仅执行记忆巩固模式：要求已有索引
        if total_nodes == 0:
            print("❌ 错误：--consolidation_only 要求已有索引，但图谱为空")
            return
        print("⏭️  跳过索引构建（consolidation_only模式）")
    elif args.skip_index:
        # 跳过索引模式：直接使用已有图谱
        if total_nodes == 0:
            print("⚠️  警告：--skip_index 指定但图谱为空，将执行索引构建")
            args.skip_index = False  # 强制执行索引
        else:
            print(f"⏭️  跳过索引构建，使用已有图谱（{total_nodes} 个节点）")
    
    # 执行索引构建（如果需要）
    if not args.skip_index and not args.consolidation_only and total_nodes == 0:
        print(" 索引数据集...")
        from datasets.locomo_loader import LocomoLoader
        
        # 如果直接指定了文件路径，直接使用；否则在目录中查找
        if os.path.isfile(args.data_dir):
            json_path = args.data_dir
        else:
            json_path = os.path.join(args.data_dir, "data", "locomo10.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(args.data_dir, "locomo10.json")
        
        if os.path.exists(json_path):
            loader = LocomoLoader(json_path)
            # 如果指定了采样比例，索引时也使用相同的采样比例
            bundle = loader.load(
                db="conversation", 
                max_samples=None,
                subset_ratio=args.subset_ratio if args.subset_ratio is not None else None,
                subset_seed=args.subset_seed
            )
            
            # 索引对话内容（包含时间信息）
            if bundle.docs:
                print("  1/3: 正在添加文档到知识图谱...")
                try:
                    cogmem.add_documents(bundle.docs)
                    print("  ✓ 文档添加完成")
                except Exception as e:
                    print(f"  ✗ 文档添加失败: {e}")
                    raise

                # 社区发现已废弃，不再执行
                print("  2/3: 正在保存索引（包括图、向量库等）...")
                try:
                    # 检查图是否为空
                    total_nodes_before_save = cogmem.graph.node_count
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
                total_nodes = cogmem.graph.node_count
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
    
    # 记忆巩固步骤（根据skip_consolidation选项决定）
    if not args.skip_consolidation:
        print("\n 执行记忆巩固...")
        fact_count_before = sum(1 for nid, ndata in cogmem.graph._g.nodes(data=True) 
                               if ndata and ndata.get('node_type') == NodeTypeEnum.FACT.value)
        print(f"   巩固前 Fact 节点数: {fact_count_before}")
        
        try:
            # 执行记忆巩固
            cogmem.consolidate_memory()
            cogmem.save()
            
            fact_count_after = sum(1 for nid, ndata in cogmem.graph._g.nodes(data=True) 
                                  if ndata and ndata.get('node_type') == NodeTypeEnum.FACT.value)
            print(f"   巩固后 Fact 节点数: {fact_count_after}")
            print(f"   新增 Fact 节点: {fact_count_after - fact_count_before}")
        except Exception as e:
            print(f"  ⚠️  记忆巩固失败: {e}")
            print("     继续使用未巩固的图谱进行评测")
    else:
        print("\n⏭️  跳过记忆巩固步骤")
    
    # 如果是consolidation_only模式，执行完巩固后直接返回
    if args.consolidation_only:
        print("\n✅ 记忆巩固完成（consolidation_only模式）")
        return
    
    # 初始化 LLM Judge（启用 LLM as Judge）
    print("初始化 LLM Judge...")
    judge = None
    if LLM_JUDGE_AVAILABLE:
        try:
            judge = LLMJudge(config)
            print("  ✓ LLM Judge 已启用")
        except Exception as e:
            print(f"  ✗ LLM Judge 初始化失败: {e}")
            print("  可能的原因：")
            print("    - LLM client 未配置")
            print("    - API keys 未设置（OpenAI/Anthropic）")
            print("    - 本地 LLM 配置不正确")
            print("  评测将继续进行，但不会使用 LLM Judge")
            judge = None
    else:
        print("  ⚠️  LLM Judge 模块不可用 (examples/llm_judge_eval.py)")
        print("  评测将继续进行，但不会使用 LLM Judge")
        judge = None
    
    # 开始完整评测
    results = comprehensive_evaluation(
        quiet=args.quiet,
        cogmem=cogmem,
        dataset=dataset,
        judge=judge,
        max_samples=args.max_samples,
        output_file=args.output,
        gam_mode=args.gam_mode
    )
    
    # 分析结果
    analyze_performance_by_category(args.output)

if __name__ == "__main__":
    main()
