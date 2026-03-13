"""
LoCoMo 数据集分类别详细评测脚本
支持加载已有索引，测试特定类别的问题，并记录详细的问答过程
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import dotenv

dotenv.load_dotenv()

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cogmem.config import CogMemConfig, DEFAULT_CONFIG
from cogmem.cogmem import CogMem
from datasets.locomo_loader import LocomoLoader
from examples.llm_judge_eval import LLMJudge

from collections import Counter
import math
import re
import string

def normalize_answer(s: str) -> str:
    """标准化答案文本，对齐 LoCoMo 官方标准"""
    if not isinstance(s, str):
        s = str(s)
    
    s = s.replace(',', "")  # 移除逗号

    def remove_articles(text):
        return re.sub(r'\b(a|an|the|and)\b', ' ', text, flags=re.IGNORECASE)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def extract_date_info(text):
        """处理日期：将 ISO 格式转为可读格式以匹配标准答案"""
        # 匹配标准 ISO 日期
        iso_pattern = r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})'
        matches = re.findall(iso_pattern, text)
        result = text
        month_names = ['', 'january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']
        
        for year, month, day, hour, minute, second in matches:
            month_name = month_names[int(month)] if 1 <= int(month) <= 12 else month
            date_str = f"{int(day)} {month_name} {year}"
            iso_date = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
            result = result.replace(iso_date, date_str)
        
        # 额外匹配普通日期 YYYY-MM-DD
        simple_iso = r'(\d{4})-(\d{2})-(\d{2})'
        simple_matches = re.findall(simple_iso, result)
        for year, month, day in simple_matches:
            if int(month) < 1 or int(month) > 12: continue
            month_name = month_names[int(month)]
            date_str = f"{int(day)} {month_name} {year}"
            iso_date = f"{year}-{month}-{day}"
            result = result.replace(iso_date, date_str)
            
        # 匹配 YYYY-MM
        month_iso = r'(\d{4})-(\d{2})'
        month_matches = re.findall(month_iso, result)
        for year, month in month_matches:
            if int(month) < 1 or int(month) > 12: continue
            month_name = month_names[int(month)]
            date_str = f"{month_name} {year}"
            iso_date = f"{year}-{month}"
            result = result.replace(iso_date, date_str)
            
        return result

    s = extract_date_info(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _clean_answer_for_evaluation(answer: str) -> str:
    """清理答案用于评估：使用 CogMem 统一的清理工具"""
    from cogmem.utils.text_utils import extract_clean_answer
    return extract_clean_answer(answer)

def calculate_em(predicted: str, gold: str, gam_mode: bool = False) -> float:
    """计算精确匹配分数"""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    
    if gam_mode:
        # GAM 模式：基于标准化后的字符串直接比较
        return 1.0 if pred_norm == gold_norm else 0.0
    
    # 原始模式：词集合比较
    return 1.0 if set(pred_norm.split()) == set(gold_norm.split()) else 0.0

def calculate_f1(predicted: str, gold: str, gam_mode: bool = False) -> float:
    """计算 F1 分数"""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    
    # 分词
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # 使用 Counter 计算重合度（GAM 风格）
    gcount = Counter(gold_tokens)
    pcount = Counter(pred_tokens)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    
    if overlap == 0:
        return 0.0
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def calculate_bleu(predicted: str, gold: str, gam_mode: bool = False) -> float:
    """计算 BLEU 分数（开启 gam_mode 时对齐 GAM 风格）"""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    gcount = Counter(gold_tokens)
    pcount = Counter(pred_tokens)
    clipped_overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped_overlap / len(pred_tokens)
    
    if gam_mode:
        # GAM 风格：带长度惩罚的 BLEU-1
        bp = 1.0 if len(pred_tokens) >= len(gold_tokens) else math.exp(1 - len(gold_tokens)/len(pred_tokens))
        return bp * precision
    
    return precision

def create_config(
    workspace_dir: str,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    use_vector_search: bool = True,
    max_iterations: int = 10
) -> CogMemConfig:
    """创建配置（默认沿用 DEFAULT_CONFIG，仅按需覆盖）"""
    config = DEFAULT_CONFIG
    config.persist_dir = workspace_dir

    if provider:
        config.model.provider = provider
        if provider == "adaptor_llm" and model_name:
            config.model.adaptor_llm_model = model_name
        elif provider == "siliconflow" and model_name:
            config.model.chat_model = model_name

    config.faiss.use_vector_search = use_vector_search
    config.agent.max_iterations = max_iterations

    return config

def load_category_questions(data_dir: str, category: str, max_samples: Optional[int] = None, subset_ratio: Optional[float] = None) -> List[Any]:
    """加载指定类别的问题"""
    loader = LocomoLoader(data_dir)
    bundle = loader.load(db="conversation", max_samples=None, subset_ratio=subset_ratio)
    
    if category == "All":
        category_qa = bundle.qa
        print(f"  找到 {len(category_qa)} 个所有类别的问题")
    else:
        # 筛选指定类别
        category_qa = [qa for qa in bundle.qa if qa.category == category]
        print(f"  找到 {len(category_qa)} 个 [{category}] 类别的问题")
    
    # 限制样本数
    if max_samples and len(category_qa) > max_samples:
        category_qa = category_qa[:max_samples]
        print(f"  限制为前 {max_samples} 个样本")
    
    return category_qa

def index_locomo(cogmem: CogMem, data_dir: str, max_samples: Optional[int] = None, subset_ratio: Optional[float] = None):
    """构建 LoCoMo 索引"""
    print(f"\n正在从 {data_dir} 构建知识图谱...")
    loader = LocomoLoader(data_dir)
    bundle = loader.load(db="conversation", max_samples=max_samples, subset_ratio=subset_ratio)
    
    if not bundle.docs:
        print("  [错误] 未提取到任何文档")
        return
    
    print(f"  提取到 {len(bundle.docs)} 个会话文档，准备索引...")
    
    # 索引对话内容
    print("  1/3: 正在添加文档到知识图谱...")
    try:
        cogmem.add_documents(bundle.docs)
        print("  ✓ 文档添加完成")
    except Exception as e:
        print(f"  ✗ 文档添加失败: {e}")
        raise

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

    # 统计节点信息
    total_nodes = cogmem.graph.node_count
    total_relations = cogmem.graph.num_relations()
    print(f"✅ 索引完成，共 {len(bundle.docs)} 个对话文档")
    print(f"   图谱统计：{total_nodes} 个节点，{total_relations} 个关系")

    # 导出知识图谱为可读格式（可选）
    try:
        print("  3/3: 正在导出知识图谱（可选步骤）...")
        workspace_dir = cogmem.workspace_dir
        graph_json_path = os.path.join(workspace_dir, "knowledge_graph.json")
        graph_graphml_path = os.path.join(workspace_dir, "knowledge_graph.graphml")

        cogmem.graph.export_json(graph_json_path, indent=2)
        cogmem.graph.export_graphml(graph_graphml_path)

        print(f"   JSON 格式: {graph_json_path}")
        print(f"   GraphML 格式: {graph_graphml_path}")
    except Exception as e:
        print(f"  ⚠️  导出图谱失败（不影响评测）: {e}")

def run_detailed_evaluation(
    cogmem: CogMem,
    qa_samples: List[Any],
    category: str,
    output_file: str,
    verbose: bool = False,
    gam_mode: bool = False
) -> Dict[str, Any]:
    """运行详细评测"""
    
    print(f"\n开始评测 [{category}] 类别，共 {len(qa_samples)} 个问题")
    if gam_mode:
        print(f"⚠️  [GAM Mode] 已开启：将执行答案精炼和 GAM 风格指标计算")
    print("="*100)
    
    results = []
    # 按类别统计
    category_stats = {} # category_name -> {em, f1, bleu, hit, steps, count}
    
    total_em = 0.0
    total_f1 = 0.0
    total_bleu = 0.0
    total_reasoning_steps = 0
    
    for idx, qa_sample in enumerate(qa_samples):
        question = qa_sample.question
        gold_answer = qa_sample.answer
        question_id = qa_sample.question_id if hasattr(qa_sample, 'question_id') else f"qa_{idx}"
        evidence_ids = qa_sample.evidence_turn_ids if hasattr(qa_sample, 'evidence_turn_ids') else []
        qa_category = qa_sample.category or "Unknown"
        
        if qa_category not in category_stats:
            category_stats[qa_category] = {"em": 0.0, "f1": 0.0, "bleu": 0.0, "steps": 0, "count": 0}
        
        # 获取类别 ID 用于 Prompt 选择
        category_id = 0
        if "Multi-hop" in qa_category: category_id = 1
        elif "Single Hop" in qa_category: category_id = 2
        elif "Temporal" in qa_category: category_id = 3
        elif "Open Domain" in qa_category: category_id = 4
        
        print(f"\n[{idx+1}/{len(qa_samples)}] [{qa_category}] ID: {question_id}")
        print(f"  Q: {question}")
        print(f"  Gold: {gold_answer}")
        
        start_time = time.time()
        
        try:
            # 执行问答，获取详细结果
            answer_result = cogmem.answer(question, return_full_result=True, verbose=verbose)
            
            predicted_answer = answer_result.get("answer", "I could not generate an answer.")
            reasoning_path = answer_result.get("reasoning_path", [])
            confidence = answer_result.get("confidence", "medium")
            
            # === GAM 模式提炼步骤 ===
            if gam_mode:
                # 收集证据（模拟 integrated_memory）
                evidence_texts = []
                for step in reasoning_path:
                    obs = ""
                    if isinstance(step, dict): obs = step.get("observation", "")
                    elif hasattr(step, "observation"): obs = step.observation
                    if obs: evidence_texts.append(f"Observation: {obs}")
                
                integrated_evidence = "\n".join(evidence_texts)
                
                # 选择精炼提示词
                from cogmem.prompt import ANSWER_REFINEMENT_PROMPT, ANSWER_REFINEMENT_CATEGORY3_PROMPT
                if category_id == 3:  # Temporal
                    refine_prompt = ANSWER_REFINEMENT_CATEGORY3_PROMPT.format(question=question, evidence=integrated_evidence)
                else:
                    refine_prompt = ANSWER_REFINEMENT_PROMPT.format(question=question, evidence=integrated_evidence)
                
                # 调用 LLM 精炼
                try:
                    refined_resp = cogmem.llm.chat(
                        system="You are a precise answer extractor.",
                        user=refine_prompt
                    )
                    # 清理并更新预测答案
                    refined_answer = _clean_answer_for_evaluation(refined_resp)
                    if refined_answer:
                        predicted_answer = refined_answer
                except Exception as e:
                    if verbose: print(f"  ⚠️  提炼失败: {e}")

            latency = time.time() - start_time
            
            # 清理预测答案用于评估
            cleaned_predicted_answer = _clean_answer_for_evaluation(predicted_answer)
            
            # 推理过程记录
            reasoning_details = []
            for step in reasoning_path:
                if isinstance(step, dict):
                    reasoning_details.append(step)
                else:
                    reasoning_details.append({"raw": str(step)})
            
            # 计算指标（传递 gam_mode）
            em_score = calculate_em(cleaned_predicted_answer, gold_answer, gam_mode=gam_mode)
            f1_score = calculate_f1(cleaned_predicted_answer, gold_answer, gam_mode=gam_mode)
            bleu_score = calculate_bleu(cleaned_predicted_answer, gold_answer, gam_mode=gam_mode)
            
            # 更新统计
            category_stats[qa_category]["em"] += em_score
            category_stats[qa_category]["f1"] += f1_score
            category_stats[qa_category]["bleu"] += bleu_score
            category_stats[qa_category]["steps"] += len(reasoning_path)
            category_stats[qa_category]["count"] += 1

            total_em += em_score
            total_f1 += f1_score
            total_bleu += bleu_score
            total_reasoning_steps += len(reasoning_path)
            
            result = {
                "question_id": question_id,
                "question": question,
                "ground_truth": gold_answer,
                "predicted_answer": predicted_answer,
                "category": qa_category,
                "evidence_ids": evidence_ids,
                "em_score": em_score,
                "f1_score": f1_score,
                "bleu_score": bleu_score,
                "confidence": confidence,
                "latency": latency,
                "reasoning_steps": len(reasoning_path),
                "reasoning_details": reasoning_details
            }
            # 单样本 skill 相关字段（与 eval_locomo_complete 一致）
            if isinstance(answer_result, dict):
                if "question_type" in answer_result:
                    result["question_type"] = answer_result["question_type"]
                if "skill_flags" in answer_result:
                    result["skill_flags"] = answer_result["skill_flags"]
                if "skill_usage" in answer_result:
                    result["skill_usage"] = answer_result["skill_usage"]
                result["effective_agent_config"] = answer_result.get("effective_agent_config") or {}

            results.append(result)
            
            print(f"  Predicted: {predicted_answer[:80]}...")
            print(f"  Metrics: EM={em_score:.0%}, F1={f1_score:.1%}, BLEU={bleu_score:.1%}")
            print(f"  Reasoning Steps: {len(reasoning_path)}, Latency: {latency:.2f}s")
            
        except Exception as e:
            print(f"  [错误] {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                "question_id": question_id,
                "question": question,
                "ground_truth": gold_answer,
                "predicted_answer": f"[ERROR] {str(e)}",
                "category": qa_category,
                "evidence_ids": evidence_ids,
                "em_score": 0.0,
                "f1_score": 0.0,
                "bleu_score": 0.0,
                "confidence": "error",
                "latency": time.time() - start_time,
                "reasoning_steps": 0,
                "reasoning_details": [],
                "error": str(e)
            }
            results.append(result)
    
    # 计算总体统计
    num_samples = len(results)
    
    # 提取性能指标
    from cogmem.utils.metrics import METRICS
    perf_snapshot = METRICS.snapshot()
    
    # 格式化分类统计
    category_summary = {}
    for cat, s in category_stats.items():
        n = s["count"]
        if n > 0:
            category_summary[cat] = {
                "count": n,
                "avg_em": s["em"] / n,
                "avg_f1": s["f1"] / n,
                "avg_bleu": s["bleu"] / n,
                "avg_steps": s["steps"] / n
            }

    summary = {
        "category": category,
        "total_samples": num_samples,
        "avg_em_score": total_em / num_samples if num_samples > 0 else 0,
        "avg_f1_score": total_f1 / num_samples if num_samples > 0 else 0,
        "avg_bleu_score": total_bleu / num_samples if num_samples > 0 else 0,
        "avg_reasoning_steps": total_reasoning_steps / num_samples if num_samples > 0 else 0,
        "category_summary": category_summary,
        "performance": perf_snapshot,
        "timestamp": datetime.now().isoformat()
    }
    
    # 顶层结果字段：实验名、skill 开关、agent 配置（与 eval_locomo_complete 一致）
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
    output_data = {
        "experiment_name": experiment_name,
        "skill_flags": skill_flags,
        "agent_base_config": agent_base_config,
        "summary": summary,
        "detailed_results": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "="*100)
    print(f"评测完成！")
    print(f"  总样本数: {num_samples}")
    print(f"  平均 EM: {summary['avg_em_score']:.2%}")
    print(f"  平均 F1: {summary['avg_f1_score']:.2%}")
    print(f"  平均 BLEU: {summary['avg_bleu_score']:.2%}")
    print(f"  平均推理步数: {summary['avg_reasoning_steps']:.1f}")
    
    if category_summary:
        print(f"\n分类统计:")
        print(f"{'Category':<20} | {'Count':<6} | {'EM':<6} | {'F1':<6} | {'BLEU':<6} | {'Steps':<6}")
        print("-" * 80)
        for cat, s in category_summary.items():
            print(f"{cat:<20} | {s['count']:<6} | {s['avg_em']:>6.2%} | {s['avg_f1']:>6.2%} | {s['avg_bleu']:>6.2%} | {s['avg_steps']:>6.1f}")

    # 打印性能统计
    print(f"\n性能统计 (Token & Latency):")
    chat_perf = perf_snapshot.get("chat", {})
    embed_perf = perf_snapshot.get("embed", {})
    
    if chat_perf.get('calls', 0) > 0:
        print(f"  [Chat]  调用数: {chat_perf.get('calls', 0)}, Token: {chat_perf.get('total_tokens', 0)}, "
              f"平均延迟: {chat_perf.get('total_latency_ms', 0)/(chat_perf.get('calls', 1)):.0f}ms")
    if embed_perf.get('calls', 0) > 0:
        print(f"  [Embed] 调用数: {embed_perf.get('calls', 0)}, Token: {embed_perf.get('total_tokens', 0)}, "
              f"平均延迟: {embed_perf.get('total_latency_ms', 0)/(embed_perf.get('calls', 1)):.0f}ms")
    
    print(f"\n结果已保存到: {output_file}")
    print("="*100)
    
    return output_data

def run_llm_judge_evaluation(
    results_file: str,
    judge_config: CogMemConfig,
    output_file: Optional[str] = None
) -> None:
    """使用 LLM Judge 进行二次评分"""
    
    print(f"\n使用 LLM Judge (provider={judge_config.model.provider}) 进行评分...")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    judge = LLMJudge(config=judge_config)
    
    detailed_results = data['detailed_results']
    judged_count = 0
    
    for idx, result in enumerate(detailed_results):
        if 'judge_is_correct' in result:
            print(f"  [{idx+1}/{len(detailed_results)}] Already judged, skipping")
            judged_count += 1
            continue
        
        if 'error' in result:
            print(f"  [{idx+1}/{len(detailed_results)}] Error sample, skipping")
            continue
        
        question = result['question']
        predicted = result['predicted_answer']
        ground_truth = result['ground_truth']
        
        print(f"  [{idx+1}/{len(detailed_results)}] Judging...")
        
        try:
            judge_result = judge.evaluate_answer(
                question=question,
                gold_answer=ground_truth,
                predicted_answer=predicted,
                category=result.get('category')
            )
            
            result['judge_is_correct'] = judge_result.is_correct
            result['judge_explanation'] = judge_result.explanation
            
            judged_count += 1
            
        except Exception as e:
            print(f"    Judging failed: {e}")
            result['judge_error'] = str(e)
    
    # Re-calculate statistics
    correct_count = 0
    valid_count = 0
    
    for result in detailed_results:
        if 'judge_is_correct' in result:
            if result['judge_is_correct']:
                correct_count += 1
            valid_count += 1
    
    if valid_count > 0:
        data['summary']['avg_judge_pass_rate'] = correct_count / valid_count
        data['summary']['judged_samples'] = valid_count
    
    # Save results
    if output_file is None:
        output_file = results_file.replace('.json', '_judged.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nJudging completed!")
    print(f"  Judged samples: {valid_count}/{len(detailed_results)}")
    if valid_count > 0:
        print(f"  Judge Pass Rate: {data['summary']['avg_judge_pass_rate']:.2%}")
    print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo 数据集分类别详细评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 测试 Single Hop 类别，限制 20 个样本
  python eval_locomo_category.py --category "Single Hop" --max-samples 20
  
  # 测试 Multi-Hop 类别，使用 Qwen 模型
  python eval_locomo_category.py --category "Multi-Hop" --provider siliconflow --model "Qwen/Qwen2.5-14B-Instruct"
  
  # 测试 Temporal 类别，并使用 LLM Judge 评分（使用相同的 LLM 配置）
  python eval_locomo_category.py --category Temporal --use-judge
  
  # 测试 Open Domain，显示详细推理过程
  python eval_locomo_category.py --category "Open Domain" --verbose --max-samples 5
  
  # 重建索引并测试（会删除旧索引）
  python eval_locomo_category.py --category "Single Hop" --rebuild-index
  
  # 跳过索引，直接使用已有图谱
  python eval_locomo_category.py --category "Multi-Hop" --skip-index
  
  # 跳过记忆巩固，使用原始图谱
  python eval_locomo_category.py --category Temporal --skip-consolidation
  
  # 仅执行记忆巩固，不评测
  python eval_locomo_category.py --consolidation-only
  
  # 查看支持的类别列表
  python eval_locomo_category.py --list-categories
        """
    )
    
    parser.add_argument(
        "--workspace",
        type=str,
        default=os.path.join("CogMem", "workspace", "kimi-locomo"),
        help="知识图谱工作区目录（已构建好的索引）"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/locomo10.json",
        help="LoCoMo 数据集路径"
    )
    
    parser.add_argument(
        "--category",
        type=str,
        choices=["Single Hop", "Multi-Hop", "Temporal", "Open Domain", "All"],
        default="All",
        help="要测试的问题类别（默认测试所有类别）"
    )
    
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="跳过构建索引，直接加载已有知识图谱"
    )
    
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重新构建索引（会删除 workspace 目录）"
    )
    
    parser.add_argument(
        "--skip-consolidation",
        action="store_true",
        help="跳过记忆巩固步骤，直接评测原始图谱"
    )
    
    parser.add_argument(
        "--consolidation-only",
        action="store_true",
        help="仅执行记忆巩固并保存，不进行评测"
    )
    
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=None,
        help="采样数据集比例 (0.0-1.0)，用于快速测试"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大测试样本数（默认测试该类别所有样本）"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["adaptor_llm", "siliconflow"],
        help="LLM 提供商（不传则使用默认配置）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称（不传则使用默认配置）"
    )
    
    parser.add_argument(
        "--use-vector-search",
        action="store_true",
        default=True,
        help="是否启用向量检索"
    )
    
    parser.add_argument(
        "--no-vector-search",
        action="store_true",
        help="禁用向量检索"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Agent 最大迭代次数"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认为 eval_{category}_{timestamp}.json）"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细的推理过程"
    )
    
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="使用 LLM Judge 进行二次评分（使用与主评测相同的 LLM 配置）"
    )
    
    parser.add_argument(
        "--gam-mode",
        default=False,
        action="store_true",
        help="启用 GAM 风格评测（精炼短语回答 + 对齐后的 F1/BLEU 计算）"
    )
    
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="列出所有可用的类别"
    )
    # Skill 消融开关（与 eval_locomo_complete 一致）
    parser.add_argument("--no-temporal-skill", action="store_true", dest="no_temporal_skill", help="消融：关闭 temporal grounding skill")
    parser.add_argument("--no-evidence-skill", action="store_true", dest="no_evidence_skill", help="消融：关闭 evidence retrieval skill")
    parser.add_argument("--no-routing-skill", action="store_true", dest="no_routing_skill", help="消融：关闭 routing skill")

    args = parser.parse_args()
    
        # 列出类别
    if args.list_categories:
        print("\n可用的问题类别:")
        print("  - Single Hop: 单跳问题，只需一步推理")
        print("  - Multi-Hop: 多跳问题，需要多步推理")
        print("  - Temporal: 时序问题，涉及时间推理")
        print("  - Open Domain: 开放域问题，需要综合推理")
        print()
        return
    
    if not args.category:
        parser.error("请指定 --category 参数或使用 --list-categories 查看可用类别")
    
    # 处理向量检索选项
    use_vector_search = args.use_vector_search and not args.no_vector_search
    
    # 生成输出文件名
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category_name = args.category.replace(" ", "_").lower()
        args.output = f"eval_{category_name}_{timestamp}.json"
    
    print("="*100)
    print(f"LoCoMo 分类别详细评测")
    print("="*100)
    print(f"\n配置:")
    print(f"  工作区: {args.workspace}")
    print(f"  数据集: {args.data_dir}")
    print(f"  类别: {args.category}")
    print(f"  最大样本数: {args.max_samples if args.max_samples else '全部'}")
    print(f"  提供商: {args.provider}")
    print(f"  模型: {args.model}")
    print(f"  向量检索: {'启用' if use_vector_search else '禁用'}")
    print(f"  最大迭代: {args.max_iterations}")
    print(f"  输出文件: {args.output}")
    print(f"  LLM Judge: {'启用' if args.use_judge else '禁用'}")
    print(f"  跳过索引: {'是' if args.skip_index else '否'}")
    print(f"  重建索引: {'是' if args.rebuild_index else '否'}")
    print(f"  跳过巩固: {'是' if args.skip_consolidation else '否'}")
    print(f"  仅巩固: {'是' if args.consolidation_only else '否'}")
    
    # 创建配置
    config = create_config(
        workspace_dir=args.workspace,
        provider=args.provider,
        model_name=args.model,
        use_vector_search=use_vector_search,
        max_iterations=args.max_iterations
    )
    # Skill 消融：与主评测脚本一致的开关
    if getattr(args, "no_temporal_skill", False):
        config.agent.enable_temporal_skill = False
        print("  消融: 已关闭 temporal_skill")
    if getattr(args, "no_evidence_skill", False):
        config.agent.enable_evidence_skill = False
        print("  消融: 已关闭 evidence_skill")
    if getattr(args, "no_routing_skill", False):
        config.agent.enable_routing_skill = False
        print("  消融: 已关闭 routing_skill")

    # 加载知识图谱
    print(f"\n初始化知识图谱...")
    cogmem = CogMem(config=config, workspace_dir=args.workspace, auto_load=True)
    
    # 检查图是否正确加载
    total_nodes = cogmem.graph.node_count
    print(f"  已加载图节点数: {total_nodes}")
    
    # 统计核心节点类型（P-E-C-F + Claim）
    from cogmem.storage.schema import NodeTypeEnum
    person_count = event_count = concept_count = fact_count = claim_count = 0
    
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

    print(f"  Person 节点数: {person_count}")
    print(f"  Event 节点数: {event_count}")
    print(f"  Concept 节点数: {concept_count}")
    print(f"  Fact 节点数: {fact_count}")
    print(f"  Claim 节点数: {claim_count}")
    
    # 构建索引逻辑（根据选项决定）
    if args.consolidation_only:
        # 仅执行记忆巩固模式：要求已有索引
        if total_nodes == 0:
            print("❌ 错误：--consolidation-only 要求已有索引，但图谱为空")
            return
        print("⏭️  跳过索引构建（consolidation-only模式）")
    elif args.skip_index:
        # 跳过索引模式：直接使用已有图谱
        if total_nodes == 0:
            print("⚠️  警告：--skip-index 指定但图谱为空，将执行索引构建")
            args.skip_index = False  # 强制执行索引
        else:
            print(f"⏭️  跳过索引构建，使用已有图谱（{total_nodes} 个节点）")
    
    # 执行索引构建（如果需要）
    if args.rebuild_index:
        import shutil
        if os.path.exists(args.workspace):
            print(f"  正在删除旧工作区: {args.workspace}")
            shutil.rmtree(args.workspace)
        os.makedirs(args.workspace, exist_ok=True)
        # 重新初始化
        cogmem = CogMem(config=config, workspace_dir=args.workspace, auto_load=False)
        index_locomo(cogmem, args.data_dir, subset_ratio=args.subset_ratio)
    elif not args.skip_index and not args.consolidation_only and total_nodes == 0:
        print(f"  未找到完整索引，开始构建...")
        index_locomo(cogmem, args.data_dir, subset_ratio=args.subset_ratio)
    else:
        print(f"  知识图谱加载完成")
    
    # 记忆巩固步骤（根据选项决定）
    if not args.skip_consolidation:
        print("\n执行记忆巩固...")
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
    
    # 如果是consolidation-only模式，执行完巩固后直接返回
    if args.consolidation_only:
        print("\n✅ 记忆巩固完成（consolidation-only模式）")
        return
    
    # 加载指定类别的问题
    print(f"\n加载 [{args.category}] 类别的问题...")
    qa_samples = load_category_questions(args.data_dir, args.category, args.max_samples, subset_ratio=args.subset_ratio)
    
    if len(qa_samples) == 0:
        print(f"\n[警告] 没有找到 [{args.category}] 类别的问题！")
        return

    
    # 运行评测
    results = run_detailed_evaluation(
        cogmem=cogmem,
        qa_samples=qa_samples,
        category=args.category,
        output_file=args.output,
        verbose=args.verbose,
        gam_mode=args.gam_mode
    )
    
    # LLM Judge 评分
    if args.use_judge:
        judge_output = args.output.replace('.json', '_judged.json')
        run_llm_judge_evaluation(
            results_file=args.output,
            judge_config=config,  # Use the same config as main evaluation
            output_file=judge_output
        )

if __name__ == "__main__":
    main()
