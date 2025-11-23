"""快速参数调优脚本：针对关键参数进行网格搜索优化"""

import json
import time
from itertools import product
from typing import Dict, List, Tuple
from pathlib import Path

from cogmem import CogMem
from cogmem.config import DEFAULT_CONFIG
from examples.llm_judge_eval import LLMJudge, compute_judge_metrics
from locomo.data_loader import load_locomo_dataset


# 参数搜索空间（精简版，只调关键参数）
PARAM_SPACE = {
    'min_similarity': [0.2, 0.3, 0.4],
    'top_k': [8, 10, 12],
    'chunk_top_k': [5, 8, 10],
    'gap_max_steps': [1, 2, 3],
    'fusion_semantic_weight': [0.5, 0.6, 0.7],
}


def create_config_variant(base_config, params: Dict) -> object:
    """根据参数创建配置变体"""
    config = base_config
    
    # 更新阈值配置
    config.thresholds.min_similarity = params['min_similarity']
    
    # 更新检索配置
    config.retrieval.topk_candidates = params['top_k']
    config.retrieval.chunk_top_k = params['chunk_top_k']
    config.retrieval.gap_max_steps = params['gap_max_steps']
    config.retrieval.fusion_semantic_weight = params['fusion_semantic_weight']
    config.retrieval.fusion_structure_weight = 1.0 - params['fusion_semantic_weight']
    
    return config


def evaluate_config(
    config, 
    test_samples: List[Dict], 
    workspace_dir: str,
    max_samples: int = 10
) -> Dict:
    """评估单个配置的性能"""
    
    # 初始化系统
    cogmem = CogMem(config=config, workspace_dir=workspace_dir, auto_load=True)
    judge = LLMJudge(config)
    
    results = []
    judge_results = []
    
    # 限制样本数量以加快调优速度
    samples = test_samples[:max_samples]
    
    for sample in samples:
        question = sample["question"]
        gold_answer = sample["answer"]
        
        try:
            # 生成答案
            predicted_answer = cogmem.answer(
                question, 
                use_enhanced_retrieval=True,
                enable_post_processing=True
            )
            
            # LLM Judge 评测
            judge_result = judge.evaluate_answer(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer
            )
            
            judge_results.append(judge_result)
            
        except Exception as e:
            print(f"评测样本时出错: {str(e)}")
            continue
    
    # 计算聚合指标
    if judge_results:
        metrics = compute_judge_metrics(judge_results)
        return {
            'avg_overall_score': metrics.get('avg_overall_score', 0.0),
            'avg_semantic_correctness': metrics.get('avg_semantic_correctness', 0.0),
            'avg_completeness': metrics.get('avg_completeness', 0.0),
            'high_quality_ratio': metrics.get('high_quality_ratio', 0.0),
            'sample_count': len(judge_results)
        }
    else:
        return {
            'avg_overall_score': 0.0,
            'avg_semantic_correctness': 0.0,
            'avg_completeness': 0.0,
            'high_quality_ratio': 0.0,
            'sample_count': 0
        }


def grid_search_optimization(
    dataset: List[Dict],
    workspace_base: str = ".tune_workspace",
    max_samples_per_config: int = 10,
    max_configs: int = 20
) -> Tuple[Dict, Dict]:
    """网格搜索参数优化
    
    Args:
        dataset: 测试数据集
        workspace_base: 工作空间基础目录
        max_samples_per_config: 每个配置测试的最大样本数
        max_configs: 最大测试配置数量
        
    Returns:
        (best_params, all_results)
    """
    
    print(f"🔧 开始参数网格搜索优化")
    print(f"参数空间大小: {len(list(product(*PARAM_SPACE.values())))}")
    print(f"每配置测试样本数: {max_samples_per_config}")
    print(f"最大配置数: {max_configs}")
    print("=" * 60)
    
    # 生成所有参数组合
    param_names = list(PARAM_SPACE.keys())
    param_values = list(PARAM_SPACE.values())
    all_combinations = list(product(*param_values))
    
    # 限制测试的配置数量
    if len(all_combinations) > max_configs:
        print(f"⚠️ 配置数量过多，随机选择 {max_configs} 个配置进行测试")
        import random
        random.shuffle(all_combinations)
        all_combinations = all_combinations[:max_configs]
    
    best_score = 0.0
    best_params = None
    all_results = []
    
    for i, param_combo in enumerate(all_combinations):
        # 创建参数字典
        params = dict(zip(param_names, param_combo))
        
        print(f"\n🧪 测试配置 {i+1}/{len(all_combinations)}")
        print(f"参数: {params}")
        
        # 创建配置
        config = create_config_variant(DEFAULT_CONFIG, params)
        config.extraction.use_enhanced_extraction = True
        
        # 创建独立的工作空间
        workspace_dir = f"{workspace_base}_config_{i}"
        
        start_time = time.time()
        
        try:
            # 评估配置
            metrics = evaluate_config(
                config, 
                dataset, 
                workspace_dir,
                max_samples_per_config
            )
            
            eval_time = time.time() - start_time
            
            # 记录结果
            result = {
                'config_id': i,
                'params': params,
                'metrics': metrics,
                'eval_time': eval_time
            }
            all_results.append(result)
            
            # 更新最佳配置
            current_score = metrics['avg_overall_score']
            if current_score > best_score:
                best_score = current_score
                best_params = params
                print(f"🎯 新的最佳配置！分数: {current_score:.3f}")
            
            print(f"   综合分数: {current_score:.3f}")
            print(f"   语义正确性: {metrics['avg_semantic_correctness']:.3f}")
            print(f"   完整性: {metrics['avg_completeness']:.3f}")
            print(f"   高质量比例: {metrics['high_quality_ratio']:.3f}")
            print(f"   评测时间: {eval_time:.1f}s")
            
        except Exception as e:
            print(f"❌ 配置测试失败: {str(e)}")
            continue
    
    return best_params, all_results


def save_tuning_results(best_params: Dict, all_results: List[Dict], output_file: str):
    """保存调优结果"""
    
    # 按分数排序
    sorted_results = sorted(
        all_results, 
        key=lambda x: x['metrics']['avg_overall_score'], 
        reverse=True
    )
    
    tuning_report = {
        'summary': {
            'total_configs_tested': len(all_results),
            'best_score': sorted_results[0]['metrics']['avg_overall_score'] if sorted_results else 0.0,
            'best_params': best_params,
            'param_space': PARAM_SPACE,
        },
        'top_5_configs': sorted_results[:5],
        'all_results': sorted_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tuning_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 调优结果已保存到: {output_file}")


def analyze_parameter_importance(results: List[Dict]):
    """分析参数重要性"""
    
    print("\n📈 参数重要性分析:")
    print("-" * 40)
    
    # 按参数分组分析
    param_impact = {}
    
    for param_name in PARAM_SPACE.keys():
        param_scores = {}
        
        for result in results:
            param_value = result['params'][param_name]
            score = result['metrics']['avg_overall_score']
            
            if param_value not in param_scores:
                param_scores[param_value] = []
            param_scores[param_value].append(score)
        
        # 计算每个参数值的平均分数
        avg_scores = {}
        for value, scores in param_scores.items():
            avg_scores[value] = sum(scores) / len(scores)
        
        param_impact[param_name] = avg_scores
        
        # 输出分析结果
        print(f"\n{param_name}:")
        for value, avg_score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {value}: {avg_score:.3f}")


def main():
    """主函数"""
    
    # 加载数据集
    print("📚 加载 LoCoMo 数据集...")
    dataset = load_locomo_dataset("locomo")
    
    if not dataset:
        print("❌ 数据集加载失败")
        return
    
    print(f"✅ 数据集加载成功，共 {len(dataset)} 个样本")
    
    # 执行网格搜索
    best_params, all_results = grid_search_optimization(
        dataset,
        max_samples_per_config=8,  # 减少样本数以加快速度
        max_configs=15  # 限制配置数量
    )
    
    if not all_results:
        print("❌ 没有成功的配置测试结果")
        return
    
    # 保存结果
    output_file = "tuning_results.json"
    save_tuning_results(best_params, all_results, output_file)
    
    # 分析参数重要性
    analyze_parameter_importance(all_results)
    
    # 输出最佳配置
    print(f"\n🏆 最佳配置:")
    print(f"参数: {best_params}")
    
    best_result = max(all_results, key=lambda x: x['metrics']['avg_overall_score'])
    print(f"性能:")
    print(f"  - 综合分数: {best_result['metrics']['avg_overall_score']:.3f}")
    print(f"  - 语义正确性: {best_result['metrics']['avg_semantic_correctness']:.3f}")
    print(f"  - 完整性: {best_result['metrics']['avg_completeness']:.3f}")
    print(f"  - 高质量比例: {best_result['metrics']['high_quality_ratio']:.3f}")
    
    print(f"\n✅ 参数调优完成！")


if __name__ == "__main__":
    main()
