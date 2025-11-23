"""LLM as Judge 评测模块：使用大模型评估开放域问答质量"""

import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from cogmem.models.llm_client import create_llm_client
from cogmem.config import CogMemConfig


@dataclass
class JudgeResult:
    """评测结果"""
    semantic_correctness: float  # 语义正确性 (0-1)
    completeness: float         # 完整性 (0-1)
    relevance: float           # 相关性 (0-1)
    overall_score: float       # 综合分数 (0-1)
    explanation: str           # 评测说明


class LLMJudge:
    """LLM 评测器：使用大模型评估问答质量"""
    
    def __init__(self, config: CogMemConfig):
        self.llm = create_llm_client(config)
        
    def evaluate_answer(
        self, 
        question: str, 
        gold_answer: str, 
        predicted_answer: str,
        category: Optional[str] = None
    ) -> JudgeResult:
        """使用 LLM 评估答案质量
        
        Args:
            question: 问题
            gold_answer: 标准答案
            predicted_answer: 预测答案
            category: 问题类别（可选）
            
        Returns:
            评测结果
        """
        system_prompt = self._get_judge_system_prompt()
        user_prompt = self._format_judge_prompt(question, gold_answer, predicted_answer, category)
        
        try:
            response = self.llm.chat(system_prompt, user_prompt)
            result_dict = json.loads(response.strip())
            
            return JudgeResult(
                semantic_correctness=float(result_dict.get("semantic_correctness", 0.0)),
                completeness=float(result_dict.get("completeness", 0.0)),
                relevance=float(result_dict.get("relevance", 0.0)),
                overall_score=float(result_dict.get("overall_score", 0.0)),
                explanation=result_dict.get("explanation", "")
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # 解析失败时返回默认结果
            return JudgeResult(
                semantic_correctness=0.0,
                completeness=0.0,
                relevance=0.0,
                overall_score=0.0,
                explanation=f"评测解析失败: {str(e)}"
            )
    
    def _get_judge_system_prompt(self) -> str:
        """获取评测系统提示"""
        return """你是专业的问答质量评估助手。你的任务是客观、准确地评估问答系统的回答质量。

评估原则：
1. 语义正确性：预测答案是否在语义上正确，即使表达方式不同
2. 完整性：预测答案是否包含了回答问题所需的关键信息
3. 相关性：预测答案是否直接回答了问题，没有偏离主题

【关键规则：无法回答的情况】
如果预测答案表示"无法确定"、"cannot determine"、"不知道"、"不确定"等，而标准答案提供了具体信息，则：
- 语义正确性: 0.0-0.2（因为无法确定 ≠ 正确答案）
- 完整性: 0.0-0.2（完全没有提供答案）
- 相关性: 0.0-0.3（虽然承认无法回答，但没有回答问题）
- 综合分数: 0.0-0.2（严重失败）

例外情况：只有当标准答案也是"无法确定"或"无信息"时，预测答案"cannot determine"才能得到高分。

评分标准：
- 1.0: 完美/优秀，完全正确且完整
- 0.8-0.9: 很好，有小瑕疵或轻微不完整
- 0.6-0.7: 一般，缺少部分信息或有轻微错误
- 0.4-0.5: 较差，信息不完整或有明显错误
- 0.0-0.3: 很差，完全错误、无关、或无法回答（当标准答案有具体内容时）

请严格按照JSON格式返回结果，不要包含任何其他文本。"""

    def _format_judge_prompt(
        self, 
        question: str, 
        gold_answer: str, 
        predicted_answer: str,
        category: Optional[str] = None
    ) -> str:
        """格式化评测提示"""
        category_info = f"\n问题类别: {category}" if category else ""
        
        # 检测预测答案是否为"无法确定"类型
        pred_lower = predicted_answer.lower().strip()
        is_cannot_determine = any(phrase in pred_lower for phrase in [
            "cannot determine", "cannot be determined", "cannot be determined from",
            "无法确定", "无法判断", "不确定", "不知道", "没有信息", "无信息",
            "no information", "not available", "not mentioned", "not found"
        ])
        
        # 检测标准答案是否为"无法确定"类型
        gold_lower = gold_answer.lower().strip()
        gold_is_unknown = any(phrase in gold_lower for phrase in [
            "cannot determine", "无法确定", "不确定", "不知道", "无信息",
            "no information", "not available", "not mentioned"
        ])
        
        # 构建特殊提示
        special_instruction = ""
        if is_cannot_determine and not gold_is_unknown:
            special_instruction = """
【⚠️ 重要：无法回答的情况】
预测答案表示"无法确定"，但标准答案提供了具体信息（"{gold_answer}"）。
这是一个严重的失败：系统应该能够回答这个问题，但选择了放弃。
请严格按照以下评分：
- 语义正确性: 0.0-0.2（无法确定 ≠ 正确答案）
- 完整性: 0.0-0.2（完全没有提供答案）
- 相关性: 0.0-0.3（虽然承认无法回答，但没有回答问题）
- 综合分数: 0.0-0.2（严重失败）

请明确说明这是"无法回答"导致的失败。
""".format(gold_answer=gold_answer)
        elif is_cannot_determine and gold_is_unknown:
            special_instruction = """
【特殊情况：双方都表示无法确定】
预测答案和标准答案都表示"无法确定"，这是正确的。
请给予高分（0.8-1.0），因为系统正确识别了无法回答的情况。
"""
        
        return f"""请评估以下问答的质量：

问题: {question}{category_info}
标准答案: {gold_answer}
预测答案: {predicted_answer}
{special_instruction}
请从以下三个维度评估（每个维度0-1分）：
1. 语义正确性: 预测答案是否语义正确（允许表达方式不同）
2. 完整性: 是否包含回答问题所需的关键信息
3. 相关性: 是否直接回答问题，没有偏离主题

综合分数 = (语义正确性 * 0.4 + 完整性 * 0.4 + 相关性 * 0.2)

【评分示例】
示例1: 问题="When did X happen?", 标准答案="2022", 预测答案="cannot determine"
→ 语义正确性: 0.1, 完整性: 0.1, 相关性: 0.2, 综合分数: 0.12（严重失败）

示例2: 问题="What is X?", 标准答案="Y", 预测答案="Y"
→ 语义正确性: 1.0, 完整性: 1.0, 相关性: 1.0, 综合分数: 1.0（完美）

返回JSON格式：
{{
    "semantic_correctness": 0.8,
    "completeness": 0.6,
    "relevance": 0.9,
    "overall_score": 0.74,
    "explanation": "预测答案语义正确但不够完整，缺少具体细节"
}}"""

    def batch_evaluate(
        self, 
        evaluations: List[Dict]
    ) -> List[JudgeResult]:
        """批量评估
        
        Args:
            evaluations: 评估数据列表，每个包含 question, gold_answer, predicted_answer
            
        Returns:
            评估结果列表
        """
        results = []
        
        for i, eval_data in enumerate(evaluations):
            print(f"评估进度: {i+1}/{len(evaluations)}")
            
            result = self.evaluate_answer(
                question=eval_data["question"],
                gold_answer=eval_data["gold_answer"],
                predicted_answer=eval_data["predicted_answer"],
                category=eval_data.get("category")
            )
            
            results.append(result)
            
            # 避免API调用过快
            time.sleep(0.1)
        
        return results


def compute_judge_metrics(results: List[JudgeResult]) -> Dict[str, float]:
    """计算LLM Judge评测指标
    
    Args:
        results: 评测结果列表
        
    Returns:
        聚合指标
    """
    if not results:
        return {}
    
    metrics = {
        "avg_semantic_correctness": sum(r.semantic_correctness for r in results) / len(results),
        "avg_completeness": sum(r.completeness for r in results) / len(results),
        "avg_relevance": sum(r.relevance for r in results) / len(results),
        "avg_overall_score": sum(r.overall_score for r in results) / len(results),
        "high_quality_ratio": sum(1 for r in results if r.overall_score >= 0.7) / len(results),
        "total_samples": len(results)
    }
    
    return metrics


if __name__ == "__main__":
    # 测试示例
    from cogmem.config import DEFAULT_CONFIG
    
    judge = LLMJudge(DEFAULT_CONFIG)
    
    # 测试单个评估
    result = judge.evaluate_answer(
        question="What did John do to help his community last year in his hometown?",
        gold_answer="Helped renovate a rundown community center.",
        predicted_answer="John volunteered to help his community last year in his hometown",
        category="Open Domain"
    )
    
    print("评测结果:")
    print(f"语义正确性: {result.semantic_correctness:.2f}")
    print(f"完整性: {result.completeness:.2f}")
    print(f"相关性: {result.relevance:.2f}")
    print(f"综合分数: {result.overall_score:.2f}")
    print(f"说明: {result.explanation}")
