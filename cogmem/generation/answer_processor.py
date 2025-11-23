"""答案后处理模块：格式化、验证、归一化和置信度评估"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client


@dataclass
class AnswerQuality:
    """答案质量评估结果"""
    confidence_score: float  # 置信度分数 (0-1)
    completeness: float     # 完整性分数 (0-1)
    coherence: float        # 连贯性分数 (0-1)
    factual_consistency: float  # 事实一致性分数 (0-1)
    overall_quality: float  # 综合质量分数 (0-1)
    issues: List[str]       # 发现的问题列表


class AnswerProcessor:
    """答案后处理器：提升生成答案的质量和可读性"""
    
    def __init__(self, config: CogMemConfig):
        self.config = config
        self.llm = create_llm_client(config)
    
    def process_answer(
        self, 
        raw_answer: str, 
        question: str, 
        evidence: List[str],
        entities: List = None
    ) -> Tuple[str, AnswerQuality]:
        """处理原始答案，返回优化后的答案和质量评估
        
        Args:
            raw_answer: 原始生成的答案
            question: 用户问题
            evidence: 支撑证据列表
            entities: 相关实体列表
            
        Returns:
            (processed_answer, quality_assessment)
        """
        # 1. 基础清理和格式化
        cleaned_answer = self._clean_and_format(raw_answer)
        
        # 2. 事实验证和修正
        verified_answer = self._verify_and_correct(cleaned_answer, evidence)
        
        # 3. 完整性检查和补充
        enhanced_answer = self._enhance_completeness(verified_answer, question, evidence)
        
        # 4. 最终格式化
        final_answer = self._final_formatting(enhanced_answer)
        
        # 5. 质量评估
        quality = self._assess_quality(final_answer, question, evidence)
        
        return final_answer, quality
    
    def _clean_and_format(self, answer: str) -> str:
        """基础清理和格式化"""
        if not answer:
            return "抱歉，我无法根据现有信息回答这个问题。"
        
        # 去除多余空白
        answer = re.sub(r'\s+', ' ', answer.strip())
        
        # 移除常见的生成瑕疵
        answer = re.sub(r'^(答案?[:：]?\s*)', '', answer)
        answer = re.sub(r'^(根据.*?[，,]\s*)', '', answer)
        
        # 确保句子完整性
        if answer and not answer[-1] in '.。!！?？':
            answer += '。'
        
        # 首字母大写（英文）
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        return answer
    
    def _verify_and_correct(self, answer: str, evidence: List[str]) -> str:
        """事实验证和修正"""
        if not evidence or not answer:
            return answer
        
        # 使用LLM进行事实核查
        verification_prompt = f"""
        请检查以下答案是否与提供的证据一致，如有不一致请修正：
        
        答案: {answer}
        
        证据:
        {chr(10).join(f"- {ev}" for ev in evidence[:3])}
        
        要求：
        1. 如果答案与证据一致，返回原答案
        2. 如果有事实错误，请修正
        3. 如果证据不足，请在答案中说明
        4. 保持答案简洁准确
        
        只返回修正后的答案，不要其他说明。
        """
        
        try:
            corrected = self.llm.chat(
                "你是专业的事实核查助手，确保答案准确性。",
                verification_prompt
            )
            return corrected.strip() if corrected.strip() else answer
        except Exception:
            return answer
    
    def _enhance_completeness(self, answer: str, question: str, evidence: List[str]) -> str:
        """完整性检查和补充"""
        if not answer or len(answer) < 10:
            return self._generate_fallback_answer(question, evidence)
        
        # 检查答案是否直接回答了问题
        completeness_prompt = f"""
        检查以下答案是否完整回答了问题，如需要请补充：
        
        问题: {question}
        当前答案: {answer}
        
        可用证据:
        {chr(10).join(f"- {ev}" for ev in evidence[:3])}
        
        要求：
        1. 如果答案已经完整，返回原答案
        2. 如果缺少关键信息，请适当补充
        3. 保持答案准确和简洁
        4. 不要编造信息
        
        只返回最终答案。
        """
        
        try:
            enhanced = self.llm.chat(
                "你是答案完整性检查助手。",
                completeness_prompt
            )
            return enhanced.strip() if enhanced.strip() else answer
        except Exception:
            return answer
    
    def _generate_fallback_answer(self, question: str, evidence: List[str]) -> str:
        """生成备用答案（当主答案质量过低时）"""
        if not evidence:
            return "抱歉，我没有足够的信息来回答这个问题。"
        
        fallback_prompt = f"""
        基于以下证据，简洁地回答问题：
        
        问题: {question}
        
        证据:
        {chr(10).join(f"- {ev}" for ev in evidence[:3])}
        
        要求：
        1. 直接回答问题
        2. 只使用提供的证据
        3. 保持简洁准确
        4. 如果证据不足，说明无法确定
        """
        
        try:
            fallback = self.llm.chat(
                "你是专业的问答助手，基于证据准确回答。",
                fallback_prompt
            )
            return fallback.strip() if fallback.strip() else "无法根据现有信息回答此问题。"
        except Exception:
            return "无法根据现有信息回答此问题。"
    
    def _final_formatting(self, answer: str) -> str:
        """最终格式化"""
        if not answer:
            return "无法提供答案。"
        
        # 统一标点符号
        answer = answer.replace('，，', '，').replace('。。', '。')
        answer = re.sub(r'([。！？])\1+', r'\1', answer)
        
        # 移除重复短语
        answer = self._remove_repetitions(answer)
        
        # 确保适当的长度
        if len(answer) > 500:
            # 截断到最后一个完整句子
            sentences = re.split(r'[。！？.!?]', answer)
            truncated = ''
            for sent in sentences:
                if len(truncated + sent) > 400:
                    break
                truncated += sent + '。'
            answer = truncated.rstrip('。') + '。'
        
        return answer.strip()
    
    def _remove_repetitions(self, text: str) -> str:
        """移除重复的短语或句子"""
        # 简单的重复检测和移除
        sentences = re.split(r'([。！？.!?])', text)
        seen = set()
        result = []
        
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence and sentence not in seen:
                    seen.add(sentence)
                    result.append(sentence)
                    if i + 1 < len(sentences):
                        result.append(sentences[i + 1])
        
        return ''.join(result)
    
    def _assess_quality(self, answer: str, question: str, evidence: List[str]) -> AnswerQuality:
        """评估答案质量"""
        issues = []
        
        # 1. 基础检查
        if not answer or len(answer.strip()) < 5:
            return AnswerQuality(0.0, 0.0, 0.0, 0.0, 0.0, ["答案过短或为空"])
        
        # 2. 长度检查
        length_score = self._assess_length(answer)
        if length_score < 0.5:
            issues.append("答案长度不合适")
        
        # 3. 完整性检查
        completeness_score = self._assess_completeness_simple(answer, question)
        if completeness_score < 0.6:
            issues.append("答案可能不够完整")
        
        # 4. 连贯性检查
        coherence_score = self._assess_coherence(answer)
        if coherence_score < 0.6:
            issues.append("答案连贯性有待改善")
        
        # 5. 事实一致性检查
        factual_score = self._assess_factual_consistency(answer, evidence)
        if factual_score < 0.7:
            issues.append("答案与证据可能不一致")
        
        # 6. 置信度评估
        confidence_score = min(completeness_score, factual_score, coherence_score)
        
        # 7. 综合质量分数
        overall_quality = (
            completeness_score * 0.3 +
            factual_score * 0.3 +
            coherence_score * 0.2 +
            length_score * 0.2
        )
        
        return AnswerQuality(
            confidence_score=confidence_score,
            completeness=completeness_score,
            coherence=coherence_score,
            factual_consistency=factual_score,
            overall_quality=overall_quality,
            issues=issues
        )
    
    def _assess_length(self, answer: str) -> float:
        """评估答案长度合理性"""
        length = len(answer)
        if 20 <= length <= 300:
            return 1.0
        elif 10 <= length < 20 or 300 < length <= 500:
            return 0.8
        elif 5 <= length < 10 or 500 < length <= 800:
            return 0.6
        else:
            return 0.3
    
    def _assess_completeness_simple(self, answer: str, question: str) -> float:
        """简单的完整性评估"""
        # 检查问题关键词是否在答案中被涉及
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # 移除停用词
        stopwords = {"what", "when", "where", "who", "how", "why", "is", "are", "was", "were", 
                    "什么", "什么时候", "哪里", "谁", "怎么", "为什么", "是", "的", "了", "在"}
        question_words -= stopwords
        
        if not question_words:
            return 0.8
        
        overlap = len(question_words.intersection(answer_words))
        coverage = overlap / len(question_words)
        
        return min(coverage + 0.3, 1.0)  # 给基础分
    
    def _assess_coherence(self, answer: str) -> float:
        """评估答案连贯性"""
        # 简单的连贯性检查
        sentences = re.split(r'[。！？.!?]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 0.9
        
        # 检查重复
        unique_sentences = set(sentences)
        repetition_penalty = len(unique_sentences) / len(sentences)
        
        # 检查句子长度变化（过于相似可能表示重复）
        lengths = [len(s) for s in sentences]
        if len(set(lengths)) == 1 and len(lengths) > 2:
            repetition_penalty *= 0.8
        
        return min(repetition_penalty, 1.0)
    
    def _assess_factual_consistency(self, answer: str, evidence: List[str]) -> float:
        """评估事实一致性"""
        if not evidence:
            return 0.7  # 无证据时给中等分
        
        # 简单的关键词匹配检查
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        evidence_words = set()
        
        for ev in evidence[:3]:
            evidence_words.update(re.findall(r'\b\w+\b', ev.lower()))
        
        if not evidence_words:
            return 0.7
        
        # 计算答案中有多少词汇得到证据支持
        supported_words = answer_words.intersection(evidence_words)
        support_ratio = len(supported_words) / max(len(answer_words), 1)
        
        return min(support_ratio + 0.4, 1.0)  # 给基础分


def format_answer_with_confidence(answer: str, quality: AnswerQuality) -> str:
    """根据置信度格式化答案"""
    if quality.confidence_score >= 0.8:
        return answer
    elif quality.confidence_score >= 0.6:
        return f"{answer}\n\n（注：此答案基于有限信息，可能不够完整）"
    elif quality.confidence_score >= 0.4:
        return f"{answer}\n\n（注：此答案置信度较低，建议进一步核实）"
    else:
        return f"基于现有信息：{answer}\n\n（注：信息不足，答案可能不准确）"
