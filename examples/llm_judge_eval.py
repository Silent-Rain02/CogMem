"""
LLM-as-Judge evaluation module for LoCoMo and LongMemEval datasets.

This module provides a standardized way to evaluate model answers against
ground truth using LLM-as-a-Judge paradigm, which is the recommended approach
for both LoCoMo and LongMemEval benchmarks.

References:
    - LoCoMo: https://snap-research.github.io/locomo
    - LongMemEval: https://github.com/Wuhan-AI/LongMemEval
    - LLM-as-a-Judge: Gu et al., 2024
"""

from __future__ import annotations

import os
import re
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass


@dataclass
class JudgeResult:
    """LLM Judge evaluation result."""
    is_correct: bool
    confidence: float
    reasoning: str
    raw_output: str


class LLMJudge:
    """
    LLM-as-Judge evaluator for conversational QA benchmarks.
    
    This judge is designed to work with:
    - LoCoMo (Long Conversational Memory) dataset
    - LongMemEval (Long-term Memory Evaluation) dataset
    
    The evaluation uses a strong LLM (e.g., GPT-4o, Gemini-1.5-Pro) to compare
    generated responses against reference answers, avoiding the brittleness of
    exact string matching while accounting for paraphrases and semantically
    equivalent answers.
    
    Usage:
        judge = LLMJudge(client)
        result = judge.evaluate(
            question="What is user's favorite color?",
            predicted="Blue",
            ground_truth="The user likes blue."
        )
        print(result.is_correct)  # True
    """
    
    # Standard LLM-as-Judge prompt based on LoCoMo and LongMemEval papers
    # This is the widely adopted prompt for evaluating QA correctness
    DEFAULT_JUDGE_PROMPT = """You are an expert evaluator for question answering systems. Your task is to determine if the predicted answer correctly answers the question, compared to the ground truth answer.

Evaluation Guidelines:
1. Semantic Equivalence: The predicted answer should convey the same meaning as the ground truth, even if worded differently.
2. Partial Credit: If the predicted answer contains the core correct information but adds extra details, it should still be considered correct.
3. Date/Time Flexibility: Different date formats (e.g., "2023-05-07", "May 7, 2023", "7th May") should be treated as equivalent.
4. Synonyms: Accept synonymous expressions (e.g., "happy" ≈ "joyful", "scared" ≈ "afraid").
5. Lists: If the ground truth is a list, the prediction must contain the key items. Missing items or extra items may be acceptable depending on context.
6. Negation: Be careful with negations. "He went" is NOT the same as "He did not go".
7. Unanswerable: If the question is unanswerable from the context, the prediction should indicate this (e.g., "I don't know", "not mentioned").

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {predicted}

First, analyze the predicted answer compared to the ground truth. Consider:
- Does it contain the key information from the ground truth?
- Are there any contradictions?
- Is the level of detail appropriate?

Then, provide your judgment:
- Output "CORRECT" if the predicted answer is semantically equivalent to the ground truth
- Output "INCORRECT" if the predicted answer is wrong, contradicts the ground truth, or misses critical information
- Output "UNCERTAIN" if you cannot confidently determine correctness

Output format:
Judgment: [CORRECT/INCORRECT/UNCERTAIN]
Confidence: [0-100]
Reasoning: [Brief explanation of your decision]"""

    # Simplified binary judge prompt (for faster evaluation)
    BINARY_JUDGE_PROMPT = """You are an impartial semantic judge evaluating a Question Answering system.

Context:
- Question: "{question}"
- Ground Truth: "{ground_truth}"
- Predicted Answer: "{predicted}"

Task: Determine if the Predicted Answer conveys the SAME meaning as the Ground Truth.

Evaluation Rules (Be Flexible):
1. Dates: Treat "2023-05-07", "May 7th, 2023", "7/5/23" as EQUIVALENT.
2. Synonyms: "Happy" == "Joyful", "Scared" == "Afraid".
3. Verbosity: If the Prediction is long but contains the correct answer, it is CORRECT.
4. Lists: If the Truth is a list, the Prediction must contain the key items.
5. Negation: Watch out for "NOT". "He went" != "He did not go".

Output: Respond ONLY with "YES" if correct, or "NO" if incorrect."""

    # LongMemEval specific judge prompt (based on the original paper)
    LONGMEMEVAL_JUDGE_PROMPT = """You are evaluating answers for a long-term conversational memory benchmark.

Question: {question}
Ground Truth Answer: {ground_truth}
Model's Answer: {predicted}

Evaluate if the model's answer correctly answers the question based on the ground truth.
The model's answer may be phrased differently but should convey the same meaning.

Consider:
- Is the core information correct?
- Are temporal references accurate (e.g., "last week", "two days ago")?
- Does the answer reflect the user's preferences/stated information correctly?
- Is there any contradiction with the ground truth?

Respond with only one word: "YES" if correct, "NO" if incorrect."""

    # LoCoMo specific judge prompt with categories support
    LOCOMO_JUDGE_PROMPT = """You are an evaluator for the LoCoMo (Long Conversational Memory) benchmark.

Question Type: {question_type}
Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {predicted}

Evaluation Criteria by Question Type:
- single_hop: Direct factual recall from one conversation turn
- multi_hop: Synthesis of information across multiple turns  
- temporal: Correct understanding of time relationships and ordering
- open_domain: General knowledge integration with conversation context
- adversarial: Correctly identifying unanswerable or trick questions

Task: Determine if the predicted answer is semantically equivalent to the ground truth.

Rules:
1. Accept paraphrases and different wordings with the same meaning
2. Treat different date/time formats as equivalent
3. Accept synonyms and equivalent expressions
4. For adversarial questions, "I don't know" or similar is correct if the question is unanswerable
5. The answer should not contradict any information in the ground truth

Output format:
Judgment: [YES/NO]
Reason: [One sentence explanation]"""

    def __init__(
        self,
        client: Any,
        prompt_type: Literal["default", "binary", "longmemeval", "locomo"] = "default",
        custom_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the LLM Judge.
        
        Args:
            client: LLM client with `chat()` method (e.g., from cogmem.models.llm_client)
            prompt_type: Which predefined prompt to use
                - "default": Standard detailed evaluation prompt
                - "binary": Fast YES/NO prompt
                - "longmemeval": LongMemEval specific prompt
                - "locomo": LoCoMo specific prompt with category support
            custom_prompt: Override with a custom prompt template
            model: Specific model to use for judging (if None, uses client's default)
        """
        self.client = client
        self.model = model
        
        if custom_prompt:
            self.prompt_template = custom_prompt
        elif prompt_type == "binary":
            self.prompt_template = self.BINARY_JUDGE_PROMPT
        elif prompt_type == "longmemeval":
            self.prompt_template = self.LONGMEMEVAL_JUDGE_PROMPT
        elif prompt_type == "locomo":
            self.prompt_template = self.LOCOMO_JUDGE_PROMPT
        else:
            self.prompt_template = self.DEFAULT_JUDGE_PROMPT
    
    def evaluate(
        self,
        question: str,
        predicted: str,
        ground_truth: str,
        question_type: Optional[str] = None,
    ) -> JudgeResult:
        """
        Evaluate a single prediction against ground truth.
        
        Args:
            question: The question being asked
            predicted: The model's predicted answer
            ground_truth: The reference/ground truth answer
            question_type: Optional question category (for LoCoMo evaluation)
            
        Returns:
            JudgeResult with is_correct, confidence, reasoning, and raw_output
        """
        # Build prompt
        prompt_kwargs = {
            "question": question,
            "predicted": predicted,
            "ground_truth": ground_truth,
        }
        if question_type:
            prompt_kwargs["question_type"] = question_type
            
        prompt = self.prompt_template.format(**prompt_kwargs)
        
        # Call LLM
        system_msg = "You are an expert evaluator for question answering systems."
        try:
            response = self.client.chat(
                system=system_msg,
                user=prompt,
                model=self.model,
            )
        except Exception as e:
            return JudgeResult(
                is_correct=False,
                confidence=0.0,
                reasoning=f"Error during evaluation: {str(e)}",
                raw_output="",
            )
        
        # Parse response
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> JudgeResult:
        """
        Parse the LLM response to extract judgment.
        
        Args:
            response: Raw LLM output
            
        Returns:
            Parsed JudgeResult
        """
        response_upper = response.upper().strip()
        
        # Check for binary YES/NO
        if response_upper in ["YES", "YES.", "YES!"]:
            return JudgeResult(
                is_correct=True,
                confidence=1.0,
                reasoning="Binary judgment: YES",
                raw_output=response,
            )
        if response_upper in ["NO", "NO.", "NO!"]:
            return JudgeResult(
                is_correct=False,
                confidence=1.0,
                reasoning="Binary judgment: NO",
                raw_output=response,
            )
        
        # Try to extract structured judgment
        # Look for "Judgment: CORRECT/INCORRECT/YES/NO"
        judgment_match = re.search(
            r'Judgment:\s*(CORRECT|INCORRECT|YES|NO|UNCERTAIN)',
            response,
            re.IGNORECASE
        )
        
        if judgment_match:
            judgment = judgment_match.group(1).upper()
            is_correct = judgment in ["CORRECT", "YES"]
            
            # Try to extract confidence
            confidence_match = re.search(
                r'Confidence:\s*(\d+)',
                response,
                re.IGNORECASE
            )
            confidence = int(confidence_match.group(1)) / 100.0 if confidence_match else 0.8
            
            # Try to extract reasoning
            reasoning_match = re.search(
                r'Reason(?:ing)?:\s*(.+?)(?:\n|$)',
                response,
                re.IGNORECASE | re.DOTALL
            )
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response
            
            return JudgeResult(
                is_correct=is_correct,
                confidence=confidence,
                reasoning=reasoning,
                raw_output=response,
            )
        
        # Fallback: check if response contains "correct" or "incorrect"
        if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
            return JudgeResult(
                is_correct=True,
                confidence=0.7,
                reasoning="Response indicates correctness",
                raw_output=response,
            )
        if "INCORRECT" in response_upper:
            return JudgeResult(
                is_correct=False,
                confidence=0.7,
                reasoning="Response indicates incorrectness",
                raw_output=response,
            )
        
        # Default to uncertain
        return JudgeResult(
            is_correct=False,
            confidence=0.0,
            reasoning=f"Could not parse judgment from response: {response[:200]}",
            raw_output=response,
        )
    
    def evaluate_batch(
        self,
        items: list[Dict[str, Any]],
        show_progress: bool = True,
    ) -> list[JudgeResult]:
        """
        Evaluate a batch of predictions.
        
        Args:
            items: List of dicts with keys "question", "predicted", "ground_truth", 
                   optionally "question_type"
            show_progress: Whether to print progress
            
        Returns:
            List of JudgeResult objects
        """
        results = []
        total = len(items)
        
        for i, item in enumerate(items):
            result = self.evaluate(
                question=item["question"],
                predicted=item["predicted"],
                ground_truth=item["ground_truth"],
                question_type=item.get("question_type"),
            )
            results.append(result)
            
            if show_progress and (i + 1) % 10 == 0:
                correct_so_far = sum(1 for r in results if r.is_correct)
                print(f"  Progress: {i+1}/{total}, Accuracy: {correct_so_far/(i+1):.2%}")
        
        return results
    
    def calculate_accuracy(self, results: list[JudgeResult]) -> Dict[str, float]:
        """
        Calculate accuracy metrics from judge results.
        
        Args:
            results: List of JudgeResult objects
            
        Returns:
            Dict with accuracy metrics
        """
        if not results:
            return {"accuracy": 0.0, "total": 0, "correct": 0}
        
        correct = sum(1 for r in results if r.is_correct)
        total = len(results)
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
        }
