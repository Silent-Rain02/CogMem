from __future__ import annotations

from typing import List, Optional
import re

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client
from ..models.embedding import EmbeddingEncoder
from ..prompt import RAG_ANSWER_SYSTEM, RAG_ANSWER_USER
from ..storage.schema import RetrievalResult


class AdaptiveGenerator:
    """Adaptive answer generator that adjusts based on evidence confidence and quality."""

    def __init__(self, cfg: CogMemConfig):
        self._cfg = cfg
        self._llm = create_llm_client(cfg)
        self._embed = EmbeddingEncoder(cfg)

    def generate(self, query: str, evidence: List[str], entities: Optional[List] = None) -> str:
        """Generate answer with adaptive confidence-based logic.

        Args:
            query: User query
            evidence: List of evidence strings
            entities: List of entity names for context

        Returns:
            Generated answer or fallback response
        """
        if not evidence:
            return self._generate_minimal_answer(query, entities)

        # Calculate evidence confidence with improved metrics
        confidence = self._calculate_evidence_confidence(evidence, query, entities)

        # Adaptive generation with much more lenient thresholds
        if confidence < 0.05:
            # Very low confidence - minimal answer
            return self._generate_minimal_answer(query, entities)
        elif confidence < 0.15:
            # Low confidence - try fallback generation
            return self._fallback_generate(query, evidence)
        elif confidence < 0.3:
            # Medium confidence - generate with uncertainty disclaimer
            answer = self._generate_with_uncertainty(query, evidence)
            return answer
        else:
            # High confidence - normal generation
            return self._generate_normal(query, evidence)

    def _calculate_evidence_confidence(self, evidence: List[str], query: str, entities: Optional[List] = None) -> float:
        """Calculate confidence score for evidence quality and relevance.

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not evidence:
            return 0.0

        # Factor 1: Evidence quantity (more evidence = higher confidence)
        quantity_score = min(len(evidence) / 3.0, 1.0)  # Cap at 3 pieces for better coverage

        # Factor 2: Evidence specificity (avoid generic "Entity: X" patterns)
        specific_evidence = [ev for ev in evidence if not ev.startswith("Entity:") and len(ev.strip()) > 10]
        specificity_score = len(specific_evidence) / max(len(evidence), 1)

        # Factor 3: Query-evidence semantic similarity
        similarity_score = self._calculate_query_evidence_similarity(query, evidence[:3])  # Top 3 evidence

        # Factor 4: Entity coverage (if entities provided)
        coverage_score = 1.0
        if entities:
            entity_coverage = sum(1 for entity in entities if any(entity.lower() in ev.lower() for ev in evidence))
            coverage_score = entity_coverage / max(len(entities), 1)

        # Weighted combination
        confidence = (
            quantity_score * 0.2 +
            specificity_score * 0.3 +
            similarity_score * 0.3 +
            coverage_score * 0.2
        )

        return min(confidence, 1.0)

    def _calculate_query_evidence_similarity(self, query: str, evidence: List[str]) -> float:
        """Calculate semantic similarity between query and evidence."""
        if not evidence:
            return 0.0

        try:
            # Get embeddings
            texts = [query] + evidence[:3]  # Query + top 3 evidence
            embeddings = self._embed.encode(texts)

            if len(embeddings) < 2:
                return 0.0

            query_emb = embeddings[0]
            evidence_embs = embeddings[1:]

            # Calculate cosine similarities
            import numpy as np
            similarities = []
            for ev_emb in evidence_embs:
                # Cosine similarity
                dot_product = np.dot(query_emb, ev_emb)
                norm_q = np.linalg.norm(query_emb)
                norm_ev = np.linalg.norm(ev_emb)
                if norm_q > 0 and norm_ev > 0:
                    sim = dot_product / (norm_q * norm_ev)
                    similarities.append(max(0, sim))  # Clamp to [0, 1]

            return np.mean(similarities) if similarities else 0.0

        except Exception as e:
            print(f"[AdaptiveGenerator] Error calculating similarity: {e}")
            return 0.0

    def _generate_normal(self, query: str, evidence: List[str]) -> str:
        """Normal generation for high-confidence evidence."""
        formatted_evidence = self._format_evidence_for_qa(evidence)

        user = RAG_ANSWER_USER.format(
            query=query,
            evidence=formatted_evidence
        )
        raw_answer = self._llm.chat(RAG_ANSWER_SYSTEM, user)

        return self._extract_clean_answer(raw_answer)

    def _generate_with_uncertainty(self, query: str, evidence: List[str]) -> str:
        """Generate answer with uncertainty disclaimer."""
        formatted_evidence = self._format_evidence_for_qa(evidence)

        # Modified system prompt: Still try to extract answer, but be more cautious
        uncertainty_system = RAG_ANSWER_SYSTEM + "\n\nIMPORTANT: The evidence quality is uncertain, but you should still attempt to extract an answer from the available evidence. Only say 'cannot determine' if there is absolutely no relevant information in the evidence."

        user = RAG_ANSWER_USER.format(
            query=query,
            evidence=formatted_evidence
        )
        raw_answer = self._llm.chat(uncertainty_system, user)

        answer = self._extract_clean_answer(raw_answer)

        # Don't add uncertainty marker - just return the answer
        return answer

    def _fallback_generate(self, query: str, evidence: List[str]) -> str:
        """Fallback generation for low-confidence evidence."""
        # Even with low confidence, try to extract an answer from evidence
        if evidence:
            formatted_evidence = self._format_evidence_for_qa(evidence)
            # Use a simpler prompt that encourages extraction
            simple_system = "You are a Q&A assistant. Answer the question based on the provided evidence. Extract the answer directly from the evidence, even if it's partial. Only say 'cannot determine' if there is absolutely no relevant information."
            
            user = RAG_ANSWER_USER.format(
                query=query,
                evidence=formatted_evidence
            )
            try:
                raw_answer = self._llm.chat(simple_system, user)
                answer = self._extract_clean_answer(raw_answer)
                if answer and answer.lower() not in ["cannot determine", "无法确定", "no information available"]:
                    return answer
            except Exception as e:
                print(f"[AdaptiveGenerator] Fallback generation failed: {e}")
        
        # Last resort - try to extract from entity names
        entity_names = []
        for ev in evidence:
            if ev.startswith("Entity:"):
                entity_names.append(ev[7:].strip())

        if entity_names:
            return f"Related entities: {', '.join(entity_names[:3])}"

        # Only return cannot determine as absolute last resort
        return "cannot determine"

    def _generate_minimal_answer(self, query: str, entities: Optional[List] = None) -> str:
        """Generate minimal answer when evidence is insufficient."""
        if entities and len(entities) > 0:
            entity_names = [e.name if hasattr(e, 'name') else str(e) for e in entities[:3]]
            return f"I have information about {', '.join(entity_names)}, but cannot determine the specific answer to this question."
        else:
            # Try to extract some basic information from the query
            query_lower = query.lower()
            if "what" in query_lower and "did" in query_lower:
                return "I cannot determine what happened based on available information."
            elif "when" in query_lower:
                return "I cannot determine when this occurred based on available information."
            elif "where" in query_lower:
                return "I cannot determine the location based on available information."
            elif "who" in query_lower:
                return "I cannot identify the person based on available information."
            else:
                return "I don't have enough specific information to answer this question."

    def _format_evidence_for_qa(self, evidence: List[str]) -> str:
        """Format evidence for QA generation."""
        formatted_parts = []
        for ev in evidence[:10]:  # Limit to top 10
            ev_clean = ev.strip()
            if ev_clean and len(ev_clean) > 5:  # Filter out too short evidence
                formatted_parts.append(ev_clean)
        return "\n\n".join(formatted_parts) if formatted_parts else "No evidence available."

    def _extract_clean_answer(self, raw_answer: str) -> str:
        """Extract clean answer from LLM response."""
        if not raw_answer:
            return "cannot determine"

        answer = raw_answer.strip()

        # Remove common prefixes
        prefixes = ["答案：", "答案是：", "Answer:", "答案:", "The answer is", "Based on the evidence"]
        for prefix in prefixes:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()

        # Take first sentence
        if "." in answer:
            answer = answer.split(".")[0].strip()
        elif "。" in answer:
            answer = answer.split("。")[0].strip()

        return answer if answer else "cannot determine"
