"""Query Processor for context-aware query rewriting and expansion.

This module provides:
1. Coreference Resolution: Resolves pronouns in queries using conversation history
2. Query Expansion: Generates synonymous queries to improve recall
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: datetime
    user_query: str
    assistant_response: str
    entities: List[str] = field(default_factory=list)  # Extracted entities from this turn


class ConversationBuffer:
    """Lightweight buffer to maintain conversation history for context awareness."""

    def __init__(self, max_turns: int = 5):
        """Initialize conversation buffer.

        Args:
            max_turns: Maximum number of conversation turns to keep in buffer
        """
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []

    def add_turn(self, user_query: str, assistant_response: str, entities: Optional[List[str]] = None):
        """Add a new conversation turn to the buffer.

        Args:
            user_query: User's query
            assistant_response: Assistant's response
            entities: Entities extracted from this turn (if available)
        """
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=user_query,
            assistant_response=assistant_response,
            entities=entities or []
        )
        self.turns.append(turn)

        # Maintain buffer size
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_recent_context(self, last_n: int = 3) -> str:
        """Get recent conversation context as a formatted string.

        Args:
            last_n: Number of recent turns to include

        Returns:
            Formatted conversation context
        """
        if not self.turns:
            return ""

        recent_turns = self.turns[-last_n:]
        context_parts = []

        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"User: {turn.user_query}")
            if turn.assistant_response:
                context_parts.append(f"Assistant: {turn.assistant_response[:200]}...")  # Truncate long responses
            if turn.entities:
                context_parts.append(f"Entities: {', '.join(turn.entities)}")
            context_parts.append("")

        return "\n".join(context_parts).strip()

    def get_recent_entities(self) -> List[str]:
        """Get entities from recent conversation turns.

        Returns:
            List of entity names mentioned in recent turns
        """
        entities = []
        for turn in self.turns[-2:]:  # Get entities from last 2 turns
            entities.extend(turn.entities)
        return list(set(entities))  # Remove duplicates

    def clear(self):
        """Clear all conversation history."""
        self.turns.clear()

    def __len__(self):
        return len(self.turns)


class QueryProcessor:
    """Processes queries with context awareness for improved retrieval."""

    def __init__(self, llm_client, embed_encoder, conversation_buffer: Optional[ConversationBuffer] = None):
        """Initialize QueryProcessor.

        Args:
            llm_client: LLM client for query rewriting
            embed_encoder: Embedding encoder for query expansion
            conversation_buffer: Optional conversation buffer for context
        """
        self.llm = llm_client
        self.embed = embed_encoder
        self.conversation_buffer = conversation_buffer or ConversationBuffer()

        # Common pronouns that need resolution
        self.PRONOUNS = {
            'he', 'she', 'it', 'they', 'them', 'his', 'her', 'its', 'their',
            'him', 'this', 'that', 'these', 'those', 'here', 'there'
        }

    def contains_pronoun(self, query: str) -> bool:
        """Check if query contains pronouns that need resolution.

        Args:
            query: User query string

        Returns:
            True if query contains pronouns
        """
        query_words = set(query.lower().split())
        return len(query_words.intersection(self.PRONOUNS)) > 0

    def resolve_coreferences(self, query: str) -> str:
        """Resolve pronouns in query using conversation history.

        Args:
            query: User query string that may contain pronouns

        Returns:
            Rewritten query with pronouns resolved
        """
        if not self.contains_pronoun(query):
            return query

        # Get recent context
        recent_context = self.conversation_buffer.get_recent_context()
        recent_entities = self.conversation_buffer.get_recent_entities()

        if not recent_context:
            return query  # No context available

        # Use LLM to rewrite the query
        system_prompt = """You are a conversational query rewriter. Your task is to resolve pronouns in user queries using conversation history.

Rules:
1. Replace pronouns (he, she, her, his, they, them, etc.) with specific entity names from conversation
2. If multiple entities were mentioned, use the most recent one that fits the context
3. Keep the original query structure and intent
4. Only make changes if you're confident about the resolution
5. Return ONLY the rewritten query, no explanations

Examples:
Conversation:
User: "What is Caroline's age?"
Assistant: "Caroline is 28 years old."
User: "What is her job?"
Rewritten: "What is Caroline's job?"

Conversation:
User: "Who is John?"
Assistant: "John is a software engineer at Google."
User: "When did he start working there?"
Rewritten: "When did John start working at Google?"
"""

        user_prompt = f"""Conversation History:
{recent_context}

Recent Entities: {', '.join(recent_entities) if recent_entities else 'None'}

Current Query: {query}

Please resolve any pronouns in the current query using the conversation history and recent entities.
Return ONLY the rewritten query."""

        try:
            rewritten_query = self.llm.chat(system_prompt, user_prompt).strip()

            # Validate that something changed
            if rewritten_query.lower() == query.lower():
                return query

            # Ensure the rewritten query is reasonable length
            if len(rewritten_query) > 500:
                return query

            print(f"[QueryProcessor] Coreference resolved: '{query}' → '{rewritten_query}'")
            return rewritten_query

        except Exception as e:
            print(f"[QueryProcessor] Coreference resolution failed: {e}")
            return query

    def expand_query(self, query: str, num_variants: int = 3) -> List[str]:
        """Generate query variants for improved recall.

        Args:
            query: Original query string
            num_variants: Number of variants to generate

        Returns:
            List of query variants (including original)
        """
        variants = [query]  # Always include original

        # Use LLM to generate synonyms and paraphrases
        num_variants_to_generate = num_variants - 1
        system_prompt = f"""You are a query expansion assistant. Generate {num_variants_to_generate} alternative ways to ask the same question.

Rules:
1. Generate natural, fluent variations of the original query
2. Use synonyms and different phrasings
3. Keep the same intent and meaning
4. Each variant should be a complete question
5. Return as a JSON array of strings

Example:
Input: "What is Caroline's job?"
Output: [
  "What does Caroline do for a living?",
  "What is Caroline's occupation?",
  "What work does Caroline do?"
]
"""

        user_prompt = f"Original query: {query}\n\nGenerate {num_variants_to_generate} alternative versions:"

        try:
            response = self.llm.chat(system_prompt, user_prompt)
            import json

            # Parse JSON response
            expanded_queries = json.loads(response)

            if isinstance(expanded_queries, list):
                # Filter out the original and add unique variants
                for variant in expanded_queries:
                    if isinstance(variant, str) and variant.lower() != query.lower():
                        variants.append(variant)

                print(f"[QueryProcessor] Generated {len(variants) - 1} query variants")

        except Exception as e:
            print(f"[QueryProcessor] Query expansion failed: {e}")

        return variants[:num_variants]  # Limit to requested number

    def generate_hypothetical_answers(self, query: str, num_answers: int = 2) -> List[str]:
        """Generate hypothetical answers for vector similarity search.

        This technique (HyDE - Hypothetical Document Embeddings) generates
        potential answers that can be used to find relevant documents.

        Args:
            query: Original query string
            num_answers: Number of hypothetical answers to generate

        Returns:
            List of hypothetical answer strings
        """
        system_prompt = f"""You are a helpful assistant. Given a question, generate {num_answers} plausible answers.

The answers should:
1. Be concise but informative (1-2 sentences)
2. Sound like they could be real answers
3. Cover different possible interpretations of the question
4. Be formatted as a JSON array of strings

Example:
Question: "What is Caroline's job?"
Answers: [
  "Caroline works as a software engineer at a tech company in San Francisco.",
  "Caroline is a marketing manager at a startup, focusing on digital campaigns."
]
"""

        user_prompt = f"Question: {query}\n\nGenerate {num_answers} plausible answers:"

        try:
            response = self.llm.chat(system_prompt, user_prompt)
            import json

            hypothetical_answers = json.loads(response)

            if isinstance(hypothetical_answers, list):
                print(f"[QueryProcessor] Generated {len(hypothetical_answers)} hypothetical answers")
                return [ans for ans in hypothetical_answers if isinstance(ans, str)]

        except Exception as e:
            print(f"[QueryProcessor] Hypothetical answer generation failed: {e}")

        return []

    def process_query(self, query: str, enable_coreference: bool = True, enable_expansion: bool = True) -> Dict[str, Any]:
        """Main entry point for query processing.

        Args:
            query: Original user query
            enable_coreference: Whether to enable coreference resolution
            enable_expansion: Whether to enable query expansion

        Returns:
            Dictionary containing:
                - rewritten_query: Query with pronouns resolved
                - query_variants: List of query variants for vector search
                - seed_queries: Combined list for seed node identification
        """
        result = {
            'original_query': query,
            'rewritten_query': query,
            'query_variants': [],
            'hypothetical_answers': [],
            'seed_queries': []
        }

        # Step 1: Coreference resolution
        if enable_coreference and self.contains_pronoun(query):
            rewritten = self.resolve_coreferences(query)
            result['rewritten_query'] = rewritten

        # Step 2: Query expansion
        if enable_expansion:
            # Generate query variants
            result['query_variants'] = self.expand_query(result['rewritten_query'])

            # Generate hypothetical answers (for HyDE technique)
            result['hypothetical_answers'] = self.generate_hypothetical_answers(result['rewritten_query'])

        # Step 3: Prepare seed queries (all variants for vector search)
        result['seed_queries'] = []
        result['seed_queries'].extend(result['query_variants'])
        result['seed_queries'].extend(result['hypothetical_answers'])

        # Ensure we have at least the rewritten query
        if not result['seed_queries']:
            result['seed_queries'] = [result['rewritten_query']]

        print(f"[QueryProcessor] Processed query: {result['original_query']} → {result['rewritten_query']}")
        print(f"[QueryProcessor] Generated {len(result['query_variants'])} variants and {len(result['hypothetical_answers'])} hypothetical answers")

        return result
