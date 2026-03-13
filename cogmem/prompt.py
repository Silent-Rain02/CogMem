ATOMIC_EXTRACTION_SYSTEM = """You are an expert information extractor. Your task is to extract structured information from the conversation.

Extract the following:
1. **Events**: Specific occurrences or happenings mentioned in the conversation.
   - Include: who participated, what happened, when (if mentioned)
   - Format: {{"type": "event", "description": "...", "participants": [...], "time": "..."}}

2. **Facts**: Objective facts about people, their attributes, preferences, or states.
   - Include: subject, predicate, object
   - Format: {{"type": "fact", "subject": "...", "predicate": "...", "object": "..."}}

3. **Claims**: Subjective statements, opinions, or beliefs expressed by speakers.
   - Include: who made the claim, what the claim is about, the claim content
   - Format: {{"type": "claim", "source": "...", "target": "...", "content": "..."}}

Guidelines:
- Be precise and extract only what is explicitly stated or can be reasonably inferred from the context.
- Use exact names and terms from the conversation.
- If time is relative (e.g., "yesterday", "last week"), preserve the relative reference.
- Output valid JSON only, no markdown formatting.
"""

ATOMIC_EXTRACTION_USER = """Extract structured information from the following conversation:

Conversation Text:
{text}

Session Date: {session_date}

Provide your extraction as a JSON object with "events", "facts", and "claims" arrays.
"""

TOOLS_DESCRIPTION = """
Available Tools:

1. semantic_search(query: str, top_k: int = 10)
   - Performs semantic search over facts, events, and claims.
   - Use for: finding specific information about entities or events.
   - Returns: ranked list of relevant nodes with scores.

2. find_anchors(query: str, top_k: int = 5)
   - Finds key entities (Person, Concept) related to the query.
   - Use for: identifying the main entities involved in a question.
   - Returns: list of anchor nodes.

3. traverse_neighbors(node_id: str, max_depth: int = 1, max_results: int = 10)
   - Explores neighbors of a node in the knowledge graph.
   - Use for: finding relationships and connected information.
   - Returns: connected nodes with relationship types.

4. find_common_ground(entity_ids: list, top_k: int = 5)
   - Finds common connections between multiple entities.
   - Use for: multi-hop questions involving multiple entities.
   - Returns: shared concepts, events, or facts.

5. get_evidence(node_ids: list)
   - Retrieves source evidence for specific nodes.
   - Use for: grounding answers in source material.
   - Returns: original text passages with context.

Tool Selection Strategy:
- Start with find_anchors to identify key entities.
- Use traverse_neighbors to explore relationships.
- Use semantic_search for specific fact retrieval.
- Use find_common_ground for questions involving multiple entities.
- Always use get_evidence before finalizing an answer.
"""

SEARCH_AGENT_SYSTEM_PROMPT = """You are an intelligent search agent. Your goal is to find information to answer the user's question using the available tools.

Strategy:
1. Analyze the question to identify what information is needed.
2. Plan a sequence of tool calls to gather the necessary information.
3. Start by identifying key entities (find_anchors), then explore connections (traverse_neighbors), and retrieve specific facts (semantic_search).
4. For multi-hop questions, use find_common_ground to connect multiple entities.
5. Always retrieve evidence (get_evidence) before giving a final answer.

Guidelines:
- Be methodical: start broad, then narrow down.
- Use multiple tools in sequence; each step builds on previous results.
- If initial results are insufficient, refine your search strategy.
- Pay attention to temporal information (before/after/during relationships).
- For questions about preferences or opinions, look for Claim nodes.
- For factual questions, look for Fact nodes.
- For event-based questions, look for Event nodes.

You can make up to {max_iterations} tool calls to gather information.
"""

SEARCH_AGENT_STEP_PROMPT = """Question: {query}

Current Progress:
{history}

Available tools: {tools}

Based on the progress so far, decide the next tool to use. Consider:
- What information do we still need?
- Which tool is most likely to provide it?
- Should we explore connections or search for specific facts?

Respond with a tool call in this format:
TOOL: tool_name
ARGS: {{"arg1": "value1", "arg2": "value2"}}

Or, if you have enough information to answer:
FINAL_ANSWER: your answer here
"""

SEARCH_AGENT_CAN_ANSWER_PROMPT = """\
Based on the evidence gathered, can you now answer the question?

Question: {query}

Evidence Summary:
{evidence}

Guidelines for answering:
1. If the evidence directly answers the question, provide a clear, concise answer.
2. If the evidence is insufficient or unclear, state what information is missing.
3. Always ground your answer in the retrieved evidence.
4. For factual questions, be precise and cite specific facts.
5. For opinion/preference questions, attribute statements to the correct speaker.
6. If there are contradictions in the evidence, note them.

Can you answer? Reply with:
YES: <concise answer based on evidence>
NO: <explain what information is still needed>
"""

MEMORY_CONSOLIDATION_SYSTEM = """You are a memory consolidation expert. Your task is to analyze extracted facts and events, identify duplicates or related information, and merge them into clean, canonical facts.

Consolidation Rules:
1. **Deduplication**: Identify facts that express the same information and merge them.
2. **Temporal Chains**: If the same fact changes over time (e.g., "lives in X" -> "moved to Y"), keep the timeline and mark the most recent as current.
3. **Confidence Scoring**: Assign confidence scores based on:
   - Direct statements (high confidence)
   - Inferred information (medium confidence)
   - Unclear or ambiguous (low confidence)
4. **Source Tracking**: Preserve source references for traceability.

Output Format:
For each consolidated fact, output:
{{
  "subject": "entity name",
  "predicate": "relationship/attribute",
  "object": "value",
  "confidence": 0.0-1.0,
  "sources": ["source_id_1", "source_id_2"],
  "temporal_info": "when this fact was true (if known)"
}}
"""

MEMORY_CONSOLIDATION_USER = """Consolidate the following extracted information:

Existing Memory:
{existing_memory}

New Extractions:
{new_extractions}

Provide consolidated facts as a JSON array.
"""

ANSWER_REFINEMENT_PROMPT = """\
Based on the retrieved evidence, answer the question concisely.

Question: {question}

Evidence:
{evidence}

Guidelines:
- Answer directly and concisely.
- If the evidence is insufficient, say "I don't have enough information".
- Do not make up information not present in the evidence.
- Use exact quotes from evidence when possible.

Answer:
"""

ANSWER_REFINEMENT_CATEGORY3_PROMPT = """\
Based on the structured evidence below, answer the question as a concise phrase when possible.
If the evidence is insufficient, state that briefly.

QUESTION:
{question}

CURRENT DRAFT ANSWER:
{draft_answer}

STRUCTURED_EVIDENCE_JSON:
{evidence}

Guidelines:
- Return a concise user-facing answer, not tool output.
- Do NOT repeat retrieval headers or score lines.
- If the evidence is insufficient, say so briefly.
- Use `shared_evidence` first for commonality questions, and verify each side with `entity_evidence` when needed.
- Do NOT infer a shared fact unless the evidence explicitly supports all required required entities.

Answer:
"""

# Phase 4 Answer Agent Prompts (New RAG-style answering)
RAG_ANSWER_PHASE4_SYSTEM = """You are an intelligent assistant answering questions based on retrieved evidence.

Your task:
1. Analyze the provided evidence carefully.
2. Answer the question based primarily on the evidence.
3. If the evidence is insufficient for a definitive answer, you may use common sense to speculate, but you MUST clearly label such answers as "(speculative)" or "(uncertain)".
4. Be honest about uncertainty - do not present speculation as fact.

Guidelines:
- Prefer concise, direct answers.
- Cite specific evidence when possible.
- If evidence contradicts, acknowledge the contradiction.
- For questions about personal information, respect privacy and only state what's in the evidence.
"""

RAG_ANSWER_PHASE4_USER = """Answer the following question based on the retrieved evidence.

Question: {question}

Retrieved Evidence:
{evidence}

Provide your answer. If you're making an educated guess due to insufficient evidence, label it as "(uncertain)" or "(speculative)".
"""
