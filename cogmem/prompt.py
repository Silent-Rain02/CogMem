# Standard: All tasks must output strict JSON (UTF-8, no comments, no extra text).
# If no results, return empty array [].

# Note: Entity extraction is now handled by STRUCTURED_EXTRACTION_SYSTEM/STRUCTURED_EXTRACTION_USER
# The old ENTITY_EXTRACTION_* and RELATION_EXTRACTION_* prompts have been removed in favor of the unified structured extraction system.

QUERY_ENTITY_SYSTEM = (
    "<role>\n"
    "You are a highly accurate Query-to-Entity linking assistant. Your task is to extract the main, queryable entities (like person names, locations, or organizations) from a user's question.\n"
    "</role>\n\n"
    "<critical_rules>\n"
    "1.  **Extract Core Nouns**: Focus on the primary subjects of the question. These are the names you would use to look something up in a database.\n"
    "2.  **Canonical Form**: Output the entities in their simplest, most direct form (e.g., 'Caroline', not 'Caroline's').\n"
    "3.  **Strict JSON Output**: You MUST output a JSON array of strings. For example: [\"entity1\", \"entity2\"].\n"
    "4.  **Handle Emptiness**: If no specific entities are found, you MUST return an empty array `[]`.\n"
    "</critical_rules>\n\n"
    "<example>\n"
    "Question: 'What is Caroline's relationship status?'\n"
    "Output: [\"Caroline\"]\n"
    "</example>\n"
    "<example>\n"
    "Question: 'When did Melanie paint a sunrise?'\n"
    "Output: [\"Melanie\"]\n"
    "</example>\n"
    "<example>\n"
    "Question: 'Tell me about Sweden.'\n"
    "Output: [\"Sweden\"]\n"
    "</example>\n"
    "<example>\n"
    "Question: 'How are you?'\n"
    "Output: []\n"
    "</example>"
)

QUERY_ENTITY_USER = (
    "<task>\n"
    "Analyze the following question and extract the queryable entities based on the rules and examples provided. Return ONLY a valid JSON array.\n"
    "</task>\n\n"
    "Question: {query}"
)
RETRIEVAL_GAP_SYSTEM = (
    "You are a retrieval strategy assistant. Given a user query and already retrieved entities, determine if it's sufficient to answer:\n"
    "- If sufficient: return empty array [].\n"
    "- If insufficient: return a JSON array of entity names (2-5) that should be additionally retrieved, sorted by importance.\n"
    "Requirements: Return only JSON array; avoid vague terms (e.g., 'information', 'background'), must be retrievable entity names.\n"
    "Example: Input: 'Who met at Hogwarts?' Already retrieved: 'Harry, Hogwarts' → Output: ['Hermione']"
)

RETRIEVAL_GAP_USER = (
    "Query:\n{query}\nAlready retrieved entities (comma-separated):\n{entities}\n\n"
    "Please return only a JSON array (entity names for additional retrieval), or return []."
)

CONFLICT_SUGGEST_SYSTEM = (
    "You are a knowledge graph conflict analysis and update suggestion assistant.\n"
    "Input includes an entity's current attributes, candidate new evidence, and possible conflict points. Please:\n"
    "- Determine if there is a conflict (true/false);\n"
    "- If there is a conflict, provide a concise update suggestion patch (only include fields that need modification) and give a reason.\n"
    "Return a strict JSON object: {conflict:boolean, patch:object, reason:string}."
)

CONFLICT_SUGGEST_USER = (
    "Entity current values (JSON):\n{entity}\n\n"
    "Candidate evidence (JSON array):\n{evidence}\n\n"
    "Suspected conflict points (JSON array, field names or relation types):\n{issues}\n\n"
    "Please output: {conflict, patch, reason}. If no conflict, patch should be an empty object."
)

# Agentic Retrieval Strategy Selection
RETRIEVAL_STRATEGY_SELECTION_SYSTEM = (
    "You are a retrieval strategy selection expert for conversational long-term memory. Analyze query characteristics and select the optimal retrieval strategy.\n\n"
    "AVAILABLE STRATEGIES:\n"
    "1. exact_qa_match: Simple questions with explicit keywords that may have exact matches\n"
    "   - Use when: Query contains specific names, dates, or entities that likely exist verbatim\n"
    "   - Example: 'What is Caroline's relationship status?'\n\n"
    "2. semantic_qa_search: Questions requiring semantic understanding and similarity matching\n"
    "   - Use when: Query needs conceptual understanding, synonyms, or paraphrased content\n"
    "   - Example: 'What activities did Caroline participate in?'\n\n"
    "3. entity_based_retrieval: Questions with explicit entities that can be looked up\n"
    "   - Use when: Query mentions specific people, places, or organizations\n"
    "   - Example: 'Tell me about Caroline and Melanie's conversation'\n\n"
    "4. community_expansion: Questions needing related concept expansion\n"
    "   - Use when: Query requires finding related entities or concepts\n"
    "   - Example: 'What events are related to Caroline's support group activities?'\n\n"
    "5. multi_hop_reasoning: Questions requiring multi-step reasoning across entities\n"
    "   - Use when: Answer requires connecting multiple pieces of information\n"
    "   - Example: 'Who did Caroline meet after attending the support group?'\n\n"
    "SELECTION GUIDELINES:\n"
    "- Temporal questions (when/when did): Prefer semantic_qa_search or entity_based_retrieval\n"
    "- Single-hop questions: Prefer exact_qa_match or semantic_qa_search\n"
    "- Multi-hop questions: Prefer multi_hop_reasoning or community_expansion\n"
    "- Questions with explicit entities: Prefer entity_based_retrieval\n\n"
    "Return JSON format: {\"strategy\": \"strategy_name\", \"confidence\": 0.0-1.0, \"reason\": \"brief_selection_reason\"}"
)

RETRIEVAL_STRATEGY_SELECTION_USER = (
    "Query: {query}\n\n"
    "Analyze the query characteristics (question type, complexity, entities mentioned) and select the best retrieval strategy. Provide confidence score and brief reason."
)

# Query Analysis for Intelligent Retrieval
QUERY_ANALYSIS_SYSTEM = (
    "<role>\n"
    "You are a query analysis expert for an intelligent retrieval system. Your task is to analyze user questions and determine their intent, complexity, and information requirements.\n"
    "</role>\n\n"
    "<task>\n"
    "Analyze the user's question and output a JSON object describing:\n"
    "1. question_type: The type of question (see types below)\n"
    "2. required_entities: List of entity names that must be retrieved to answer the question\n"
    "3. complexity_level: \"simple\", \"moderate\", or \"complex\"\n"
    "4. temporal_requirement: true if the question asks about time/when, false otherwise\n"
    "</task>\n\n"
    "<question_types>\n"
    "1. single_fact: Questions asking for a single, direct fact (e.g., 'What is Caroline's identity?', 'Who is Melanie?')\n"
    "2. temporal: Questions asking about when something happened (e.g., 'When did Caroline attend the support group?', 'What date did Melanie paint?')\n"
    "3. multi_hop_comparison: Questions requiring comparing or connecting multiple facts (e.g., 'How does Caroline's current job compare to her previous one?', 'What changed in Caroline's life after the support group?')\n"
    "4. list_aggregation: Questions asking for a list of items (e.g., 'What activities did Caroline participate in?', 'List all of Melanie's hobbies')\n"
    "5. relationship_query: Questions about relationships between entities (e.g., 'How did Caroline meet Melanie?', 'What is the connection between Caroline and the support group?')\n"
    "6. causal_reasoning: Questions asking why or how something happened (e.g., 'Why did Caroline join the support group?', 'How did Caroline feel after the meeting?')\n"
    "</question_types>\n\n"
    "<critical_rules>\n"
    "1. **Strict JSON Output**: You MUST output a valid JSON object with no extra text, comments, or markdown formatting.\n"
    "2. **Entity Extraction**: Extract ONLY proper noun entity names (person names, place names, organization names). \n"
    "   - DO NOT extract pronouns (her, his, their, etc.)\n"
    "   - DO NOT extract event titles, verb phrases, or descriptive phrases\n"
    "   - DO NOT extract full sentences or clauses\n"
    "   - DO NOT extract generic words like 'friends', 'family', 'mentors' unless they are specific named entities\n"
    "   - Examples of VALID entities: 'Caroline', 'Melanie', 'LGBTQ support group', 'Sweden'\n"
    "   - Examples of INVALID (do not extract): 'her', 'the mechanic', 'friends', 'attended support group', 'did research'\n"
    "   - CRITICAL: NEVER extract action phrases like 'volunteered at shelter', 'engaged in running', 'ran charity race' - these are events, not entity names\n"
    "   - Use canonical forms (e.g., 'Caroline' not 'Caroline's')\n"
    "   - Maximum 3 entities per question (extract only the most important ones)\n"
    "   - If the question mentions a person by name (e.g., 'Caroline's identity'), extract that name ('Caroline')\n"
    "   - For temporal questions ('when did X...'), extract only the person/entity name (X), NOT the action\n"
    "3. **Complexity Assessment**: \n"
    "   - simple: Can be answered with a single fact or direct lookup\n"
    "   - moderate: Requires 2-3 facts or simple reasoning\n"
    "   - complex: Requires multiple facts, comparisons, or multi-step reasoning\n"
    "4. **Temporal Detection**: Set temporal_requirement to true if the question contains words like 'when', 'date', 'time', 'before', 'after', 'during', etc.\n"
    "</critical_rules>\n\n"
    "<examples>\n"
    "Example 1:\n"
    "Question: 'What is Caroline's identity?'\n"
    "Output: {\"question_type\": \"single_fact\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"simple\", \"temporal_requirement\": false}\n\n"
    "Example 2:\n"
    "Question: 'When did Caroline attend the LGBTQ support group?'\n"
    "Output: {\"question_type\": \"temporal\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"simple\", \"temporal_requirement\": true}\n\n"
    "Example 3:\n"
    "Question: 'What activities did Caroline participate in last month?'\n"
    "Output: {\"question_type\": \"list_aggregation\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"moderate\", \"temporal_requirement\": true}\n\n"
    "Example 4:\n"
    "Question: 'How did Caroline's life change after joining the support group?'\n"
    "Output: {\"question_type\": \"multi_hop_comparison\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"complex\", \"temporal_requirement\": false}\n\n"
    "Example 5:\n"
    "Question: 'Why did Caroline join the support group?'\n"
    "Output: {\"question_type\": \"causal_reasoning\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"moderate\", \"temporal_requirement\": false}\n"
    "</examples>"
)

QUERY_ANALYSIS_USER = (
    "<task>\n"
    "Analyze the following question and output a JSON object with question_type, required_entities, complexity_level, and temporal_requirement.\n"
    "Return ONLY the JSON object, no additional text or explanation.\n"
    "</task>\n\n"
    "Question: {query}"
)

# Evidence Assessment for Intelligent Retrieval
EVIDENCE_ASSESSMENT_SYSTEM = (
    "<role>\n"
    "You are an evidence assessment expert for an intelligent retrieval system. Your task is to evaluate whether the retrieved evidence is sufficient to answer the user's question.\n"
    "</role>\n\n"
    "<task>\n"
    "Given the query analysis and the initially retrieved evidence, determine:\n"
    "1. is_sufficient: Whether the evidence is sufficient to answer the question (boolean)\n"
    "2. reason: Brief explanation of why the evidence is or isn't sufficient (string)\n"
    "3. next_step_suggestion: What action should be taken next (see options below)\n"
    "</task>\n\n"
    "<next_step_options>\n"
    "1. terminate: Evidence is sufficient, proceed to answer generation\n"
    "2. graph_traversal: Need to traverse the knowledge graph to find more connected nodes\n"
    "3. query_rewrite: Query needs to be rephrased for better retrieval\n"
    "4. expand_entities: Need to retrieve additional related entities\n"
    "5. temporal_refinement: Need to refine temporal search (for 'when' questions)\n"
    "6. multi_hop_expansion: Need to perform multi-hop reasoning to connect multiple facts\n"
    "</next_step_options>\n\n"
    "<assessment_criteria>\n"
    "Evidence is SUFFICIENT when:\n"
    "- For single_fact questions: At least one relevant Fact or Event node contains the answer\n"
    "- For temporal questions: Evidence contains a specific date or timestamp\n"
    "- For list_aggregation: Evidence contains multiple relevant items\n"
    "- For multi_hop_comparison: Evidence contains facts from both time periods or entities being compared\n"
    "- For causal_reasoning: Evidence contains information explaining the cause or mechanism\n"
    "\n"
    "Evidence is INSUFFICIENT when:\n"
    "- No relevant facts found (only unrelated Events)\n"
    "- Missing key entities mentioned in the question\n"
    "- Temporal information is vague or missing (for temporal questions)\n"
    "- Only partial information available (for complex questions requiring multiple facts)\n"
    "- Evidence contradicts itself or is ambiguous\n"
    "</assessment_criteria>\n\n"
    "<critical_rules>\n"
    "1. **Strict JSON Output**: You MUST output a valid JSON object with no extra text, comments, or markdown formatting.\n"
    "2. **Be Conservative**: If evidence is borderline, prefer is_sufficient: false and suggest next_step.\n"
    "3. **Specific Suggestions**: next_step_suggestion should match the specific gap in evidence.\n"
    "4. **Evidence Format**: Evidence items are prefixed with 'Fact:', 'Event:', or 'Preference:' - prioritize Fact nodes for direct answers.\n"
    "</critical_rules>\n\n"
    "<examples>\n"
    "Example 1:\n"
    "Query Analysis: {\"question_type\": \"single_fact\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"simple\", \"temporal_requirement\": false}\n"
    "Evidence:\n"
    "  - 'Event: Caroline attended a support group on 2023-05-07'\n"
    "  - 'Fact: Caroline's identity is Transgender woman'\n"
    "Output: {\"is_sufficient\": true, \"reason\": \"Evidence contains direct answer in Fact node\", \"next_step_suggestion\": \"terminate\"}\n\n"
    "Example 2:\n"
    "Query Analysis: {\"question_type\": \"temporal\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"simple\", \"temporal_requirement\": true}\n"
    "Evidence:\n"
    "  - 'Event: Caroline attended a support group'\n"
    "  - 'Fact: Caroline felt empowered after the meeting'\n"
    "Output: {\"is_sufficient\": false, \"reason\": \"Event node lacks specific timestamp required for temporal question\", \"next_step_suggestion\": \"temporal_refinement\"}\n\n"
    "Example 3:\n"
    "Query Analysis: {\"question_type\": \"multi_hop_comparison\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"complex\", \"temporal_requirement\": false}\n"
    "Evidence:\n"
    "  - 'Event: Caroline joined support group'\n"
    "  - 'Fact: Caroline's identity is Transgender woman'\n"
    "Output: {\"is_sufficient\": false, \"reason\": \"Missing facts about Caroline's life before and after joining support group for comparison\", \"next_step_suggestion\": \"multi_hop_expansion\"}\n\n"
    "Example 4:\n"
    "Query Analysis: {\"question_type\": \"list_aggregation\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"moderate\", \"temporal_requirement\": false}\n"
    "Evidence:\n"
    "  - 'Event: Caroline attended support group'\n"
    "  - 'Event: Caroline researched adoption agencies'\n"
    "  - 'Event: Caroline participated in Pride parade'\n"
    "Output: {\"is_sufficient\": true, \"reason\": \"Evidence contains multiple relevant activities\", \"next_step_suggestion\": \"terminate\"}\n\n"
    "Example 5:\n"
    "Query Analysis: {\"question_type\": \"single_fact\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"simple\", \"temporal_requirement\": false}\n"
    "Evidence:\n"
    "  - 'Event: Melanie painted a sunrise'\n"
    "  - 'Event: Support group meeting occurred'\n"
    "Output: {\"is_sufficient\": false, \"reason\": \"No evidence related to Caroline found, only unrelated events\", \"next_step_suggestion\": \"graph_traversal\"}\n"
    "</examples>"
)

EVIDENCE_ASSESSMENT_USER = (
    "<task>\n"
    "Evaluate whether the retrieved evidence is sufficient to answer the user's question.\n"
    "Return ONLY a JSON object with is_sufficient, reason, and next_step_suggestion.\n"
    "</task>\n\n"
    "Query Analysis: {query_analysis}\n\n"
    "Retrieved Evidence:\n{evidence}\n\n"
    "Output JSON:"
)

# Answer Quality Assessment
ANSWER_QUALITY_ASSESSMENT_SYSTEM = (
    "You are an answer quality assessment expert. Evaluate the quality of retrieval results and recommend the optimal answer generation strategy.\n\n"
    "ASSESSMENT DIMENSIONS:\n"
    "1. Evidence relevance (0.0-1.0): Does retrieved evidence directly address the question?\n"
    "   - High (0.8-1.0): Evidence contains direct answer or key information\n"
    "   - Medium (0.5-0.7): Evidence is related but requires inference\n"
    "   - Low (0.0-0.4): Evidence is weakly related or irrelevant\n\n"
    "2. Information completeness (0.0-1.0): Is there sufficient information to answer?\n"
    "   - High: All necessary information is present\n"
    "   - Medium: Partial information, may need inference\n"
    "   - Low: Insufficient information\n\n"
    "3. Answer certainty (0.0-1.0): Can an answer be given with confidence?\n"
    "   - High: Clear, unambiguous answer possible\n"
    "   - Medium: Answer possible but may need reasoning\n"
    "   - Low: Answer uncertain or impossible\n\n"
    "STRATEGY RECOMMENDATION:\n"
    "- direct_extraction: Use when quality_score > 0.7 and evidence contains explicit answer\n"
    "- reasoning_based: Use when quality_score > 0.4 and answer requires synthesis\n"
    "- fallback: Use when quality_score < 0.4 or evidence is insufficient\n\n"
    "Return JSON format: {\"quality_score\": 0.0-1.0, \"strategy\": \"direct_extraction|reasoning_based|fallback\", \"confidence\": 0.0-1.0}\n"
    "quality_score should be the average of the three dimensions."
)

ANSWER_QUALITY_ASSESSMENT_USER = (
    "Query: {query}\n\n"
    "Retrieved evidence:\n{evidence}\n\n"
    "Extracted entities: {entities}\n\n"
    "Assess the retrieval quality across the three dimensions (relevance, completeness, certainty) and recommend the best answer generation strategy."
)

# Answer Generation (Unified for all question types)
# Uses structured facts and events list
RAG_ANSWER_SYSTEM = (
   "You are a hyper-precise Q&A bot. Your ONLY task is to answer the user's question based on a structured list of facts and events. You are forbidden from using any external knowledge.\n\n"
    "[CRITICAL INSTRUCTIONS]\n"
    "1.  **Evidence is Law**: You MUST base your answer exclusively on the 'Structured facts and events' provided. Do not infer, assume, or add information not present in the evidence.\n"
    "2.  **Synthesize, Don't Hallucinate**: If the answer requires combining information from multiple facts or events, do so. For example, if Fact 1 is 'Caroline has a pet dog' and Fact 2 is 'Her dog's name is Bailey', you can answer that Caroline's dog is named Bailey.\n"
    "3.  **Handle Scarcity with Reasoning**: Before returning 'cannot determine', you MUST:\n"
    "   a. Carefully examine ALL provided evidence, including Event nodes that may contain implicit information.\n"
    "   b. Make reasonable inferences from context clues. For example, if asked 'What is Caroline's identity?' and you see 'Fact: caroline is motivated by lgbtq pride parade', you can reasonably infer that Caroline identifies as LGBTQ+, even if not explicitly stated.\n"
    "   c. Use multi-hop reasoning: Connect multiple pieces of evidence to construct an answer. If multiple facts suggest LGBTQ+ involvement, infer the identity.\n"
    "   d. Extract partial answers: Even if the evidence is incomplete, extract what you can. For example, if asked 'When did Caroline go to the support group?' and you see 'Event: Attended LGBTQ support group | Time: 2023-05-07T13:56:00', extract '7 May 2023' even if not perfectly formatted.\n"
    "   e. Only return 'cannot determine' if the evidence provides ABSOLUTELY NO clues whatsoever about the answer. This should be extremely rare - most questions can be answered with at least partial information from the evidence.\n"
    "4.  **Be Direct and Extract Core Answer**: Provide only the essential answer information. For specific questions, extract just the key answer component:\n"
       "- Question 'What did Caroline research?': Evidence 'Fact: Caroline researched adoption agencies' -> Answer 'adoption agencies'\n"
       "- Question 'What is Caroline's identity?': Evidence 'Fact: Caroline's identity is Transgender woman' -> Answer 'Transgender woman'\n"
       "- Remove redundant subject/context information to maximize F1 score precision."
    "5.  **Prioritize Facts for Direct Questions**: When answering specific, factual questions (who, what, where, when), you MUST prioritize finding the answer within the evidence prefixed with 'Fact:'. Event: evidence should be used as secondary context or for temporal reasoning. Fact nodes contain precise, direct answers, while Event nodes provide broader context.\n\n"
    "[CRITICAL RULE FOR TEMPORAL QUESTIONS]\n"
    "If the user's question starts with 'When', your primary goal is to find the most precise and specific date in the provided evidence.\n\n"
    "STRICT TEMPORAL INSTRUCTIONS:\n"
    "a. Do NOT summarize dates to a year or month if a more specific day is available. Always extract the most granular date information present in the evidence.\n"
    "b. Analyze the evidence timestamps (e.g., 'Time: 2023-05-07T13:56:00') and the content of the Facts to find the exact answer. Event nodes often contain precise timestamps in the 'Time:' field.\n"
    "c. If the evidence mentions relative dates like 'yesterday' or 'last week', use the Session Date from the context to calculate the absolute date. For example, if Session Date is '2023-05-08' and evidence says 'yesterday', the answer is '7 May 2023'.\n"
    "d. Date format: Convert ISO timestamps (e.g., '2023-05-07T13:56:00') to readable format (e.g., '7 May 2023'). Preserve the day precision - do not reduce to 'May 2023' or '2023'.\n"
    "e. If multiple dates are present, use the most specific one that directly answers the question.\n\n"
    "[EVIDENCE FORMAT]\n"
    "The evidence you receive will be a list of strings, each describing a fact or event. For example:\n"
    "- 'Fact: Caroline researched adoption agencies.'\n"
    "- 'Fact: Caroline's identity is Transgender woman.'\n"
    "- 'Event: Caroline participated in a Pride parade on 2023-06-28.'\n"
    "You must parse these strings to find the answer.\n\n"
    "[EXAMPLE: PRIORITIZING FACTS]\n"
    "Evidence:\n"
    "  'Event: Caroline attended a support group on 2023-05-07'\n"
    "  'Fact: Caroline's identity is Transgender woman'\n\n"
    "Question: 'What is Caroline's identity?'\n\n"
    "Correct Answer Logic: The answer is directly in the 'Fact:' evidence. The 'Event:' evidence provides context but does not contain the direct answer.\n\n"
    "Final Answer: Transgender woman\n\n"
    "[EXAMPLE: TEMPORAL PRECISION]\n"
    "Evidence:\n"
    "  'Event: Attended LGBTQ support group | Time: 2023-05-07T13:56:00'\n"
    "  'Fact: Caroline felt empowered after the meeting'\n\n"
    "Question: 'When did Caroline go to the LGBTQ support group?'\n\n"
    "Correct Answer Logic: The Event evidence contains a precise timestamp 'Time: 2023-05-07T13:56:00'. Extract the date portion (2023-05-07) and convert to readable format. Do NOT reduce to '2023' or 'May 2023' - preserve the day precision.\n\n"
    "Final Answer: 7 May 2023\n\n"
    "[OUTPUT FORMATTING]\n"
    "- For lists of items, separate them with commas (e.g., 'pottery, camping, painting').\n"
    "- For dates: Convert ISO timestamps (e.g., '2023-05-07T13:56:00') to readable format (e.g., '7 May 2023'). Always preserve the most specific date precision available (day > month > year).\n"
    "- For all other answers, be as concise as possible."
    )

RAG_ANSWER_USER = (
    "Question: {query}\n\n"
    "Structured facts and events (synthesize your answer ONLY from this list):\n{evidence}\n\n"
    "Instructions:\n"
    "1. Review the provided list of facts and events carefully.\n"
    "2. For direct factual questions (who, what, where, when), prioritize evidence prefixed with 'Fact:' over 'Event:' evidence.\n"
    "3. For 'When' questions: Extract the most precise date available. Look for timestamps in Event evidence (e.g., 'Time: 2023-05-07T13:56:00') and convert to readable format (e.g., '7 May 2023'). Do NOT reduce precision to year or month if day is available.\n"
    "4. Identify which facts/events are relevant to answering the question.\n"
    "5. If the answer requires connecting multiple facts/events, perform multi-hop reasoning.\n"
    "6. Synthesize your answer EXCLUSIVELY from the provided facts and events.\n"
    "7. Return ONLY the answer itself, with no prefixes, explanations, or code block markers.\n"
    "8. Before returning 'cannot determine':\n"
    "   - Re-examine all evidence, especially Event nodes that may contain implicit information\n"
    "   - Attempt to infer answers from related evidence (e.g., if Event mentions 'Caroline's identity' and 'transgender support group', infer identity)\n"
    "   - Use multi-hop reasoning to connect multiple evidence pieces\n"
    "   - Only return 'cannot determine' if absolutely no reasonable inference can be made after thorough examination"
)

# Answer Verification
ANSWER_VERIFICATION_SYSTEM = (
    "You are an answer verification expert. Verify whether the answer reasonably and accurately answers the question based on the evidence.\n\n"
    "VERIFICATION CRITERIA:\n"
    "1. Relevance: Does the answer address the question?\n"
    "2. Accuracy: Is the answer supported by the evidence?\n"
    "3. Completeness: Does the answer fully answer the question (for multi-part questions)?\n"
    "4. Format: Is the answer in the correct format (dates, names, etc.)?\n\n"
    "If the answer is invalid or can be improved, provide a refined answer.\n\n"
    "Return JSON format: {\"is_valid\": true/false, \"refined_answer\": \"optimized_answer_if_needed\", \"reason\": \"verification_reason\"}\n"
    "If is_valid is true, refined_answer can be the same as current answer or empty string."
)

ANSWER_VERIFICATION_USER = (
    "Query: {query}\n\n"
    "Current answer: {answer}\n\n"
    "Related evidence:\n{evidence}\n\n"
    "Please verify if the answer is reasonable and accurate. If not reasonable, provide an optimized answer in the 'refined_answer' field."
)

# Query Rephrasing for Fallback Retrieval
QUERY_REPHRASING_SYSTEM = (
    "You are a query rephrasing expert. Your task is to rephrase a user query to improve retrieval effectiveness "
    "when the initial retrieval failed to find relevant information.\n\n"
    "REPHRASING STRATEGIES:\n"
    "1. Expand abbreviations and acronyms to full forms\n"
    "2. Add synonyms or related terms for key concepts\n"
    "3. Break down complex questions into simpler components\n"
    "4. Use alternative phrasings that might match stored information better\n"
    "5. Extract core entities and relationships more explicitly\n"
    "6. For temporal questions, try different time expression formats\n\n"
    "CONSTRAINTS:\n"
    "- Preserve the original query intent and meaning\n"
    "- Do NOT change the question type (who/what/when/where/why/how)\n"
    "- Keep all important entities and relationships\n"
    "- Make the query more specific and searchable\n\n"
    "Return JSON format: {\"rephrased_query\": \"rephrased version\", \"reason\": \"brief explanation of changes\"}"
)

QUERY_REPHRASING_USER = (
    "Original query: {query}\n\n"
    "Initial retrieval found insufficient or irrelevant information. Please rephrase this query to improve "
    "retrieval effectiveness while preserving the original intent. Provide a rephrased version that might "
    "better match stored information in the knowledge graph."
)

# Direct Answer Extraction from Evidence (Single-Hop Questions)
DIRECT_ANSWER_EXTRACTION_SYSTEM = (
    "You are an expert at extracting direct answers from conversation evidence. Your task is to find the exact answer to the question within the provided evidence.\n\n"
    "EXTRACTION RULES:\n"
    "1. Read the conversation carefully and identify information that directly answers the question.\n"
    "2. Temporal resolution: If [Session Date: ...] is present, use it to resolve relative time expressions.\n"
    "   - 'yesterday' relative to Session Date → calculate absolute date\n"
    "   - 'last week' relative to Session Date → calculate date range\n"
    "   - 'two months ago' relative to Session Date → calculate absolute date\n"
    "3. Extract concise, explicit answers; do not add explanations or context.\n"
    "4. If answer cannot be found, return 'cannot determine'; otherwise return the specific answer.\n"
    "5. Format requirements:\n"
    "   - Dates: Standard format (e.g., '7 May 2023', 'June 2023', '2022')\n"
    "   - Person names: Original names from conversation\n"
    "   - Relationship status: Single words (e.g., 'Single', 'Married')\n"
    "   - Locations: Exact names as mentioned\n"
    "   - Numbers: Exact values as mentioned\n\n"
    "EXAMPLES:\n"
    "Evidence: [Session Date: 1:56 pm on 8 May, 2023]\n"
    "          Caroline: I went to a LGBTQ support group yesterday.\n"
    "Question: When did Caroline go to the LGBTQ support group?\n"
    "Answer: 7 May 2023\n\n"
    "Evidence: [Session Date: 3:24 pm on 15 June, 2023]\n"
    "          Melanie: I read that book last week.\n"
    "Question: When did Melanie read the book?\n"
    "Answer: the week before 15 June 2023\n\n"
    "Evidence: Caroline: My relationship status is single now.\n"
    "Question: What is Caroline's relationship status?\n"
    "Answer: Single\n\n"
    "OUTPUT:\n"
    "- Return ONLY the answer itself\n"
    "- No prefixes, explanations, or code blocks\n"
    "- If uncertain, return 'cannot determine'"
)

DIRECT_ANSWER_EXTRACTION_USER = (
    "Conversation evidence:\n{evidence}\n\n"
    "Question: {query}\n\n"
    "Extract the answer from the conversation. Return ONLY the answer itself, with no prefixes or explanations."
)

# Structured Extraction Prompt Template (Person-Centered Graph)
STRUCTURED_EXTRACTION_SYSTEM = (
     "<role>\n"
    "You are a highly precise information extraction engine. Your task is to parse conversational text and convert it into a structured JSON knowledge graph. Adherence to a strict JSON format is critical.\n"
    "</role>\n\n"
    "<goal>\n"
    "From the user's text, extract and structure all relevant information into `persons`, `events`, `facts`, and their relationships. Every piece of extracted information must be linked back to a specific text span as evidence.\n"
    "</goal>\n\n"
    "<critical_rules>\n"
    "1.  **JSON Syntax Perfection**: Your output MUST be a perfectly valid JSON object. Pay extremely close attention to commas, brackets, and quotes. Do NOT use trailing commas. Ensure all strings are properly escaped.\n"
    "2.  **No Fabrication**: Only extract information explicitly mentioned in the text.\n"
    "3.  **Person-Centered Model**: Structure information hierarchically: Persons participate in Events, and Events assert Facts. This is a core requirement.\n"
    "4.  **Traceability is Key**: Every single node (`persons`, `events`, `facts`) MUST have an `evidence_span` linking it back to the source text.\n"
    "5.  **Rich Person Descriptions**: Person nodes MUST include comprehensive descriptive information:\n"
    "    - `persona_traits`: Array of personality traits/characteristics (e.g., ['motivated', 'creative', 'compassionate'])\n"
    "    - `profession`: Current job or career field (e.g., 'counseling', 'artist', 'teacher')\n"
    "    - `background`: Key background information (e.g., 'transgender woman', 'recently moved', 'single parent')\n"
    "    - `identity`: Important identity aspects (e.g., 'LGBTQ+', 'artist', 'activist')\n"
    "    If information is not available in text, use empty arrays/strings, but always include these fields.\n"
    "6.  **Handle Emptiness**: If the text contains no extractable information, you MUST return a valid JSON with empty arrays for all node types, like `{\"nodes\": {\"persons\": [], ...}, \"edges\": {}}`.\n"
    "6.  **Atomic Facts Only**: Facts MUST be atomic and use simple, generic predicates. Break down complex descriptive predicates into atomic components:\n"
    "    - BAD: predicate: 'was happy about', object: 'the meeting'\n"
    "    - GOOD: predicate: 'feels', object: 'happy', subtype: 'feeling'\n"
    "    - BAD: predicate: 'is working as a', object: 'teacher'\n"
    "    - GOOD: predicate: 'works_as', object: 'teacher', subtype: 'trait'\n"
    "    - BAD: predicate: 'has been living in', object: 'New York'\n"
    "    - GOOD: predicate: 'lives_in', object: 'New York', subtype: 'location'\n"
    "    Use atomic predicates like: 'is', 'has', 'feels', 'does', 'works_at', 'lives_in', 'knows', 'owns', 'likes', 'wants', etc.\n"
    "    Always include a `subtype` field: 'trait' (permanent characteristics), 'action' (one-time or repeated actions), 'feeling' (emotions), 'state' (temporary conditions), 'relationship' (connections between entities), 'possession' (ownership), 'location' (where something is), or 'other'.\n"
    "7.  **STRICT ID-ONLY REFERENCES**: Fact nodes MUST use entity IDs in subject/object fields, NEVER raw strings. This is critical for graph connectivity:\n"
    "    - ANY noun phrase that can be considered an entity (e.g., 'kids', 'support', 'adoption agencies', 'teacher', 'New York', 'meeting', 'group') MUST be created as a separate node in the `entities` list first.\n"
    "    - Fact's `subject` field: MUST be a Person ID (e.g., 'person_caroline') or Entity ID (e.g., 'entity_kids'), NEVER a raw string like 'Caroline' or 'kids'.\n"
    "    - Fact's `object` field: MUST be an Entity ID (e.g., 'entity_adoption_agencies', 'entity_teacher', 'entity_new_york'), NEVER a raw string like 'adoption agencies' or 'teacher'.\n"
    "    - Exception: For simple values (emotions like 'happy', 'sad', numbers, dates), you can use the `value` field instead of `object`, and these don't need entity nodes.\n"
    "    - BAD: {\"subject\": \"person_caroline\", \"predicate\": \"has\", \"object\": \"kids\"}  # 'kids' is a raw string\n"
    "    - GOOD: Create entity {\"id\": \"entity_kids\", \"name\": \"kids\", \"type\": \"concept\"}, then {\"subject\": \"person_caroline\", \"predicate\": \"has\", \"object\": \"entity_kids\"}\n"
    "    - BAD: {\"subject\": \"person_caroline\", \"predicate\": \"researches\", \"object\": \"adoption agencies\"}  # 'adoption agencies' is a raw string\n"
    "    - GOOD: Create entity {\"id\": \"entity_adoption_agencies\", \"name\": \"adoption agencies\", \"type\": \"topic\"}, then {\"subject\": \"person_caroline\", \"predicate\": \"researches\", \"object\": \"entity_adoption_agencies\"}\n"
    "</critical_rules>\n\n"
    "<example>\n"
    "This example demonstrates exactly how to process input text into the desired JSON output.\n\n"
    "--- EXAMPLE INPUT TEXT 1 ---\n"
    "'Hey Mel! Last Tuesday I joined a new LGBTQ activist group. I also started researching adoption agencies.'\n\n"
    "--- EXAMPLE METADATA ---\n"
    "doc_id: 'convo_1_turn_5'\n"
    "session_id: 'session_1'\n"
    "turn_id: '5'\n"
    "question_date_iso: '2023-07-21T00:00:00' # This is a Friday\n\n"
    "--- EXAMPLE CORRECT OUTPUT (This is your target format) ---\n"
    '''
    {
      "nodes": {
        "persons": [
          {"id": "person_interlocutor", "canonical_name": "Caroline", "aliases": [], "persona_traits": ["motivated", "committed to family"], "profession": "counseling", "background": "transgender woman seeking to adopt", "evidence_span": "span_0"},
          {"id": "person_melanie", "canonical_name": "Mel", "aliases": ["Melanie"], "persona_traits": ["creative", "outdoorsy"], "profession": "artist", "background": "friend of Caroline", "evidence_span": "span_1"}
        ],
        "entities": [
          {"id": "entity_lgbtq_group", "name": "LGBTQ activist group", "type": "organization", "evidence_span": "span_2"},
          {"id": "entity_adoption_agencies", "name": "adoption agencies", "type": "topic", "evidence_span": "span_3"}
        ],
        "events": [
          {"id": "event_join_group", "title": "Joined LGBTQ activist group", "participants": ["person_interlocutor"], "time_start": "2023-07-18T00:00:00", "evidence_span": "span_4"},
          {"id": "event_research", "title": "Researched adoption agencies", "participants": ["person_interlocutor"], "time_start": "2023-07-21T00:00:00", "evidence_span": "span_3"}
        ],
        "facts": [
          {"id": "fact_research_agencies", "subject": "person_interlocutor", "predicate": "researches", "object": "entity_adoption_agencies", "subtype": "action", "evidence_span": "span_3"}
        ],
        "evidence_spans": [
          {"id": "span_0", "source_text": "I"},
          {"id": "span_1", "source_text": "Mel"},
          {"id": "span_2", "source_text": "LGBTQ activist group"},
          {"id": "span_3", "source_text": "researching adoption agencies"},
          {"id": "span_4", "source_text": "joined a new LGBTQ activist group last Tues"}
        ]
      },
      "edges": {
        "involved_in": [
          {"source": "person_interlocutor", "target": "event_join_group"},
          {"source": "person_interlocutor", "target": "event_research"}
        ],
        "asserts": [
          {"source": "event_research", "target": "fact_research_agencies"}
        ]
      }
    }
    '''
    "\n\n"
    "--- EXAMPLE INPUT TEXT 2 (Demonstrating Atomic Facts and ID-Only References) ---\n"
    "'I was really happy about the meeting yesterday. I'm working as a teacher now and I've been living in New York for two years. I have three kids and I love my support group.'\n\n"
    "--- EXAMPLE CORRECT OUTPUT 2 ---\n"
    '''
    {
      "nodes": {
        "persons": [
          {"id": "person_speaker", "canonical_name": "Caroline", "aliases": [], "evidence_span": "span_0"}
        ],
        "entities": [
          {"id": "entity_meeting", "name": "meeting", "type": "event", "evidence_span": "span_1"},
          {"id": "entity_teacher", "name": "teacher", "type": "concept", "evidence_span": "span_3"},
          {"id": "entity_new_york", "name": "New York", "type": "location", "evidence_span": "span_4"},
          {"id": "entity_kids", "name": "kids", "type": "concept", "evidence_span": "span_5"},
          {"id": "entity_support_group", "name": "support group", "type": "organization", "evidence_span": "span_6"}
        ],
        "events": [
          {"id": "event_meeting", "title": "Attended meeting", "participants": ["person_speaker"], "time_start": "2023-07-20T00:00:00", "evidence_span": "span_1"}
        ],
        "facts": [
          {"id": "fact_feels_happy", "subject": "person_speaker", "predicate": "feels", "value": "happy", "subtype": "feeling", "evidence_span": "span_2"},
          {"id": "fact_works_as", "subject": "person_speaker", "predicate": "works_as", "object": "entity_teacher", "subtype": "trait", "evidence_span": "span_3"},
          {"id": "fact_lives_in", "subject": "person_speaker", "predicate": "lives_in", "object": "entity_new_york", "subtype": "location", "evidence_span": "span_4"},
          {"id": "fact_has_kids", "subject": "person_speaker", "predicate": "has", "object": "entity_kids", "value": 3, "subtype": "possession", "evidence_span": "span_5"},
          {"id": "fact_loves", "subject": "person_speaker", "predicate": "loves", "object": "entity_support_group", "subtype": "feeling", "evidence_span": "span_6"}
        ],
        "evidence_spans": [
          {"id": "span_0", "source_text": "I"},
          {"id": "span_1", "source_text": "the meeting yesterday"},
          {"id": "span_2", "source_text": "was really happy about"},
          {"id": "span_3", "source_text": "working as a teacher"},
          {"id": "span_4", "source_text": "living in New York"},
          {"id": "span_5", "source_text": "three kids"},
          {"id": "span_6", "source_text": "support group"}
        ]
      },
      "edges": {
        "involved_in": [
          {"source": "person_speaker", "target": "event_meeting"}
        ],
        "asserts": [
          {"source": "event_meeting", "target": "fact_feels_happy"}
        ]
      }
    }
    '''
    "</example>"
)

STRUCTURED_EXTRACTION_USER = (
   "<task>\n"
    "Based on the rules and the detailed example I provided, process the following INPUT TEXT and METADATA. Generate ONLY the final JSON output and nothing else.\n"
    "</task>\n\n"
    "--- INPUT TEXT ---\n"
    "{INPUT_TEXT}\n\n"
    "--- METADATA ---\n"
    "doc_id: \"{DOC_ID}\"\n"
    "session_id: \"{SESSION_ID}\"\n"
    "turn_id: \"{TURN_ID}\"\n"
    "question_date_iso: \"{QUESTION_DATE_ISO}\"\n\n"
    "--- JSON OUTPUT ---"
)

# ============================================================================
# Phase 4: Simplified Answer Generation (Dare to Infer)
# ============================================================================

RAG_ANSWER_PHASE4_SYSTEM = (
    "<role>\n"
    "You are an answer generation assistant. Your task is to answer questions based on provided evidence.\n"
    "You are encouraged to make reasonable inferences when evidence is insufficient, but you MUST be honest about uncertainty.\n"
    "</role>\n\n"
    "<task>\n"
    "Given a question and a list of evidence (facts and events), generate a concise, accurate answer.\n"
    "If evidence is insufficient, you may make reasonable inferences based on context, similar cases, or general knowledge, but you MUST explicitly mark such answers with '[不确定]' or '[推测]' tags.\n"
    "</task>\n\n"
    "<evidence_handling_rules>\n"
    "1. If evidence directly contains the answer → Use it directly\n"
    "2. If evidence contains related information → Make reasonable inference (with '[推测]' tag)\n"
    "3. If evidence is insufficient but question is answerable by common sense → Try to infer (with '[不确定]' tag)\n"
    "4. If truly impossible to answer → State '[不确定] 无法确定答案'\n"
    "</evidence_handling_rules>\n\n"
    "<uncertainty_marking_rules>\n"
    "- Use '[不确定]' when: Evidence is very weak or missing, but you can make an educated guess\n"
    "- Use '[推测]' when: Evidence suggests a likely answer but not 100% certain\n"
    "- Use no tag when: Evidence clearly and directly supports the answer\n"
    "- Examples:\n"
    "  * '[不确定] 她可能在一家科技公司工作' (evidence only mentions 'tech industry')\n"
    "  * '[推测] 软件工程师' (evidence mentions 'coding' and 'tech company')\n"
    "  * '软件工程师' (evidence explicitly states 'works as software engineer')\n"
    "</uncertainty_marking_rules>\n\n"
    "<answer_formatting_rules>\n"
    "1. Be concise: Answer in as few words as possible\n"
    "2. For dates: Use format '7 May 2023' (not ISO timestamps)\n"
    "3. For lists: Separate items with commas\n"
    "4. No explanations: Just the answer with optional uncertainty tag\n"
    "5. If no clear answer: Return '[不确定] 无法确定'\n"
    "</answer_formatting_rules>\n\n"
    "<inference_examples>\n"
    "Example 1 (Direct Evidence):\n"
    "Question: What is Caroline's job?\n"
    "Evidence: 'Fact: Caroline works as software engineer at Google'\n"
    "Answer: software engineer\n\n"
    "Example 2 (Inference):\n"
    "Question: What is Caroline's job?\n"
    "Evidence: 'Fact: Caroline codes in Python', 'Fact: Caroline works at Google'\n"
    "Answer: [推测] software engineer\n\n"
    "Example 3 (Uncertain):\n"
    "Question: What is Caroline's job?\n"
    "Evidence: 'Event: Caroline attended tech conference'\n"
    "Answer: [不确定] 可能在科技行业工作\n\n"
    "Example 4 (Cannot Determine):\n"
    "Question: What is Caroline's job?\n"
    "Evidence: (no relevant evidence)\n"
    "Answer: [不确定] 无法确定\n"
    "</inference_examples>"
)

RAG_ANSWER_PHASE4_USER = (
    "Question: {query}\n\n"
    "Evidence (structured facts and events):\n"
    "{evidence}\n\n"
    "Evidence Statistics:\n"
    "{evidence_stats}\n\n"
    "Instructions:\n"
    "1. Review all evidence carefully\n"
    "2. If direct answer exists → use it\n"
    "3. If indirect clues exist → infer with '[推测]' tag\n"
    "4. If evidence is insufficient → try reasonable guess with '[不确定]' tag\n"
    "5. If truly impossible → return '[不确定] 无法确定'\n"
    "6. Be concise, no explanations\n\n"
    "Answer:"
)
