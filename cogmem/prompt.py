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
# QUERY_ANALYSIS_SYSTEM = (
#     "<role>\n"
#     "You are a query analysis expert for an intelligent retrieval system. Your task is to analyze user questions and determine their intent, complexity, and information requirements.\n"
#     "</role>\n\n"
#     "<task>\n"
#     "Analyze the user's question and output a JSON object describing:\n"
#     "1. question_type: The type of question (see types below)\n"
#     "2. required_entities: List of entity names that must be retrieved to answer the question\n"
#     "3. complexity_level: \"simple\", \"moderate\", or \"complex\"\n"
#     "4. temporal_requirement: true if the question asks about time/when, false otherwise\n"
#     "</task>\n\n"
#     "<question_types>\n"
#     "1. single_fact: Questions asking for a single, direct fact (e.g., 'What is Caroline's identity?', 'Who is Melanie?')\n"
#     "2. temporal: Questions asking about when something happened (e.g., 'When did Caroline attend the support group?', 'What date did Melanie paint?')\n"
#     "3. multi_hop_comparison: Questions requiring comparing or connecting multiple facts (e.g., 'How does Caroline's current job compare to her previous one?', 'What changed in Caroline's life after the support group?')\n"
#     "4. list_aggregation: Questions asking for a list of items (e.g., 'What activities did Caroline participate in?', 'List all of Melanie's hobbies')\n"
#     "5. relationship_query: Questions about relationships between entities (e.g., 'How did Caroline meet Melanie?', 'What is the connection between Caroline and the support group?')\n"
#     "6. causal_reasoning: Questions asking why or how something happened (e.g., 'Why did Caroline join the support group?', 'How did Caroline feel after the meeting?')\n"
#     "</question_types>\n\n"
#     "<critical_rules>\n"
#     "1. **Strict JSON Output**: You MUST output a valid JSON object with no extra text, comments, or markdown formatting.\n"
#     "2. **Entity Extraction**: Extract ONLY proper noun entity names (person names, place names, organization names). \n"
#     "   - DO NOT extract pronouns (her, his, their, etc.)\n"
#     "   - DO NOT extract event titles, verb phrases, or descriptive phrases\n"
#     "   - DO NOT extract full sentences or clauses\n"
#     "   - DO NOT extract generic words like 'friends', 'family', 'mentors' unless they are specific named entities\n"
#     "   - Examples of VALID entities: 'Caroline', 'Melanie', 'LGBTQ support group', 'Sweden'\n"
#     "   - Examples of INVALID (do not extract): 'her', 'the mechanic', 'friends', 'attended support group', 'did research'\n"
#     "   - CRITICAL: NEVER extract action phrases like 'volunteered at shelter', 'engaged in running', 'ran charity race' - these are events, not entity names\n"
#     "   - Use canonical forms (e.g., 'Caroline' not 'Caroline's')\n"
#     "   - Maximum 3 entities per question (extract only the most important ones)\n"
#     "   - If the question mentions a person by name (e.g., 'Caroline's identity'), extract that name ('Caroline')\n"
#     "   - For temporal questions ('when did X...'), extract only the person/entity name (X), NOT the action\n"
#     "3. **Complexity Assessment**: \n"
#     "   - simple: Can be answered with a single fact or direct lookup\n"
#     "   - moderate: Requires 2-3 facts or simple reasoning\n"
#     "   - complex: Requires multiple facts, comparisons, or multi-step reasoning\n"
#     "4. **Temporal Detection**: Set temporal_requirement to true if the question contains words like 'when', 'date', 'time', 'before', 'after', 'during', etc.\n"
#     "</critical_rules>\n\n"
#     "<examples>\n"
#     "Example 1:\n"
#     "Question: 'What is Caroline's identity?'\n"
#     "Output: {\"question_type\": \"single_fact\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"simple\", \"temporal_requirement\": false}\n\n"
#     "Example 2:\n"
#     "Question: 'When did Caroline attend the LGBTQ support group?'\n"
#     "Output: {\"question_type\": \"temporal\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"simple\", \"temporal_requirement\": true}\n\n"
#     "Example 3:\n"
#     "Question: 'What activities did Caroline participate in last month?'\n"
#     "Output: {\"question_type\": \"list_aggregation\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"moderate\", \"temporal_requirement\": true}\n\n"
#     "Example 4:\n"
#     "Question: 'How did Caroline's life change after joining the support group?'\n"
#     "Output: {\"question_type\": \"multi_hop_comparison\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"complex\", \"temporal_requirement\": false}\n\n"
#     "Example 5:\n"
#     "Question: 'Why did Caroline join the support group?'\n"
#     "Output: {\"question_type\": \"causal_reasoning\", \"required_entities\": [\"Caroline\"], \"complexity_level\": \"moderate\", \"temporal_requirement\": false}\n"
#     "</examples>"
# )
QUERY_ANALYSIS_SYSTEM = (
    "<role>\n"
    "You are a Query Analyst. Your job is to break down a user question into search parameters.\n"
    "</role>\n\n"
    "<task>\n"
    "Output a JSON object with: `question_type`, `required_entities`, `complexity_level`, `temporal_requirement`.\n"
    "</task>\n\n"
    "<critical_rules>\n"
    "1. **Entity Extraction (BROADENED SCOPE)**:\n"
    "   - Extract Person names (e.g., 'Caroline', 'Melanie').\n"
    "   - Extract **Activities/Events** (e.g., 'camping', 'charity race', 'support group').\n"
    "   - Extract **Topics** (e.g., 'career', 'education', 'relationship').\n"
    "   - If the user asks 'When is the camping?', extract ['camping'].\n"
    "2. **Question Types**:\n"
    "   - `temporal`: When, what date, how long.\n"
    "   - `single_fact`: What is X, Who is Y.\n"
    "   - `multi_hop`: Connects two pieces of info (e.g. 'When did the person who likes art go camping?').\n"
    "</critical_rules>\n\n"
    "<example>\n"
    "Q: 'When did Melanie run a charity race?'\n"
    "JSON: {\n"
    "  \"question_type\": \"temporal\",\n"
    "  \"required_entities\": [\"Melanie\", \"charity race\"],\n"
    "  \"complexity_level\": \"moderate\",\n"
    "  \"temporal_requirement\": true\n"
    "}\n"
    "</example>"
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
# 
RAG_ANSWER_SYSTEM = (
    "<role>\n"
    "You are an intelligent Memory Assistant. You answer questions based on retrieved history.\n"
    "You strike a balance between accuracy and helpfulness. You prefer giving a likely answer with a confidence score over refusing to answer.\n"
    "</role>\n\n"
    "<input_context>\n"
    "You will receive a list of Evidence (Facts and Events) containing timestamps.\n"
    "</input_context>\n\n"
    "<answering_rules>\n"
    "1. **Direct Answer First**: If the evidence explicitly states the answer, output it directly.\n"
    "2. **Reasonable Inference**: If evidence says 'Caroline planned to camp in June' and the question is 'When did she camp?', infer 'June 2023' (marked as [Likely]).\n"
    "3. **Temporal Logic**:\n"
    "   - If you see `Event: Camping | Time: 2023-06-XX` and question is 'When?', answer 'June 2023'.\n"
    "   - If exact day is missing, answer with the Month/Year available.\n"
    "   - Do NOT say 'cannot determine' just because the exact day is missing.\n"
    "4. **Formatting**:\n"
    "   - Output ONLY the answer string.\n"
    "   - Date format: 'D Month YYYY' (e.g., '7 May 2023').\n"
    "   - If uncertainty exists, prefix with '[Estimated]'.\n"
    "5. **Fallback**: Only return 'cannot determine' if there is absolutely NO relevant entity or event in the evidence.\n"
    "</answering_rules>\n\n"
    "<example>\n"
    "Evidence: [Event: Went camping | Time: 2023-06-XX]\n"
    "Q: When did she go camping?\n"
    "A: June 2023\n"
    "</example>"
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
# DIRECT_ANSWER_EXTRACTION_SYSTEM = (
#     "You are an expert at extracting direct answers from conversation evidence. Your task is to find the exact answer to the question within the provided evidence.\n\n"
#     "EXTRACTION RULES:\n"
#     "1. Read the conversation carefully and identify information that directly answers the question.\n"
#     "2. Temporal resolution: If [Session Date: ...] is present, use it to resolve relative time expressions.\n"
#     "   - 'yesterday' relative to Session Date → calculate absolute date\n"
#     "   - 'last week' relative to Session Date → calculate date range\n"
#     "   - 'two months ago' relative to Session Date → calculate absolute date\n"
#     "3. Extract concise, explicit answers; do not add explanations or context.\n"
#     "4. If answer cannot be found, return 'cannot determine'; otherwise return the specific answer.\n"
#     "5. Format requirements:\n"
#     "   - Dates: Standard format (e.g., '7 May 2023', 'June 2023', '2022')\n"
#     "   - Person names: Original names from conversation\n"
#     "   - Relationship status: Single words (e.g., 'Single', 'Married')\n"
#     "   - Locations: Exact names as mentioned\n"
#     "   - Numbers: Exact values as mentioned\n\n"
#     "EXAMPLES:\n"
#     "Evidence: [Session Date: 1:56 pm on 8 May, 2023]\n"
#     "          Caroline: I went to a LGBTQ support group yesterday.\n"
#     "Question: When did Caroline go to the LGBTQ support group?\n"
#     "Answer: 7 May 2023\n\n"
#     "Evidence: [Session Date: 3:24 pm on 15 June, 2023]\n"
#     "          Melanie: I read that book last week.\n"
#     "Question: When did Melanie read the book?\n"
#     "Answer: the week before 15 June 2023\n\n"
#     "Evidence: Caroline: My relationship status is single now.\n"
#     "Question: What is Caroline's relationship status?\n"
#     "Answer: Single\n\n"
#     "OUTPUT:\n"
#     "- Return ONLY the answer itself\n"
#     "- No prefixes, explanations, or code blocks\n"
#     "- If uncertain, return 'cannot determine'"
# )
DIRECT_ANSWER_EXTRACTION_SYSTEM = (
    "Extract the concise answer from the provided evidence text.\n"
    "Rules:\n"
    "- If the question asks 'When', look for time expressions relative to the Session Date.\n"
    "- Output ONLY the final value (e.g., '7 May 2023', 'Psychology', 'Single').\n"
    "- Do not form a sentence.\n"
    "- If evidence implies the answer, output it.\n"
    "- If truly unknown, return 'cannot determine'."
)

DIRECT_ANSWER_EXTRACTION_USER = (
    "Conversation evidence:\n{evidence}\n\n"
    "Question: {query}\n\n"
    "Extract the answer from the conversation. Return ONLY the answer itself, with no prefixes or explanations."
)

# Structured Extraction Prompt Template (Person-Centered Graph)
# STRUCTURED_EXTRACTION_SYSTEM = (
#      "<role>\n"
#     "You are a highly precise information extraction engine. Your task is to parse conversational text and convert it into a structured JSON knowledge graph. Adherence to a strict JSON format is critical.\n"
#     "</role>\n\n"
#     "<goal>\n"
#     "From the user's text, extract and structure all relevant information into `persons`, `events`, `facts`, and their relationships. Every piece of extracted information must be linked back to a specific text span as evidence.\n"
#     "</goal>\n\n"
#     "<critical_rules>\n"
#     "1.  **JSON Syntax Perfection**: Your output MUST be a perfectly valid JSON object. Pay extremely close attention to commas, brackets, and quotes. Do NOT use trailing commas. Ensure all strings are properly escaped.\n"
#     "2.  **No Fabrication**: Only extract information explicitly mentioned in the text.\n"
#     "3.  **Person-Centered Model**: Structure information hierarchically: Persons participate in Events, and Events assert Facts. This is a core requirement.\n"
#     "4.  **Traceability is Key**: Every single node (`persons`, `events`, `facts`) MUST have an `evidence_span` linking it back to the source text.\n"
#     "5.  **Rich Person Descriptions**: Person nodes MUST include comprehensive descriptive information:\n"
#     "    - `persona_traits`: Array of personality traits/characteristics (e.g., ['motivated', 'creative', 'compassionate'])\n"
#     "    - `profession`: Current job or career field (e.g., 'counseling', 'artist', 'teacher')\n"
#     "    - `background`: Key background information (e.g., 'transgender woman', 'recently moved', 'single parent')\n"
#     "    - `identity`: Important identity aspects (e.g., 'LGBTQ+', 'artist', 'activist')\n"
#     "    If information is not available in text, use empty arrays/strings, but always include these fields.\n"
#     "6.  **CRITICAL: Event vs Fact Distinction - THIS IS THE MOST IMPORTANT RULE FOR CONVERSATIONAL DATA**:\n"
#     "    \n"
#     "    **THE FUNDAMENTAL PRINCIPLE**:\n"
#     "    - Events = ACTIONS that HAPPENED (past tense verbs describing what someone DID)\n"
#     "    - Facts = STATES, TRAITS, RELATIONSHIPS, or PREFERENCES that are TRUE (present tense or timeless)\n"
#     "    \n"
#     "    **MANDATORY DECISION TREE**:\n"
#     "    1. Does the text contain a PAST TENSE ACTION VERB? (went, attended, joined, visited, participated, volunteered, met, saw, helped, started, finished, etc.)\n"
#     "       → YES: Extract as EVENT (even if no explicit time is mentioned)\n"
#     "       → NO: Continue to step 2\n"
#     "    \n"
#     "    2. Does it describe a STATE, TRAIT, or PREFERENCE? (is, has, likes, works_as, lives_in, feels, etc.)\n"
#     "       → YES: Extract as FACT\n"
#     "       → NO: If it's an action without past tense, still extract as EVENT\n"
#     "    \n"
#     "    **CRITICAL RULE FOR CONVERSATIONAL DATA**:\n"
#     "    When you see dialogue like 'Caroline said: I went camping', you MUST:\n"
#     "    - Extract 'Caroline went camping' as an EVENT (the actual action)\n"
#     "    - IGNORE 'Caroline said' (the speech act is just delivery, not content)\n"
#     "    - Extract the time 'June' and associate it with the Event\n"
#     "    \n"
#     "    **Event Extraction Criteria (MUST extract as Event if ANY applies):**\n"
#     "    ✓ Past tense action verbs: 'went', 'attended', 'joined', 'visited', 'participated', 'volunteered', 'met', 'saw', 'helped', 'started', 'finished', 'completed', 'researched', 'organized', 'created', 'ran', 'walked', 'drove', 'painted', 'wrote', 'read', 'cooked', 'ate', 'slept', 'woke', 'left', 'arrived', 'returned'\n"
#     "    ✓ Actions with time references: 'yesterday', 'last week', 'on Tuesday', 'in June', 'recently', 'last month', 'on Monday'\n"
#     "    ✓ Actions describing participation: 'went to support group', 'attended parade', 'joined program', 'visited center'\n"
#     "    ✓ Actions describing activities: 'went camping', 'painted a picture', 'ran a marathon', 'organized event'\n"
#     "    ✓ Progressive actions: 'started researching', 'began working', 'continued studying'\n"
#     "    ✓ Even without explicit time, if it's a specific past action → Event\n"
#     "    \n"
#     "    **Fact Extraction Criteria (MUST extract as Fact):**\n"
#     "    ✓ Permanent characteristics: 'is a teacher', 'works as engineer', 'lives in New York' (present tense states)\n"
#     "    ✓ Preferences/states: 'likes hiking', 'loves music', 'enjoys reading' (present tense preferences)\n"
#     "    ✓ Possessions: 'has three kids', 'owns a car', 'has a dog' (present tense possession)\n"
#     "    ✓ Feelings/emotions: 'feels happy', 'is sad', 'worries about' (present tense states)\n"
#     "    ✓ Relationships: 'knows John', 'is friends with', 'married to' (present tense relationships)\n"
#     "    \n"
#     "    **Examples - CORRECT Event Extraction:**\n"
#     "    - 'Caroline said: I went camping in June' → Event: {title: 'Went camping', participants: ['person_caroline'], time_start: '2023-06-XX'}\n"
#     "    - 'I joined a new LGBTQ activist group last Tuesday' → Event: {title: 'Joined LGBTQ activist group', participants: ['person_interlocutor'], time_start: '2023-07-18'}\n"
#     "    - 'Caroline attended the support group' → Event: {title: 'Attended support group', participants: ['person_caroline']}\n"
#     "    - 'She volunteered at the shelter' → Event: {title: 'Volunteered at shelter', participants: ['person_she']}\n"
#     "    - 'I started researching adoption agencies' → Event: {title: 'Started researching adoption agencies', participants: ['person_interlocutor']}\n"
#     "    \n"
#     "    **Examples - CORRECT Fact Extraction:**\n"
#     "    - 'Caroline is a teacher' → Fact: {subject: 'person_caroline', predicate: 'works_as', object: 'entity_teacher', subtype: 'trait'}\n"
#     "    - 'Caroline likes hiking' → Fact: {subject: 'person_caroline', predicate: 'likes', object: 'entity_hiking', subtype: 'preference'}\n"
#     "    - 'Caroline has three kids' → Fact: {subject: 'person_caroline', predicate: 'has', object: 'entity_kids', subtype: 'possession'}\n"
#     "    - 'Caroline feels happy' → Fact: {subject: 'person_caroline', predicate: 'feels', value: 'happy', subtype: 'feeling'}\n"
#     "    \n"
#     "    **Event Requirements (MANDATORY):**\n"
#     "    - Every Event MUST have a `title` describing the action (e.g., 'Went camping', 'Attended support group', 'Started researching')\n"
#     "    - Every Event MUST have at least one `participant` (the person who performed the action)\n"
#     "    - Every Event SHOULD have `time_start` if temporal information is available (even if approximate like '2023-06-XX')\n"
#     "    - For conversational data without explicit time, infer from context (session date, 'yesterday', 'last week', etc.)\n"
#     "    - Use temporal parser to normalize time expressions (e.g., 'last Tuesday' → ISO date)\n"
#     "7.  **Handle Emptiness**: If the text contains no extractable information, you MUST return a valid JSON with empty arrays for all node types, like `{\"nodes\": {\"persons\": [], ...}, \"edges\": {}}`.\n"
#     "8.  **Atomic Facts Only**: Facts MUST be atomic and use simple, generic predicates. Break down complex descriptive predicates into atomic components:\n"
#     "    - BAD: predicate: 'was happy about', object: 'the meeting'\n"
#     "    - GOOD: predicate: 'feels', object: 'happy', subtype: 'feeling'\n"
#     "    - BAD: predicate: 'is working as a', object: 'teacher'\n"
#     "    - GOOD: predicate: 'works_as', object: 'teacher', subtype: 'trait'\n"
#     "    - BAD: predicate: 'has been living in', object: 'New York'\n"
#     "    - GOOD: predicate: 'lives_in', object: 'New York', subtype: 'location'\n"
#     "    Use atomic predicates like: 'is', 'has', 'feels', 'does', 'works_at', 'lives_in', 'knows', 'owns', 'likes', 'wants', etc.\n"
#     "    Always include a `subtype` field: 'trait' (permanent characteristics), 'action' (one-time or repeated actions), 'feeling' (emotions), 'state' (temporary conditions), 'relationship' (connections between entities), 'possession' (ownership), 'location' (where something is), or 'other'.\n"
#     "    **NOTE**: If a Fact has subtype='action' AND has temporal information, it should have been extracted as an Event instead. Review your extraction.\n"
#     "9.  **STRICT ID-ONLY REFERENCES**: Fact nodes MUST use entity IDs in subject/object fields, NEVER raw strings. This is critical for graph connectivity:\n"
#     "    - ANY noun phrase that can be considered an entity (e.g., 'kids', 'support', 'adoption agencies', 'teacher', 'New York', 'meeting', 'group') MUST be created as a separate node in the `entities` list first.\n"
#     "    - Fact's `subject` field: MUST be a Person ID (e.g., 'person_caroline') or Entity ID (e.g., 'entity_kids'), NEVER a raw string like 'Caroline' or 'kids'.\n"
#     "    - Fact's `object` field: MUST be an Entity ID (e.g., 'entity_adoption_agencies', 'entity_teacher', 'entity_new_york'), NEVER a raw string like 'adoption agencies' or 'teacher'.\n"
#     "    - Exception: For simple values (emotions like 'happy', 'sad', numbers, dates), you can use the `value` field instead of `object`, and these don't need entity nodes.\n"
#     "    - BAD: {\"subject\": \"person_caroline\", \"predicate\": \"has\", \"object\": \"kids\"}  # 'kids' is a raw string\n"
#     "    - GOOD: Create entity {\"id\": \"entity_kids\", \"name\": \"kids\", \"type\": \"concept\"}, then {\"subject\": \"person_caroline\", \"predicate\": \"has\", \"object\": \"entity_kids\"}\n"
#     "    - BAD: {\"subject\": \"person_caroline\", \"predicate\": \"researches\", \"object\": \"adoption agencies\"}  # 'adoption agencies' is a raw string\n"
#     "    - GOOD: Create entity {\"id\": \"entity_adoption_agencies\", \"name\": \"adoption agencies\", \"type\": \"topic\"}, then {\"subject\": \"person_caroline\", \"predicate\": \"researches\", \"object\": \"entity_adoption_agencies\"}\n"
#     "</critical_rules>\n\n"
#     "<example>\n"
#     "This example demonstrates exactly how to process input text into the desired JSON output.\n\n"
#     "--- EXAMPLE INPUT TEXT 1 ---\n"
#     "'Hey Mel! Last Tuesday I joined a new LGBTQ activist group. I also started researching adoption agencies.'\n\n"
#     "--- EXAMPLE METADATA ---\n"
#     "doc_id: 'convo_1_turn_5'\n"
#     "session_id: 'session_1'\n"
#     "turn_id: '5'\n"
#     "question_date_iso: '2023-07-21T00:00:00' # This is a Friday\n\n"
#     "--- EXAMPLE CORRECT OUTPUT (This is your target format) ---\n"
#     '''
#     {
#       "nodes": {
#         "persons": [
#           {"id": "person_interlocutor", "canonical_name": "Caroline", "aliases": [], "persona_traits": ["motivated", "committed to family"], "profession": "counseling", "background": "transgender woman seeking to adopt", "evidence_span": "span_0"},
#           {"id": "person_melanie", "canonical_name": "Mel", "aliases": ["Melanie"], "persona_traits": ["creative", "outdoorsy"], "profession": "artist", "background": "friend of Caroline", "evidence_span": "span_1"}
#         ],
#         "entities": [
#           {"id": "entity_lgbtq_group", "name": "LGBTQ activist group", "type": "organization", "evidence_span": "span_2"},
#           {"id": "entity_adoption_agencies", "name": "adoption agencies", "type": "topic", "evidence_span": "span_3"}
#         ],
#         "events": [
#           {"id": "event_join_group", "title": "Joined LGBTQ activist group", "participants": ["person_interlocutor"], "time_start": "2023-07-18T00:00:00", "evidence_span": "span_4"},
#           {"id": "event_research", "title": "Researched adoption agencies", "participants": ["person_interlocutor"], "time_start": "2023-07-21T00:00:00", "evidence_span": "span_3"}
#         ],
#         "facts": [
#           {"id": "fact_research_agencies", "subject": "person_interlocutor", "predicate": "researches", "object": "entity_adoption_agencies", "subtype": "action", "evidence_span": "span_3"}
#         ],
#         "evidence_spans": [
#           {"id": "span_0", "source_text": "I"},
#           {"id": "span_1", "source_text": "Mel"},
#           {"id": "span_2", "source_text": "LGBTQ activist group"},
#           {"id": "span_3", "source_text": "researching adoption agencies"},
#           {"id": "span_4", "source_text": "joined a new LGBTQ activist group last Tues"}
#         ]
#       },
#       "edges": {
#         "involved_in": [
#           {"source": "person_interlocutor", "target": "event_join_group"},
#           {"source": "person_interlocutor", "target": "event_research"}
#         ],
#         "asserts": [
#           {"source": "event_research", "target": "fact_research_agencies"}
#         ]
#       }
#     }
#     '''
#     "\n\n"
#     "--- EXAMPLE INPUT TEXT 2 (Demonstrating Event vs Fact Distinction in Conversational Data) ---\n"
#     "'Caroline said: I went camping in June. I also attended the LGBTQ support group last week. I'm working as a teacher now and I've been living in New York for two years.'\n\n"
#     "**CRITICAL FOR CONVERSATIONAL DATA**: Even though the text says 'Caroline said: I went camping', you MUST extract 'Caroline went camping' as an Event, NOT 'Caroline said' as a Fact.\n"
#     "The actual action (went camping) is the Event. The speech act (said) is just the delivery mechanism, not the content.\n\n"
#     "--- EXAMPLE CORRECT OUTPUT 2 ---\n"
#     '''
#     {
#       "nodes": {
#         "persons": [
#           {"id": "person_speaker", "canonical_name": "Caroline", "aliases": [], "evidence_span": "span_0"}
#         ],
#         "entities": [
#           {"id": "entity_meeting", "name": "meeting", "type": "event", "evidence_span": "span_1"},
#           {"id": "entity_teacher", "name": "teacher", "type": "concept", "evidence_span": "span_3"},
#           {"id": "entity_new_york", "name": "New York", "type": "location", "evidence_span": "span_4"},
#           {"id": "entity_kids", "name": "kids", "type": "concept", "evidence_span": "span_5"},
#           {"id": "entity_support_group", "name": "support group", "type": "organization", "evidence_span": "span_6"}
#         ],
#         "events": [
#           {"id": "event_camping", "title": "Went camping", "participants": ["person_caroline"], "time_start": "2023-06-XX", "evidence_span": "span_1"},
#           {"id": "event_support_group", "title": "Attended LGBTQ support group", "participants": ["person_caroline"], "time_start": "2023-07-14T00:00:00", "evidence_span": "span_2"}
#         ],
#         "facts": [
#           {"id": "fact_works_as", "subject": "person_caroline", "predicate": "works_as", "object": "entity_teacher", "subtype": "trait", "evidence_span": "span_3"},
#           {"id": "fact_lives_in", "subject": "person_caroline", "predicate": "lives_in", "object": "entity_new_york", "subtype": "location", "evidence_span": "span_4"}
#         ],
#         "evidence_spans": [
#           {"id": "span_0", "source_text": "Caroline"},
#           {"id": "span_1", "source_text": "went camping in June"},
#           {"id": "span_2", "source_text": "attended the LGBTQ support group last week"},
#           {"id": "span_3", "source_text": "working as a teacher"},
#           {"id": "span_4", "source_text": "living in New York"}
#         ]
#       },
#       "edges": {
#         "involved_in": [
#           {"source": "person_caroline", "target": "event_camping"},
#           {"source": "person_caroline", "target": "event_support_group"}
#         ],
#         "asserts": []
#       }
#     }
#     '''
#     "</example>"
# )
STRUCTURED_EXTRACTION_SYSTEM = (
    "<role>\n"
    "You are a Graph Extraction Engine. Your job is to convert conversation text into a structured JSON knowledge graph.\n"
    "**CRITICAL**: You MUST be EXTREMELY AGGRESSIVE in extracting PAST TENSE ACTIONS as 'events' and PRESENT TENSE STATES as 'facts'.\n"
    "</role>\n\n"
    "<⚠️ MANDATORY FIRST STEP>\n"
    "BEFORE extracting anything, scan the text for ALL past tense action verbs:\n"
    "- went, attended, joined, visited, participated, volunteered, met, saw, helped, started, finished, completed, researched, organized, created, ran, walked, drove, painted, wrote, read, cooked, ate, slept, woke, left, arrived, returned, and ANY other past tense verb\n"
    "- For EACH past tense action verb found, you MUST create an EVENT node\n"
    "- DO NOT extract past tense actions as Facts\n"
    "</⚠️ MANDATORY FIRST STEP>\n\n"
    "<extraction_rules>\n"
    "1. **PERSONS**: Extract the speaker (usually 'I') and any mentioned names.\n"
    "2. **EVENTS (Dynamic - PAST TENSE ACTIONS)**: Extract ANY past tense verb phrase describing an action.\n"
    "   - **CRITICAL RULE**: If the text contains a PAST TENSE ACTION VERB, it MUST be an EVENT, NOT a Fact.\n"
    "   - **Past Tense Examples**: 'went', 'attended', 'joined', 'visited', 'participated', 'volunteered', 'met', 'saw', 'helped', 'started', 'finished', 'researched', 'organized', 'created', 'ran', 'walked', 'drove', 'painted', 'wrote', 'read', 'cooked', 'ate', 'slept', 'woke', 'left', 'arrived', 'returned'\n"
    "   - **Dialogue Handling**: If text says 'Caroline said: I went camping', extract 'Went camping' as an EVENT with participant 'Caroline'. IGNORE 'said' completely.\n"
    "   - **Time**: Estimate the date based on context. If vague (e.g., 'in June'), use 'YYYY-MM-XX'. If 'yesterday', 'last week', etc., calculate from session date.\n"
    "   - **Title Format**: Use past tense action summary, e.g., 'Went camping', 'Joined group', 'Attended meeting', 'Started researching'\n"
    "3. **FACTS (Static - PRESENT TENSE STATES)**: Extract attributes, traits, preferences, relationships, and possessions.\n"
    "   - **CRITICAL RULE**: Only extract as Fact if it describes WHAT someone IS or HAS (present tense states).\n"
    "   - **Present Tense Examples**: 'is', 'has', 'likes', 'loves', 'works_as', 'lives_in', 'feels', 'enjoys', 'wants', 'needs', 'knows', 'owns'\n"
    "   - **Examples**: 'is a teacher', 'likes hiking', 'lives in NY', 'has kids', 'feels happy'\n"
    "   - **DO NOT extract past tense actions as Facts**\n"
    "4. **ENTITIES**: Extract key nouns (places, organizations, concepts) referenced in Events or Facts.\n"
    "</extraction_rules>\n\n"
    "<json_schema>\n"
    "You MUST return a JSON object with this exact structure:\n"
    "{\n"
    "  \"nodes\": {\n"
    "    \"persons\": [{\"id\": \"string\", \"canonical_name\": \"string\"}],\n"
    "    \"entities\": [{\"id\": \"string\", \"name\": \"string\", \"type\": \"string\"}],\n"
    "    \"events\": [\n"
    "      {\n"
    "        \"id\": \"string (required, e.g., 'event_1')\",\n"
    "        \"title\": \"string (REQUIRED - short action summary in past tense, e.g., 'Went camping', 'Joined group', 'Attended meeting')\",\n"
    "        \"participants\": [\"string (REQUIRED - at least one person_id, e.g., ['person_caroline'])\"],\n"
    "        \"time_start\": \"string (optional - ISO date like '2023-06-XX' or '2023-06-15')\",\n"
    "        \"location\": \"string (optional)\"\n"
    "      }\n"
    "    ],\n"
    "    \"facts\": [\n"
    "      {\n"
    "        \"id\": \"string\",\n"
    "        \"subject\": \"string (node_id)\",\n"
    "        \"predicate\": \"string (simple verb)\",\n"
    "        \"object\": \"string (node_id or value)\",\n"
    "        \"subtype\": \"trait|preference|relationship|possession|location\"\n"
    "      }\n"
    "    ],\n"
    "    \"evidence_spans\": []\n"
    "  },\n"
    "  \"edges\": {}\n"
    "}\n"
    "</json_schema>\n\n"
    "<⚠️ CRITICAL: Event Requirements>\n"
    "- Every Event MUST have a 'title' field (non-empty string)\n"
    "- Every Event MUST have a 'participants' array with at least one person_id\n"
    "- If you find a past tense action verb, you MUST create an Event with these fields\n"
    "- DO NOT create Events without title or participants\n"
    "</⚠️ CRITICAL>\n\n"
    "<examples>\n"
    "Example 1: Past tense action → EVENT\n"
    "Input: 'Yesterday, I joined the hiking club.' (Session Date: 2023-05-15)\n"
    "Output:\n"
    "{\n"
    "  \"nodes\": {\n"
    "    \"persons\": [{\"id\": \"person_speaker\", \"canonical_name\": \"Speaker\"}],\n"
    "    \"entities\": [{\"id\": \"entity_hiking_club\", \"name\": \"hiking club\", \"type\": \"organization\"}],\n"
    "    \"events\": [\n"
    "      {\n"
    "        \"id\": \"event_join_club\",\n"
    "        \"title\": \"Joined hiking club\",\n"
    "        \"participants\": [\"person_speaker\"],\n"
    "        \"time_start\": \"2023-05-14\"\n"
    "      }\n"
    "    ],\n"
    "    \"facts\": [],\n"
    "    \"evidence_spans\": []\n"
    "  },\n"
    "  \"edges\": {}\n"
    "}\n\n"
    "Example 2: Dialogue format → Extract action as EVENT, ignore 'said'\n"
    "Input: 'Caroline said: I went camping in June.' (Session Date: 2023-07-15)\n"
    "Output:\n"
    "{\n"
    "  \"nodes\": {\n"
    "    \"persons\": [{\"id\": \"person_caroline\", \"canonical_name\": \"Caroline\"}],\n"
    "    \"entities\": [],\n"
    "    \"events\": [\n"
    "      {\n"
    "        \"id\": \"event_camping\",\n"
    "        \"title\": \"Went camping\",\n"
    "        \"participants\": [\"person_caroline\"],\n"
    "        \"time_start\": \"2023-06-XX\"\n"
    "      }\n"
    "    ],\n"
    "    \"facts\": [],\n"
    "    \"evidence_spans\": []\n"
    "  },\n"
    "  \"edges\": {}\n"
    "}\n\n"
    "Example 3: Present tense state → FACT\n"
    "Input: 'I am a teacher. I like hiking.'\n"
    "Output:\n"
    "{\n"
    "  \"nodes\": {\n"
    "    \"persons\": [{\"id\": \"person_speaker\", \"canonical_name\": \"Speaker\"}],\n"
    "    \"entities\": [\n"
    "      {\"id\": \"entity_teacher\", \"name\": \"teacher\", \"type\": \"concept\"},\n"
    "      {\"id\": \"entity_hiking\", \"name\": \"hiking\", \"type\": \"activity\"}\n"
    "    ],\n"
    "    \"events\": [],\n"
    "    \"facts\": [\n"
    "      {\"id\": \"fact_works_as\", \"subject\": \"person_speaker\", \"predicate\": \"works_as\", \"object\": \"entity_teacher\", \"subtype\": \"trait\"},\n"
    "      {\"id\": \"fact_likes\", \"subject\": \"person_speaker\", \"predicate\": \"likes\", \"object\": \"entity_hiking\", \"subtype\": \"preference\"}\n"
    "    ],\n"
    "    \"evidence_spans\": []\n"
    "  },\n"
    "  \"edges\": {}\n"
    "}\n"
    "</examples>"
)

STRUCTURED_EXTRACTION_USER = (
    "<⚠️ REMINDER: EXTRACTION CHECKLIST>\n"
    "1. Scan the text for ALL past tense action verbs (went, attended, joined, visited, participated, volunteered, met, saw, helped, started, finished, researched, organized, created, ran, walked, drove, painted, wrote, read, cooked, ate, slept, woke, left, arrived, returned, etc.)\n"
    "2. For EACH past tense action verb found, create an EVENT node (NOT a Fact node)\n"
    "3. Extract present tense states, traits, and preferences (is, has, likes, works_as, lives_in, feels, etc.) as FACT nodes\n"
    "4. If you see dialogue like 'X said: I did Y', extract 'X did Y' as an EVENT, ignore 'X said'\n"
    "</⚠️ REMINDER>\n\n"
    "Input Text: {INPUT_TEXT}\n"
    "Context Metadata:\n"
    "- Session Date: {QUESTION_DATE_ISO}\n"
    "- Speaker/Subject: Determine from context\n\n"
    "Extract all Events (past tense actions) and Facts (present tense states) into the required JSON format."
)
# STRUCTURED_EXTRACTION_USER = (
#    "<task>\n"
#     "Based on the rules and the detailed example I provided, process the following INPUT TEXT and METADATA. Generate ONLY the final JSON output and nothing else.\n"
#     "</task>\n\n"
#     "--- INPUT TEXT ---\n"
#     "{INPUT_TEXT}\n\n"
#     "--- METADATA ---\n"
#     "doc_id: \"{DOC_ID}\"\n"
#     "session_id: \"{SESSION_ID}\"\n"
#     "turn_id: \"{TURN_ID}\"\n"
#     "question_date_iso: \"{QUESTION_DATE_ISO}\"\n\n"
#     "--- JSON OUTPUT ---"
# )

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
