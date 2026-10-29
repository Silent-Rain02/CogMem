import copy
from types import SimpleNamespace

import pytest

from cogmem.agents.search_agent import SearchAgent, ThoughtActionObservation
from cogmem.config import CogMemConfig
from cogmem.memory.claim_reconciler import ClaimReconciler
from cogmem.memory.concept_merger import ConceptMerger
from cogmem.memory.consolidator import MemoryConsolidator, _make_fact_id
from cogmem.memory.fact_deduplicator import FactDeduplicator
from cogmem.memory.indexer import Indexer
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.schema import Event
from cogmem.storage.evidence_store import EvidenceStore
from cogmem.utils.concept_manager import ConceptManager
from cogmem.utils.temporal import normalize_event_time


@pytest.mark.parametrize('value,reference,expected', [
    ('2020', None, '2020'), ('2020-XX-XX', None, '2020'),
    ('2020-02', None, '2020-02'), ('2020-02-XX', None, '2020-02'),
    ('2020-02-30', None, None), ('yesterday', None, None),
    ('yesterday', '2020-03-01', '2020-02-29'),
    ('next month', '2020-12-18', '2021-01'),
    ('last month', '2020-03-31', '2020-02'),
    ('recently', '2020-03-01', None), ('last week', '2020-03-01', None),
    ('last Friday', '2020-03-06', '2020-02-28'),
    ('yesterday (relative to 2020-03-01)', '2020-03-01', '2020-02-29'),
    ('yesterday (relative to 2019-03-01)', '2020-03-01', None),
])
def test_time_resolution_requires_anchor_and_preserves_precision(value, reference, expected):
    assert normalize_event_time(value, reference) == expected


def builder():
    indexer = object.__new__(Indexer)
    indexer._cfg = CogMemConfig()
    indexer._concept_manager = ConceptManager()
    return indexer


def test_distinct_occurrences_survive_and_raw_time_is_not_overwritten():
    indexer = builder()
    payload = {'persons': ['Alice'], 'events': [
        {'title': 'Alice visits the museum', 'participants': ['Alice'], 'time': 'yesterday', 'concepts': ['museum']}],
        'facts': [], 'claims': []}
    before = copy.deepcopy(payload)
    nodes1, _ = indexer._build_event_mediated_graph(payload, 'session_1', '2020-03-01', '2020-03-01', source_id='doc1')
    nodes2, _ = indexer._build_event_mediated_graph(payload, 'session_2', '2020-03-02', '2020-03-02', source_id='doc2')
    e1 = next(n for n in nodes1 if isinstance(n, Event))
    e2 = next(n for n in nodes2 if isinstance(n, Event))
    assert e1.id != e2.id
    assert e1.time_start_raw == 'yesterday'
    assert e1.time_resolved == '2020-02-29'
    assert e2.time_resolved == '2020-03-01'
    assert payload == before


def test_missing_event_time_is_valid_and_session_date_is_not_event_date():
    event = Event(id='event_unknown', title='Something happened', session_time='2020-04-05')
    assert event.effective_time is None
    coarse = Event(id='event_year', title='Something happened', time_start='2020')
    assert coarse.time_start == '2020'


def test_shared_node_scoring_has_no_query_independent_topic_or_named_person_exception():
    instance = object.__new__(SearchAgent)
    art = {'node_type':'concept','name':'painting'}
    music = {'node_type':'concept','name':'music'}
    assert instance._score_common_ground_node(art, {'walking'}) == instance._score_common_ground_node(music, {'walking'})
    assert instance._extract_focus_terms('What does Caroline do?') == instance._extract_focus_terms('What does Morgan do?').difference({'morgan'}) | {'caroline'}


def test_timeline_filters_use_event_date_not_recording_date():
    instance = object.__new__(SearchAgent)
    graph = GraphStore()
    graph._g.add_node('fact', node_type='fact', subject_id='p_a', content='Museum visit', session_time='2023-12-01')
    events = [SimpleNamespace(target_id='event_old', target_node={
        'node_type':'event', 'title':'Museum visit', 'time_start':'2019-07', 'session_time':'2023-12-01'}),
        SimpleNamespace(target_id='event_unknown', target_node={
        'node_type':'event', 'title':'Museum visit', 'session_time':'2023-12-01'})]
    instance.graph_service = SimpleNamespace(traverse_associations=lambda *args, **kwargs:events, graph_store=graph)
    instance._last_tool_node_hits = {}
    instance._current_query = 'Which museum visit occurred?'
    instance.traverse_neighbors_max_results = 10
    assert instance._tool_scan_timeline(['p_a'], '2023').startswith('No events found')
    summer = instance._tool_scan_timeline(['p_a'], 'summer')
    assert 'event_old' in summer and 'event_unknown' not in summer
    assert 'event_old' in instance._tool_scan_timeline(['p_a'], '2019-07-15')


def test_fact_time_filter_does_not_invent_validity_from_recording_time():
    instance = object.__new__(SearchAgent)
    graph = GraphStore()
    graph._g.add_node('unknown', node_type='fact', subject_id='p_a', content='Alice likes tea', session_time='2023-06-01')
    graph._g.add_node('known', node_type='fact', subject_id='p_a', content='Alice likes coffee', valid_from='2020-01', valid_to='2020-12', session_time='2023-06-01')
    instance.graph_service = SimpleNamespace(graph_store=graph)
    instance._last_tool_node_hits = {}
    instance._current_query = ''
    assert instance._tool_fact_lookup(subject_id='p_a', at_time='2023').startswith('No facts found')
    result = instance._tool_fact_lookup(subject_id='p_a', at_time='2020-06')
    assert 'known' in result and '[ID: unknown]' not in result


def raw_step(text='[Alice] The meeting was yesterday.'):
    return ThoughtActionObservation(1, '', 'get_evidence', {},
        'Original Text Evidence for p_alice:\n[Evidence 1] [Session Date: 2020-03-01]\n' + text)


def agent():
    instance = object.__new__(SearchAgent)
    instance.require_evidence_for_answer = True
    instance._current_query = 'When was the meeting?'
    instance._current_question_type = 'single_hop'
    instance._extract_query_entities = lambda query: []
    instance._build_run_result = lambda answer, path, confidence: {'answer': answer, 'confidence': confidence}
    return instance


def test_every_answer_exit_requires_actual_source_not_a_success_flag():
    instance = agent()
    instance._evidence_collected = True
    derived = ThoughtActionObservation(1, '', 'semantic_search', {}, 'Fact: meeting happened today')
    assert not instance._should_attempt_stepwise_answer('semantic_search', [derived])
    result = instance._return_run_result([derived], 'Invented answer', 'high')
    assert result == {'answer': "I don't have enough information.", 'confidence': 'low'}
    assert instance._has_raw_evidence([raw_step()])
    assert not instance._has_raw_evidence([ThoughtActionObservation(1, '', 'get_evidence', {}, 'Tool Error: failure')])


def test_synthesis_view_excludes_derived_claims_and_keeps_complete_source():
    instance = agent()
    derived = ThoughtActionObservation(1, '', 'semantic_search', {}, 'Fact: THIS IS AN UNVERIFIED CLAIM')
    package = instance._build_synthesis_evidence_package([derived, raw_step()])
    assert len(package['steps']) == 1
    assert package['steps'][0]['action'] == 'get_evidence'
    assert 'yesterday' in str(package)
    assert 'UNVERIFIED CLAIM' not in str(package)


def test_semantic_similarity_cannot_delete_or_expire_nodes():
    graph = GraphStore()
    graph._g.add_node('a', node_type='fact', subject_id='alice', content='Alice likes tea')
    graph._g.add_node('b', node_type='fact', subject_id='bob', content='Bob likes tea')
    before = copy.deepcopy(dict(graph._g.nodes(data=True)))
    dedup = FactDeduplicator(graph)
    assert dedup.merge_duplicates({'same-topic': [('a', before['a']), ('b', before['b'])]}, {('a', 'b'): 1.0}) == {}
    assert dict(graph._g.nodes(data=True)) == before
    assert list(graph._g.edges()) == []
    merger = ConceptMerger(graph)
    assert merger.merge_concepts([{'a', 'b'}], {('a', 'b'): 1.0}) == {}
    assert set(graph._g.nodes) == {'a', 'b'}


def test_keep_all_claims_never_establishes_supersession():
    graph = GraphStore()
    graph._g.add_node('old', node_type='claim', source_id='alice', target_id='work', content='I enjoy quiet offices', session_time='2020-01-01')
    graph._g.add_node('new', node_type='claim', source_id='alice', target_id='work', content='I enjoy team brainstorming', session_time='2021-01-01')
    reconciler = ClaimReconciler(graph)
    assert reconciler.run(strategy='keep_all') == {}
    assert not graph._g.nodes['old'].get('is_archived')
    assert not graph._g.nodes['new'].get('is_archived')
    assert all(data.get('rel_type') != 'SUPERSEDES' for _, _, data in graph._g.edges(data=True))


def test_synthesis_identity_depends_on_exact_evidence_set():
    assert _make_fact_id('p', 'c', 'likes', ['e1']) != _make_fact_id('p', 'c', 'likes', ['e2'])
    assert _make_fact_id('p', 'c', 'likes', ['e1', 'e2']) == _make_fact_id('p', 'c', 'likes', ['e2', 'e1'])


def test_an_existing_observation_does_not_block_new_source_synthesis():
    consolidator = object.__new__(MemoryConsolidator)
    consolidator.graph = GraphStore()
    consolidator.graph._g.add_node('observation', node_type='fact', subject_id='p', object_id='c')
    assert not consolidator._has_synthesis_for('p', 'c', [{'id': 'e1'}])
    consolidator.graph._g.add_node('derived', node_type='fact', subject_id='p', object_id='c', metadata={'source': 'memory_consolidation'})
    consolidator.graph._g.add_edge('derived', 'e1', rel_type='SUPPORTED_BY')
    assert consolidator._has_synthesis_for('p', 'c', [{'id': 'e1'}])
    assert not consolidator._has_synthesis_for('p', 'c', [{'id': 'e2'}])


def test_participant_list_order_does_not_invent_primary_and_secondary_roles():
    indexer = builder()
    payload = {'events': [{'title': 'Shared visit', 'participants': ['Alice', 'Bob'], 'time': '2020'}]}
    before = copy.deepcopy(payload)
    nodes, relations = indexer._build_event_mediated_graph(payload, 's', source_id='document')
    event = next(n for n in nodes if isinstance(n, Event))
    participant_edges = [r for r in relations if r.tail_id == event.id and r.head_id in event.participants]
    assert len(participant_edges) == 2
    assert all(r.rel_type == 'PARTICIPATED_IN' for r in participant_edges)
    assert event.metadata.source == 'document'
    payload['events'][0]['participants'].reverse()
    other, _ = indexer._build_event_mediated_graph(payload, 's', source_id='document')
    assert next(n for n in other if isinstance(n, Event)).id == event.id
    assert before['events'][0]['participants'] == ['Alice', 'Bob']


def test_semantic_ranking_does_not_prioritize_type_over_relevance():
    instance = agent()
    instance._last_tool_node_hits = {}
    data = {'f': {'node_type': 'fact', 'content': 'Less relevant'},
            'e': {'node_type': 'event', 'title': 'More relevant'}}
    instance.graph_service = SimpleNamespace(use_vector_search=True,
        embedding_encoder=SimpleNamespace(encode=lambda _: [[1.0]]),
        vector_store=SimpleNamespace(search_by_type=lambda vector, nt, top_k: [('f', .2)] if nt == 'fact' else [('e', .9)]),
        graph_store=SimpleNamespace(get_node=lambda key: data.get(key)))
    instance._format_node_to_rich_text = lambda value, key: key
    result = instance._tool_semantic_search('query', ['fact', 'event'], top_k=1)
    assert '[ID: e]' in result
    assert '[ID: f]' not in result


def test_operator_budget_applies_to_automatic_calls_too():
    instance = agent()
    instance.max_iterations = 1
    instance.max_tool_calls = 1
    instance.skill_runtime = None
    calls = []
    instance.tools = {'get_evidence': lambda **params: calls.append(params) or 'No evidence'}
    assert instance._execute_tool('get_evidence', {'entity_id': 'p'}) == 'No evidence'
    assert instance._execute_tool('get_evidence', {'entity_id': 'p'}) == 'Tool budget exhausted.'
    assert len(calls) == 1


def test_all_support_links_and_original_chunks_survive_synthesis():
    consolidator = object.__new__(MemoryConsolidator)
    consolidator.config = CogMemConfig()
    consolidator.graph = GraphStore()
    consolidator.embedding_encoder = SimpleNamespace(encode=lambda texts: [[1.0] for _ in texts])
    consolidator.vector_store = SimpleNamespace(add_typed=lambda **kwargs: None)
    consolidator.evidence_store = EvidenceStore()
    consolidator.graph._g.add_node('p', node_type='person', canonical_name='Person')
    consolidator.graph._g.add_node('c', node_type='concept', name='Concept')
    events = [{'id': f'event_{i}'} for i in range(8)]
    for item in events:
        consolidator.graph._g.add_node(item['id'], node_type='event')
        consolidator.evidence_store.add_chunk_for_entity(item['id'], f"source {item['id']}")
    fact_id = consolidator._create_fact_node('p', 'c', {'content': 'A supported observation', 'predicate': 'observed'}, events, max_supported_by=5)
    targets = {target for _, target, edge in consolidator.graph._g.out_edges(fact_id, data=True)
               if edge.get('rel_type') == 'SUPPORTED_BY'}
    assert targets == {item['id'] for item in events}
    assert len(consolidator.evidence_store.get_chunks(fact_id, limit=None)) == 8


def test_coarse_event_time_overlaps_its_entire_calendar_interval():
    event = Event(id='event_year', title='A historical visit', time_start='2020')
    assert event.overlaps_with('2020-06-12', '2020-06-13')
    assert not event.overlaps_with('2021', '2021')
    month = Event(id='event_month', title='A visit', time_start='2020-02')
    assert month.overlaps_with('2020-02-29', '2020-02-29')
    assert not month.overlaps_with('2020-03-01', '2020-03-02')


def test_fact_validity_never_invents_first_day_of_month():
    from cogmem.storage.schema import Fact
    fact = Fact(id='fact_1', subject_id='p', predicate='works_at', object_id='c', valid_from='2020-02')
    assert fact.valid_from == '2020-02'
    consolidator = object.__new__(MemoryConsolidator)
    assert consolidator._normalize_fact_date('2020') == '2020'
    assert consolidator._normalize_fact_date('2020-02') == '2020-02'


def test_all_claim_entrypoints_preserve_coexisting_claims():
    graph = GraphStore()
    graph._g.add_node('a', node_type='claim', source_id='p', target_id='work', content='Quiet office')
    graph._g.add_node('b', node_type='claim', source_id='p', target_id='work', content='Team discussions')
    claims = list(graph._g.nodes(data=True))
    before = copy.deepcopy(dict(claims))
    reconciler = ClaimReconciler(graph)
    assert reconciler.reconcile_group(claims) is None
    with pytest.raises(ValueError, match='source-backed'):
        reconciler.establish_temporal_chain({'group': claims})
    with pytest.raises(ValueError, match='source-backed'):
        reconciler.reconcile_group(claims, strategy='newest_wins')
    assert dict(graph._g.nodes(data=True)) == before
    assert list(graph._g.edges()) == []


def test_preinsertion_dedup_keeps_distinct_observations_and_rejects_unvalidated_update():
    from cogmem.storage.schema import Fact
    facts = [Fact(id=f'fact_{i}', subject_id='p', predicate='visited', object_id='c', valid_from=year)
             for i, year in enumerate(['2018', '2022'])]
    dedup = FactDeduplicator(GraphStore())
    assert dedup.deduplicate(facts) == facts
    with pytest.raises(ValueError, match='source-backed'):
        dedup.update_fact(facts[0].id, facts[1])
    assert list(dedup.graph._g.nodes()) == []


def test_consumer_failure_never_returns_controller_draft():
    instance = agent()
    instance._has_raw_evidence = lambda history: True
    instance._synthesize_answer_with_llm = lambda *args: None
    answer, confidence = instance._postprocess_final_answer('Question', 'Unsupported graph draft', 'high', [])
    assert answer == "I don't have enough information."
    assert confidence == 'low'


def test_raw_view_does_not_attach_first_relative_date_to_entire_passage():
    instance = agent()
    step = ThoughtActionObservation(step=1, thought='', action='get_evidence', action_input={},
        observation='Original Text Evidence for p:\n[Evidence 1] [Session Date: 2020-03-01]\nAlice: Yesterday I walked. Bob: Last year I moved.')
    payload = instance._structured_observation_payload(step)
    item = payload['items'][0]
    assert item['session_date'] == '2020-03-01'
    assert 'normalized_time' not in item
    assert 'Yesterday I walked' in item['content']
    assert 'Last year I moved' in item['content']


def test_retired_fact_consolidation_preserves_pipeline_call_contract():
    dedup = FactDeduplicator(GraphStore())
    assert dedup.run(create_supersedes=True, high_threshold=.95, max_supported_by=5) == {}


def test_unanchored_extraction_never_injects_wall_clock_date():
    instance = builder()
    prompts = []
    instance._llm = SimpleNamespace(chat=lambda **kwargs: prompts.append(kwargs['user']) or '{}')
    instance._extract_structured_from_chunk('Morgan: Yesterday I walked.', 'undated')
    assert 'unknown (do not resolve relative dates)' in prompts[0]


def test_source_registration_does_not_require_lexical_overlap_or_drop_late_chunks():
    from cogmem.storage.schema import Concept
    from cogmem.storage.chunk_store import Chunk
    instance = builder()
    instance._evidence = EvidenceStore()
    node = Concept(id='c_abstract', name='Circadian homeostasis')
    chunks = [Chunk(id=str(i), text=f'Alice: I went to bed {i}.', doc_id='doc') for i in range(7)]
    instance._attach_relevant_evidence_chunks([node], chunks)
    instance._attach_relevant_evidence_chunks([node], chunks)
    assert instance._evidence.get_chunks(node.id, limit=None) == [c.text for c in chunks]


def test_new_retrieval_source_is_reinstated_after_earlier_successful_evidence():
    instance = agent()
    instance._last_tool_node_hits = {'semantic_search': ['node_beta']}
    instance._maybe_finalize_after_step = lambda *args, **kwargs: None
    calls = []
    instance._execute_tool = lambda action, params: calls.append(params['entity_id']) or 'Original Text Evidence for node_beta:\n[Evidence 1] Second independent source.'
    history = [ThoughtActionObservation(1, '', 'get_evidence', {'entity_id': 'node_alpha'},
        'Original Text Evidence for node_alpha:\n[Evidence 1] First independent source.')]
    instance._record_and_maybe_finalize(query='A question needing the second source', history=history,
        scratchpad=[], step_counter=1, thought='', action='semantic_search', action_input={}, observation='New candidate node_beta')
    assert calls == ['node_beta']
    assert not instance._needs_evidence_for_target('node_beta', history)


def test_existing_source_content_is_not_refetched_through_another_graph_address():
    instance = agent()
    instance.evidence_store = EvidenceStore()
    instance.evidence_store.add_chunk_for_entity('different_address', '[Session Date: 2020-03-01]\n[Alice] A shared original passage.')
    assert not instance._needs_evidence_for_target('different_address', [raw_step('[Alice] A shared original passage.')])


def test_planning_rounds_do_not_get_consumed_by_automatic_source_operators():
    instance = agent()
    instance.max_iterations = 2
    instance.skill_runtime = None
    instance.tools = {'get_evidence': lambda **params: 'A raw retrieval'}
    instance._tool_calls_used = 3  # anchors + initial retrieval + its source
    assert instance._operator_budget() == 7
    for _ in range(4):
        assert instance._execute_tool('get_evidence', {}) == 'A raw retrieval'
    assert instance._execute_tool('get_evidence', {}) == 'Tool budget exhausted.'


def test_initial_retrieval_and_source_do_not_prevent_first_planner_round():
    cfg = CogMemConfig()
    calls = []
    llm = SimpleNamespace(chat=lambda *args, **kwargs: calls.append(args) or '{"final_answer":"A grounded answer"}')
    instance = SearchAgent(SimpleNamespace(find_anchors=lambda *args, **kwargs: []), llm, config=cfg)
    instance._build_seed_retrieval_plan = lambda *args: ('semantic_search', {'query':'query'}, '')
    def retrieve(**kwargs):
        instance._remember_tool_hits('semantic_search', ['node_alpha'])
        return 'Semantic Search Results: node_alpha'
    instance.tools['semantic_search'] = retrieve
    instance.tools['get_evidence'] = lambda **kwargs: 'Original Text Evidence for node_alpha:\n[Evidence 1] A recorded source.'
    instance._maybe_finalize_after_step = lambda *args, **kwargs: None
    instance._postprocess_final_answer = lambda *args: ('A grounded answer', 'medium')
    result = instance.run('Where is Alice?')
    assert len(calls) == 1
    assert result['effective_agent_config']['max_iterations'] == 3
    assert result['effective_agent_config']['max_tool_calls'] == 9
    assert result['operator_calls_used'] == 3
    assert result['planning_rounds_used'] == 1


def test_shared_term_ranking_has_no_query_independent_art_topic_bonus():
    instance = agent()
    groups = [[{'content':'art zoology'}], [{'content':'art zoology'}]]
    result = instance._extract_shared_fact_terms(groups, set())
    assert [term for term, examples in result] == ['zoology', 'art']
