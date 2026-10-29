"""Pure contract regressions; no provider, labels or benchmark-specific rules."""
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from cogmem.agents.search_agent import SearchAgent, ThoughtActionObservation
from cogmem.storage.graph_service import GraphService
from cogmem.storage.evidence_store import EvidenceStore
from cogmem.storage.graph_store import GraphStore






def instance():
    agent = object.__new__(SearchAgent)
    graph = GraphStore()
    graph._g.add_nodes_from([
        ('p_a', {'node_type': 'person', 'canonical_name': 'A'}),
        ('p_b', {'node_type': 'person', 'canonical_name': 'B'}),
        ('f_a', {'id': 'f_a', 'node_type': 'fact', 'subject_id': 'p_a',
                 'predicate': 'enjoys', 'content': 'A enjoys mountain hiking'}),
        ('f_b', {'id': 'f_b', 'node_type': 'fact', 'subject_id': 'p_b',
                 'predicate': 'enjoys', 'content': 'B enjoys mountain hiking'}),
    ])
    graph._g.add_edge('p_a', 'f_a', rel_type='HAS_FACT', weight=1, confidence=1)
    graph._g.add_edge('p_b', 'f_b', rel_type='HAS_FACT', weight=1, confidence=1)
    agent.graph_service = GraphService(graph, None, None, use_vector_search=False)
    agent.evidence_store = EvidenceStore()
    agent.evidence_store.add_chunk_for_entity('f_a', 'Source A: I enjoy hiking in the mountains.')
    agent.evidence_store.add_chunk_for_entity('f_b', 'Source B: I enjoy hiking in the mountains.')
    agent._last_tool_node_hits = {}
    agent._current_query = 'What do A and B enjoy?'
    agent._current_question_type = 'single_hop'
    agent.find_common_ground_top_k = 3
    agent.traverse_neighbors_max_depth = 1
    agent.traverse_neighbors_max_results = 3
    agent.require_evidence_for_answer = True
    return agent


def test_transient_common_pattern_resolves_both_actual_supporting_facts():
    agent = instance()
    result = agent._tool_find_common_ground(['p_a', 'p_b'], node_types=['fact'])
    assert 'Common Nodes Found' in result
    assert set(agent._last_tool_node_hits['find_common_ground']) == {'f_a', 'f_b'}
    raw = agent._tool_get_evidence(entity_ids=agent._last_tool_node_hits['find_common_ground'])
    assert 'Source A' in raw and 'Source B' in raw


def test_explicit_anchor_tool_preserves_its_real_graph_address():
    agent = instance()
    agent.find_anchors_top_k = 3
    result = agent._tool_find_anchors('A', types=['person'])
    assert '[ID: p_a]' in result
    packaged = agent._structured_observation_payload(ThoughtActionObservation(1, '', 'find_anchors', {}, result))
    assert any('[ID: p_a]' in item for item in packaged['items'])


def test_common_ground_ranking_precedes_limit_and_is_permutation_invariant():
    agent = instance()
    nodes = [{'id': 'c_chess', 'node_type': 'concept', 'name': 'chess'},
             {'id': 'c_hiking', 'node_type': 'concept', 'name': 'hiking'},
             {'id': 'c_swimming', 'node_type': 'concept', 'name': 'swimming'}]
    agent.graph_service.graph_store._g.add_nodes_from((node['id'], node) for node in nodes)
    agent._current_query = 'What do both share about hiking?'
    agent.find_common_ground_top_k = 1
    outputs = set()
    for order in permutations(nodes):
        agent.graph_service.find_common_ground = lambda *args, **kwargs: SimpleNamespace(common_nodes=list(order))
        outputs.add(agent._tool_find_common_ground(['p_a', 'p_b'], node_types=['concept']))
        assert agent._last_tool_node_hits['find_common_ground'] == ['c_hiking']
    assert len(outputs) == 1


def test_service_intersection_uses_stable_node_order():
    agent = instance()
    graph = agent.graph_service.graph_store._g
    for i in range(9):
        key = 'c_' + str(i)
        graph.add_node(key, node_type='concept', name=key)
        graph.add_edge('p_a', key, rel_type='LIKES')
        graph.add_edge('p_b', key, rel_type='LIKES')
    result = agent.graph_service.find_common_ground(['p_a', 'p_b'], ['concept'], max_depth=1)
    assert [node['id'] for node in result.common_nodes] == sorted('c_' + str(i) for i in range(9))


def test_traversal_returns_actual_address_and_auto_grounds_it():
    agent = instance()
    observation = agent._tool_traverse_neighbors(node_ids=['p_a'])
    assert '[ID: f_a]' in observation
    assert agent._last_tool_node_hits['traverse_neighbors'] == ['f_a']
    agent._maybe_finalize_after_step = lambda *args, **kwargs: None
    agent._execute_tool = lambda tool, params: agent._tool_get_evidence(**params)
    history = []
    agent._record_and_maybe_finalize(query=agent._current_query, history=history, scratchpad=[],
        step_counter=0, thought='', action='traverse_neighbors', action_input={'node_ids': ['p_a']},
        observation=observation)
    assert agent._has_substantive_retrieval(history)
    assert history[1].action_input['entity_id'] == 'f_a'
    assert 'Source A' in history[1].observation and 'Source B' not in history[1].observation


def test_delegated_empty_timeline_cannot_reuse_a_previous_hit_set():
    agent = instance()
    agent._last_tool_node_hits['timeline_query'] = ['stale']
    agent._tool_traverse_neighbors(node_ids=['p_a'], time_query='2099')
    assert agent._last_tool_node_hits['timeline_query'] == []
    assert agent._last_tool_node_hits['traverse_neighbors'] == []


@pytest.mark.parametrize('refusal', ["I don't have enough information", "I don't have enough information."])
def test_positive_ready_followed_by_reader_abstention_does_not_certify_early_success(refusal):
    agent = instance()
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        if kwargs['system'].startswith('You are a strict evidence sufficiency judge'):
            return json.dumps({'can_answer': True, 'answer': 'A draft', 'confidence': 'medium'})
        return refusal
    agent.llm = SimpleNamespace(chat=chat)
    history = [ThoughtActionObservation(1, '', 'semantic_search', {}, 'A retrieved cue'),
               ThoughtActionObservation(2, '', 'get_evidence', {'entity_id': 'f_a'},
                   'Original Text Evidence for f_a:\n[Evidence 1] Source A')]
    assert agent._maybe_finalize_after_step('What is the answer?', history, action='get_evidence') is None
    assert len(calls) == 2
    # Abstention is still a legal final result after the controller is exhausted.
    agent._build_run_result = lambda answer, path, confidence: {'answer': answer, 'confidence': confidence}
    final = agent._return_run_result(history, "I don't have enough information.", 'low')
    assert final == {'answer': "I don't have enough information.", 'confidence': 'low'}
