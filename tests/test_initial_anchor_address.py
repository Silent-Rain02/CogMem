from types import SimpleNamespace
import pytest

from cogmem.config import CogMemConfig
from cogmem.agents.search_agent import SearchAgent




def check_initial_address(kind, node_type='event'):
    opaque_id = 'event_opaque_9f18d206'
    anchor = SimpleNamespace(node_id=opaque_id, node_type=node_type,
                             metadata={'node_type': node_type, 'title': 'A stored visit', 'text': 'An original cue preview'})
    calls = []
    llm = SimpleNamespace(chat=lambda *args, **kwargs: calls.append(args) or '{"final_answer":"A grounded answer"}')
    instance = kind(SimpleNamespace(find_anchors=lambda *args, **kwargs: [anchor]), llm, config=CogMemConfig())
    instance._build_seed_retrieval_plan = lambda *args: ('semantic_search', {'query': 'query'}, '')
    def retrieve(**kwargs):
        instance._remember_tool_hits('semantic_search', ['node_alpha'])
        return 'Semantic Search Results: node_alpha'
    instance.tools['semantic_search'] = retrieve
    instance.tools['get_evidence'] = lambda **kwargs: 'Original Text Evidence for node_alpha:\n[Evidence 1] A recorded source.'
    instance._maybe_finalize_after_step = lambda *args, **kwargs: None
    instance._postprocess_final_answer = lambda *args: ('A grounded answer', 'medium')
    result = instance.run('Where is Ari?')
    assert len(calls) == 1
    assert '[ID: ' + opaque_id + ']' in result['reasoning_path'][0]['observation']
    assert '[ID: ' + opaque_id + ']' in calls[0][1]
    assert result['operator_calls_used'] == 3
    assert result['planning_rounds_used'] == 1


@pytest.mark.parametrize('node_type', ['event', 'chunk'])
def test_candidate_initial_run_exposes_opaque_anchor_address_to_the_planner(node_type):
    check_initial_address(SearchAgent, node_type)
