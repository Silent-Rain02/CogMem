from types import SimpleNamespace
import pytest
from cogmem.config import CogMemConfig
from cogmem.agents.search_agent import SearchAgent
from cogmem.storage.graph_service import GraphService
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.chunk_store import Chunk


def agent(graph):
    return SearchAgent(GraphService(graph,None,None,use_vector_search=False),None,config=CogMemConfig())


def test_explicit_address_resolution_still_respects_requested_type():
    g=GraphStore();g._g.add_node('same',node_type='concept',name='Same')
    assert agent(g)._resolve_entity_id('same',types=['person']) is None


def test_id_prefix_cannot_override_declared_node_type():
    g=GraphStore();g._g.add_node('p_iris',node_type='concept',name='Iris')
    assert agent(g).graph_service.find_anchors('Iris',types=['person'])==[]


def test_typed_person_query_cannot_return_a_raw_chunk_fallback():
    g=GraphStore();chunks=SimpleNamespace(size=lambda:1,search=lambda *args,**kwargs:[(Chunk('chunk','Original','source'),.99)])
    encoder=SimpleNamespace(encode=lambda texts:[[1.,0.]])
    service=GraphService(g,SimpleNamespace(search=lambda *args,**kwargs:[]),encoder,chunk_store=chunks)
    assert service.find_anchors('Unknown',types=['person'])==[]
    assert service.find_anchors('Unknown',types=None)[0].node_type=='chunk'


def test_timestamp_occurrence_can_be_selected_by_calendar_window():
    g=GraphStore();g._g.add_node('event_known',node_type='event',title='Meeting',time_start='2020-03-04T14:00:00Z')
    a=agent(g);a._current_query='meeting'
    assert '[ID: event_known]' in a._tool_scan_timeline(['event_known'],'2020-03')


def test_explicit_event_addresses_do_not_require_an_id_spelling_convention():
    g=GraphStore();g._g.add_node('opaque:17',node_type='event',title='Meeting',time_start='2020-03-04')
    a=agent(g);a._current_query='meeting'
    assert '[ID: opaque:17]' in a._tool_scan_timeline(['opaque:17'],'2020')


@pytest.mark.parametrize('side',['source','target'])
def test_unresolved_explicit_claim_filter_never_broadens_to_unattributed_lookup(side):
    a=agent(GraphStore());calls=[]
    a._resolve_entity_id=lambda value,types: None if value=='Missing' else 'known'
    a.graph_service.find_claims_about_target=lambda **kwargs:calls.append(kwargs) or []
    params={'source':'Known','target':'Known'};params[side]='Missing'
    assert 'No '+side+' found' in a._tool_claim_lookup(**params)
    assert calls==[]


def test_time_normalization_cannot_drop_a_bound_qualifier_or_a_season_year():
    a=agent(GraphStore())
    assert a._normalize_tool_time_query('2020-01-01 before the meeting')=='2020-01-01 before the meeting'
    assert a._normalize_tool_time_query('summer 2020')=='summer 2020'
    assert a._normalize_tool_time_query('March 2020')=='2020-03'


def test_routing_time_span_preserves_operator_order_and_all_bounds():
    from cogmem.agents.skill_runtime import SkillRuntime
    runtime=object.__new__(SkillRuntime)
    assert runtime._extract_time_query('What happened before 2020?')=='before 2020'
    assert runtime._extract_time_query('What happened after 2019 and before 2022?')=='after 2019 and before 2022'


def test_timeline_fact_fallback_requires_validity_not_a_year_mentioned_in_content():
    g=GraphStore();g._g.add_node('wrong',node_type='fact',subject_id='p',content='Plans for 2020',valid_from='2022',valid_to='2022')
    g._g.add_node('right',node_type='fact',subject_id='p',content='Resident of Lisbon',valid_from='2020',valid_to='2020')
    g._g.add_node('archived',node_type='fact',subject_id='p',content='Resident of Porto',valid_from='2020',is_archived=True)
    a=agent(g);a._current_query='';output=a._tool_scan_timeline(['p'],'2020')
    assert '[ID: right]' in output and '[ID: wrong]' not in output and '[ID: archived]' not in output
