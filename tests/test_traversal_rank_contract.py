from cogmem.agents.search_agent import SearchAgent
from cogmem.config import CogMemConfig
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.graph_service import GraphService
from cogmem.storage.schema import Relation


def selected(order):
    graph=GraphStore()
    graph._g.add_node('person_mara',node_type='person',canonical_name='Mara')
    graph._g.add_node('fact_irrelevant',node_type='fact',content='Mara likes grocery shopping')
    graph._g.add_node('fact_relevant',node_type='fact',content='Mara attended a ceramics workshop')
    graph._g.add_node('outside_scope',node_type='fact',content='Mara ceramics workshop')
    graph.add_relations([Relation(head_id='person_mara',tail_id=node,rel_type='HAS_ATTRIBUTE') for node in order])
    agent=SearchAgent(GraphService(graph,None,None,use_vector_search=False),None,config=CogMemConfig())
    agent._current_query='Which ceramics workshop did Mara attend?'
    agent.traverse_neighbors_max_results=1
    agent._tool_traverse_neighbors(node_ids=['person_mara'],max_depth=1)
    return agent._last_tool_node_hits['traverse_neighbors']


def test_relevant_neighbor_is_not_starved_by_insertion_order_and_scope_is_preserved():
    assert selected(['fact_irrelevant','fact_relevant'])==['fact_relevant']
    assert selected(['fact_relevant','fact_irrelevant'])==['fact_relevant']


def test_common_ground_keeps_its_existing_type_bonus():
    agent=object.__new__(SearchAgent)
    assert agent._score_query_node({'node_type':'concept','name':'ceramics'},{'ceramics'})==3
    assert agent._score_common_ground_node({'node_type':'concept','name':'ceramics'},{'ceramics'})==5
    assert agent._score_common_ground_node({'node_type':'pattern','content':'ceramics'},{'ceramics'})==7
    assert agent._score_common_ground_node({'node_type':'fact','content':'ceramics'},{'ceramics'})==3
