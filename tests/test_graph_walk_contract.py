from types import SimpleNamespace
from cogmem.config import CogMemConfig
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.graph_service import GraphService
from cogmem.storage.schema import Relation
from cogmem.agents.search_agent import SearchAgent
from cogmem.memory.consolidator import MemoryConsolidator


def graph():
    g=GraphStore()
    for key in ['a','b','c']:g._g.add_node(key,node_type='event',title='Relevant meeting')
    return g


def test_incoming_walk_visits_actual_neighbors_and_reaches_second_hop():
    g=graph();g.add_relations([Relation(head_id='c',tail_id='b',rel_type='SUPPORTED_BY'),Relation(head_id='b',tail_id='a',rel_type='SUPPORTED_BY')])
    rows=GraphService(g,None,None,use_vector_search=False).traverse_associations(['a'],direction='in',max_depth=2)
    assert {r.target_id for r in rows}=={'b','c'}
    assert all(r.target_node['id']==r.target_id for r in rows)


def test_both_direction_walk_preserves_incoming_and_outgoing_edge_types():
    g=graph();g.add_relations([Relation(head_id='a',tail_id='b',rel_type='INVOLVES'),Relation(head_id='b',tail_id='a',rel_type='SUPPORTED_BY')])
    rows=GraphService(g,None,None,use_vector_search=False).traverse_associations(['a'],direction='both',max_depth=1)
    assert {(r.target_id,r.relation_type) for r in rows}=={('b','INVOLVES'),('b','SUPPORTED_BY')}


def test_multiple_roots_have_independent_walk_state():
    g=graph();g.add_relations([Relation(head_id='a',tail_id='c',rel_type='INVOLVES'),Relation(head_id='b',tail_id='c',rel_type='INVOLVES')])
    rows=GraphService(g,None,None,use_vector_search=False).traverse_associations(['a','b'],max_depth=1)
    assert {(r.source_id,r.target_id) for r in rows}=={('a','c'),('b','c')}


def test_identical_relation_insertion_is_idempotent_but_new_versions_survive():
    g=graph();edge=Relation(head_id='a',tail_id='b',rel_type='INVOLVES');g.add_relations([edge,edge]);g.add_relations([edge])
    assert g.num_relations()==1
    g.add_relations([edge.model_copy(update={'version':2})]);assert g.num_relations()==2


def test_neighbor_budget_counts_distinct_nodes_not_parallel_edge_records():
    g=graph();g.add_relations([Relation(head_id='a',tail_id='b',rel_type='INVOLVES'),Relation(head_id='a',tail_id='b',rel_type='SUPPORTED_BY'),Relation(head_id='a',tail_id='c',rel_type='INVOLVES')])
    a=SearchAgent(GraphService(g,None,None,use_vector_search=False),None,config=CogMemConfig());a._current_query='meeting';a.traverse_neighbors_max_results=2
    a._tool_traverse_neighbors(node_ids=['a'],max_depth=1)
    assert set(a._last_tool_node_hits['traverse_neighbors'])=={'b','c'}


def test_parallel_edges_do_not_count_as_repeated_independent_events():
    g=graph();g._g.add_node('person',node_type='person');g._g.add_node('concept',node_type='concept',name='meeting')
    for _ in range(3):
        g._g.add_edge('person','a',rel_type='PARTICIPATED_IN');g._g.add_edge('a','concept',rel_type='INVOLVES')
    instance=object.__new__(MemoryConsolidator);instance.graph=g
    events=instance._get_participated_events('person');assert len(events)==1
    assert len(instance._cluster_events_by_concept(events)['concept'])==1
