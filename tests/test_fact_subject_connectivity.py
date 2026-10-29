from cogmem.config import CogMemConfig
from cogmem.memory.indexer import Indexer
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.schema import Fact
from cogmem.utils.concept_manager import ConceptManager
import pytest


def build(payload, source):
    indexer=object.__new__(Indexer)
    indexer._cfg=CogMemConfig()
    indexer._concept_manager=ConceptManager()
    return indexer._build_event_mediated_graph(payload,'session_1','2020-03-01','2020-03-01',source_id=source)


@pytest.mark.parametrize('value', [{'object':'tea'}, {'purpose':'stay fit'}])
def test_atomic_fact_is_reachable_from_its_declared_subject(value):
    nodes,relations=build({'facts':[{'subject':'Mara','predicate':'likes',**value}]},'doc1')
    fact=next(n for n in nodes if isinstance(n,Fact))
    links=[r for r in relations if r.head_id==fact.subject_id and r.tail_id==fact.id and r.rel_type=='HAS_ATTRIBUTE']
    assert len(links)==1
    graph=GraphStore()
    for node in nodes:graph.add_node(node)
    graph.add_relations(relations)
    assert graph._g.has_edge(fact.subject_id,fact.id)


def test_distinct_owners_and_occurrences_keep_separate_reachable_fact_sources(tmp_path):
    graph=GraphStore();facts=[]
    payload={'facts':[{'subject':'Mara','predicate':'likes','object':'tea'},
                      {'subject':'Leo','predicate':'likes','object':'tea'}]}
    for source in ['first_source','second_source']:
        nodes,relations=build(payload,source)
        facts.extend(n for n in nodes if isinstance(n,Fact))
        for node in nodes:graph.add_node(node)
        graph.add_relations(relations)
    assert len({n.id for n in facts})==4
    for fact in facts:
        owners=[a for a,b,data in graph._g.in_edges(fact.id,data=True) if data['rel_type']=='HAS_ATTRIBUTE']
        assert owners==[fact.subject_id]
        assert fact.metadata.source in {'first_source','second_source'}
    graph.save(str(tmp_path/'graph.pkl'))
    restored=GraphStore();restored.load(str(tmp_path/'graph.pkl'))
    assert all(restored._g.has_edge(f.subject_id,f.id) for f in facts)
