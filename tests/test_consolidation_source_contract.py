from types import SimpleNamespace
import json
import pytest
from cogmem.config import CogMemConfig
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.typed_vector_store import TypedVectorStore
from cogmem.storage.evidence_store import EvidenceStore
from cogmem.memory.consolidator import MemoryConsolidator


def instance():
    c=object.__new__(MemoryConsolidator);c.config=CogMemConfig();c.graph=GraphStore()
    c.graph._g.add_node('p',node_type='person',canonical_name='Person');c.graph._g.add_node('c',node_type='concept',name='Concept');c.graph._g.add_node('event',node_type='event')
    c.vector_store=TypedVectorStore(2);c.embedding_encoder=SimpleNamespace(encode=lambda _: [[1.,0.]]);c.evidence_store=EvidenceStore();c.evidence_store.add_chunk_for_entity('event','Original')
    return c


@pytest.mark.parametrize('object_id',['foreign','invented_concept'])
def test_synthesis_cannot_invent_another_concept_address(object_id):
    c=instance();c.llm=SimpleNamespace(chat=lambda **kwargs:json.dumps(dict(content='Proposition',predicate='attribute',object_id=object_id)))
    assert c._synthesize_fact('p','Person','c',[{'id':'event'}]) is None


def test_embedding_failure_leaves_no_half_created_fact_or_edges():
    c=instance();c.embedding_encoder=SimpleNamespace(encode=lambda _: [])
    before=(c.graph.num_entities(),c.graph.num_relations())
    with pytest.raises(ValueError):c._create_fact_node('p','c',dict(content='Proposition',predicate='attribute'),[{'id':'event'}])
    assert (c.graph.num_entities(),c.graph.num_relations())==before and c.vector_store.size()==0


def test_fact_without_original_support_is_rejected_before_projection_write():
    c=instance();c.evidence_store=EvidenceStore();before=(c.graph.num_entities(),c.graph.num_relations())
    with pytest.raises(ValueError):c._create_fact_node('p','c',dict(content='Proposition',predicate='attribute'),[{'id':'event'}])
    assert (c.graph.num_entities(),c.graph.num_relations())==before and c.vector_store.size()==0


def test_empty_support_cannot_create_an_unattributed_summary():
    c=instance()
    with pytest.raises(ValueError):c._create_fact_node('p','c',dict(content='Proposition',predicate='attribute'),[])
    assert c.vector_store.size()==0
