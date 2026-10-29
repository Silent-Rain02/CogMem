from cogmem.config import CogMemConfig
from cogmem.memory.indexer import Indexer
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.schema import Claim,Person,Concept,Metadata

def builder():
    indexer=object.__new__(Indexer)
    indexer._cfg=CogMemConfig();indexer._graph=GraphStore()
    for name in ['Mira','Niko']:
        indexer._graph.add_node(Person(id='p_'+name.lower(),canonical_name=name))
    indexer._graph.add_node(Concept(id='c_crowds',name='crowds'))
    return indexer

def claim(**changes):
    values=dict(id='claim_first',source_id='p_mira',target_id='p_niko',content='dislikes crowds',
                metadata=Metadata(source='turn-session_1-D1:1',created_at='2020-01-01'))
    values.update(changes)
    return Claim(**values)

def test_claim_embedding_uses_semantics_and_excludes_occurrence_bookkeeping():
    indexer=builder();first=claim()
    second=claim(id='claim_other',metadata=Metadata(source='turn-session_2-D2:8',created_at='2030-09-08'))
    a=indexer._generate_rich_node_text(first)
    b=indexer._generate_rich_node_text(second)
    assert a==b
    assert 'Mira' in a and 'Niko' in a and first.content in a
    for noise in ['claim_first','claim_other','created_at','timestamp','confidence','turn-session','p_mira','p_niko']:
        assert noise not in a

def test_claim_embedding_keeps_source_target_direction_and_topic_targets():
    indexer=builder()
    forward=indexer._generate_rich_node_text(claim())
    reverse=indexer._generate_rich_node_text(claim(source_id='p_niko',target_id='p_mira'))
    topic=indexer._generate_rich_node_text(claim(target_id='c_crowds',content='are overwhelming'))
    assert forward!=reverse
    assert forward.index('Mira')<forward.index('Niko')
    assert reverse.index('Niko')<reverse.index('Mira')
    assert 'crowds' in topic and 'are overwhelming' in topic
    assert indexer._generate_rich_node_text(claim(content='loves crowds'))!=forward
