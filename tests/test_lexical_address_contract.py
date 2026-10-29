from cogmem.storage.graph_service import GraphService
from cogmem.storage.graph_store import GraphStore
from cogmem.utils.bm25_retriever import BM25Retriever


def test_fact_body_is_searchable_without_exposing_occurrence_address():
    graph = GraphStore()
    graph._g.add_node('fact_opaque_occurrence', node_type='fact',
                      content='Iris repairs harpsichords')
    graph._g.add_node('unrelated', node_type='person', canonical_name='Nora')
    graph._g.add_node('another', node_type='person', canonical_name='Milo')
    service = GraphService(graph, None, None, use_vector_search=False)
    service._build_bm25_index()
    corpus = service._bm25_retriever.corpus
    assert 'Iris repairs harpsichords' in corpus
    assert all('opaque_occurrence' not in text for text in corpus)
    assert [r.node_id for r in service.find_anchors('harpsichords', types=['fact'], top_k=1)] == ['fact_opaque_occurrence']


def test_concept_lexical_address_uses_display_name_not_storage_key():
    graph = GraphStore()
    graph._g.add_node('c_opaque_storage', node_type='concept', name='glassblowing')
    graph._g.add_node('other', node_type='person', canonical_name='Nora')
    service = GraphService(graph, None, None, use_vector_search=False)
    service._build_bm25_index()
    assert 'glassblowing' in service._bm25_retriever.corpus
    assert all('opaque storage' not in text for text in service._bm25_retriever.corpus)


def test_lexical_scope_precedes_top_k_and_preserves_global_scores():
    retriever = BM25Retriever(['harpsichord', 'harpsichord',
                              'a long account about a harpsichord repair',
                              'ordinary groceries', 'a commute', 'reading', 'swimming'])
    global_scores = dict(retriever.search('harpsichord', top_k=7))
    result = retriever.search('harpsichord', top_k=1, allowed_indices={2})
    assert result == [(2, global_scores[2])]
    assert retriever.search('harpsichord', top_k=1, allowed_indices=set()) == []


def test_typed_anchor_is_not_starved_by_unrelated_or_archived_nodes():
    graph = GraphStore()
    for index in range(4):
        graph._g.add_node(f'person_{index}', node_type='person', canonical_name='harpsichord')
    graph._g.add_node('archived', node_type='claim', content='harpsichord', is_archived=True)
    graph._g.add_node('wanted', node_type='claim', content='an account of a harpsichord repair')
    for index in range(8):
        graph._g.add_node(f'filler_{index}', node_type='event', title=f'commute number {index}')
    service = GraphService(graph, None, None, use_vector_search=False)
    result = service.find_anchors('harpsichord', types=['claim'], top_k=1)
    assert [r.node_id for r in result] == ['wanted']
