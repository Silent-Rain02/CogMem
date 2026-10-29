from cogmem.storage.graph_service import GraphService
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.schema import Event, Fact


def lexical(graph):
    return GraphService(graph, None, None, use_vector_search=False)


def people(aliases, name='Iris'):
    graph = GraphStore()
    graph._g.add_node('person_opaque', node_type='person', canonical_name=name, aliases=aliases)
    return lexical(graph)


def test_weaker_alias_cannot_downgrade_exact_canonical_name():
    result = people(['Iris Park']).find_anchors('Iris', top_k=1)
    assert [(r.node_id, r.score) for r in result] == [('person_opaque', 1.)]


def test_alias_order_does_not_hide_a_stronger_match():
    for aliases in [['Iri', 'Iris'], ['Iris', 'Iri']]:
        result = people(aliases, name='Nora').find_anchors('Iris', top_k=1)
        assert [(r.node_id, r.score) for r in result] == [('person_opaque', 1.)]


def test_blank_alias_cannot_match_every_question():
    assert people(['', '   '], name='Nora').find_anchors('harpsichord', top_k=1) == []


def test_existing_partial_alias_match_is_preserved():
    result = people(['Iris'], name='Nora').find_anchors('Where is Iris now?', top_k=1)
    assert [(r.node_id, r.score) for r in result] == [('person_opaque', .85)]


def bank(content=None):
    graph = GraphStore()
    graph.add_nodes([Event(id=f'event_{i}', title=title) for i, title in
                     enumerate(['A commute', 'A meal', 'A walk'])])
    if content:
        graph.add_node(fact(content))
    return graph


def fact(content):
    return Fact(id='fact_opaque', subject_id='person_iris', predicate='repairs',
                object_id='concept_instruments', content=content)


def ids(service, query):
    return [r.node_id for r in service.find_anchors(query, types=['fact'], top_k=1)]


def test_lexical_cache_sees_a_new_fact_after_first_query():
    graph = bank(); service = lexical(graph)
    assert ids(service, 'harpsichords') == []
    graph.add_node(fact('Iris repairs harpsichords'))
    assert ids(service, 'harpsichords') == ['fact_opaque']


def test_lexical_cache_sees_upsert_without_node_count_change():
    graph = bank('Iris repairs harpsichords'); service = lexical(graph)
    assert ids(service, 'harpsichords') == ['fact_opaque']
    before = graph.node_count
    graph.add_node(fact('Iris repairs theremins'))
    assert graph.node_count == before
    assert ids(service, 'theremins') == ['fact_opaque']
    assert ids(service, 'harpsichords') == []


def test_lexical_cache_sees_reload_with_same_node_and_changelog_counts(tmp_path):
    graph = bank('Iris repairs harpsichords'); service = lexical(graph)
    assert ids(service, 'harpsichords') == ['fact_opaque']
    other = bank('Iris repairs theremins')
    assert other.node_count == graph.node_count and len(other._changelog) == len(graph._changelog)
    path = tmp_path/'graph.pkl'; other.save(str(path))
    graph.load(str(path))
    assert ids(service, 'theremins') == ['fact_opaque']


def test_unchanged_graph_reuses_the_real_bm25_index():
    service = lexical(bank('Iris repairs harpsichords'))
    assert ids(service, 'harpsichords') == ['fact_opaque']
    cached = service._bm25_retriever
    assert ids(service, 'harpsichords') == ['fact_opaque']
    assert service._bm25_retriever is cached
