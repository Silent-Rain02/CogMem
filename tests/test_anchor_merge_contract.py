from types import SimpleNamespace

from cogmem.storage.graph_service import GraphService
from cogmem.storage.graph_store import GraphStore


def service(graph, hits):
    vectors = SimpleNamespace(search=lambda query, top_k: hits[:top_k])
    encoder = SimpleNamespace(encode=lambda texts: [[1., 0.] for _ in texts])
    return GraphService(graph, vectors, encoder)


def test_weaker_lexical_hit_cannot_hide_the_same_nodes_stronger_vector_score():
    graph = GraphStore()
    graph._g.add_node('target', node_type='fact', content='harpsichord repair')
    graph._g.add_node('other', node_type='event', title='a recital')
    for i in range(5):
        graph._g.add_node(f'fill{i}', node_type='event', title=f'commute {i}')
    result = service(graph, [('target', .99), ('other', .8)]).find_anchors('harpsichord', top_k=1)
    assert [(r.node_id, r.score) for r in result] == [('target', .99)]


def test_vector_candidates_are_merged_before_final_top_k_even_with_many_lexical_hits():
    graph = GraphStore()
    names = ['amber', 'birch', 'cedar', 'dahlia', 'elm', 'fir', 'gingko', 'hazel']
    for name in names:
        graph._g.add_node('concept_' + name, node_type='concept', name=name)
    graph._g.add_node('target', node_type='event', title='A relevant gathering')
    result = service(graph, [('target', .99)]).find_anchors(' '.join(names), top_k=1)
    assert [(r.node_id, r.score) for r in result] == [('target', .99)]


def test_merging_a_weaker_vector_hit_preserves_exact_name_score():
    graph = GraphStore()
    graph._g.add_node('person_opaque', node_type='person', canonical_name='Iris')
    result = service(graph, [('person_opaque', .7)]).find_anchors('Iris', top_k=1)
    assert [(r.node_id, r.score) for r in result] == [('person_opaque', 1.)]


def test_archived_vector_candidate_still_cannot_enter_result():
    graph = GraphStore()
    graph._g.add_node('inactive', node_type='event', title='An old gathering', is_archived=True)
    graph._g.add_node('active', node_type='event', title='A new gathering')
    result = service(graph, [('inactive', .99), ('active', .8)]).find_anchors('gathering', top_k=1)
    assert [r.node_id for r in result] == ['active']
