from cogmem.memory.entity_matcher import EntityMatcher
from cogmem.storage.evidence_store import EvidenceStore
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.schema import Person


def test_person_registration_preserves_preassigned_source_and_relation_addresses():
    graph = GraphStore()
    matcher = EntityMatcher(graph)
    evidence = EvidenceStore()
    prefix = 'An observer with a sufficiently long descriptive name shared by '
    ids = []
    for suffix in ('one participant', 'another participant'):
        name = prefix + suffix
        key = 'p_' + name.lower().replace(' ', '_')
        person = Person(id=key, canonical_name=name)
        evidence.add_chunk_for_entity(key, 'Raw source for ' + suffix)
        actual, registered = matcher.find_or_create_person(person)
        graph.add_nodes([registered])
        graph._g.add_edge('event_' + suffix, key, rel_type='PARTICIPANT')
        assert actual == key and evidence.get_chunks(actual) == ['Raw source for ' + suffix]
        assert graph.get_node(key)['canonical_name'] == name
        ids.append(actual)
    assert ids[0] != ids[1]


def test_new_person_id_is_not_rederived_from_display_name():
    matcher = EntityMatcher(GraphStore())
    person = Person(id='p_source_scoped_opaque_address', canonical_name='A display name')
    actual, registered = matcher.find_or_create_person(person)
    assert actual == registered.id == 'p_source_scoped_opaque_address'
