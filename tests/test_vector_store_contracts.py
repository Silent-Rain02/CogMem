import pytest
from cogmem.storage.vector_store import VectorStore
from cogmem.storage.typed_vector_store import TypedVectorStore


def test_upsert_replaces_stale_vector_and_preserves_other_ids_across_reload(tmp_path):
    store = VectorStore(2)
    store.add(['a', 'b'], [[1., 0.], [0., 1.]])
    store.add(['a', 'c', 'c'], [[-1., 0.], [1., 1.], [-1., 0.]])
    assert store.size() == 3
    assert [key for key, _ in store.search([1., 0.], 3)].count('a') == 1
    assert store.get_vector('a') == pytest.approx([-1., 0.])
    store.save(str(tmp_path / 'index'), str(tmp_path / 'ids.npy'))
    restored = VectorStore(2)
    restored.load(str(tmp_path / 'index'), str(tmp_path / 'ids.npy'))
    assert restored.get_vector('a') == pytest.approx([-1., 0.])
    restored.remove('a')
    assert restored.size() == 2
    assert restored.get_vector('a') is None
    assert {key for key, _ in restored.search([1., 0.], 3)} == {'b', 'c'}


def test_typed_counts_and_global_index_follow_upserts_and_removal():
    store = TypedVectorStore(2, enable_global_index=True)
    store.add_typed('a', [1., 0.], 'event')
    store.add_typed('a', [0., 1.], 'event')
    assert store.size('event') == 1
    assert store.get_vector('a') == pytest.approx([0., 1.])
    store.remove_typed('a', 'event')
    assert store.get_vector('a') is None
    assert store.search_global([1., 0.]) == []
    assert store.size('event') == 0


def test_vector_input_mismatch_fails_before_index_mutation():
    store = VectorStore(2)
    with pytest.raises(ValueError):
        store.add(['a', 'b'], [[1., 0.]])
    assert store.size() == 0


def test_empty_typed_indexes_persist_removal_and_statistics_after_reload(tmp_path):
    store = TypedVectorStore(2)
    assert store.size() == 0
    store.add_typed('a', [1., 0.], 'event')
    store.save(str(tmp_path))
    store.remove_typed('a', 'event')
    store.save(str(tmp_path))
    restored = TypedVectorStore(2)
    restored.load(str(tmp_path))
    assert restored.size() == 0
    assert restored.get_type_stats()['event'] == 0
    assert restored.search_by_type([1., 0.], 'event') == []
