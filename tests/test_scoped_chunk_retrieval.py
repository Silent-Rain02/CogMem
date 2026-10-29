from types import SimpleNamespace

import pytest

from cogmem.storage.chunk_store import Chunk, ChunkStore
from cogmem.storage.evidence_store import EvidenceStore


def test_source_scope_is_ranked_before_global_top_k_truncation():
    store = ChunkStore(2)
    distractors = [Chunk(str(i), f'unrelated-{i}', 'other') for i in range(20)]
    permitted = [Chunk('a', 'source-a', 'doc'), Chunk('b', 'source-b', 'doc')]
    store.add(distractors+permitted, [[1., 0.]]*20+[[.5, .5], [.2, .8]])
    evidence = EvidenceStore()
    for chunk in permitted:
        evidence.add_chunk_for_entity('node', chunk.text)
    calls = []
    def encode(texts):
        calls.append(texts)
        assert texts == ['query']  # Reuse stored vectors; no re-embedding sources.
        return [[1., 0.]]
    assert evidence.get_chunks_by_query('node', 'query', 2, SimpleNamespace(encode=encode), store) == ['source-a', 'source-b']
    assert len(calls) == 1


def test_scoped_search_handles_l2_and_detects_missing_sources():
    store = ChunkStore(2, metric='l2')
    store.add([Chunk('a', 'A', 'doc'), Chunk('b', 'B', 'doc')], [[10., 0.], [2., 0.]])
    assert [chunk.id for chunk, _ in store.search_in_scope([1., 0.], {'A', 'B'}, top_k=2)] == ['b', 'a']
    with pytest.raises(ValueError):
        store.search_in_scope([1., 0.], {'missing'})
