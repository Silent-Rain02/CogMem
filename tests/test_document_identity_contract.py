"""Immutable source IDs cannot alias distinct documents in an indexing batch."""
from types import SimpleNamespace

import pytest

from cogmem.memory.indexer import Indexer
from cogmem.storage.schema import Doc


def indexer(processed):
    instance = object.__new__(Indexer)
    reads = []
    def known_ids():
        reads.append(True)
        return set(processed)
    instance._evidence = SimpleNamespace(get_processed_doc_ids=known_ids)
    return instance, reads


@pytest.mark.parametrize('incremental', [False, True])
def test_conflicting_source_ids_reject_entire_batch_before_processing_or_skip(incremental):
    instance, reads = indexer(['source'])
    docs = (Doc(id='source', text=t) for t in ['Original account.', 'A different account.'])
    with pytest.raises(ValueError, match='Conflicting source document id'):
        instance.add_documents(docs, incremental=incremental)
    assert not reads  # An incremental skip must not hide contradictory input IDs.


def test_an_identical_repeated_source_is_still_accepted_by_incremental_skip():
    instance, reads = indexer(['source'])
    instance.add_documents([Doc(id='source', text='Exact original.')]*2, incremental=True)
    assert reads == [True]


def test_distinct_source_namespaces_can_use_equal_local_record_numbers():
    instance, reads = indexer(['collection-a:1', 'collection-b:1'])
    instance.add_documents([Doc(id='collection-a:1', text='First collection.'),
                            Doc(id='collection-b:1', text='Second collection.')], incremental=True)
    assert reads == [True]
