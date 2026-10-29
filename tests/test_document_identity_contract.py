"""Immutable source IDs cannot alias distinct documents in an indexing batch."""
from types import SimpleNamespace

import pytest

from cogmem.memory.indexer import Indexer
from cogmem.storage.schema import Doc
from cogmem.storage.evidence_store import EvidenceStore


def indexer(processed):
    instance = object.__new__(Indexer)
    reads = []
    store = EvidenceStore()
    texts = {'source': 'Exact original.', 'collection-a:1': 'First collection.', 'collection-b:1': 'Second collection.'}
    store.register_documents({key: texts[key] for key in processed})
    for key in processed:store.mark_doc_processed(key)
    def known_ids():
        reads.append(True)
        return set(processed)
    store.get_processed_doc_ids = known_ids
    instance._evidence = store
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


@pytest.mark.parametrize('incremental',[False,True])
def test_conflicting_document_identity_across_calls_fails_before_any_new_processing(incremental):
    instance,_=indexer(['source'])
    with pytest.raises(ValueError,match='different original content'):
        instance.add_documents([Doc(id='source',text='A changed original')],incremental=incremental)


def test_document_fingerprints_survive_save_reload(tmp_path):
    store=EvidenceStore();store.register_documents({'source':'Original'});store.mark_doc_processed('source');store.save(str(tmp_path/'evidence'))
    loaded=EvidenceStore();loaded.load(str(tmp_path/'evidence'));loaded.register_documents({'source':'Original'})
    with pytest.raises(ValueError):loaded.register_documents({'source':'Changed'})


def test_legacy_processed_identity_cannot_silently_hide_a_changed_original():
    store=EvidenceStore();store.mark_doc_processed('legacy')
    with pytest.raises(ValueError,match='unverifiable'):store.register_documents({'legacy':'Unverifiable original'})


def test_rejected_identity_batch_does_not_partially_reserve_other_documents():
    store=EvidenceStore();store.register_documents({'existing':'Original'})
    with pytest.raises(ValueError):store.register_documents({'new':'First version','existing':'Changed'})
    store.register_documents({'new':'Different version'})  # Rejected batch did not reserve 'new'.
