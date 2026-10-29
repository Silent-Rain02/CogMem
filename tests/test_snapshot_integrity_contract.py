"""Corrupt or incompatible projections must fail before replacing usable state."""
import pickle
import faiss
import numpy as np
import pytest
from cogmem.storage.vector_store import VectorStore
from cogmem.storage.typed_vector_store import TypedVectorStore
from cogmem.storage.chunk_store import ChunkStore,Chunk
from cogmem.storage.evidence_store import EvidenceStore


def vector_pair(path,ids=('disk',),dim=2,metric='cosine'):
    store=VectorStore(dim,metric);store.add(list(ids),[[1.]*dim for _ in ids]);store.save(str(path/'index'),str(path/'ids.npy'));return store


def test_vector_count_mismatch_is_rejected_before_replacing_live_state(tmp_path):
    vector_pair(tmp_path)
    with (tmp_path/'ids.npy').open('wb') as f:np.save(f,np.array([],dtype=object))
    live=VectorStore(2);live.add(['live'],[[1.,0.]])
    with pytest.raises(ValueError):live.load(str(tmp_path/'index'),str(tmp_path/'ids.npy'))
    assert live.get_vector('live')==pytest.approx([1.,0.])


def test_vector_missing_half_snapshot_is_rejected(tmp_path):
    vector_pair(tmp_path);(tmp_path/'ids.npy').unlink();live=VectorStore(2);live.add(['live'],[[1.,0.]])
    with pytest.raises((ValueError,FileNotFoundError)):live.load(str(tmp_path/'index'),str(tmp_path/'ids.npy'))
    assert live.get_vector('live')==pytest.approx([1.,0.])


def test_vector_metric_mismatch_is_not_silently_reinterpreted(tmp_path):
    vector_pair(tmp_path,metric='l2');live=VectorStore(2,'cosine')
    with pytest.raises(ValueError):live.load(str(tmp_path/'index'),str(tmp_path/'ids.npy'))
    assert live.size()==0


def test_duplicate_persisted_addresses_are_rejected(tmp_path):
    vector_pair(tmp_path,ids=('a','b'))
    with (tmp_path/'ids.npy').open('wb') as f:np.save(f,np.array(['same','same'],dtype=object))
    with pytest.raises(ValueError):VectorStore(2).load(str(tmp_path/'index'),str(tmp_path/'ids.npy'))


def test_chunk_cardinality_failure_does_not_silently_drop_originals():
    live=ChunkStore(2)
    with pytest.raises(ValueError):live.add([Chunk('a','First','source'),Chunk('b','Second','source')],[[1.,0.]])
    assert live.size()==0


def test_identical_chunk_addresses_within_one_batch_are_idempotent():
    live=ChunkStore(2);chunk=Chunk('a','Exact original','source')
    live.add([chunk,chunk],[[1.,0.],[1.,0.]])
    assert live.size()==1 and live._index.ntotal==1


def test_original_chunk_address_cannot_silently_change_content():
    live=ChunkStore(2);live.add([Chunk('a','Original','source')],[[1.,0.]])
    with pytest.raises(ValueError):live.add([Chunk('a','Changed','source')],[[0.,1.]])
    assert [c.text for c in live._chunks]==['Original']


def test_chunk_load_rejects_dimension_mismatch_without_replacing_live_state(tmp_path):
    disk=ChunkStore(3);disk.add([Chunk('disk','Disk','source')],[[1.,0.,0.]]);disk.save(str(tmp_path/'index'),str(tmp_path/'chunks'))
    live=ChunkStore(2);live.add([Chunk('live','Live','source')],[[1.,0.]])
    with pytest.raises(ValueError):live.load(str(tmp_path/'index'),str(tmp_path/'chunks'))
    assert live._index.d==2 and live._chunks[0].id=='live'


def test_chunk_load_rejects_vector_metadata_count_mismatch(tmp_path):
    disk=ChunkStore(2);disk.add([Chunk('disk','Disk','source')],[[1.,0.]]);disk.save(str(tmp_path/'index'),str(tmp_path/'chunks'))
    (tmp_path/'chunks').write_bytes(pickle.dumps([]))
    with pytest.raises(ValueError):ChunkStore(2).load(str(tmp_path/'index'),str(tmp_path/'chunks'))


def test_typed_bad_later_vector_cannot_partially_commit_earlier_type():
    live=TypedVectorStore(2,enable_global_index=True)
    with pytest.raises(ValueError):live.add_batch_typed(['a','b'],[[1.,0.],[1.,0.,0.]],['event','fact'])
    assert live.size()==0 and live.search_global([1.,0.])==[]


def test_unsupported_type_cannot_create_a_ghost_global_vector():
    live=TypedVectorStore(2,enable_global_index=True)
    with pytest.raises(ValueError):live.add_batch_typed(['a'],[[1.,0.]],['unsupported'])
    assert live.size()==0 and live.search_global([1.,0.])==[]


def test_one_address_cannot_exist_in_multiple_type_shards():
    live=TypedVectorStore(2);live.add_typed('a',[1.,0.],'event')
    with pytest.raises(ValueError):live.add_typed('a',[0.,1.],'fact')
    assert live.get_type_stats()['event']==1 and live.get_type_stats()['fact']==0


def test_reloading_missing_type_shard_does_not_keep_previous_bank_state(tmp_path):
    directory=tmp_path/'new';directory.mkdir();only=VectorStore(2);only.add(['new_fact'],[[0.,1.]])
    only.save(str(directory/'fact_index.faiss'),str(directory/'fact_ids.npy'))
    live=TypedVectorStore(2);live.add_typed('old_event',[1.,0.],'event');live.load(str(directory))
    assert live.get_vector('old_event') is None and live.get_vector('new_fact') is not None


def test_typed_load_is_transactional_when_a_later_shard_is_corrupt(tmp_path):
    disk=TypedVectorStore(2);disk.add_typed('disk',[1.,0.],'event');disk.save(str(tmp_path))
    (tmp_path/'fact_ids.npy').write_bytes(b'corrupt')
    live=TypedVectorStore(2);live.add_typed('live',[0.,1.],'event')
    with pytest.raises(Exception):live.load(str(tmp_path))
    assert live.get_vector('live')==pytest.approx([0.,1.]) and live.get_vector('disk') is None


def test_corrupt_evidence_load_raises_and_preserves_existing_sources(tmp_path):
    f=tmp_path/'evidence';f.write_bytes(b'corrupt')
    live=EvidenceStore();live.add_chunk_for_entity('a','Original')
    with pytest.raises(Exception):live.load(str(f))
    assert live.get_chunks('a')==['Original']
