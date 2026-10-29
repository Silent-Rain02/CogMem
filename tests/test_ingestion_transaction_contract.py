"""Provider and partial-batch failures cannot publish incomplete source banks."""
from types import SimpleNamespace
import pytest
from cogmem.config import CogMemConfig
from cogmem.memory.indexer import Indexer
from cogmem.storage.schema import Doc
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.chunk_store import ChunkStore, Chunk
from cogmem.storage.typed_vector_store import TypedVectorStore
from cogmem.storage.evidence_store import EvidenceStore
from cogmem.utils.concept_manager import ConceptManager


def instance():
    i=object.__new__(Indexer);i._cfg=CogMemConfig();i._cfg.model.embedding_dim=2
    i._graph=GraphStore();i._typed_vstore=TypedVectorStore(2);i._vstore=None
    i._chunk_store=ChunkStore(2);i._evidence=EvidenceStore();i._concept_manager=ConceptManager();i._cogmem=None
    return i


def write_staged(self,docs,incremental):
    for d in docs:
        self._graph._g.add_node(d.id,node_type='event')
        self._typed_vstore.add_typed(d.id,[1.,0.],'event')
        self._chunk_store.add([Chunk('chunk-'+d.id,d.text,d.id)],[[1.,0.]])
        self._evidence.add_chunk_for_entity(d.id,d.text)
        self._evidence.mark_doc_processed(d.id)


def test_failed_embedding_leaves_live_sources_addresses_and_processed_ids_unchanged(monkeypatch):
    i=instance();before=(i._graph,i._typed_vstore,i._chunk_store,i._evidence)
    def fail(self,docs,incremental):
        write_staged(self,docs,incremental)
        raise RuntimeError('Embedding failed after source extraction')
    monkeypatch.setattr(Indexer,'_add_documents_uncommitted',fail)
    with pytest.raises(RuntimeError,match='Embedding failed'):i.add_documents([Doc(id='new',text='Original')])
    assert before==(i._graph,i._typed_vstore,i._chunk_store,i._evidence)
    assert not i._graph.num_entities() and not i._evidence.get_processed_doc_ids()
    assert not i._evidence._document_hashes and i._chunk_store.size()==0


def test_partial_batch_cannot_look_like_a_successful_commit(monkeypatch):
    i=instance()
    monkeypatch.setattr(Indexer,'_add_documents_uncommitted',lambda self,docs,incremental:write_staged(self,docs[:1],incremental))
    with pytest.raises(RuntimeError,match='Ingestion incomplete'):i.add_documents([Doc(id='a',text='A'),Doc(id='b',text='B')])
    assert i._graph.num_entities()==0 and not i._evidence.get_processed_doc_ids()


def test_complete_commit_deduplicates_identical_input_and_is_retry_safe(monkeypatch):
    i=instance();calls=[]
    def complete(self,docs,incremental):
        calls.append(len(docs));write_staged(self,docs,incremental)
    monkeypatch.setattr(Indexer,'_add_documents_uncommitted',complete)
    docs=[Doc(id='a',text='A')]*2;i.add_documents(docs)
    assert calls==[1] and i._graph.num_entities()==i._typed_vstore.size()==i._chunk_store.size()==1
    i.add_documents(docs,incremental=True)
    assert calls==[1] and i._evidence.get_processed_doc_ids()=={'a'}


def test_missing_vector_fails_before_live_bank_or_query_stack_is_rebound(monkeypatch):
    i=instance();commits=[];i._cogmem=SimpleNamespace(_install_workspace=commits.append,save=lambda:commits.append('save'))
    def incomplete(self,docs,incremental):
        write_staged(self,docs,incremental);self._typed_vstore=TypedVectorStore(2)
    monkeypatch.setattr(Indexer,'_add_documents_uncommitted',incomplete)
    with pytest.raises(ValueError,match='without vector'):i.add_documents([Doc(id='a',text='A')])
    assert not commits and i._graph.num_entities()==0


def test_empty_extraction_with_chunk_embedding_failure_never_commits(monkeypatch):
    i=instance();i._cfg.runtime.max_concurrency=1;i._cfg.chunk.enable_chunk_embedding=False
    i._llm=SimpleNamespace(chat=lambda **kwargs:'{"events":[],"facts":[],"claims":[]}')
    def invalid_vectors(texts):raise ValueError('Invalid embedding shape')
    i._embed=SimpleNamespace(encode=invalid_vectors)
    with pytest.raises(ValueError,match='Invalid embedding shape'):
        i.add_documents([Doc(id='source',text='Hello there.')])
    assert i._chunk_store.size()==0 and not i._evidence.get_processed_doc_ids()
    assert not i._evidence._document_hashes and i._graph.num_entities()==0
