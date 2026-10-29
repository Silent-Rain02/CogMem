"""Whole-workspace loads cannot mix two banks or bypass original provenance."""
from types import SimpleNamespace
import pickle
import pytest
from cogmem.config import CogMemConfig
from cogmem.cogmem import CogMem
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.typed_vector_store import TypedVectorStore
from cogmem.storage.chunk_store import ChunkStore,Chunk
from cogmem.storage.evidence_store import EvidenceStore
from cogmem.storage.workspace import read_workspace
from cogmem.memory.indexer import Indexer


def config():
    cfg=CogMemConfig();cfg.model.embedding_dim=2;return cfg


def save_bank(root, name='new'):
    root.mkdir();cfg=config();g=GraphStore();g._g.add_node(name,node_type='event',title=name);g.save(str(root/cfg.graph_file))
    v=TypedVectorStore(2);v.add_typed(name,[1.,0.],'event');v.save(str(root/'typed_indices'))
    chunks=ChunkStore(2);chunks.add([Chunk('chunk_'+name,name,name)],[[1.,0.]]);chunks.save(str(root/cfg.chunk_index_file),str(root/cfg.chunk_meta_file))
    e=EvidenceStore();e.add_chunk_for_entity(name,name);e.save(str(root/cfg.evidence_store_file));return cfg


def memory(root):
    cm=object.__new__(CogMem);cm.cfg=config();cm.workspace_dir=str(root);cm.graph=GraphStore();cm.graph._g.add_node('live',node_type='event')
    cm.vstore=TypedVectorStore(2);cm.chunk_store=ChunkStore(2);cm.indexer=object.__new__(Indexer);cm.indexer._graph=cm.graph;cm.indexer._typed_vstore=cm.vstore;cm.indexer._chunk_store=cm.chunk_store;cm.indexer._evidence=EvidenceStore()
    cm._embed=object();cm.llm=object();cm.qa_orchestrator=object();return cm


def test_corrupt_later_snapshot_keeps_entire_previous_live_bank(tmp_path):
    bank=tmp_path/'new';cfg=save_bank(bank);(bank/cfg.evidence_store_file).write_bytes(b'corrupt')
    cm=memory(tmp_path);before=(cm.graph,cm.vstore,cm.chunk_store,cm.indexer._evidence,cm.qa_orchestrator)
    with pytest.raises(Exception):cm.load(str(bank))
    assert before==(cm.graph,cm.vstore,cm.chunk_store,cm.indexer._evidence,cm.qa_orchestrator)
    assert 'live' in cm.graph._g


def test_successful_reload_rebinds_all_components_to_one_bank(tmp_path):
    bank=tmp_path/'new';save_bank(bank);cm=memory(tmp_path);cm.load(str(bank))
    agent=cm.qa_orchestrator.search_agent
    assert set(cm.graph._g)=={'new'} and cm.indexer._graph is cm.graph
    assert cm.indexer._typed_vstore is cm.vstore
    assert agent.graph_service.graph_store is cm.graph and agent.graph_service.vector_store is cm.vstore
    assert agent.graph_service.chunk_store is cm.chunk_store
    assert agent.evidence_store is cm.indexer._evidence
    assert agent._tool_get_evidence('new').endswith('[Evidence 1] new')


def test_derived_vector_cannot_address_a_missing_graph_node(tmp_path):
    bank=tmp_path/'new';save_bank(bank);v=TypedVectorStore(2);v.add_typed('foreign',[1.,0.],'event');v.save(str(bank/'typed_indices'))
    with pytest.raises(ValueError,match='graph node'):read_workspace(config(),bank)


def test_evidence_cannot_inject_an_unrelated_banks_source_text(tmp_path):
    bank=tmp_path/'new';cfg=save_bank(bank);e=EvidenceStore();e.add_chunk_for_entity('new','Another bank original');e.save(str(bank/cfg.evidence_store_file))
    with pytest.raises(ValueError,match='original sources'):read_workspace(cfg,bank)


def test_source_only_snapshot_can_be_loaded_without_provider_calls(tmp_path):
    bank=tmp_path/'new';cfg=save_bank(bank);view=read_workspace(cfg,bank)
    assert view.evidence.get_chunks('new')==['new']


def test_new_workspace_export_preserves_complete_sources_and_all_projections(tmp_path):
    from cogmem.storage.workspace import export_workspace
    source=tmp_path/'source';cfg=save_bank(source);view=read_workspace(cfg,source);target=tmp_path/'target'
    export_workspace(cfg,target,view);restored=read_workspace(cfg,target)
    assert restored.evidence.get_chunks('new')==['new'] and restored.chunks._chunks[0].text=='new'
    assert restored.vectors.get_vector('new')==pytest.approx([1.,0.])
    assert restored.graph.num_entities()==view.graph.num_entities()


def test_failed_export_never_publishes_a_partial_workspace(tmp_path,monkeypatch):
    from cogmem.storage.workspace import export_workspace
    source=tmp_path/'source';cfg=save_bank(source);view=read_workspace(cfg,source);target=tmp_path/'target'
    def fail(*args,**kwargs):raise OSError('disk failure')
    monkeypatch.setattr(view.evidence,'save',fail)
    with pytest.raises(OSError):export_workspace(cfg,target,view)
    assert not target.exists() and not list(tmp_path.glob('.cogmem-export-*'))
    assert read_workspace(cfg,source).evidence.get_chunks('new')==['new']


def test_consolidation_source_and_target_do_not_construct_default_model_clients(tmp_path,monkeypatch):
    from cogmem.memory.consolidator import MemoryConsolidator
    import cogmem.cogmem as module
    def forbidden(*args,**kwargs):raise AssertionError('No CogMem/model probing should occur during workspace transfer')
    monkeypatch.setattr(module,'CogMem',forbidden)
    source=tmp_path/'source';cfg=save_bank(source);target=tmp_path/'target'
    instance=object.__new__(MemoryConsolidator);instance.config=cfg;instance.graph=GraphStore();instance.vector_store=TypedVectorStore(2);instance.embedding_encoder=None;instance.llm=None;instance.evidence_store=EvidenceStore();instance.chunk_store=ChunkStore(2)
    instance.run_consolidation(source_workspace=str(source),target_workspace=str(target),merge_concepts=False,deduplicate_facts=False,reconcile_claims=False)
    restored=read_workspace(cfg,target)
    assert restored.evidence.get_chunks('new')==['new'] and instance.evidence_store.get_chunks('new')==['new']


def test_consolidation_rejects_existing_target_before_changing_live_state(tmp_path):
    from cogmem.memory.consolidator import MemoryConsolidator
    target=tmp_path/'target';target.mkdir();(target/'keep').write_text('Existing user data')
    instance=object.__new__(MemoryConsolidator);instance.config=config();original=GraphStore();instance.graph=original
    with pytest.raises(FileExistsError):instance.run_consolidation(target_workspace=str(target))
    assert instance.graph is original and (target/'keep').read_text()=='Existing user data'


def test_existing_workspace_save_rolls_back_a_mid_publication_io_error(tmp_path,monkeypatch):
    import os
    from cogmem.storage.workspace import save_workspace
    source=tmp_path/'bank';cfg=save_bank(source);before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    replacement=tmp_path/'replacement';save_bank(replacement,'replacement');view=read_workspace(cfg,replacement)
    real=os.replace;failed=[]
    def fail_once(src,dest):
        if str(dest)==str(source/'typed_indices') and not failed:
            failed.append(True);raise OSError('publication interrupted')
        return real(src,dest)
    monkeypatch.setattr(os,'replace',fail_once)
    with pytest.raises(OSError,match='publication interrupted'):save_workspace(cfg,source,view)
    after={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert failed and before==after and not list(tmp_path.glob('.cogmem-save-*'))
    assert set(read_workspace(cfg,source).graph._g)=={'new'}


def test_save_success_publishes_complete_generation_and_preserves_unmanaged_files(tmp_path):
    from cogmem.storage.workspace import save_workspace
    bank=tmp_path/'bank';cfg=save_bank(bank);(bank/'notes.txt').write_text('User notes')
    new=tmp_path/'new';save_bank(new,'replacement');save_workspace(cfg,bank,read_workspace(cfg,new))
    assert set(read_workspace(cfg,bank).graph._g)=={'replacement'}
    assert (bank/'notes.txt').read_text()=='User notes' and (bank/'.cogmem-generation').exists()
    assert not (bank/'.cogmem-save-pending').exists()


def test_interrupted_generation_is_neither_read_nor_overwritten(tmp_path):
    from cogmem.storage.workspace import save_workspace
    bank=tmp_path/'bank';cfg=save_bank(bank);view=read_workspace(cfg,bank)
    marker=bank/'.cogmem-save-pending';marker.write_text('Retained recovery directory')
    with pytest.raises(RuntimeError,match='interrupted'):read_workspace(cfg,bank)
    with pytest.raises(RuntimeError,match='interrupted'):save_workspace(cfg,bank,view)
    assert marker.read_text()=='Retained recovery directory'


def test_generation_change_during_load_is_rejected(tmp_path,monkeypatch):
    bank=tmp_path/'bank';cfg=save_bank(bank);(bank/'.cogmem-generation').write_text('before')
    real=EvidenceStore.load
    def concurrent_save(self,path):
        real(self,path);(bank/'.cogmem-generation').write_text('after')
    monkeypatch.setattr(EvidenceStore,'load',concurrent_save)
    with pytest.raises(RuntimeError,match='changed during load'):read_workspace(cfg,bank)


def test_failed_consolidation_does_not_publish_partial_mutations(tmp_path,monkeypatch):
    import cogmem.memory.consolidator as module
    cm=memory(tmp_path);original=cm.graph;before=cm.qa_orchestrator
    class Failed:
        def __init__(self,**kwargs):self.graph=kwargs['graph']
        def run_consolidation(self,**kwargs):
            self.graph._g.add_node('half-created',node_type='fact')
            return {'errors':['An optional phase failed'],'facts_created':1}
    monkeypatch.setattr(module,'MemoryConsolidator',Failed)
    cm.save=lambda:pytest.fail('A failed phase must not save')
    stats=cm.consolidate_memory()
    assert stats['committed'] is False and cm.graph is original and 'half-created' not in cm.graph._g
    assert cm.qa_orchestrator is before


def test_forced_in_place_rebuild_cannot_silently_merge_or_overwrite_sources(tmp_path):
    cm=memory(tmp_path)
    with pytest.raises(ValueError,match='new workspace'):cm.add_documents_incremental([],force_rebuild=True)
    assert 'live' in cm.graph._g


def test_corrupt_index_is_not_treated_as_a_missing_bank(tmp_path):
    bank=tmp_path/'bank';cfg=save_bank(bank);(bank/cfg.chunk_meta_file).unlink();cm=memory(bank)
    with pytest.raises(ValueError,match='Incomplete chunk'):cm.is_indexed()


def test_failed_rollback_keeps_a_recoverable_previous_generation(tmp_path,monkeypatch):
    import os
    from cogmem.storage.workspace import save_workspace,recover_workspace
    bank=tmp_path/'bank';cfg=save_bank(bank);before={str(p.relative_to(bank)):p.read_bytes() for p in bank.rglob('*') if p.is_file()}
    new=tmp_path/'new';save_bank(new,'replacement');view=read_workspace(cfg,new);real=os.replace
    def fail_typed(src,dest):
        if str(dest)==str(bank/'typed_indices'):raise OSError('Device unavailable during publication and rollback')
        return real(src,dest)
    monkeypatch.setattr(os,'replace',fail_typed)
    with pytest.raises(RuntimeError,match='rollback incomplete'):save_workspace(cfg,bank,view)
    with pytest.raises(RuntimeError):read_workspace(cfg,bank)
    monkeypatch.setattr(os,'replace',real)
    assert recover_workspace(cfg,bank) is True
    assert {str(p.relative_to(bank)):p.read_bytes() for p in bank.rglob('*') if p.is_file()}==before
    assert set(read_workspace(cfg,bank).graph._g)=={'new'}


def test_processed_empty_extraction_still_requires_its_original_document(tmp_path):
    from cogmem.storage.workspace import WorkspaceView,validate_workspace
    cfg=config();evidence=EvidenceStore();evidence.register_documents({'source':'An original with no extracted entities'})
    evidence.mark_doc_processed('source');view=WorkspaceView(GraphStore(),TypedVectorStore(2),ChunkStore(2),evidence)
    with pytest.raises(ValueError,match='missing their original'):validate_workspace(cfg,view)
    view.chunks.add([Chunk('raw','An original with no extracted entities','source')],[[1.,0.]])
    validate_workspace(cfg,view)  # A legitimate empty graph with its source is valid.
