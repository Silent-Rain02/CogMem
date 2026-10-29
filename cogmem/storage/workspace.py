"""Read a coherent set of existing stores without constructing model clients."""
from pathlib import Path
from typing import NamedTuple
from .graph_store import GraphStore
from .vector_store import VectorStore
from .typed_vector_store import TypedVectorStore
from .chunk_store import ChunkStore
from .evidence_store import EvidenceStore


class WorkspaceView(NamedTuple):
    graph: GraphStore
    vectors: object
    chunks: ChunkStore
    evidence: EvidenceStore


def read_workspace(config, directory, *, _allow_pending=False):
    """Load and validate into temporary stores; caller commits only on success.

    No persistent schema is introduced. Existing index pairs and source files
    remain authoritative; unsupported or cross-bank addresses fail closed.
    """
    root = Path(directory)
    if (root / '.cogmem-save-pending').exists() and not _allow_pending:
        raise RuntimeError('Workspace save is in flight or interrupted; mixed generations cannot be loaded')
    generation_path = root / '.cogmem-generation'
    generation = generation_path.read_bytes() if generation_path.exists() else None
    graph = GraphStore()
    graph.load(str(root / config.graph_file), str(root / config.graph_changelog_file))
    if (root / 'typed_indices').is_dir():
        vectors = TypedVectorStore(config.model.embedding_dim, config.faiss.metric,
                                   config.faiss.enable_global_index)
        vectors.load(str(root / 'typed_indices'))
    elif (root / config.faiss_index_file).exists() or (root / config.entity_store_file).exists():
        vectors = VectorStore(config.model.embedding_dim, config.faiss.metric)
        vectors.load(str(root / config.faiss_index_file), str(root / config.entity_store_file))
    else:
        vectors = TypedVectorStore(config.model.embedding_dim, config.faiss.metric,
                                   config.faiss.enable_global_index)
    chunks = ChunkStore(config.model.embedding_dim, config.faiss.metric)
    chunks.load(str(root / config.chunk_index_file), str(root / config.chunk_meta_file))
    evidence = EvidenceStore()
    evidence.load(str(root / config.evidence_store_file))
    view = WorkspaceView(graph, vectors, chunks, evidence)
    validate_workspace(config, view)
    after = generation_path.read_bytes() if generation_path.exists() else None
    if ((root / '.cogmem-save-pending').exists() and not _allow_pending) or generation != after:
        raise RuntimeError('Workspace changed during load; retry after saving completes')
    return view


def validate_workspace(config, view):
    """Cross-check one in-memory set of stores before publication/binding."""
    graph, vectors, chunks, evidence = view
    if isinstance(vectors, TypedVectorStore):
        for kind, store in vectors._stores.items():
            for key in store._ids:
                node = graph.get_node(key)
                if not node or node.get('node_type') != kind:
                    raise ValueError('Vector address does not resolve to a graph node of the declared type')
    elif any(graph.get_node(key) is None for key in vectors._ids):
        raise ValueError('Vector address is missing from the graph')
    originals = {chunk.text for chunk in chunks._chunks}
    for key, texts in evidence._entity_to_chunks.items():
        if graph.get_node(key) is None or not set(texts) <= originals:
            raise ValueError('Evidence references do not resolve to this workspace original sources')
    if config.faiss.use_vector_search:
        ids = {key for store in vectors._stores.values() for key in store._ids} if isinstance(vectors, TypedVectorStore) else set(vectors._ids)
        expected = {key for key, node in graph._g.nodes(data=True)
                    if node.get('node_type') in {'person','concept','event','fact','claim'} and not node.get('is_archived')}
        if not expected <= ids:
            raise ValueError('Graph contains eligible nodes without vector projections')


def export_workspace(config, directory, view):
    """Publish a new complete workspace directory without overwriting another bank."""
    import os
    import shutil
    import tempfile
    target = Path(directory)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError('Workspace export requires a new or empty target directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.cogmem-export-', dir=target.parent))
    try:
        view.graph.save(str(staging / config.graph_file), str(staging / config.graph_changelog_file))
        if isinstance(view.vectors, TypedVectorStore):
            view.vectors.save(str(staging / 'typed_indices'))
        else:
            view.vectors.save(str(staging / config.faiss_index_file), str(staging / config.entity_store_file))
        view.chunks.save(str(staging / config.chunk_index_file), str(staging / config.chunk_meta_file))
        view.evidence.save(str(staging / config.evidence_store_file))
        read_workspace(config, staging)  # Verify source/index closure before publication.
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def save_workspace(config, directory, view):
    """Serialize before publication, roll back I/O failures, detect interruption.

    Single writer contract: caller serializes saves. Readers refuse an in-flight
    publication instead of accepting a mixture of generations. The recovery
    directory is retained if rollback itself fails or the process is killed.
    This is not a power-loss durability guarantee or a multi-process database.
    """
    import os
    import shutil
    import tempfile
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / '.cogmem-save-pending'
    if marker.exists():
        raise RuntimeError('Workspace has an interrupted save; restore its recorded recovery directory before writing')
    names = list(dict.fromkeys([config.graph_file, config.graph_changelog_file,
        config.chunk_index_file, config.chunk_meta_file, config.evidence_store_file,
        config.faiss_index_file, config.entity_store_file, 'typed_indices',
        'knowledge_graph.json', '.cogmem-generation']))
    for name in names:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError('Workspace file paths must stay inside their workspace')
    recovery = Path(tempfile.mkdtemp(prefix='.cogmem-save-', dir=root.parent))
    staging, backup = recovery / 'new', recovery / 'previous'
    backup.mkdir()
    touched = []
    try:
        export_workspace(config, staging, view)
        # Include the human-readable projection in the same generation.
        view.graph.export_json(str(staging / 'knowledge_graph.json'), indent=2)
        from uuid import uuid4
        (staging / '.cogmem-generation').write_text(uuid4().hex, encoding='ascii')
        import json
        (recovery / 'journal.json').write_text(json.dumps({
            'managed': names, 'previous_present': [name for name in names if (root / name).exists()]
        }), encoding='utf-8')
        # A killed process leaves an explicit, inspectable recovery location.
        with marker.open('x', encoding='utf-8') as stream:
            stream.write(str(recovery.resolve()))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            for name in names:
                old, new, saved = root / name, staging / name, backup / name
                existed = old.exists()
                if existed:
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(old, saved)
                touched.append((name, existed))
                if new.exists():
                    old.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(new, old)
            marker.unlink()  # Commit: only now may a reader load the generation.
        except BaseException:
            try:
                for name, existed in reversed(touched):
                    dest, saved = root / name, backup / name
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    elif dest.exists():
                        dest.unlink()
                    if existed:
                        os.replace(saved, dest)
                marker.unlink()
            except BaseException as recovery_error:
                raise RuntimeError(f'Workspace rollback incomplete; recovery retained at {recovery}') from recovery_error
            raise
    finally:
        if not marker.exists():
            shutil.rmtree(recovery, ignore_errors=True)


def recover_workspace(config, directory):
    """Explicitly restore the pre-save generation after interrupted publication.

    Call only after the writer has stopped. Retain recovery files for inspection;
    this operation restores known previous paths, never rebuilds model output.
    """
    import json
    import os
    import shutil
    root = Path(directory).resolve()
    marker = root / '.cogmem-save-pending'
    if not marker.exists():
        return False
    recovery = Path(marker.read_text(encoding='utf-8')).resolve()
    if recovery.parent != root.parent or not recovery.name.startswith('.cogmem-save-'):
        raise ValueError('Recovery directory is outside this workspace parent')
    journal = json.loads((recovery / 'journal.json').read_text(encoding='utf-8'))
    for name in journal['managed']:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Invalid recovery address')
    for name in reversed(journal['managed']):
        dest, saved = root / name, recovery / 'previous' / name
        if saved.exists() or name not in journal['previous_present']:
            if dest.is_dir():
                shutil.rmtree(dest)
            elif dest.exists():
                dest.unlink()
            if saved.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(saved, dest)
    read_workspace(config, root, _allow_pending=True)
    os.replace(marker, recovery / 'recovered-marker')
    return True
