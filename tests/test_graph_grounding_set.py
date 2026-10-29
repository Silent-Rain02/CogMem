from copy import deepcopy
from types import SimpleNamespace

from cogmem.agents.search_agent import SearchAgent
from cogmem.storage.chunk_store import Chunk, ChunkStore
from cogmem.storage.evidence_store import EvidenceStore
from cogmem.storage.graph_store import GraphStore


def instance():
    agent=object.__new__(SearchAgent)
    graph=GraphStore()
    graph._g.add_nodes_from([('a',{'node_type':'fact'}),('b',{'node_type':'fact'})])
    evidence=EvidenceStore()
    evidence.add_chunk_for_entity('a','Original source A')
    evidence.add_chunk_for_entity('b','Original source B')
    chunks=ChunkStore(2)
    chunks.add([Chunk('ca','Original source A','da'),Chunk('cb','Original source B','db'),Chunk('cx','Unrelated source X','dx')],[[1.,0.],[.9,.1],[1.,0.]])
    agent.evidence_store=evidence
    agent.graph_service=SimpleNamespace(graph_store=graph,chunk_store=chunks,
        embedding_encoder=SimpleNamespace(encode=lambda texts:[[1.,0.]]*len(texts)))
    return agent


def test_batch_grounding_uses_both_graph_sources_with_one_shared_limit_and_no_global_fallback():
    agent=instance();before=deepcopy(agent.evidence_store._entity_to_chunks)
    result=agent._tool_get_evidence(entity_ids=['a','b','a'],query='query',limit=2)
    assert 'Original source A' in result and 'Original source B' in result
    assert 'Unrelated source X' not in result and result.count('[Evidence ')==2
    limited=agent._tool_get_evidence(entity_ids=['a','b'],query='query',limit=1)
    assert limited.count('[Evidence ')==1
    assert agent.evidence_store._entity_to_chunks==before
    assert agent._tool_get_evidence('a').endswith('Original source A')


def test_source_resolution_follows_only_provenance_and_terminates_cycles():
    agent=instance();graph=agent.graph_service.graph_store._g
    graph.add_edge('derived','b',rel_type='SUPPORTED_BY')
    graph.add_edge('b','derived',rel_type='SUPPORTED_BY')
    graph.add_edge('unknown','a',rel_type='LIKES')
    assert 'Original source B' in agent._tool_get_evidence('derived')
    assert agent._tool_get_evidence('unknown').startswith('No original text evidence')
    graph.add_edge('cycle1','cycle2',rel_type='SUPPORTED_BY')
    graph.add_edge('cycle2','cycle1',rel_type='SUPPORTED_BY')
    assert agent._tool_get_evidence('cycle1').startswith('No original text evidence')


def test_auto_grounding_forwards_the_actual_result_set_in_one_operator():
    agent=instance();agent.require_evidence_for_answer=True
    agent._last_tool_node_hits={'semantic_search':['a','b']}
    agent._maybe_finalize_after_step=lambda *args,**kwargs:None
    calls=[]
    def execute(name,params):
        calls.append((name,params))
        return agent._tool_get_evidence(**params)
    agent._execute_tool=execute
    history=[]
    agent._record_and_maybe_finalize(query='query',history=history,scratchpad=[],step_counter=0,
        thought='',action='semantic_search',action_input={'query':'query'},observation='Retrieved both graph nodes')
    assert len(calls)==1 and calls[0][0]=='get_evidence'
    assert calls[0][1]['entity_ids']==['a','b'] and calls[0][1]['limit']==3
    assert agent._has_raw_evidence(history)


def test_an_empty_retrieval_cannot_reuse_a_previous_hit_set():
    agent=instance();agent._last_tool_node_hits={'semantic_search':['old']}
    agent.tools={'semantic_search':lambda **kwargs:'No results'}
    agent._tool_calls_used=0;agent._operator_budget=lambda:5;agent.skill_runtime=None
    assert agent._execute_tool('semantic_search',{})=='No results'
    assert agent._last_tool_node_hits['semantic_search']==[]
