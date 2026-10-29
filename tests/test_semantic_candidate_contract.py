"""Semantic operators must rank eligible candidates before applying output limits."""
from types import SimpleNamespace
from cogmem.agents.search_agent import SearchAgent
from cogmem.config import CogMemConfig
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.graph_service import GraphService


def agent(hits, metadata, expansion=2.):
    graph=GraphStore()
    for node,attrs in metadata.items():graph._g.add_node(node,**attrs)
    calls=[]
    def search(vector, kind, top_k):
        calls.append((kind,top_k))
        return hits[kind][:top_k]
    store=SimpleNamespace(search_by_type=search)
    encoder=SimpleNamespace(encode=lambda texts:[[1.,0.] for _ in texts])
    cfg=CogMemConfig();cfg.agent.vector_search_expansion_factor=expansion
    instance=SearchAgent(GraphService(graph,store,encoder),None,config=cfg)
    return instance,calls


def test_eligible_candidate_is_not_starved_by_archived_candidates_of_other_types():
    hits={'event':[('a',.99),('b',.98)],'fact':[('c',.97),('target',.96)]}
    metadata={n:dict(node_type='event' if n in 'ab' else 'fact',title=n,content=n,is_archived=n!='target') for n in ['a','b','c','target']}
    value,calls=agent(hits,metadata);value._tool_semantic_search('q',node_types=['event','fact'],top_k=1)
    assert value._last_tool_node_hits.get('semantic_search')==['target']
    assert calls==[('event',2),('fact',2)]


def test_existing_importance_ranking_precedes_global_output_truncation():
    hits={'event':[('a',.99),('b',.98)],'fact':[('c',.97),('target',.96)]}
    metadata={n:dict(node_type='event' if n in 'ab' else 'fact',title=n,content=n,importance_score=1. if n=='target' else 0.) for n in ['a','b','c','target']}
    value,calls=agent(hits,metadata);value._tool_semantic_search('q',node_types=['event','fact'],top_k=1)
    assert value._last_tool_node_hits['semantic_search']==['target']
    assert calls==[('event',2),('fact',2)]


def test_effective_runtime_expansion_factor_is_honored():
    value,calls=agent({'event':[]},{},expansion=1.)
    value.vector_search_expansion_factor=3.
    value._tool_semantic_search('q',node_types=['event'],top_k=2)
    assert calls==[('event',6)]


def test_existing_final_limit_and_score_are_preserved_without_competing_candidates():
    value,calls=agent({'fact':[('one',.9)]},{'one':dict(node_type='fact',content='A proposition',importance_score=.5)})
    text=value._tool_semantic_search('q',node_types=['fact'],top_k=1)
    assert value._last_tool_node_hits['semantic_search']==['one']
    assert '[Score: 0.765]' in text
