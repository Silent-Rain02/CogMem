from test_fact_subject_connectivity import build
from cogmem.storage.schema import Claim,Concept
from cogmem.storage.graph_store import GraphStore
import pytest

@pytest.mark.parametrize('target',['Niko','c_crowds'])
def test_claim_speaker_is_separate_from_evidence_source_and_occurrence(target,tmp_path):
    payload={'claims':[{'source':'Mira','target':target,'content':'dislikes crowds'}]}
    all_claims=[];graph=GraphStore()
    for address in ['turn-session_1-D1:1','turn-session_1-D1:2']:
        nodes,relations=build(payload,address)
        claim=next(n for n in nodes if isinstance(n,Claim));all_claims.append(claim)
        assert claim.metadata.source==address
        assert claim.source_id!=address
        assert any(r.head_id==claim.source_id and r.tail_id==claim.id and r.rel_type=='CLAIMS' for r in relations)
        assert all(n.metadata.source==address for n in nodes if isinstance(n,Concept))
        for node in nodes:graph.add_node(node)
        graph.add_relations(relations)
    assert all_claims[0].id!=all_claims[1].id
    graph.save(str(tmp_path/'graph.pkl'))
    loaded=GraphStore();loaded.load(str(tmp_path/'graph.pkl'))
    for claim in all_claims:
        restored=Claim.model_validate(loaded.get_node(claim.id))
        assert restored.metadata.source==claim.metadata.source
        assert loaded._g.has_edge(claim.source_id,claim.id)
        assert loaded._g.has_edge(claim.id,claim.target_id)

def test_identical_claims_in_distinct_source_turns_have_distinct_occurrences():
    payload={'claims':[{'source':'Mira','target':'c_crowds','content':'dislikes crowds'}]}
    first,_=build(payload,'turn-session_1-D1:1')
    second,_=build(payload,'turn-session_1-D1:2')
    a=next(n for n in first if isinstance(n,Claim))
    b=next(n for n in second if isinstance(n,Claim))
    assert a.id!=b.id
    assert a.source_id==b.source_id
