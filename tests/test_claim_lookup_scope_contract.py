"""Speaker attribution and node lifecycle constrain opinion lookup."""
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.graph_service import GraphService


def test_parallel_edges_are_one_claim_and_misattribution_is_rejected():
    service=object.__new__(GraphService);service.graph_store=GraphStore();g=service.graph_store._g
    g.add_node('c',node_type='claim',source_id='alice',target_id='bob')
    g.add_node('archived',node_type='claim',source_id='alice',target_id='bob',is_archived=True)
    g.add_edge('alice','c',rel_type='CLAIMS');g.add_edge('alice','c',rel_type='CLAIMS',version=2)
    g.add_edge('eve','c',rel_type='CLAIMS');g.add_edge('alice','archived',rel_type='CLAIMS')
    assert len(service.find_claims_about_target('alice','bob'))==1
    assert service.find_claims_about_target('eve','bob')==[]
    assert len(service.find_claims_about_target(target_id='bob'))==1
