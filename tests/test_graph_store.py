from cogmem.storage.graph_store import GraphStore
from cogmem.storage.schema import Concept, NodeTypeEnum


def test_add_node_round_trips_a_typed_concept() -> None:
    graph = GraphStore()
    concept = Concept(id="c_birdwatching", name="birdwatching")

    graph.add_node(concept)

    node = graph.get_node("c_birdwatching", NodeTypeEnum.CONCEPT)
    assert node is not None
    assert node["id"] == "c_birdwatching"
    assert node["node_type"] == NodeTypeEnum.CONCEPT.value
