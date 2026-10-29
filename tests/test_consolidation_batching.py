from cogmem.memory.concept_merger import ConceptMerger
from cogmem.memory.fact_deduplicator import FactDeduplicator


class BoundaryEncoder:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return [self.mapping[text] for text in texts]


def test_concept_batching_preserves_pairwise_decisions_and_texts():
    concepts = [(str(i), {"name": f"name{i}", "description": f"description{i}"}) for i in range(12)]
    mapping = {f"name{i} description{i}": [1.0, float(i % 2)] for i in range(12)}
    encoder = BoundaryEncoder(mapping)
    batched = ConceptMerger(None, embedding_encoder=encoder)
    batched._get_all_concepts = lambda: concepts
    reference = ConceptMerger(None, embedding_encoder=encoder)
    reference._get_all_concepts = lambda: concepts
    reference._concept_embeddings = {cid: mapping[f"{data['name']} {data['description']}"]
                                     for cid, data in concepts}
    assert batched.find_synonyms() == reference.find_synonyms()
    assert [len(call) for call in encoder.calls] == [10, 2]
    assert [text for call in encoder.calls for text in call] == list(mapping)


def test_fact_batching_preserves_duplicate_decisions_and_texts():
    facts = [(str(i), {"content": f"fact{i}", "predicate": f"predicate{i}", "object_id": f"object{i}"})
             for i in range(12)]
    mapping = {f"fact{i} predicate{i} object{i}": [1.0, float(i % 2)] for i in range(12)}
    encoder = BoundaryEncoder(mapping)
    batched = FactDeduplicator(None, embedding_encoder=encoder)
    batched._get_all_facts = lambda: facts
    reference = FactDeduplicator(None, embedding_encoder=encoder)
    reference._get_all_facts = lambda: facts
    reference._fact_embeddings = {fid: mapping[f"{d['content']} {d['predicate']} {d['object_id']}"]
                                 for fid, d in facts}
    assert batched.find_duplicates() == reference.find_duplicates()
    assert [len(call) for call in encoder.calls] == [10, 2]
    assert [text for call in encoder.calls for text in call] == list(mapping)
