import json
from types import SimpleNamespace

import pytest

from cogmem.agents.search_agent import SearchAgent, ThoughtActionObservation
from cogmem.memory.consolidator import MemoryConsolidator
from cogmem.storage.evidence_store import EvidenceStore
from datasets.locomo_loader import load_locomo_dataset


@pytest.mark.parametrize("response", [
    'TOOL: get_evidence\nARGS: {"entity_id": "p_alice"}',
    'TOOL: get_evidence\nARGS: ```json\n{"entity_id": "p_alice"}\n```',
    '{"tool": "get_evidence", "args": {"entity_id": "p_alice"}}',
])
def test_action_protocol_preserves_tool_and_arguments(response):
    result = SearchAgent._parse_step_response(None, response)
    assert result["action"] == "get_evidence"
    assert result["params"] == {"entity_id": "p_alice"}


def test_argument_object_is_not_an_action():
    assert SearchAgent._parse_step_response(None, '{"entity_id": "p_alice"}') == {}
    assert SearchAgent._parse_step_response(None, 'FINAL_ANSWER: Blue.') == {"final_answer": "Blue."}


def test_evidence_payload_keeps_complete_multiline_blocks():
    step = ThoughtActionObservation(1, "", "get_evidence", {},
        "Original Text Evidence for p_alice:\n[Evidence 1] header\n[Alice] First line.\n"
        "[Bob] Second line.\n\n[Evidence 2]\n[Alice] Another block.")
    payload = SearchAgent._structured_observation_payload(None, step)
    assert [item["content"] for item in payload["items"]] == [
        "header\n[Alice] First line.\n[Bob] Second line.", "[Alice] Another block."]


def test_evidence_storage_is_lossless_across_save_load(tmp_path):
    store = EvidenceStore()
    chunks = ["x" * 1600 + f"tail-{i}" for i in range(20)]
    for chunk in chunks:
        store.add_chunk_for_entity("p_alice", chunk)
    assert store.get_chunks("p_alice", limit=20) == chunks
    path = str(tmp_path / "evidence.pkl")
    store.save(path)
    restored = EvidenceStore()
    restored.load(path)
    assert restored.get_chunks("p_alice", limit=20) == chunks
    # Storage completeness does not remove the read budget.
    assert store.get_chunks("p_alice", limit=2) == chunks[:2]


def test_consolidation_prompt_matches_consumer_contract():
    calls = []
    fact = {"content": "Alice runs.", "predicate": "runs", "confidence": 0.8}
    def chat(**kwargs):
        calls.append(kwargs)
        return json.dumps(fact)
    consolidator = object.__new__(MemoryConsolidator)
    consolidator.graph = SimpleNamespace(get_node=lambda _: {"name": "Running"})
    consolidator.llm = SimpleNamespace(chat=chat)
    result = consolidator._synthesize_fact("p_alice", "Alice", "c_running", [
        {"id": "event_1", "description": "Alice runs."}])
    assert result == {**fact, "object_id": "c_running"}
    assert len(calls) == 1
    assert "event_1" in calls[0]["user"]
    assert '"predicate"' in calls[0]["system"]


def test_consolidation_batch_reports_failures_with_concurrency(capsys):
    consolidator = object.__new__(MemoryConsolidator)
    consolidator.config = SimpleNamespace(runtime=SimpleNamespace(max_concurrency=2))
    def fail(*args):
        raise KeyError("broken contract")
    consolidator._synthesize_fact = fail
    assert consolidator._synthesize_facts_batch([("p_a", "Alice", "c_x", [])], 1) == []
    assert "broken contract" in capsys.readouterr().out


def test_evaluation_loader_preserves_official_numeric_category(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([{
        "conversation": {"session_1": [{"text": "Hello", "speaker": "Alice"}]},
        "qa": [{"question": "Who?", "answer": "Alice", "category": 1}],
    }]))
    assert load_locomo_dataset(str(path))[0]["category"] == 1
