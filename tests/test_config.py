from cogmem.config import CogMemConfig, classify_question_type
from cogmem.agents.qa_orchestrator import QAOrchestrator


def test_question_type_routing_uses_strong_signals() -> None:
    assert classify_question_type("What do Alice and Bob have in common?") == "multi_hop"
    assert classify_question_type("When did Alice start the new role?") == "temporal"
    assert classify_question_type("What does Bob believe about the plan?") == "open_domain"
    assert classify_question_type("Where does Alice live?") == "single_hop"


def test_config_instances_are_independent() -> None:
    first = CogMemConfig()
    second = CogMemConfig()

    first.agent.max_iterations = 9

    assert second.agent.max_iterations == 4
    assert second.model.temperature == 0.0


def test_answer_cleaning_handles_nested_json() -> None:
    answer = QAOrchestrator._clean_answer('{"response": {"content": "Grounded answer"}}')

    assert answer == "Grounded answer"
