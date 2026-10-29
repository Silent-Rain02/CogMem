import json

from datasets.locomo_loader import LocomoLoader, map_category


def test_loader_preserves_session_metadata_and_filters_blank_answers(tmp_path) -> None:
    payload = [
        {
            "conversation": {
                "session_1_date_time": "2024-01-01T10:00:00",
                "session_1": [{"dia_id": "1", "speaker": "Alice", "text": "Hello Bob."}],
            },
            "qa": [
                {"question": "Who said hello?", "answer": "Alice", "evidence": ["1"], "category": 1},
                {"question": "Ignored?", "answer": None, "evidence": [], "category": 5},
            ],
        }
    ]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = LocomoLoader(str(path)).load(db="conversation")

    assert len(bundle.docs) == 1
    assert "[Session Date: 2024-01-01T10:00:00]" in bundle.docs[0].text
    assert bundle.docs[0].metadata.doc_type == "dialog"
    assert len(bundle.qa) == 1
    assert bundle.qa[0].category == "Multi-Hop"


def test_category_mapping_matches_loader_contract() -> None:
    assert map_category(1) == "Multi-Hop"
    assert map_category("2") == "Temporal"
    assert map_category(None) == "Unknown"


def test_loader_preserves_dialog_ids_and_source_image_captions(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps([{"conversation": {"session_1": [
        {"dia_id": "D1:1", "speaker": "Alice", "text": "See this.", "blip_caption": "A red boat."},
        {"speaker": "Bob", "text": "Nice."},
        {"dia_id": "D1:3", "speaker": "Alice", "text": "", "blip_caption": "A blue boat."}
    ]}}]))
    text = LocomoLoader(str(path)).load(db="conversation").docs[0].text
    assert "[D1:1] [Alice] See this." in text
    assert "[Image caption] A red boat." in text
    assert "[Bob] Nice." in text
    assert "[D1:3] [Alice] [Image caption] A blue boat." in text
    assert map_category(3) == "Open Domain"
    assert map_category(4) == "Single Hop"
