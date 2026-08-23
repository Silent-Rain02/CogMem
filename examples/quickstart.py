"""Index two short dialogue turns and ask a grounded question.

Before running, configure an available provider in ``.env`` or edit the
``CogMemConfig`` below. This example performs real model calls.
"""

from cogmem import CogMem, CogMemConfig
from cogmem.storage import Doc


def main() -> None:
    config = CogMemConfig()
    config.model.provider = "ollama"
    config.model.embedding_provider = "ollama"
    config.model.ollama_chat_model = "qwen2.5:14b-instruct"
    config.model.ollama_embedding_model = "bge-m3"

    memory = CogMem(config=config, workspace_dir="./workspace")
    memory.add_documents(
        [
            Doc(id="turn-1", text="Alice and Bob visited the park on Saturday."),
            Doc(id="turn-2", text="Bob said that he enjoys birdwatching."),
        ]
    )
    memory.save()
    print(memory.answer("What does Bob enjoy?"))


if __name__ == "__main__":
    main()
