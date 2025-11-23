
from cogmem import CogMem
from cogmem.storage.schema import Doc

if __name__ == "__main__":
    cm = CogMem()
    docs = [
        Doc(id="b1", text="Harry met Hermione at Hogwarts. They liked magic."),
        Doc(id="b2", text="Frodo and Sam traveled from Shire to Mordor."),
    ]
    cm.add_documents(docs)

    # 发现语义社区并持久化
    cm.discover_communities(semantic=True)
    cm.save()

    # 社区优先检索 + 一次迭代 agentic
    q = "谁在霍格沃茨见面？"
    ret = cm.retrieve(q)
    print("[Evidence]", ret.evidence)
    print("[Entities]", [e.name for e in ret.entities])

    # 最小 RAG 答案
    ans = cm.answer(q)

    print("[Answer]", ans)
