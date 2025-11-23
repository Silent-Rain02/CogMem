from cogmem import CogMem
from cogmem.storage.schema import Doc

if __name__ == "__main__":
    cm = CogMem()
    docs = [
        Doc(id="b1", text="Harry met Hermione at Hogwarts. They liked magic."),
        Doc(id="b2", text="Frodo and Sam traveled from Shire to Mordor."),
    ]
    cm.add_documents(docs)
    cm.discover_communities(semantic=True)
    cm.save()

    # 导出 GraphML/JSON
    cm.graph.export_graphml(".cogmem/graph.graphml")
    cm.graph.export_json(".cogmem/graph.json")
    print("Exported to .cogmem/graph.graphml and .cogmem/graph.json")
