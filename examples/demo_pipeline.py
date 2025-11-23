"""
完整的 CogMem 功能演示流程：
导入文档 → 社区发现（语义+结构）→ 检索（融合+N步）→ 冲突检测 → RAG 答案
"""

from cogmem import CogMem
from cogmem.storage.schema import Doc


def main():
    print("=== CogMem 完整 Pipeline 演示 ===\n")
    
    # 1. 初始化（使用独立工作空间）
    workspace = ".cogmem-demo"
    cm = CogMem(workspace_dir=workspace, auto_load=False)
    
    # 2. 导入文档
    print("[1] 导入文档...")
    docs = [
        Doc(id="d1", text="张三在北京大学与李四会面，讨论学术问题。"),
        Doc(id="d2", text="李四是清华大学的研究生，擅长机器学习。"),
        Doc(id="d3", text="张三和李四都参加了2024年人工智能会议。"),
    ]
    entities = cm.add_documents(docs)
    print(f"    抽取到 {len(entities)} 个实体")
    for e in entities[:3]:
        print(f"      - {e.name} ({e.type}): {e.description}")
    
    # 3. 发现社区（语义+结构）
    print("\n[2] 发现社区...")
    cm.discover_communities(semantic=True, structure=True)
    sem_coms = cm.community_store.get_semantic()
    str_coms = cm.community_store.get_structure()
    print(f"    语义社区: {len(sem_coms)} 个")
    print(f"    结构社区: {len(str_coms)} 个")
    
    # 4. 保存索引
    print("\n[3] 保存索引...")
    cm.save()
    print(f"    索引已保存到: {workspace}")
    
    # 5. 检索（融合+N步）
    print("\n[4] 检索测试...")
    query = "谁在北京大学？"
    ret = cm.retrieve(query, top_k=5)
    print(f"    查询: {query}")
    print(f"    检索路径: {ret.evidence}")
    print(f"    返回实体: {[e.name for e in ret.entities]}")
    
    # 6. RAG 答案生成
    print("\n[5] RAG 答案生成...")
    answer = cm.answer(query)
    print(f"    问题: {query}")
    print(f"    答案: {answer}")
    
    # 7. 冲突检测示例（模拟）
    print("\n[6] 冲突检测示例...")
    from cogmem.memory.conflict import ConflictResolver
    resolver = ConflictResolver(cm.cfg)
    
    # 模拟一个冲突场景：同名不同描述
    test_entities = [
        cm.graph.get_entity(entities[0].id) if entities else None,
    ]
    if test_entities[0]:
        issues = resolver.detect_conflicts_rule([test_entities[0]])
        print(f"    检测到问题: {issues if issues else '无冲突'}")
    
    # 8. 图查询示例
    print("\n[7] 图查询示例...")
    if entities:
        eid = entities[0].id
        neighbors = cm.graph.neighbors(eid, direction="out", min_confidence=0.5)
        print(f"    实体 '{entities[0].name}' 的邻居（置信度>=0.5）: {[e.name for e in neighbors]}")
    
    # 9. 导出图谱
    print("\n[8] 导出图谱...")
    cm.graph.export_json(f"{workspace}/graph_export.json")
    print(f"    图谱已导出到: {workspace}/graph_export.json")
    
    # 10. 指标统计
    print("\n[9] 指标统计...")
    metrics = cm.get_metrics()
    print(f"    Chat 调用: {metrics.get('chat', {}).get('calls', 0)} 次")
    print(f"    Embed 调用: {metrics.get('embed', {}).get('calls', 0)} 次")
    print(f"    缓存命中率: Chat {metrics.get('chat', {}).get('cache_hits', 0) / max(1, metrics.get('chat', {}).get('calls', 1)):.1%}, "
          f"Embed {metrics.get('embed', {}).get('cache_hits', 0) / max(1, metrics.get('embed', {}).get('calls', 1)):.1%}")
    
    print("\n=== Pipeline 演示完成 ===")


if __name__ == "__main__":
    main()

