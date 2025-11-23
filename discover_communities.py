#!/usr/bin/env python3
"""
独立社区发现脚本

在索引构建完成后，单独运行社区发现功能。
这对于已经存在的索引特别有用，无需重新构建整个知识图谱。
"""

import argparse
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cogmem.cogmem import CogMem


def main():
    parser = argparse.ArgumentParser(description="独立社区发现功能")
    parser.add_argument(
        "--workspace",
        type=str,
        default="./.cogmem_complete_eval",
        help="工作空间目录（包含索引文件）"
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        default=True,
        help="是否发现语义社区（基于embedding相似度）"
    )
    parser.add_argument(
        "--structure",
        action="store_true",
        default=False,
        help="是否发现结构社区（基于图连接）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="强制重新发现社区（覆盖现有社区）"
    )

    args = parser.parse_args()

    print(f"[INFO] 开始社区发现...")
    print(f"   工作空间: {args.workspace}")
    print(f"   语义社区: {'启用' if args.semantic else '禁用'}")
    print(f"   结构社区: {'启用' if args.structure else '禁用'}")
    print(f"   强制模式: {'是' if args.force else '否'}")
    print()

    try:
        # 初始化 CogMem（加载现有索引）
        cogmem = CogMem(workspace_dir=args.workspace, auto_load=True)

        # 检查是否已存在社区
        if not args.force:
            try:
                # 尝试加载现有社区
                cogmem.community_store.load(
                    community_meta_file=cogmem.cfg.community_meta_file,
                    community_centroid_file=cogmem.cfg.community_centroid_file
                )

                # 检查是否有社区数据
                semantic_count = len(cogmem.community_store.semantic) if hasattr(cogmem.community_store, 'semantic') else 0
                structure_count = len(cogmem.community_store.structure) if hasattr(cogmem.community_store, 'structure') else 0

                if semantic_count > 0 or structure_count > 0:
                    print(f"[WARN] 发现现有社区数据:")
                    print(f"   语义社区: {semantic_count} 个")
                    print(f"   结构社区: {structure_count} 个")
                    print("   使用 --force 参数强制重新发现")
                    return

            except Exception as e:
                print(f"[INFO] 无法加载现有社区数据: {e}")
                print("   将创建新的社区数据")

        # 执行社区发现
        print("[INFO] 开始执行社区发现...")
        start_time = time.time()

        cogmem.discover_communities(
            semantic=args.semantic,
            structure=args.structure
        )

        discovery_time = time.time() - start_time
        print(".2f")

        # 保存社区数据
        print("[INFO] 保存社区数据...")
        save_start = time.time()
        cogmem.save()
        save_time = time.time() - save_start
        print(".2f")

        # 显示统计信息
        semantic_count = len(cogmem.community_store.semantic) if hasattr(cogmem.community_store, 'semantic') else 0
        structure_count = len(cogmem.community_store.structure) if hasattr(cogmem.community_store, 'structure') else 0

        print("[SUCCESS] 社区发现完成!")
        print(f"   语义社区: {semantic_count} 个")
        print(f"   结构社区: {structure_count} 个")
        print()
        print("[STATS] 社区统计:")

        if semantic_count > 0:
            print("   [SEMANTIC] 语义社区详情:")
            for i, community in enumerate(cogmem.community_store.semantic[:5]):  # 只显示前5个
                print(f"      社区 {i+1}: {len(community.member_ids)} 个成员")
            if semantic_count > 5:
                print(f"      ... 还有 {semantic_count - 5} 个社区")

        if structure_count > 0:
            print("   [STRUCTURE] 结构社区详情:")
            for i, community in enumerate(cogmem.community_store.structure[:5]):  # 只显示前5个
                print(f"      社区 {i+1}: {len(community.member_ids)} 个成员")
            if structure_count > 5:
                print(f"      ... 还有 {structure_count - 5} 个社区")

    except Exception as e:
        print(f"[ERROR] 社区发现失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
