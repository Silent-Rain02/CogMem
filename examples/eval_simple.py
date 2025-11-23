from typing import List, Dict, Tuple
import math

from cogmem import CogMem
from cogmem.storage.schema import Doc


def dcg(scores: List[int]) -> float:
    return sum((s / math.log2(i + 2)) for i, s in enumerate(scores))


def ndcg_at_k(gold: List[str], pred: List[str], k: int) -> float:
    gold_set = set(gold)
    rel = [1 if p in gold_set else 0 for p in pred[:k]]
    ideal = sorted(rel, reverse=True)
    return (dcg(rel) / (dcg(ideal) or 1.0)) if rel else 0.0


def average_precision(gold: List[str], pred: List[str], k: int) -> float:
    gold_set = set(gold)
    hits = 0
    ap_sum = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in gold_set:
            hits += 1
            ap_sum += hits / i
    denom = min(len(gold_set), k)
    return (ap_sum / denom) if denom else 0.0


def evaluate(cm: CogMem, queries: List[str], expected: List[List[str]], top_k: int = 10) -> None:
    assert len(queries) == len(expected)
    total = len(queries)
    hits_any = 0
    total_hits = 0
    mAP = 0.0
    nDCG = 0.0

    for q, exp in zip(queries, expected):
        ret = cm.retrieve(q, top_k=top_k)
        got = [e.name for e in ret.entities]
        got_set = set(got)
        exp_set = set(exp)
        hit_set = got_set & exp_set
        if hit_set:
            hits_any += 1
        total_hits += len(hit_set)
        mAP += average_precision(exp, got, top_k)
        nDCG += ndcg_at_k(exp, got, top_k)
        print(f"Q: {q}")
        print("  Expected:", list(exp_set))
        print("  Got:     ", got)
        print("  Hit:     ", list(hit_set))
        print("  Evidence:", ret.evidence)

    hit_rate = hits_any / total if total else 0.0
    avg_hits = total_hits / total if total else 0.0
    mAP /= total if total else 1
    nDCG /= total if total else 1
    print("\n=== Eval Summary ===")
    print(f"Queries: {total}")
    print(f"Top-{top_k} Hit-Any Rate: {hit_rate:.3f}")
    print(f"Avg Hits per Query: {avg_hits:.3f}")
    print(f"mAP@{top_k}: {mAP:.3f}")
    print(f"nDCG@{top_k}: {nDCG:.3f}")


if __name__ == "__main__":
    # 构造一个最小评测集
    cm = CogMem()
    docs = [
        Doc(id="b1", text="Harry met Hermione at Hogwarts. They liked magic."),
        Doc(id="b2", text="Frodo and Sam traveled from Shire to Mordor."),
        Doc(id="b3", text="Alice loves Bob in Wonderland."),
    ]
    cm.add_documents(docs)
    cm.discover_communities(semantic=True)

    queries = [
        "谁在霍格沃茨见面？",
        "弗罗多和谁从夏尔去魔多？",
        "爱丽丝喜欢谁？",
    ]
    expected = [
        ["Harry", "Hermione", "Hogwarts"],
        ["Frodo", "Sam", "Shire", "Mordor"],
        ["Alice", "Bob"],
    ]

    evaluate(cm, queries, expected, top_k=10)
