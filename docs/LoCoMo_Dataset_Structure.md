# LoCoMo数据集结构说明

## 数据集基本信息

根据 `locomo10.json` 分析结果：

### 整体结构
- **总Conversation数**：10 个（顶级样本）
- **总Session数**：272 个会话
- **总Turn数**：5882 个对话轮次
- **总问题（QA）数**：1986 个

### 单个Conversation的平均值
- 每个Conversation包含约 **27.2 个Session**
- 每个Session包含约 **21.6 个Turn**
- 每个Conversation包含约 **198.6 个QA问题**

### 问题类别分布（原始数据）
- **Multi-Hop（多跳）**：282 个
- **Single Hop（单跳）**：321 个
- **Temporal（时间推理）**：96 个
- **Open Domain（开放域）**：841 个
- **Adversarial（对抗性）**：446 个

### ⚠️ 重要：有效问题数
- **原始QA总数**：1986 个
- **有效QA总数**：1542 个（过滤了答案为空的QA）
- **被过滤的QA**：444 个（主要是Adversarial类别，答案本身就是None）

**过滤逻辑**：`_build_qa()` 函数会过滤掉答案为空的QA（`if q and a:`），因为这些问题的答案本身就是None，无法用于评测。

## 数据层次结构

```
locomo10.json (列表，10个元素)
└── Conversation (样本)
    ├── conversation (对话内容)
    │   ├── session_1 (会话1)
    │   │   └── [turn1, turn2, ...] (轮次列表)
    │   ├── session_2 (会话2)
    │   │   └── [turn1, turn2, ...]
    │   └── ...
    ├── qa (问题列表)
    │   ├── {question, answer, evidence, category}
    │   └── ...
    ├── observation (观察数据)
    ├── session_summary (会话摘要)
    └── event_summary (事件摘要)
```

## max_samples 参数说明

### 在 `load_locomo_dataset()` 中
- **参数含义**：限制加载的 **Conversation 数量**（不是问题数）
- **影响**：如果 `max_samples=5`，会加载前5个Conversation，包含约 5 × 198.6 ≈ 993 个问题

### 在 `comprehensive_evaluation()` 中
- **参数含义**：限制评测的 **问题（QA）数量**
- **影响**：如果 `max_samples=100`，只评测前100个问题

### ⚠️ 注意：参数含义不一致

在 `eval_locomo_complete.py` 中：
1. `load_locomo_dataset(args.data_dir)` - 没有传递 `max_samples`，会加载所有10个Conversation，包含1986个问题
2. `comprehensive_evaluation(..., max_samples=args.max_samples)` - 限制的是问题数量

**建议修复**：
- 如果要在加载阶段限制Conversation数量，应在 `load_locomo_dataset()` 中传递参数
- 如果要在评测阶段限制问题数量，应在 `comprehensive_evaluation()` 中传递参数
- 建议统一使用问题数量作为限制标准，更直观

## 当前代码流程

```
main()
  └── load_locomo_dataset(args.data_dir)
      └── 加载所有10个Conversation
          └── 返回1986个问题
  └── comprehensive_evaluation(..., max_samples=args.max_samples)
      └── 如果 max_samples=100，只评测前100个问题
```

## 实际评测数据量

根据终端输出：
- **已索引文档数**：1283 个（对话文档，合并了多个turns）
- **已索引实体数**：4544 个
- **评测问题数**：如果使用默认值（None），会评测全部 1986 个问题

