# CogMem - 长期记忆系统

基于知识图谱和向量检索的长期记忆系统，采用 **知识图谱 + 向量检索（FAISS + NetworkX）** 的混合架构，用于支持对话系统中的长期记忆和问答任务。

## ✨ 核心特性

- 🧠 **知识图谱存储**: 基于 NetworkX 的多类型节点图谱（Person/Event/Fact/Preference/Topic/Episode）
- 🔍 **智能检索**: Topic-Scoped 检索 + 向量检索 + 图遍历的混合检索策略
- 🤖 **Agentic 问答**: 基于 Think-Act 循环的智能检索和回答 Agent
- 📊 **社区发现**: 语义社区和结构社区自动发现，支持 Topic 节点生成
- ⚡ **高性能**: FAISS 向量索引 + 批量处理 + 并发支持
- 🔄 **增量索引**: 支持增量文档添加和索引更新
- 📝 **结构化抽取**: 基于 LLM 的结构化实体和关系抽取

## 📦 安装

### 环境要求

- Python 3.8+
- 支持 OpenAI 兼容 API 的 LLM 服务（百炼、OpenAI、Ollama 等）

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd cogmem

# 安装依赖
pip install -r requirements.txt

# 激活虚拟环境（Windows）
.venv\Scripts\activate.ps1

# 或使用 uv（推荐）
uv pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件或设置环境变量：

```bash
# OpenAI/百炼 兼容 API
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-xxxxxxxx

# 或使用 Ollama 本地模型
OLLAMA_BASE_URL=http://localhost:11434
```

## 🚀 快速开始

### 基础使用

```python
from cogmem import CogMem
from cogmem.storage.schema import Doc

# 初始化 CogMem
cm = CogMem(workspace_dir="./workspace")

# 添加文档
docs = [
    Doc(id="d1", text="Alice likes Bob. They went to the park together."),
    Doc(id="d2", text="Bob works at Google. He is a software engineer.")
]
cm.add_documents(docs)

# 检索
result = cm.retrieve("Where does Bob work?")
print([e.name for e in result.entities])
print(result.evidence)

# 问答
answer = cm.answer("Where does Bob work?")
print(answer)
```

### 完整示例

```python
from cogmem import CogMem
from cogmem.storage.schema import Doc

# 初始化
cm = CogMem(workspace_dir="./my_workspace", auto_load=True)

# 添加文档并索引
docs = [Doc(id=f"doc_{i}", text=text) for i, text in enumerate(texts)]
cm.add_documents(docs)

# 发现社区（生成 Topic 节点）
cm.discover_communities(semantic=True, structure=False)

# 保存索引
cm.save()

# 检索和问答
result = cm.retrieve("What events did Alice attend?")
answer = cm.answer("What events did Alice attend?")
```

更多示例见 `examples/` 目录。

## 🏗️ 系统架构

### 核心组件

```
┌─────────────────────────────────────┐
│   CogMem (主入口)                   │
│   - 文档索引                        │
│   - 检索和问答                      │
│   - 社区发现                        │
└─────────────────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼───┐    ┌───▼──────┐
│Indexer│    │Retriever │
│索引器  │    │检索器     │
└───┬───┘    └───┬──────┘
    │            │
┌───▼────────────▼───┐
│  存储层              │
│  - GraphStore       │
│  - VectorStore      │
│  - ChunkStore       │
│  - CommunityStore   │
└────────────────────┘
```

### Agent 架构

- **QAOrchestrator**: 问答协调器，整合检索和回答流程
- **RetrievalAgent**: 智能检索 Agent，支持 Topic-Scoped 检索
- **AnswerAgent**: 智能回答 Agent，基于检索结果生成答案

### 检索流程

1. **Topic Matching**: 向量检索 Top-K Topic 节点
2. **Seed Finding**: 在 Event/Fact 索引中检索种子节点
3. **Graph Traversal**: 从种子节点出发，优先遍历 Topic 邻居
4. **Evidence Formatting**: 格式化证据，包含时间、主题、类型信息

## 📚 核心功能

### 文档索引

```python
# 添加文档
cm.add_documents(docs)

# 增量索引
cm.add_documents_incremental(docs, force_rebuild=False)

# 发现社区
cm.discover_communities(semantic=True, structure=True)
```

### 检索

```python
# 基础检索
result = cm.retrieve(query="Alice 喜欢谁？", top_k=10)

# 检索结果包含
# - entities: 检索到的实体列表
# - evidence: 证据文本列表
# - matched_topics: 匹配的 Topic 节点
# - retrieval_path: 检索路径
```

### 问答

```python
# 生成答案
answer = cm.answer(query="Where does Bob work?", top_k=10)
```

## ⚙️ 配置

### 主要配置项

```python
from cogmem.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG

# 检索配置
config.retrieval.use_topic_scoping = True  # 启用 Topic-Scoped 检索
config.retrieval.topic_top_k = 3          # Topic 匹配数量
config.retrieval.seed_top_k = 15          # 种子节点数量

# 社区配置
config.community.semantic_k = 20          # 语义聚类数量
config.community.enable_structure = True   # 启用结构社区

# 并发配置
config.runtime.max_concurrency = 4         # 最大并发数
```

## 📊 评测

### LoCoMo 数据集评测

```bash
# 运行完整评测
python eval_locomo_complete.py --max_samples 50 --output complete_eval_results.json

# 仅分析已有结果
python eval_locomo_complete.py --analyze_only --output complete_eval_results.json
```

评测指标：
- **EM (Exact Match)**: 精确匹配分数
- **F1 Score**: F1 分数
- **Evidence Hit**: 证据命中率
- **LLM Judge**: LLM 评估分数

## 📁 项目结构

```
cogmem/
├── cogmem/                    # 核心代码
│   ├── agents/               # Agent 层
│   │   ├── retrieval_agent.py
│   │   ├── answer_agent.py
│   │   └── qa_orchestrator.py
│   ├── storage/              # 存储层
│   │   ├── graph_store.py
│   │   ├── vector_store.py
│   │   ├── chunk_store.py
│   │   └── schema.py
│   ├── retrieval/            # 检索层
│   │   ├── retriever.py
│   │   └── query_processor.py
│   ├── memory/               # 索引层
│   │   └── indexer.py
│   ├── extraction/           # 抽取层
│   │   ├── structured_extractor.py
│   │   └── entity_extractor.py
│   └── community_module/     # 社区发现
│       └── discovery.py
├── examples/                  # 示例代码
├── datasets/                 # 数据集加载器
├── eval_locomo_complete.py   # 评测脚本
└── requirements.txt          # 依赖列表
```

## 🔧 开发

### 运行测试

```bash
# 运行示例
python examples/demo_books.py

# 运行评测
python eval_locomo_complete.py --max_samples 10
```

### 调试

启用调试模式：

```bash
export DEBUG=1
python your_script.py
```

## 📖 文档

- [架构文档](architecture.md) - 详细的系统架构和关键函数清单
- [算法描述](des.md) - 核心算法流程和实现细节
- [评测指南](docs/LoCoMo_Evaluation_Guide.md) - LoCoMo 数据集评测指南

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[添加许可证信息]

## 🙏 致谢

- 使用 [NetworkX](https://networkx.org/) 进行图存储
- 使用 [FAISS](https://github.com/facebookresearch/faiss) 进行向量检索
- 基于 [LoCoMo](https://github.com/...) 数据集进行评测
