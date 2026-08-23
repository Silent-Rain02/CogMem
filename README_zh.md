# CogMem：从检索走向重构

CogMem 是面向长期对话的、保留来源信息的认知记忆系统。它将对话构建为可演化的 **PEC²F** 图谱：**人**（Person）、**事件**（Event）、**概念**（Concept）、**主张**（Claim）与**事实**（Fact），并通过受限的智能体检索过程，从可追溯的证据中重构答案。

> **预发布说明。** 本仓库正在整理为正式开源版本。在论文正式公开前，仓库不会提供论文链接或 BibTeX；请不要将此分支视为归档发布版本。

[English](README.md)

<p align="center">
  <img src="assets/cogmem_overview.png" alt="CogMem 总体架构：记忆生成、巩固与认知推理" width="100%">
</p>

## 为什么是 CogMem？

扁平检索容易混淆“某人表达的观点”和“客观发生的事件或稳定事实”。CogMem 显式保存它们的语义差异，并使用带类型、带时间信息的边连接这些记忆。

| 阶段 | 作用 |
| --- | --- |
| 记忆写入 | 将对话抽取为带来源与时间戳的图谱节点和关系。 |
| 记忆巩固 | 从情景事件合成稳定事实，同时保留指向原始证据的链接。 |
| 认知重构 | 根据问题动态组合锚定、遍历、交集和证据溯源等算子。 |

该设计支持多跳推理、时序问题、知识更新和带来源归属的观点问题，无需训练专门的检索策略。

## 安装

需要 Python 3.12 或更高版本。推荐使用 [`uv`](https://docs.astral.sh/uv/) 获得可复现环境，也可以使用 `pip`。

```bash
git clone https://github.com/Silent-Rain02/CogMem.git
cd CogMem
uv sync
# 或：python -m pip install -e .
```

复制环境变量模板，并选择可用的模型服务：

```bash
cp .env.example .env
```

CogMem 支持 OpenAI 兼容接口、SiliconFlow、ModelScope 和 Ollama。请在代码中设置模型服务与模型名称；环境变量示例见 [`.env.example`](.env.example)。请始终将 `.env` 保持为私有文件，绝不要提交密钥。

## 快速开始

下面的示例需要可用的对话模型和嵌入模型服务。它会创建一个新的本地工作目录，并在其中保存记忆图谱。

```python
from cogmem import CogMem, CogMemConfig
from cogmem.storage import Doc

config = CogMemConfig()
config.model.provider = "ollama"
config.model.embedding_provider = "ollama"
config.model.ollama_chat_model = "qwen2.5:14b-instruct"
config.model.ollama_embedding_model = "bge-m3"

memory = CogMem(config=config, workspace_dir="./workspace")
memory.add_documents([
    Doc(id="turn-1", text="Alice and Bob visited the park on Saturday."),
    Doc(id="turn-2", text="Bob said that he enjoys birdwatching."),
])
memory.save()

print(memory.answer("What does Bob enjoy?"))
```

可直接运行的版本见 [`examples/quickstart.py`](examples/quickstart.py)。

## 配置

`CogMemConfig` 将模型、存储、抽取与智能体设置分开管理。建议新建独立配置对象，而不是直接修改 `DEFAULT_CONFIG`：

```python
from cogmem import CogMemConfig

config = CogMemConfig()
config.model.provider = "siliconflow"
config.model.embedding_provider = "siliconflow"
config.model.siliconflow_chat_model = "Qwen/Qwen2.5-14B-Instruct"
config.model.siliconflow_embedding_model = "BAAI/bge-m3"
config.runtime.max_concurrency = 2
```

系统会在创建 FAISS 索引前自动检测嵌入维度。若切换到输出维度不同的嵌入模型，请重新构建工作目录。

## 复现实验

仓库提供 LoCoMo 与 LongMemEval 的评测脚本。camera-ready 使用 BGE-M3（1024 维）嵌入、温度 `0.0`、15 个锚点、2 跳遍历、每跳最多 15 个邻居，以及最多 5 次控制器调用：

```python
from cogmem import CogMemConfig

config = CogMemConfig()
config.model.temperature = 0.0
config.agent.find_anchors_top_k = 15
config.agent.traverse_neighbors_max_depth = 2
config.agent.traverse_neighbors_max_results = 15
config.agent.max_iterations = 5
```

请先从官方来源下载基准数据：[LoCoMo 论文](https://aclanthology.org/2024.acl-long.747/)、[LoCoMo 官方仓库](https://github.com/snap-research/locomo)和 [LongMemEval 论文](https://openreview.net/forum?id=pZiyCaVuti)。本仓库不分发基准数据。

```bash
# LoCoMo：构建图谱、执行巩固并评测。
uv run python scripts/eval_locomo_complete.py \
  --data_dir /path/to/locomo10.json \
  --workspace .cogmem/locomo \
  --output artifacts/locomo_results.json

# LongMemEval：使用官方准备的 JSON 数据。
uv run python scripts/eval_longmemeval_complete.py \
  --data_dir /path/to/longmemeval_s_cleaned.json \
  --workspace .cogmem/longmemeval \
  --output artifacts/longmemeval_results.json
```

完整实验前，请先为 LoCoMo 添加 `--max_samples 1` 运行单题 smoke test。不同模型版本、服务提供方或部署栈、数据版本、提示词和 judge 配置均可能导致输出差异。每次报告实验时，请保存 Git 提交、数据校验和、模型标识、命令与完整输出 JSON。LongMemEval 的论文指标需要对生成结果再运行其官方 LLM-as-judge 协议。

## 已报告结果

以下结果来自 camera-ready 实验记录。LoCoMo 指标为 F1 / BLEU-1，LongMemEval 指标为 LLM-as-judge Accuracy。结果受模型版本、数据版本、提示词与评测协议影响。

| 基准 | GPT-4o-mini | Qwen2.5-14B-Instruct |
| --- | ---: | ---: |
| LoCoMo - Single Hop | 59.20 / 56.40 | 60.40 / 55.80 |
| LoCoMo - Multi Hop | 50.70 / 46.10 | 48.42 / 44.65 |
| LoCoMo - Temporal | 63.80 / 57.90 | 56.20 / 49.10 |
| LoCoMo - Open Domain | 56.10 / 51.20 | 55.80 / 50.10 |
| LongMemEval - Accuracy | 68.40 | 66.50 |

## 仓库结构

```text
cogmem/       核心库：图谱模式、写入、巩固与智能体
datasets/     数据适配器
examples/     可直接运行的小示例
scripts/      基准评测脚本
tests/        无模型依赖的快速测试
```

## 安全与隐私

CogMem 可能持久化敏感对话内容、关系、偏好以及来源归属的观点。在实际部署中，请按用户隔离工作目录，控制图谱文件访问权限，避免记录原始对话日志，并提供保留与删除机制。发现漏洞或密钥泄露时，请通过维护者的私有渠道报告，而不要创建公开 issue。

## 贡献

正式公开前，请先与维护者私下协调改动。公开前需要确认论文状态并补充最终 URL 与 BibTeX；确保基准数据、密钥、私密对话和再分发许可不明确的素材均不进入 Git 历史；在干净环境中运行 `uv sync --extra dev`、`uv run pytest` 和上述 smoke test，并核对 README 的所有结果与归档实验输出一致。

## 许可证

CogMem 使用 [MIT License](LICENSE)。
