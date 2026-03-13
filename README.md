# CogMem: From Retrieval to Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b)](https://arxiv.org/abs/TBD)

CogMem 是一个用于长期对话记忆的认知图谱系统，强调从“被动检索”走向“主动重构”。系统以 **PEC$^2$F（Person-Event-Concept-Claim-Fact）** 结构化记忆为核心，将对话中的人物、事件、概念、主张与事实组织为可演化的认知图谱，支持跨事件、跨时间的复杂推理与一致性维护。

## 论文

**From Retrieval to Reconstruction: Constructing Evolvable Cognitive Memory for Long-term Dialogue**  
arXiv: `TBD`

### 摘要

Large Language Models (LLMs) are evolving into long-term personal companions, necessitating memory systems that go beyond simple text retrieval. However, existing Retrieval-Augmented Generation (RAG) frameworks typically treat memory as a flat, passive repository. This leads to semantic isolation, where the temporal links between events and the logical connections between entities are lost, hindering complex reasoning in multi-turn dialogues. In this paper, we introduce **CogMem**, a cognitive graph architecture designed to reconstruct long-term context fidelity. CogMem proposes a human-centric **PEC$^2$F (Person-Event-Concept-Claim-Fact)** schema that structurally organizes dialogue into interconnected episodic traces and semantic knowledge. By explicitly integrating **Claim nodes** within this framework, CogMem ensures epistemic clarity, distinguishing subjective attributions from objective records. Shifting from static retrieval pipelines to agentic active recall, we further develop a Cognitive Search Agent that dynamically navigates this graph using atomic operators (e.g., intersection, temporal scanning). Experiments on the LoCoMo and LongMemEval benchmarks demonstrate that CogMem significantly outperforms state-of-the-art baselines in multi-hop reasoning and temporal consistency, validating the necessity of structural reconstruction over passive vector matching.

## 亮点

- 🧠 **认知图谱记忆**：以 PEC$^2$F 为记忆组织骨架，显式区分 Claim 与 Fact
- 🔎 **主动回忆检索**：Cognitive Search Agent 通过原子操作进行多步搜索
- 🧩 **多跳推理**：支持跨事件与时间线的推理链路追踪
- ⚡ **高性能索引**：FAISS + NetworkX 的混合索引与图遍历

## 主流程图

![CogMem Overall Architecture (placeholder)](assets/figure2_placeholder.svg)

**The Overall Architecture of CogMem.** The framework operates across two layers: (1) **Memory Generation & Evolution (Top)**: (a) *Memorize*: The system ingests raw dialogue to construct the initial cognitive graph, converting episodic traces to graph nodes. (b) *Consolidate*: An offline mechanism (mimicking sleep) refines the graph by synthesizing high-level Facts from Events, deduplicating Facts and Concepts, and reconciling Claims. (2) **Cognitive Reasoning & Application (Bottom)**: (c) *Reasoning*: Shifting from passive retrieval to active recall, the Search Agent utilizes a specialized **Tools Set** (Anchoring, Intersection, Traversal, Evidence Grounding) to dynamically navigate the graph and reconstruct answers via a ReAct loop.

## 安装

### 环境要求

- Python 3.12+
- 可用的 LLM/Embedding 服务（OpenAI 兼容、Ollama、SiliconFlow 或 AdaptorLLM）

### 安装依赖

```bash
git clone <repository-url>
cd Cogmem-opensourse

pip install -e .
# 或者
uv pip install -e .
```

### 环境变量

```bash
# OpenAI/兼容服务
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-xxxxxxxx

# SiliconFlow
SILICONFLOW_API_KEY=sk-xxxxxxxx

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# AdaptorLLM
CHAT_URL=https://your-adaptor-endpoint
CHAT_AK=your-ak
CHAT_SK=your-sk
```

## 快速开始

```python
from cogmem import CogMem
from cogmem.storage.schema import Doc

cm = CogMem(workspace_dir="./workspace")

docs = [
    Doc(id="d1", text="Alice likes Bob. They went to the park together."),
    Doc(id="d2", text="Bob works at Google. He is a software engineer."),
]
cm.add_documents(docs)

result = cm.retrieve("Where does Bob work?")
print([e.name for e in result.entities])
print(result.evidence)

answer = cm.answer("Where does Bob work?")
print(answer)
```

更多示例见 `examples/demo_full_pipeline.py`。

## 配置

```python
from cogmem.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG
config.model.provider = "siliconflow"  # ollama | siliconflow | adaptor_llm | modelscope
config.runtime.max_concurrency = 2
```

## 项目结构

```
Cogmem-opensourse/
├── assets/                 # 图像与可视化资源
├── cogmem/                 # 核心库
│   ├── agents/             # Cognitive Search Agent
│   ├── memory/             # 索引/巩固/去重
│   ├── storage/            # 图谱与向量存储
│   ├── models/             # LLM/Embedding 客户端
│   └── utils/              # 工具与指标
├── examples/               # 示例脚本
├── scripts/                # 分析与调试工具
├── tests/                  # 单元测试
├── LICENSE                 # 许可证
├── README.md               # 项目说明
└── pyproject.toml          # 项目依赖与配置
```

## 评测

评测脚本入口：`scripts/eval_locomo_category_detailed.py`（LoCoMo 分类别评测）；`scripts/eval_locomo_complete.py`（LoCoMo 完整测评 包括离线记忆巩固步骤）。

LoCoMo 数据集下载：请从官方仓库获取 `locomo10.json` 并放置到本项目可访问路径（例如 `datasets/locomo10.json`）。数据与说明见官方仓库：[https://github.com/snap-research/locomo](https://github.com/snap-research/locomo)。  
参考论文引用（LoCoMo）：
```
@misc{locomo,
      title={Evaluating Very Long-Term Conversational Memory of LLM Agents}, 
      author={Adyasha Maharana and Dong-Ho Lee and Sergey Tulyakov and Mohit Bansal and Francesco Barbieri and Yuwei Fang},
      year={2024},
      eprint={2402.17753},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.17753}, 
}
```

### LoCoMo 主实验结果

Performance comparison on the **LoCoMo** benchmark. Metrics are F1 Score and BLEU-1. Best results are in **bold**, second best are <u>underlined</u>.

<table>
  <thead>
    <tr>
      <th rowspan="2">Backbone</th>
      <th rowspan="2">Method</th>
      <th colspan="2">Single Hop</th>
      <th colspan="2">Multi Hop</th>
      <th colspan="2">Temporal</th>
      <th colspan="2">Open Domain</th>
    </tr>
    <tr>
      <th>F1</th><th>BLEU-1</th>
      <th>F1</th><th>BLEU-1</th>
      <th>F1</th><th>BLEU-1</th>
      <th>F1</th><th>BLEU-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10"><strong>GPT-4o-mini</strong></td>
      <td>long-LLM</td>
      <td>46.68</td><td>37.54</td><td>29.23</td><td>22.76</td><td>25.97</td><td>19.42</td><td>16.87</td><td>13.70</td>
    </tr>
    <tr>
      <td>Naive RAG</td>
      <td><u>52.45</u></td><td><u>47.94</u></td><td>27.50</td><td>20.13</td><td><u>46.07</u></td><td><u>40.35</u></td><td>23.23</td><td>17.94</td>
    </tr>
    <tr>
      <td>LightRAG</td>
      <td>42.57</td><td>33.82</td><td>28.46</td><td>23.75</td><td>22.85</td><td>16.18</td><td><strong>54.33</strong></td><td><strong>49.61</strong></td>
    </tr>
    <tr>
      <td>HippoRAG2</td>
      <td>39.81</td><td>31.19</td><td>39.79</td><td>37.40</td><td>26.74</td><td>22.31</td><td>51.41</td><td>50.15</td>
    </tr>
    <tr>
      <td>A-Mem</td>
      <td>44.65</td><td>37.06</td><td>27.02</td><td>20.09</td><td>45.85</td><td>36.67</td><td>12.14</td><td>12.00</td>
    </tr>
    <tr>
      <td>Mem0</td>
      <td>47.65</td><td>38.72</td><td>38.72</td><td>27.13</td><td>48.93</td><td>40.51</td><td>28.64</td><td>21.58</td>
    </tr>
    <tr>
      <td>MemoryOS</td>
      <td>48.62</td><td>42.99</td><td>35.27</td><td>25.22</td><td>41.15</td><td>30.76</td><td>20.02</td><td>16.52</td>
    </tr>
    <tr>
      <td>LightMem</td>
      <td>41.79</td><td>37.83</td><td>29.78</td><td>24.80</td><td>43.71</td><td>39.72</td><td>16.89</td><td>13.92</td>
    </tr>
    <tr>
      <td>GAM</td>
      <td><strong>57.75</strong></td><td><strong>52.10</strong></td><td><u>42.29</u></td><td><u>34.44</u></td><td><strong>59.45</strong></td><td><strong>53.11</strong></td><td><u>33.30</u></td><td><u>26.97</u></td>
    </tr>
    <tr>
      <td><strong>CogMem</strong></td>
      <td>51.20</td><td>46.50</td><td><strong>55.12</strong></td><td><strong>50.80</strong></td><td>29.15</td><td>23.40</td><td>36.50</td><td>41.20</td>
    </tr>
    <tr>
      <td rowspan="10"><strong>Qwen2.5-14B</strong></td>
      <td>long-LLM</td>
      <td>46.05</td><td>39.56</td><td>32.08</td><td>24.46</td><td>30.51</td><td>24.45</td><td>14.89</td><td>11.41</td>
    </tr>
    <tr>
      <td>Naive RAG</td>
      <td><u>47.87</u></td><td><u>42.79</u></td><td>26.38</td><td>19.54</td><td><u>30.78</u></td><td>25.97</td><td>14.16</td><td>10.52</td>
    </tr>
    <tr>
      <td>LightRAG</td>
      <td>41.50</td><td>34.42</td><td>26.03</td><td>21.73</td><td>20.86</td><td>16.83</td><td><strong>54.03</strong></td><td><strong>48.65</strong></td>
    </tr>
    <tr>
      <td>HippoRAG2</td>
      <td>33.10</td><td>25.74</td><td>34.78</td><td>31.15</td><td>24.88</td><td>18.56</td><td><u>53.42</u></td><td><u>48.98</u></td>
    </tr>
    <tr>
      <td>A-Mem</td>
      <td>33.75</td><td>30.04</td><td>22.09</td><td>15.28</td><td>27.19</td><td>22.05</td><td>13.49</td><td>10.74</td>
    </tr>
    <tr>
      <td>Mem0</td>
      <td>42.58</td><td>35.15</td><td>31.73</td><td>24.82</td><td>28.96</td><td><u>26.24</u></td><td>15.03</td><td>11.28</td>
    </tr>
    <tr>
      <td>MemoryOS</td>
      <td>46.33</td><td>41.62</td><td>38.19</td><td>29.26</td><td>32.24</td><td>27.86</td><td>20.27</td><td>15.94</td>
    </tr>
    <tr>
      <td>LightMem</td>
      <td>34.92</td><td>31.22</td><td>25.45</td><td>19.61</td><td>32.03</td><td>27.70</td><td>15.81</td><td>11.81</td>
    </tr>
    <tr>
      <td>GAM</td>
      <td><strong>58.93</strong></td><td><strong>53.74</strong></td><td><u>42.96</u></td><td><u>34.48</u></td><td><strong>51.52</strong></td><td><strong>44.43</strong></td><td>30.63</td><td>26.04</td>
    </tr>
    <tr>
      <td><strong>CogMem</strong></td>
      <td>30.10</td><td>26.77</td><td><strong>53.49</strong></td><td><strong>49.11</strong></td><td>25.12</td><td>28.56</td><td>34.95</td><td>39.66</td>
    </tr>
  </tbody>
</table>

## 引用

arXiv 链接上线后将补充 BibTeX。

## 许可证

MIT License. See `LICENSE`.
