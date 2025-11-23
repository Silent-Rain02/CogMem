# CogMem (Local SDK MVP)

最小可行的长期记忆系统：知识图谱 + 向量检索（FAISS + networkx）。

## 安装

```bash
pip install -r requirements.txt
D:\project\cogmem\.venv\Scripts\activate.ps1
```

配置环境变量（任选其一）：
- 百炼/OpenAI 兼容：`OPENAI_BASE_URL`, `OPENAI_API_KEY`
- Ollama 本地：`OLLAMA_BASE_URL`（默认 `http://localhost:11434`）

也可使用 `.env` 文件（已自动加载）：

```
# OpenAI/百炼 兼容
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-xxxxxxxx

# 或者使用 Ollama 本地
# OLLAMA_BASE_URL=http://localhost:11434
```

## 快速开始

```python
from cogmem import CogMem
from cogmem.storage.schema import Doc

cm = CogMem()
cm.add_documents([Doc(id="d1", text="Alice likes Bob in Wonderland.")])
res = cm.retrieve("Alice 喜欢谁？")
print([e.name for e in res.entities])
```

更多见 `examples/demo_books.py`。
