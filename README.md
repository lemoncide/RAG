# Local RAG Knowledge Base (LlamaIndex + FastAPI + Streamlit)

这是一个高性能的本地 RAG (检索增强生成) 系统，旨在作为 AI Agent 的知识库后端或独立问答服务。

它集成了 **混合检索 (Hybrid Search)**、**重排序 (Reranking)** 和 **句子窗口检索 (Sentence Window Retrieval)** 等高级 RAG 技术，并支持通过 **LM Studio** 连接本地 LLM。

## ✨ 主要特性

- **混合检索**: 结合向量检索 (FAISS/Dense) 和关键词检索 (BM25/Sparse)，利用 RRF (倒数排名融合) 算法合并结果。
- **高级重排序**: 集成 `BAAI/bge-reranker` 模型，对检索结果进行二次精排，大幅提升相关性。
- **句子窗口检索**: 索引时切分细粒度句子，检索时返回包含前后文的完整窗口，减少断章取义。
- **本地 LLM 支持**: 无缝对接 LM Studio (兼容 OpenAI 协议)，数据不出本地。
- **可视化调试**: 提供 Streamlit 前端，方便直观地查看检索分数、上下文窗口和生成结果。

## 🛠️ 环境准备

- Python 3.10+
- LM Studio (用于运行本地大模型)

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 准备数据

将你的知识库文档（支持 PDF, TXT, MD）放入以下目录：

```text
data/embodia/pdf/
```

### 2. 构建索引

运行构建脚本，它会处理文档、生成向量索引 (FAISS) 和关键词索引 (BM25)：

```bash
python scripts/build_index.py
```

> **注意**: 每次添加新文档后都需要重新运行此步骤。

### 3. 启动本地 LLM (LM Studio)

1. 打开 **LM Studio**。
2. 下载并加载一个模型 。
3. 点击左侧 **Local Server** 图标 (双向箭头)。
4. 确保端口为 `1234`，点击 **Start Server**。

### 4. 启动后端 API

启动 FastAPI 服务：

```bash
uvicorn app.main:app --reload
```

- API 文档地址: http://127.0.0.1:8000/docs

### 5. 启动调试前端 (可选)

启动 Streamlit 界面进行对话测试：

```bash
streamlit run streamlit_app.py
```

浏览器将自动打开 http://localhost:8501。

## 🔌 API 接口说明

### 1. 对话接口 (RAG + LLM)

- **Endpoint**: `POST /api/chat`
- **描述**: 检索相关文档并生成回答。
- **请求示例**:
  ```json
  {
    "query": "机器人驾驶车辆竞赛是什么？",
    "top_k": 5
  }
  ```

### 2. 纯检索接口 (Retriever Only)

- **Endpoint**: `POST /api/query`
- **描述**: 仅返回相关的文档片段，不生成回答。适合作为 Agent 的工具调用。
- **请求示例**:
  ```json
  {
    "query": "机器人竞赛规则",
    "top_k": 10
  }
  ```
