# 本地 RAG 知识库系统 (Local RAG System)

这是一个本地 RAG (检索增强生成) 系统。它使用 **LlamaIndex** 构建，后端采用 **FastAPI**，前端提供 **Streamlit** 界面，并支持通过 **LM Studio** 连接本地运行的大模型（如 Qwen, Llama 3）。

针对资源受限环境（如 4GB 显存）进行了一些优化，确保在普通配置下也能流畅运行。

## ✨ 主要功能

1.  **混合检索 (Hybrid Search)**
    *   **向量检索**: 使用 ChromaDB 存储和检索语义向量。
    *   **关键词检索**: 使用 BM25 算法匹配精确关键词。
    *   **动态融合**: 根据你的问题长度自动调整权重（短问题看重关键词，长问题看重语义）。

2.  **重排序 (Reranking)**
    *   使用 `BAAI/bge-reranker` 模型对初步检索到的结果进行二次打分，把最相关的排到最前面。

3.  **精细化切分**
    *   将文档切分为较小的片段（256 token），既节省显存，又能更精准地定位信息。
    *   支持 PDF 自动转 Markdown，保留文档标题结构。

4.  **流式输出 (Streaming)**
    *   支持打字机效果，即使本地模型生成较慢，也能立刻看到反馈。

## 🛠️ 环境准备

*   Python 3.10+
*   LM Studio (用于运行本地 LLM)
*   推荐配置：8GB+ 内存，4GB+ 显存

### 安装依赖

```bash
pip install -r requirements.txt
```

*(如果缺少某些库，请根据报错安装，主要依赖包括：`llama-index`, `chromadb`, `fastapi`, `uvicorn`, `streamlit`, `rank_bm25`, `pymupdf4llm`)*

## 🚀 使用指南

### 1. 准备数据

将你的 PDF 文档放入 `data/embodia/pdf/` 目录中。

### 2. 构建索引

运行以下命令。它会将文档处理成向量和关键词索引，存入本地的 `chroma_db` 文件夹。

```bash
python scripts/build_index.py
```

> **注意**: 每次添加新文件后，都需要重新运行这一步。

### 3. 启动本地模型 (LM Studio)

1.  打开 LM Studio。
2.  加载一个模型（推荐 小型模型例如 **Phi-3-mini** 以获得最佳速度，默认使用 Qwen-4B）。
3.  点击左侧 **Server** 图标。
4.  点击 **Start Server**，保持默认端口 `1234`。

### 4. 启动后端服务

启动 FastAPI 后端：

```bash
uvicorn app.main:app --reload
```
- API 文档地址: http://127.0.0.1:8000/docs
### 5. 启动对话界面

另开一个终端，启动 Streamlit 前端：

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

如果你想在其他程序中调用：

*   **流式对话**: `POST /api/chat/stream` 
*   **普通对话**: `POST /api/chat`
*   **纯检索**: `POST /api/query` (只返回文档，不生成回答)