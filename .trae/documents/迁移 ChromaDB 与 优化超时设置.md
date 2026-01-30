# 架构升级与优化计划

根据你的需求，我们将从底层存储、数据处理到检索策略进行全面的“工业级”改造。

## 1. 核心架构变更
### 向量数据库迁移 (Faiss -> ChromaDB)
- **目标**: 解决本地文件存储难以扩展和持久化的问题。
- **方案**:
    - 引入 `chromadb` 和 `llama-index-vector-stores-chroma`。
    - 替换 `scripts/build_index.py` 和 `app/core/llama_pipeline.py` 中的 Faiss 实现。
    - 数据将持久化存储在 `./chroma_db` 目录，支持元数据过滤和增量更新。

### 数据处理流重构 (PDF -> Markdown)
- **目标**: 保留文档结构（表格、标题层级），为未来接入 Agent 和图数据库做准备。
- **方案**:
    - **解析器**: 替换 `unstructured` 为 `pymupdf4llm`。它能直接将 PDF 转换为高质量的 Markdown，自动处理表格和图片占位符。
    - **切分器**: 使用 LlamaIndex 的 `MarkdownNodeParser`。它会根据 Markdown 标题（# H1, ## H2）智能切分，保证语义块的完整性，优于简单的固定窗口或字符递归切分。
    - **清理**: 简化 `cleaner.py`，因为 Markdown 转换器已经处理了大部分乱码和排版问题。
    - **超时调整**: 将解析超时时间从 60s 提升至 **600s**。

## 2. 检索策略优化 (Dynamic RRF)
- **目标**: 根据用户输入意图动态调整关键词与向量检索的权重。
- **策略**:
    - **短查询模式 (Short Query)**: 输入长度 < 5 个词。用户通常是在搜专有名词（如 "Transformer架构"）。
        - **权重**: BM25 (1.5) > Vector (1.0)。
    - **长查询模式 (Long Query)**: 输入长度 >= 5 个词。用户通常是在询问复杂问题（如 "Transformer 架构中自注意力机制是如何解决长距离依赖的？"）。
        - **权重**: Vector (2.0) > BM25 (1.0)。
- **实现**: 在 `LlamaIndexRAGPipeline.run` 中增加简单的启发式判断逻辑。

## 3. 依赖更新
- 安装 `chromadb`, `pymupdf4llm`, `llama-index-vector-stores-chroma`。

---

## 你的疑问解答 (Q&A)

1.  **RRF 权重与输入长度**:
    *   **是的，非常有必要**。短查询（关键词）不仅包含信息少，而且语义模糊，BM25 的精确匹配往往更准。长查询语义丰富，向量检索优势更大。我们将实现这个动态逻辑。

2.  **结构化信息 (Markdown) 与未来扩展**:
    *   **绝对值得**。如果你未来要做 **GraphRAG**（图数据库），Markdown 的标题层级（H1/H2/H3）就是天然的“树状结构”骨架。将 PDF 转为 Markdown 是从“非结构化”迈向“半结构化”的关键一步，能极大提升 Agent 对文档逻辑的理解。

3.  **切分策略 (递归字符 vs 句子窗口)**:
    *   **句子窗口 (Sentence Window)**: 适合“只看局部”。比如问“API 的参数是什么”，它定位准。
    *   **Markdown 切分**: 适合“理解全貌”。它按章节切分，保留了上下文的完整性。
    *   **结论**: 既然转了 Markdown，用 `MarkdownNodeParser` 是最优解。

4.  **Entity Linking (实体链接)**:
    *   **暂时不做**。在没有知识图谱（KB）的情况下，做实体链接（把 "Apple" 链接到 "Apple Inc." 实体 ID）意义不大。
    *   **替代方案**: 可以在检索前加一步 **Query Expansion (查询扩展)**，利用 LLM 把用户模糊的词扩充清楚（例如把 "它怎么部署" 改写为 "RAG 系统如何部署"），这比实体链接更实惠有效。

## 执行步骤
1.  **环境准备**: 安装新依赖库。
2.  **Reader 改造**: 重写 `reader.py`，集成 `pymupdf4llm`，调整超时。
3.  **Index 改造**: 重写 `build_index.py`，对接 ChromaDB 和 Markdown 解析。
4.  **Pipeline 改造**: 升级检索逻辑，支持 Chroma 加载和动态 RRF 权重。