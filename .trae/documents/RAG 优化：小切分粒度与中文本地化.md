# RAG 系统优化计划：减小 Token 与中文注释

用户反馈 512 token 的切分粒度在本地推理时依然偏慢，希望进一步减小切分大小，同时保持 `top_k=5` 的召回数量。此外，需要将代码注释全面汉化。

## 1. 调整索引切分策略 (Critical)
为了在保持 `top_k=5` 的同时减少总 Token 输入量，我们需要显著减小每个 chunk 的大小。
- **Action**: 修改 `scripts/build_index.py`。
- **参数调整**: 将 `chunk_size` 从 512 降至 **256**（甚至更小，视效果而定，建议 256）。
    - 计算：5 个 chunk * 256 tokens ≈ 1280 tokens。比之前的 ~2500 tokens 减少一半，推理速度将提升一倍。
- **影响**: 需要重新构建索引。

## 2. 恢复默认 Top K
- **Action**: 修改 `app/api/router.py` 和 `app/core/llama_pipeline.py`。
- **参数调整**: 将默认 `top_k` 恢复为 **5**。因为单块变小了，我们需要更多的块来覆盖上下文。

## 3. 代码注释汉化
- **Action**: 遍历核心文件，将英文注释翻译为中文，方便维护。
    - `app/core/llama_pipeline.py`
    - `scripts/build_index.py`
    - `app/components/reader.py` (如有)
    - `app/api/router.py`

## 执行步骤
1.  **修改构建脚本**: 调整 `chunk_size=256` 并汉化 `scripts/build_index.py`。
2.  **重建索引**: 运行 `python scripts/build_index.py`。
3.  **调整 Pipeline**: 汉化 `app/core/llama_pipeline.py` 并恢复 `top_k=5`。
4.  **调整 Router**: 汉化 `app/api/router.py` 并恢复 `top_k=5`。
5.  **调整 Streamlit**: 确保前端默认值也同步更新。