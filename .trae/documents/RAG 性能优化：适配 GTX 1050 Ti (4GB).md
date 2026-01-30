# 性能优化计划：解决本地推理过慢问题

根据你的 LM Studio 日志分析：
- **Prompt Eval Time**: 16.9s (处理输入 1944 tokens)。
- **Eval Time**: 77.4s (生成 387 tokens，速度 **5 tokens/s**)。
- **总耗时**: 94.7s。
- **显存瓶颈**: 你使用的是 **GTX 1050 Ti (4GB)**。日志显示 `CUDA0 (GTX 1050 Ti) | 4095 = 189 + (3144 = 2267 + 576 + 301) + 761`。
    - 模型权重占了 2.2GB。
    - KV Cache (Context) 占了 576MB。
    - Compute Buffer 占了 301MB。
    - 显存只剩 189MB，非常危险，可能已经触发了部分 Offload 到 CPU 或者显存交换，导致速度只有 5 tokens/s。

对于 GTX 1050 Ti (4GB)，Qwen-4B 即使量化后也有些吃力。我们需要从**减少输入量**和**优化模型**两方面入手。

## 1. 减少输入上下文 (Critical)
目前的 `top_k=5` 导致输入近 2000 tokens，直接填满了显存的上下文窗口。
- **Action**: 将默认 `top_k` 从 5 降为 **3**。这能直接减少 40% 的 Context 占用，显著降低显存压力和 Prompt 处理时间。
- **Action**: 在 `LlamaIndexRAGPipeline` 中增加 `context_window` 限制，强制截断过长的 Prompt。

## 2. 优化 Prompt 模板
目前的 Prompt 模板包含了一些冗余描述。
- **Action**: 精简 `synthesize` 方法中的 Prompt Template，减少系统指令的 Token 消耗。

## 3. 建议用户操作 (模型侧)
- **更换模型**: 建议换用更小的 **Qwen-1.5-1.8B-Chat** 或 **Phi-3-Mini (3.8B, 4-bit)**。对于 4GB 显存，1.8B 模型可以跑得飞快（30+ tokens/s），而 RAG 的效果主要取决于检索质量，小模型做总结通常足够。
- **调整 Quantization**: 确保使用的是 `q4_k_m` 或 `q4_0` 量化版本，不要用 fp16。

## 执行步骤
1.  **修改 `app/api/router.py`**: 将 `QueryRequest` 的默认 `top_k` 改为 3。
2.  **修改 `app/core/llama_pipeline.py`**: 
    - 默认 `top_k` 改为 3。
    - 优化 `synthesize` 和 `synthesize_stream` 中的 Prompt Template，使其更短更直接。
3.  **验证**: 重启服务，观察响应速度。