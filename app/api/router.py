from fastapi import APIRouter, Request, HTTPException
# APIRouter 用于创建模块化的路由；Request 用于在函数中获取全局应用状态（如我们存好的 pipeline）
from pydantic import BaseModel
# 用于定义数据模型，确保用户发送的数据格式是正确的
from typing import List, Dict

# Pydantic model for the query request body
# 定义请求体模型。它规定了用户发过来的 JSON 必须长什么样。
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Pydantic model for the response
# 定义响应体模型。它规定了系统返回给用户的数据格式。
class DocumentResponse(BaseModel):
    text: str # 检索到的原始句子
    window: str # 包含上下文的窗口
    source: str
    page_number: int | None = None

router = APIRouter()# 实例化 APIRouter

@router.post("/query", response_model=List[DocumentResponse])
# 定义一个 POST 类型的接口，路径是 /query。这个接口最后返回的一定是一个包含多个 DocumentResponse 对象的列表
async def perform_query(request: Request, payload: QueryRequest):
    # Request：让函数能访问到整个 FastAPI 应用的上下文
    # QueryRequest：FastAPI 会自动解析用户发来的 JSON，并验证是否符合 QueryRequest 的定义
    """
    Accepts a query and returns the most relevant document chunks.
    """
    # 检查 app.state 里面有没有我们在 main.py 里塞进去的 pipeline
    if not hasattr(request.app.state, 'pipeline') or request.app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized or loaded.")
    
    # 从全局状态中取出RAGPipeline 实例
    pipeline = request.app.state.pipeline
    
    # Run the pipeline
    try:
        # Pass top_k to the run method
        # 调用 run 方法，传入用户的查询词和数量。results 将会是 retriever 找回来的文档块列表。
        results = pipeline.run(query=payload.query, top_k=payload.top_k)
        return results
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during pipeline execution: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during query processing.")
