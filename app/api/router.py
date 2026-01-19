from fastapi import APIRouter, Request, HTTPException
# APIRouter 用于创建模块化的路由；Request 用于在函数中获取全局应用状态（如我们存好的 pipeline）
from pydantic import BaseModel
# 用于定义数据模型，确保用户发送的数据格式是正确的
from typing import List, Dict, Optional, Any

# Pydantic model for the query request body
# 定义请求体模型。它规定了用户发过来的 JSON 必须长什么样。
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None # "filters" 字段是可选的，可以接受任意键值对

    class Config:
        json_schema_extra = {
            "example": {
                "query": "机器人驾驶车辆竞赛",
                "top_k": 5,
                "filters": {}
            }
        }

# Pydantic model for the response
# 定义响应体模型。它规定了系统返回给用户的数据格式。
class DocumentResponse(BaseModel):
    text: str
    window: str
    source: str
    page_number: int | None = None
    
    # Scores from different stages of the pipeline
    distance: float | None = None        # Lower is better. From Dense (vector) Retriever.
    bm25_score: float | None = None      # Higher is better. From Sparse (keyword) Retriever.
    rerank_score: float | None = None    # Higher is better. From Reranker.
    
    is_reranked: bool | None = None      # Flag to indicate if the result came from the reranker.

router = APIRouter()# 实例化 APIRouter

@router.post("/query", response_model=List[DocumentResponse])
# 定义一个 POST 类型的接口，路径是 /query。这个接口最后返回的一定是一个包含多个 DocumentResponse 对象的列表
async def perform_query(request: Request, payload: QueryRequest):
    # Request：让函数能访问到整个 FastAPI 应用的上下文
    # QueryRequest：FastAPI 会自动解析用户发来的 JSON，并验证是否符合 QueryRequest 的定义
    """
    Accepts a query and returns the most relevant document chunks.
    Can also accept an optional 'filters' dictionary for metadata filtering.
    """
    # 检查 app.state 里面有没有我们在 main.py 里塞进去的 pipeline
    if not hasattr(request.app.state, 'pipeline') or request.app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized or loaded.")
    
    # 从全局状态中取出RAGPipeline 实例
    pipeline = request.app.state.pipeline
    
    # Run the pipeline
    try:
        # Pass the query, top_k, and optional filters to the run method
        results = pipeline.run(
            query=payload.query, 
            top_k=payload.top_k, 
            filters=payload.filters
        )
        return results
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during pipeline execution: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during query processing.")
