from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# Pydantic model for the query request body
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Pydantic model for the response
class DocumentResponse(BaseModel):
    text: str
    source: str
    page_number: int | None = None

router = APIRouter()

@router.post("/query", response_model=List[DocumentResponse])
async def perform_query(request: Request, payload: QueryRequest):
    """
    Accepts a query and returns the most relevant document chunks.
    """
    if not hasattr(request.app.state, 'pipeline') or request.app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized or loaded.")
    
    pipeline = request.app.state.pipeline
    
    # Run the pipeline
    try:
        # Pass top_k to the run method
        results = pipeline.run(query=payload.query, top_k=payload.top_k)
        return results
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during pipeline execution: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during query processing.")
