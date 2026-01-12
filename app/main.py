import json
from contextlib import asynccontextmanager
from fastapi import FastAPI #Web框架核心

# Add project root to the Python path to allow absolute imports from app/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.pipeline import RAGPipeline
from app.components.retriever import DenseRetriever
from app.components.sparse_retriever import SparseRetriever
from app.components.reranker import Reranker
from app.api.router import router as api_router

# A 'lifespan' function is best practice for loading models/resources
# on startup and releasing them on shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the RAG pipeline on startup.
    """
    print("--- Loading RAG pipeline and resources ---")
    
    # 1. Define paths
    faiss_index_path = "faiss_index.bin"
    bm25_index_path = "bm25_index.pkl"
    documents_path = "documents.json"
    
    # 2. Load the shared documents
    try:
        with open(documents_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{documents_path}' not found. Please run 'scripts/build_index.py' first.")
        app.state.pipeline = None
        yield
        return

    # 3. Initialize Dense Retriever
    try:
        dense_retriever = DenseRetriever(model_name='paraphrase-multilingual-mpnet-base-v2')
        dense_retriever.load_index(faiss_index_path)
        dense_retriever.documents = documents
    except Exception as e:
        print(f"Error loading Dense Retriever: {e}")
        print("Please ensure 'faiss_index.bin' and model exist and are valid.")
        app.state.pipeline = None
        yield
        return

    # 4. Initialize Sparse Retriever
    try:
        sparse_retriever = SparseRetriever()
        sparse_retriever.load_index(bm25_index_path, documents)
    except Exception as e:
        print(f"Error loading Sparse Retriever: {e}")
        print("Please ensure 'bm25_index.pkl' exists and is valid.")
        app.state.pipeline = None
        yield
        return

    # 5. Initialize the reranker (optional)
    reranker = None
    try:
        reranker = Reranker()
    except ImportError:
        print("--- Reranker not available. Install 'FlagEmbedding' to enable it. ---")
    except Exception as e:
        print(f"--- Error initializing Reranker: {e} ---")
        print("--- Proceeding without Reranker. ---")

    # 6. Initialize the main RAG pipeline with both retrievers
    app.state.pipeline = RAGPipeline(
        reader=None, 
        preprocessor=None, 
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker
    )
    
    print("--- RAG pipeline and resources loaded successfully ---")
    
    yield #分割线,其上方的代码在启动时运行，下方的在关闭时运行
    
    # Clean up the models and release the resources 服务器停止时清空 pipeline
    print("Cleaning up resources.")
    app.state.pipeline = None

# 实例化 FastAPI
app = FastAPI(
    title="RAG From Scratch API",
    description="An API for interacting with a custom RAG system.",
    lifespan=lifespan
)

@app.get("/", tags=["General"])
# 告诉 FastAPI：当用户在浏览器访问根路径时，请执行下面的 read_root
async def read_root():
    """A simple health check endpoint."""
    return {"status": "ok"}

# Include the router for query handling
app.include_router(api_router, prefix="/api")

print("FastAPI app created. To run the server, use the command: uvicorn app.main:app --reload")

