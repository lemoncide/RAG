import json
from contextlib import asynccontextmanager
from fastapi import FastAPI #Web框架核心

# Add project root to the Python path to allow absolute imports from app/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.pipeline import RAGPipeline
from app.components.retriever import DenseRetriever
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
    index_path = "faiss_index.bin"
    documents_path = "documents.json"
    
    # 2. Load the text chunks
    try:
        with open(documents_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{documents_path}' not found. Please run 'scripts/build_index.py' first.")
        app.state.pipeline = None
        yield
        return

    # 3. Initialize components
    retriever = DenseRetriever(model_name='all-MiniLM-L6-v2')
    
    # 4. Load the pre-built index 初始化检索器,加载模型，将文本转换为向量
    try:
        retriever.load_index(index_path)
        retriever.documents = documents # IMPORTANT: associate text with the index
        #读取 FAISS 索引文件，并将刚才加载的文本块关联到检索器中。这样检索器在搜到向量 ID 后，能直接返回对应的文字。
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print("Please ensure 'faiss_index.bin' exists and is valid.")
        app.state.pipeline = None
        yield
        return

    # 5. Initialize the main RAG pipeline
    # Reader and Preprocessor are not needed for serving, only for indexing
    # 封装 Pipeline。将检索器装入 RAGPipeline 对象，并挂载到 app.state
    app.state.pipeline = RAGPipeline(reader=None, preprocessor=None, retriever=retriever)
    
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

