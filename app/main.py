import json
from contextlib import asynccontextmanager
from fastapi import FastAPI

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
    
    # 4. Load the pre-built index
    try:
        retriever.load_index(index_path)
        retriever.documents = documents # IMPORTANT: associate text with the index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print("Please ensure 'faiss_index.bin' exists and is valid.")
        app.state.pipeline = None
        yield
        return

    # 5. Initialize the main RAG pipeline
    # Reader and Preprocessor are not needed for serving, only for indexing
    app.state.pipeline = RAGPipeline(reader=None, preprocessor=None, retriever=retriever)
    
    print("--- RAG pipeline and resources loaded successfully ---")
    
    yield
    
    # Clean up the models and release the resources
    print("Cleaning up resources.")
    app.state.pipeline = None


app = FastAPI(
    title="RAG From Scratch API",
    description="An API for interacting with a custom RAG system.",
    lifespan=lifespan
)

@app.get("/", tags=["General"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "ok"}

# Include the router for query handling
app.include_router(api_router, prefix="/api")

print("FastAPI app created. To run the server, use the command: uvicorn app.main:app --reload")

