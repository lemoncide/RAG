from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

# Add project root to the Python path to allow absolute imports from app/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the new LlamaIndex-based pipeline
from app.core.llama_pipeline import LlamaIndexRAGPipeline
from app.api.router import router as api_router

# A 'lifespan' function is best practice for loading models/resources
# on startup and releasing them on shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the RAG pipeline on startup.
    This new version loads the self-contained LlamaIndex pipeline.
    """
    try:
        # Initialize the main RAG pipeline.
        # This class now handles its own resource loading internally.
        app.state.pipeline = LlamaIndexRAGPipeline(persist_dir="./vector_store")
    except (FileNotFoundError, ConnectionError) as e:
        print(f"FATAL: Could not initialize RAG pipeline: {e}")
        app.state.pipeline = None
    except Exception as e:
        print(f"An unexpected error occurred during pipeline initialization: {e}")
        app.state.pipeline = None
    
    yield # The application runs while the 'yield' is active
    
    # Clean up the models and release the resources on shutdown
    print("Cleaning up resources.")
    app.state.pipeline = None

# Instantiate FastAPI
app = FastAPI(
    title="LlamaIndex RAG API",
    description="An API for interacting with a RAG system powered by LlamaIndex and self-querying.",
    lifespan=lifespan
)

@app.get("/", tags=["General"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "ok"}

# Include the router for query handling
app.include_router(api_router, prefix="/api")

print("FastAPI app created. To run the server, use the command: uvicorn app.main:app --reload")
