import os
from pathlib import Path
import faiss
import shutil
import json

# Add project root to the Python path to allow absolute imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.reader import DocumentReader
from app.components.preprocessor import SentenceWindowPreprocessor
from app.components.sparse_retriever import SparseRetriever

# LlamaIndex components
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def main():
    """
    Main script to build the LlamaIndex vector index.
    
    1.  Point to the 'data' directory.
    2.  Read all supported files (PDF, TXT, MD) using DocumentReader.
    3.  Process them into sentence nodes with metadata using SentenceWindowPreprocessor.
    4.  Convert custom nodes to LlamaIndex TextNode objects.
    5.  Setup a FAISS-based VectorStore and a HuggingFace embedding model.
    6.  Build and persist the index to disk.
    """
    print("--- Starting LlamaIndex Building Process ---")
    
    # 1. Setup paths
    data_dir = Path("./data/embodia/pdf")
    persist_dir = Path("./vector_store")
    
    # Clean up previous storage
    if persist_dir.exists():
        print(f"Removing existing storage directory: {persist_dir}")
        shutil.rmtree(persist_dir)
        
    persist_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"Error: The '{data_dir}' directory is empty or does not exist.")
        print("Please add your documents (PDF, TXT, MD) to it and run again.")
        return

    # 2. Initialize components
    reader = DocumentReader(input_dir=data_dir)
    preprocessor = SentenceWindowPreprocessor(window_size=2)
    
    # Use the same sentence-transformer model for consistency
    embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-mpnet-base-v2")
    Settings.embed_model = embed_model
    Settings.llm = None # We are not using an LLM during indexing
    Settings.chunk_size = 512 # Set a reasonable chunk size
    
    # 3. Read documents and create custom sentence nodes
    structured_docs = reader.read()
    if not structured_docs:
        print("No processable documents found.")
        return
        
    custom_nodes = preprocessor.process(structured_docs)
    
    # 4. Convert custom nodes to LlamaIndex TextNode objects
    llama_nodes = []
    for node_dict in custom_nodes:
        # Merge window into metadata to ensure it persists correctly
        node_metadata = node_dict["metadata"].copy()
        node_metadata["window"] = node_dict.get("window", "")

        node = TextNode(
            text=node_dict["text"],
            metadata=node_metadata
        )
        llama_nodes.append(node)
        
    if not llama_nodes:
        print("No nodes were created after processing. Aborting.")
        return
        
    print(f"Successfully converted {len(llama_nodes)} custom nodes to LlamaIndex TextNodes.")

    # 5. Setup FAISS VectorStore
    embedding_dim = embed_model.get_text_embedding("test") # Get embedding dim
    faiss_index = faiss.IndexFlatL2(len(embedding_dim))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    # 6. Build and persist the index
    print("\n--- Building and Persisting LlamaIndex VectorStoreIndex ---")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(
        nodes=llama_nodes,
        storage_context=storage_context
    )
    
    index.storage_context.persist(persist_dir=persist_dir)
        
    # 7. Save documents.json for BM25 Retriever
    # This avoids the need to reconstruct documents from the vector index at runtime
    print("\n--- Saving documents.json for BM25 Retriever ---")
    documents_json_path = persist_dir / "documents.json"
    bm25_documents = []
    for node in custom_nodes:
        bm25_documents.append({
            "text": node["text"],
            "window": node.get("window", ""),
            "source": node["metadata"].get("source", "N/A"),
            "page_number": node["metadata"].get("page_number", None)
        })
    
    with open(documents_json_path, "w", encoding="utf-8") as f:
        json.dump(bm25_documents, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(bm25_documents)} documents to {documents_json_path}")
    
    # 8. Build and Persist BM25 Index (New Step)
    print("\n--- Building and Persisting BM25 Index ---")
    sparse_retriever = SparseRetriever()
    sparse_retriever.build_index(bm25_documents)
    bm25_index_path = persist_dir / "bm25_index.pkl"
    sparse_retriever.save_index(bm25_index_path)

    print("\n--- Index Building Complete! ---")
    print(f"LlamaIndex with FAISS store persisted to: {persist_dir}")
    print("You can now start the FastAPI application.")


if __name__ == "__main__":
    main()
