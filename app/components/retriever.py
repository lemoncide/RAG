import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

class DenseRetriever:
    """
    A component for creating and querying a dense vector index using FAISS.
    
    This class handles embedding creation and similarity search, getting inspiration
    from 'rag-from-scratch' for understanding the core mechanics.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print("Initializing SentenceTransformer model...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        print("Model initialized.")

    def build_index(self, documents: List[Dict[str, str]]):
        """
        Creates a FAISS index from a list of document chunks.
        
        Args:
            documents: A list of dictionaries, each with a "text" key.
        """
        self.documents = documents
        texts = [doc["text"] for doc in self.documents]
        
        print(f"Embedding {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        
        embedding_dim = embeddings.shape[1]
        print(f"Embeddings created with dimension: {embedding_dim}")
        
        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(np.array(embeddings, dtype=np.float32))
        print(f"FAISS index built. Total vectors in index: {self.index.ntotal}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Searches the index for the most relevant document chunks for a given query.
        """
        if self.index is None:
            raise RuntimeError("Index is not built. Please call 'build_index' first.")
            
        print(f"Retrieving top {top_k} documents for query: '{query}'")
        query_embedding = self.model.encode([query])
        
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        results = [self.documents[i] for i in indices[0]]
        return results

    def save_index(self, file_path: str):
        """Saves the FAISS index to a file."""
        print(f"Saving FAISS index to {file_path}")
        faiss.write_index(self.index, file_path)

    def load_index(self, file_path: str):
        """Loads a FAISS index from a file."""
        print(f"Loading FAISS index from {file_path}")
        self.index = faiss.read_index(file_path)
        print(f"Index loaded. Total vectors: {self.index.ntotal}")
