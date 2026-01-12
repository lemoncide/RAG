import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

#将文字变成向量，并负责快速找到最相关的内容。
class DenseRetriever:
    """
    A component for creating and querying a dense vector index using FAISS.
    
    This class handles embedding creation and similarity search, getting inspiration
    from 'rag-from-scratch' for understanding the core mechanics.
    """
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
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
        self.index = faiss.IndexFlatL2(embedding_dim) # 创建基于欧式距离（L2）的索引
        self.index.add(np.array(embeddings, dtype=np.float32))
        print(f"FAISS index built. Total vectors in index: {self.index.ntotal}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Searches the index for the most relevant document chunks for a given query.
        """
        if self.index is None:
            raise RuntimeError("Index is not built. Please call 'build_index' first.")
            
        print(f"Retrieving top {top_k} documents for query: '{query}'")
        query_embedding = self.model.encode([query]) # 用户的提问用同一个模型转化成query_embedding
        
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)

        results = []
        if indices is not None and len(indices) > 0:
            # 过滤掉无效的索引（-1是FAISS在找不到邻居时返回的值）
            # 并创建一个包含文档和其距离的字典列表
            for i, dist in zip(indices[0], distances[0]):
                if i != -1:
                    doc = self.documents[i].copy()  # 使用 .copy() 避免修改原始文档
                    doc['distance'] = float(dist)
                    results.append(doc)
        
        # FAISS返回的结果已经按距离排序，所以我们不需要重新排序
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
