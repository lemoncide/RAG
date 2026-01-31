import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class LlamaIndexRetriever:
    """
    基于 LlamaIndex 和 ChromaDB 的向量检索器。
    负责加载持久化索引并执行语义检索。
    """
    def __init__(self, persist_dir: str, model_name: str):
        self.persist_dir = Path(persist_dir)
        self.model_name = model_name
        self.index = None
        self._load_index()

    def _load_index(self):
        """加载 ChromaDB 索引和 Embedding 模型"""
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"未找到存储目录 '{self.persist_dir}'")

        print(f"正在初始化向量检索器 (Model: {self.model_name})...")
        
        # 1. 设置 Embedding 模型
        # 注意：这里设置全局 Settings，虽然不是最优雅的，但在 LlamaIndex 中是常规做法
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        
        # 2. 连接 ChromaDB
        db_client = chromadb.PersistentClient(path=str(self.persist_dir))
        chroma_collection = db_client.get_or_create_collection("rag_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # 3. 加载索引
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=Settings.embed_model
        )
        print("向量检索器加载成功。")

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行向量检索。
        """
        print(f"正在执行向量检索: '{query}' (top_k={top_k})")
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        # 应用元数据过滤器
        if filters:
            filter_objects = [
                ExactMatchFilter(key=key, value=value) 
                for key, value in filters.items()
                if isinstance(value, (str, int, float, list))
            ]
            if filter_objects:
                retriever.filters = MetadataFilters(filters=filter_objects)
                print(f"  应用过滤器: {filter_objects}")

        nodes_with_scores = retriever.retrieve(query)
        
        # 格式化结果
        results = []
        for node in nodes_with_scores:
            metadata = node.node.metadata or {}
            doc = {
                "text": node.node.get_content(),
                "source": metadata.get("source", "N/A"),
                "window": metadata.get("window", ""), # 如果有窗口信息也带上
                "distance": node.score, # LlamaIndex 中这是相似度分数
                "retrieval_mode": "vector"
            }
            results.append(doc)
            
        print(f"  检索到 {len(results)} 个向量结果。")
        return results
