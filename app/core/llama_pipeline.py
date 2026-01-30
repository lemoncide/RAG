from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import urllib.request

import chromadb
# LlamaIndex components
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Components for hybrid retrieval and reranking
from app.components.sparse_retriever import SparseRetriever
from app.components.reranker import Reranker

class LlamaIndexRAGPipeline:
    """
    Enhanced RAG pipeline using LlamaIndex with optional hybrid retrieval (vector + BM25)
    and reranking capabilities. Now supports ChromaDB and Dynamic RRF weights.
    
    Features:
    - LlamaIndex vector retrieval with metadata filtering (ChromaDB)
    - Optional BM25 sparse retrieval for keyword matching
    - Reciprocal Rank Fusion (RRF) with dynamic weighting based on query length
    - Optional reranking using cross-encoder models
    """
    def __init__(
        self, 
        persist_dir: str = "./chroma_db", 
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        enable_hybrid: bool = True,
        enable_reranking: bool = True,
        llm_api_base: str = "http://127.0.0.1:1234/v1", # LM Studio default
        llm_model_name: str = "local-model" # Name doesn't matter much for LM Studio
    ):
        self.persist_dir = Path(persist_dir)
        self.model_name = model_name
        self.enable_hybrid = enable_hybrid
        self.enable_reranking = enable_reranking
        self.llm_api_base = llm_api_base
        self.llm_model_name = llm_model_name
        self.sparse_retriever = None
        self.reranker = None
        self._load_resources()

    def _load_resources(self):
        """
        Loads the index (ChromaDB) and embedding model from disk, and optionally sets up
        BM25 retrieval and reranking components.
        """
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"Storage directory '{self.persist_dir}' not found. Please run 'scripts/build_index.py' first.")

        print("--- Loading LlamaIndex RAG pipeline and resources ---")
        
        # Initialize the embedding model
        print(f"Loading embedding model: {self.model_name}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        
        # Initialize LLM (LM Studio via OpenAI compatible API)
        if self.llm_api_base:
            print(f"Initializing LLM pointing to: {self.llm_api_base} (LM Studio)")
            
            # Auto-detect model name from LM Studio
            try:
                # Construct models endpoint (e.g., http://localhost:1234/v1/models)
                models_url = f"{self.llm_api_base.rstrip('/')}/models"
                with urllib.request.urlopen(models_url, timeout=3) as response:
                    data = json.loads(response.read().decode())
                    # OpenAI format: {"data": [{"id": "model-name", ...}]}
                    if "data" in data and len(data["data"]) > 0:
                        detected_model = data["data"][0]["id"]
                        print(f"Automatically detected loaded model: {detected_model}")
                        self.llm_model_name = detected_model
            except Exception as e:
                print(f"Warning: Could not auto-detect model name: {e}. Using default: {self.llm_model_name}")

            try:
                from llama_index.llms.openai import OpenAI
                
                # FIX: OpenAI 库会严格校验模型名称。
                # 当连接本地 LM Studio 时，我们需要传入一个合法的 OpenAI 模型名称（如 gpt-3.5-turbo）
                # 来绕过这个客户端校验。LM Studio 服务端会忽略这个参数，直接使用当前加载的模型。
                client_model_name = "gpt-3.5-turbo" if "localhost" in self.llm_api_base or "127.0.0.1" in self.llm_api_base else self.llm_model_name
                
                # Check for Qwen model to apply specific optimizations
                # Qwen often requires a different stop token or chat format if not handled by the server
                is_qwen = "qwen" in self.llm_model_name.lower()
                
                Settings.llm = OpenAI(
                    model=client_model_name,
                    api_base=self.llm_api_base,
                    api_key="lm-studio", # Dummy key required by client
                    temperature=0.7,
                    timeout=120.0, # Increased timeout to 120s for slow local inference
                    max_retries=1 # Reduce retries to fail faster if truly stuck
                )
            except ImportError:
                print("Warning: 'llama-index-llms-openai' not found. LLM generation will be disabled.")
                Settings.llm = None
        else:
            Settings.llm = None 

        # Load the index from ChromaDB
        print(f"Loading index from ChromaDB: {self.persist_dir}")
        db_client = chromadb.PersistentClient(path=str(self.persist_dir))
        chroma_collection = db_client.get_or_create_collection("rag_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=Settings.embed_model
        )
        print("--- LlamaIndex vector index loaded successfully ---")
        
        # Load optional components for hybrid retrieval
        if self.enable_hybrid:
            print("--- Setting up BM25 sparse retrieval ---")
            self._setup_bm25_retriever()
        
        # Load optional reranker
        if self.enable_reranking:
            print("--- Initializing reranker ---")
            try:
                self.reranker = Reranker()
            except ImportError as e:
                print(f"Warning: Reranker initialization failed: {e}")
                print("Continuing without reranker...")
                self.enable_reranking = False
                self.reranker = None
        
        print("--- LlamaIndex RAG pipeline and resources loaded successfully ---")
    
    def _setup_bm25_retriever(self):
        """
        Sets up BM25 retriever by loading documents from JSON file or extracting from LlamaIndex index.
        """
        self.sparse_retriever = SparseRetriever()
        
        # Paths
        documents_json_path = self.persist_dir / "documents.json"
        bm25_index_path = self.persist_dir / "bm25_index.pkl"
        
        documents = []
        
        # 1. Load documents (Required for both index loading and rebuilding)
        if documents_json_path.exists():
            try:
                with open(documents_json_path, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                print(f"Loaded {len(documents)} documents from JSON file.")
            except Exception as e:
                print(f"Failed to load documents from JSON: {e}")
        
        # Fallback: Extract from LlamaIndex if JSON load failed
        if not documents:
            print("Extracting documents from LlamaIndex index for BM25...")
            documents = self._extract_documents_from_index()
            
        if not documents:
            print("Warning: No documents found. BM25 retrieval will be disabled.")
            self.enable_hybrid = False
            self.sparse_retriever = None
            return

        # 2. Try to load pre-built BM25 index
        if bm25_index_path.exists():
            print(f"Loading pre-built BM25 index from {bm25_index_path}...")
            try:
                self.sparse_retriever.load_index(bm25_index_path, documents)
                return
            except Exception as e:
                print(f"Failed to load pre-built BM25 index: {e}. Rebuilding...")

        # 3. Build index from scratch (if pre-built missing or failed)
        print("Building BM25 index from scratch...")
        self.sparse_retriever.build_index(documents)
    
    def _extract_documents_from_index(self) -> List[Dict[str, Any]]:
        """
        Extracts all documents from the LlamaIndex index for BM25 indexing.
        This is done by retrieving all nodes from the index.
        """
        try:
            # Get all nodes from the index
            # Note: This requires accessing the underlying index structure
            all_nodes = self.index.storage_context.docstore.docs
            documents = []
            
            for node_id, node in all_nodes.items():
                if hasattr(node, 'get_content') and hasattr(node, 'metadata'):
                    metadata = node.metadata or {}
                    window = metadata.get("window", "")
                    
                    doc = {
                        "text": node.get_content(),
                        "window": window,
                        "source": metadata.get("source", "N/A")
                    }
                    documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error extracting documents from index: {e}")
            return []


    def run(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行增强版 RAG 管道查询。
        
        如果启用了混合检索：
        1. 从密集（向量）和稀疏（BM25）检索器检索文档
        2. 使用倒数排名融合（RRF）融合结果
        3. 可选地对融合结果进行重排序
        
        如果禁用了混合检索：
        - 仅使用带有可选元数据过滤的向量检索
        
        参数:
            query: 用于向量搜索的语义查询字符串。
            top_k: 返回的结果数量。默认为 5。
            filters: 要应用的元数据过滤器字典，例如 {"authors": "Paolillo"}。
        """
        print(f"正在运行增强管道，查询: '{query}'，top_k={top_k}，过滤器: {filters}")
        
        # 如果启用了混合检索，使用混合方法
        if self.enable_hybrid and self.sparse_retriever:
            return self._run_hybrid(query, top_k, filters)
        else:
            # 回退到简单的向量检索
            return self._run_vector_only(query, top_k, filters)
    
    def _run_vector_only(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simple vector-only retrieval with optional metadata filtering.
        """
        print("Using vector-only retrieval...")
        
        # 1. Create a standard vector store retriever
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        # 2. Construct metadata filters if provided
        if filters:
            filter_objects = [
                ExactMatchFilter(key=key, value=value) 
                for key, value in filters.items()
                if isinstance(value, (str, int, float, list))
            ]
            if filter_objects:
                retriever.filters = MetadataFilters(filters=filter_objects)
                print(f"Applying metadata filters: {filter_objects}")

        # 3. Execute the retrieval
        print("Executing vector retrieval...")
        nodes_with_scores = retriever.retrieve(query)
        print("Vector retrieval complete.")

        # 4. Format the results
        results = []
        for node in nodes_with_scores:
            metadata = node.node.metadata or {}
            
            # Since we use Recursive Character Splitting, window IS text.
            # No need to duplicate it in the output structure if they are identical.
            
            doc = {
                "text": node.node.get_content(),
                # "window": window, # Removed redundancy
                "source": metadata.get("source", "N/A"),
                "distance": node.score, # In LlamaIndex, this is cosine similarity
                "bm25_score": 0.0,
                "rerank_score": None,
                "is_reranked": False,
                "retrieval_mode": "vector_only"
            }
            results.append(doc)
            
        return results
    
    def _run_hybrid(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: vector + BM25 with RRF fusion and optional reranking.
        Now supports Dynamic RRF weights based on query length.
        """
        print("Using hybrid retrieval (vector + BM25)...")
        
        # We retrieve more documents initially to give fusion and reranking a better pool
        retrieval_top_k = top_k * 5
        
        # 1. Vector retrieval (with optional metadata filtering)
        print(f"Step 1: Retrieving top {retrieval_top_k} docs from vector retriever...")
        dense_results = self._retrieve_vector(query, retrieval_top_k, filters)
        print(f"Retrieved {len(dense_results)} vector results.")
        
        # 2. BM25 retrieval
        print(f"Step 2: Retrieving top {retrieval_top_k} docs from BM25 retriever...")
        sparse_results = self.sparse_retriever.retrieve(query, top_k=retrieval_top_k)
        print(f"Retrieved {len(sparse_results)} BM25 results.")
        
        # 3. Fuse results using RRF with Dynamic Weights
        print("Step 3: Fusing results with Reciprocal Rank Fusion...")
        
        # --- Dynamic RRF Logic ---
        # For Chinese support, we should count characters, not space-separated words.
        # Simple heuristic: if any Chinese char is present, count chars; otherwise count words.
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
        
        if is_chinese:
            query_len = len(query) # Character count for Chinese
            threshold = 6 # e.g. "Transformer" (11 chars) vs "架构" (2 chars)
        else:
            query_len = len(query.split()) # Word count for English
            threshold = 15

        if query_len < threshold:
            # Short query: Boost BM25 (Keywords)
            # Vector: 1.0, BM25: 1.5
            print(f"Short query detected ({query_len} units). Boosting BM25.")
            weights = [1.0, 1.5]
            mode = "hybrid_short_query_boost"
        else:
            # Long query: Boost Vector (Semantics)
            # Vector: 2.0, BM25: 1.0
            print(f"Long query detected ({query_len} units). Boosting Vector.")
            weights = [2.0, 1.0]
            mode = "hybrid_long_query_boost"
            
        fused_docs = self._fuse_results([dense_results, sparse_results], weights=weights)
        print(f"Fused to {len(fused_docs)} documents.")
        
        # 4. Optional reranking
        if self.reranker:
            reranker_pool_size = top_k * 4
            fused_docs_for_reranking = fused_docs[:reranker_pool_size]
            print(f"Step 4: Re-ranking top {len(fused_docs_for_reranking)} documents...")
            reranked_docs = self.reranker.rerank(query, fused_docs_for_reranking, top_n=top_k)
            print("Re-ranking complete.")
            
            # Add metadata to indicate reranking details
            for doc in reranked_docs:
                doc['is_reranked'] = True
                doc['retrieval_mode'] = mode
                # Ensure original scores are preserved if available, otherwise 0.0
                doc['distance'] = doc.get('distance', 0.0) 
                doc['bm25_score'] = doc.get('bm25_score', 0.0)
                
            return reranked_docs
        
        # Return top_k from fused results if no reranker
        for doc in fused_docs:
             doc['retrieval_mode'] = mode
             
        return fused_docs[:top_k]
    
    def _retrieve_vector(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Helper method to perform vector retrieval and format results.
        """
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        if filters:
            filter_objects = [
                ExactMatchFilter(key=key, value=value) 
                for key, value in filters.items()
                if isinstance(value, (str, int, float, list))
            ]
            if filter_objects:
                retriever.filters = MetadataFilters(filters=filter_objects)
        
        nodes_with_scores = retriever.retrieve(query)
        
        results = []
        for node in nodes_with_scores:
            metadata = node.node.metadata or {}
            
            doc = {
                "text": node.node.get_content(),
                # "window": window, # Removed
                "source": metadata.get("source", "N/A"),
                "distance": node.score, # This IS the vector score
                "bm25_score": 0.0, # Placeholder, will be filled if fused
                "rerank_score": None,
                "is_reranked": False
            }
            results.append(doc)
        
        return results
    
    def _fuse_results(self, results_lists: list, k: int = 60, weights: Optional[List[float]] = None) -> list:
        """
        Fuses multiple ranked lists of documents using Reciprocal Rank Fusion (RRF).
        
        Args:
            results_lists: A list of lists, where each inner list contains ranked documents.
            k: A constant used in the RRF formula, defaults to 60.
            weights: Optional list of weights for each result list.
        
        Returns:
            A single, re-ranked list of documents.
        """
        rrf_scores = {}
        doc_map = {}
        
        if weights is None:
            weights = [1.0] * len(results_lists)
        
        for i, results in enumerate(results_lists):
            weight = weights[i] if i < len(weights) else 1.0
            
            for rank, doc in enumerate(results):
                # Use a combination of source and text as a unique identifier
                doc_id = (doc["source"], doc["text"])
                
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                    # Create a copy to avoid modifying the original
                    doc_map[doc_id] = doc.copy()
                    
                    # Ensure score fields exist and are initialized
                    if 'distance' not in doc_map[doc_id] or doc_map[doc_id]['distance'] is None:
                        doc_map[doc_id]['distance'] = 0.0
                    if 'bm25_score' not in doc_map[doc_id] or doc_map[doc_id]['bm25_score'] is None:
                        doc_map[doc_id]['bm25_score'] = 0.0
                
                # Update specific scores based on source list to preserve original values
                # List 0 is Vector, List 1 is BM25
                if i == 0: 
                    doc_map[doc_id]['distance'] = doc.get('distance', 0.0)
                elif i == 1:
                    doc_map[doc_id]['bm25_score'] = doc.get('bm25_score', 0.0)

                # RRF formula: weight * (1 / (k + rank + 1)), where rank is 0-based
                rrf_scores[doc_id] += weight * (1 / (k + rank + 1))
        
        # Sort documents by combined RRF score (descending)
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)
        
        # Create the final sorted list
        fused_list = [doc_map[doc_id] for doc_id in sorted_doc_ids]
        
        return fused_list

    def synthesize(self, query: str, nodes: List[Dict[str, Any]]) -> str:
        """
        基于检索到的节点，使用 LLM 综合生成答案。
        """
        if not Settings.llm:
            return "未配置 LLM。请确保 LM Studio 正在运行并且已安装 'llama-index-llms-openai'。"

        print(f"正在为查询综合答案: '{query}' 使用 {len(nodes)} 个上下文节点...")
        
        # 限制总上下文长度以避免小显存 GPU OOM
        # 如果我们有 5 个 256 tokens 的节点，大约是 1280 tokens。对于 4GB 显存是安全的。
        # 构造上下文内容字符串
        context_str = "\n\n".join([f"来源: {n['source']}\n内容: {n['text']}" for n in nodes])
        
        # 定义一个简化的 QA 提示模板以节省 tokens
        template_str = (
            "上下文:\n"
            "{context_str}\n\n"
            "问题: {query_str}\n"
            "请根据上下文回答:"
        )
        prompt_tmpl = PromptTemplate(template_str)
        
        # 生成回答
        try:
            fmt_prompt = prompt_tmpl.format(context_str=context_str, query_str=query)
            response = Settings.llm.complete(fmt_prompt)
            return str(response)
        except Exception as e:
            print(f"调用 LLM 时出错: {e}")
            return f"生成答案时出错: {e}"

    def synthesize_stream(self, query: str, nodes: List[Dict[str, Any]]):
        """
        synthesize 的流式版本。生成响应块。
        """
        if not Settings.llm:
            yield "未配置 LLM。"
            return

        context_str = "\n\n".join([f"来源: {n['source']}\n内容: {n['text']}" for n in nodes])
        
        # 定义一个简化的 QA 提示模板以节省 tokens
        template_str = (
            "上下文:\n"
            "{context_str}\n\n"
            "问题: {query_str}\n"
            "请根据上下文回答:"
        )
        prompt_tmpl = PromptTemplate(template_str)
        
        try:
            fmt_prompt = prompt_tmpl.format(context_str=context_str, query_str=query)
            # 使用 stream_complete
            response_stream = Settings.llm.stream_complete(fmt_prompt)
            for chunk in response_stream:
                yield chunk.delta
        except Exception as e:
            print(f"调用 LLM 流式输出时出错: {e}")
            yield f"错误: {e}"