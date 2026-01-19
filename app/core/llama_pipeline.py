from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import urllib.request

# LlamaIndex components
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, ExactMatchFilter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Components for hybrid retrieval and reranking
from app.components.sparse_retriever import SparseRetriever
from app.components.reranker import Reranker

class LlamaIndexRAGPipeline:
    """
    Enhanced RAG pipeline using LlamaIndex with optional hybrid retrieval (vector + BM25)
    and reranking capabilities.
    
    Features:
    - LlamaIndex vector retrieval with metadata filtering
    - Optional BM25 sparse retrieval for keyword matching
    - Reciprocal Rank Fusion (RRF) for combining results
    - Optional reranking using cross-encoder models
    """
    def __init__(
        self, 
        persist_dir: str = "./vector_store", 
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
        Loads the index and embedding model from disk, and optionally sets up
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
                
                Settings.llm = OpenAI(
                    model=client_model_name,
                    api_base=self.llm_api_base,
                    api_key="lm-studio", # Dummy key required by client
                    temperature=0.7
                )
            except ImportError:
                print("Warning: 'llama-index-llms-openai' not found. LLM generation will be disabled.")
                Settings.llm = None
        else:
            Settings.llm = None 

        # Load the index from the FAISS vector store on disk
        print(f"Loading index from: {self.persist_dir}")
        vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=self.persist_dir
        )
        self.index = load_index_from_storage(storage_context)
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
                        "source": metadata.get("source", "N/A"),
                        "page_number": metadata.get("page_number", None)
                    }
                    documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error extracting documents from index: {e}")
            return []


    def run(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes the enhanced RAG pipeline for a given query.
        
        If hybrid retrieval is enabled:
        1. Retrieves documents from both dense (vector) and sparse (BM25) retrievers
        2. Fuses results using Reciprocal Rank Fusion (RRF)
        3. Optionally re-ranks the fused results
        
        If hybrid retrieval is disabled:
        - Only uses vector retrieval with optional metadata filtering
        
        Args:
            query: The semantic query string for vector search.
            top_k: The number of results to return.
            filters: A dictionary of metadata filters to apply, e.g., {"authors": "Paolillo"}.
        """
        print(f"Running enhanced pipeline for query: '{query}' with top_k={top_k} and filters: {filters}")
        
        # If hybrid retrieval is enabled, use the hybrid approach
        if self.enable_hybrid and self.sparse_retriever:
            return self._run_hybrid(query, top_k, filters)
        else:
            # Fall back to simple vector retrieval
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
            window = metadata.get("window", "")

            doc = {
                "text": node.node.get_content(),
                "window": window,
                "source": metadata.get("source", "N/A"),
                "page_number": metadata.get("page_number", None),
                "distance": node.score,
                "bm25_score": None,
                "rerank_score": None,
                "is_reranked": False
            }
            results.append(doc)
            
        return results
    
    def _run_hybrid(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: vector + BM25 with RRF fusion and optional reranking.
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
        
        # 3. Fuse results using RRF
        print("Step 3: Fusing results with Reciprocal Rank Fusion...")
        # Apply weights to favor vector retrieval (index 0) over BM25 (index 1)
        # Example: Vector weight = 2.0, BM25 weight = 1.0
        fused_docs = self._fuse_results([dense_results, sparse_results], weights=[2.0, 1.0])
        print(f"Fused to {len(fused_docs)} documents.")
        
        # 4. Optional reranking
        if self.reranker:
            reranker_pool_size = top_k * 4
            fused_docs_for_reranking = fused_docs[:reranker_pool_size]
            print(f"Step 4: Re-ranking top {len(fused_docs_for_reranking)} documents...")
            reranked_docs = self.reranker.rerank(query, fused_docs_for_reranking, top_n=top_k)
            print("Re-ranking complete.")
            # Add flag to indicate reranking
            for doc in reranked_docs:
                doc['is_reranked'] = True
            return reranked_docs
        
        # Return top_k from fused results if no reranker
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
            window = metadata.get("window", "")

            doc = {
                "text": node.node.get_content(),
                "window": window,
                "source": metadata.get("source", "N/A"),
                "page_number": metadata.get("page_number", None),
                "distance": node.score,
                "bm25_score": None,
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
                
                # RRF formula: weight * (1 / (k + rank + 1)), where rank is 0-based
                rrf_scores[doc_id] += weight * (1 / (k + rank + 1))
        
        # Sort documents by combined RRF score (descending)
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)
        
        # Create the final sorted list
        fused_list = [doc_map[doc_id] for doc_id in sorted_doc_ids]
        
        return fused_list

    def synthesize(self, query: str, nodes: List[Dict[str, Any]]) -> str:
        """
        Synthesizes an answer using the LLM based on the retrieved nodes.
        """
        if not Settings.llm:
            return "LLM is not configured. Please ensure LM Studio is running and 'llama-index-llms-openai' is installed."

        print(f"Synthesizing answer for query: '{query}' using {len(nodes)} context nodes...")
        
        # Construct context string from the 'window' (rich context) of the nodes
        context_str = "\n\n".join([f"Source: {n['source']}\nContent: {n['window']}" for n in nodes])
        
        # Define a simple QA prompt template
        template_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        prompt_tmpl = PromptTemplate(template_str)
        
        # Generate response
        try:
            fmt_prompt = prompt_tmpl.format(context_str=context_str, query_str=query)
            response = Settings.llm.complete(fmt_prompt)
            return str(response)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"Error generating answer: {e}"