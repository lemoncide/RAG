from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import urllib.request

from llama_index.core import Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer

# Components
from app.components.retriever import LlamaIndexRetriever
from app.components.sparse_retriever import SparseRetriever
from app.components.reranker import Reranker
from app.components.fusion import reciprocal_rank_fusion

class LlamaIndexRAGPipeline:
    """
    RAG 管道编排器 (Orchestrator)。
    负责组装各个组件：Retrievers -> Fusion -> Reranker -> Optimization -> Generation。
    """
    def __init__(
        self, 
        persist_dir: str = "./chroma_db", 
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        enable_hybrid: bool = True,
        enable_reranking: bool = True,
        llm_api_base: str = "http://127.0.0.1:1234/v1",
        llm_model_name: str = "local-model"
    ):
        self.persist_dir = Path(persist_dir)
        self.model_name = model_name
        self.enable_hybrid = enable_hybrid
        self.enable_reranking = enable_reranking
        self.llm_api_base = llm_api_base
        self.llm_model_name = llm_model_name
        
        # 组件占位符
        self.vector_retriever = None
        self.sparse_retriever = None
        self.reranker = None
        self.optimizer = None
        
        self._load_components()

    def _load_components(self):
        """初始化所有子组件"""
        print("--- 正在初始化 RAG 管道组件 ---")
        
        # 1. 向量检索器 (LlamaIndex + ChromaDB)
        self.vector_retriever = LlamaIndexRetriever(
            persist_dir=str(self.persist_dir),
            model_name=self.model_name
        )
        
        # 2. 句子嵌入优化器 (复用 Embedding Model)
        # 注意：LlamaIndexRetriever 内部已经设置了 Settings.embed_model
        print("正在初始化句子嵌入优化器...")
        self.optimizer = SentenceEmbeddingOptimizer(
            embed_model=Settings.embed_model,
            percentile_cutoff=0.5 
        )
        
        # 3. 稀疏检索器 (BM25)
        if self.enable_hybrid:
            print("--- 正在设置 BM25 稀疏检索 ---")
            self._setup_bm25()
            
        # 4. 重排序器 (BGE)
        if self.enable_reranking:
            print("--- 正在初始化重排序器 ---")
            try:
                self.reranker = Reranker()
            except ImportError as e:
                print(f"警告: 重排序器初始化失败: {e}。将禁用重排序。")
                self.enable_reranking = False
        
        # 5. LLM 设置
        self._setup_llm()
        
        print("--- RAG 管道组件初始化完成 ---")

    def _setup_bm25(self):
        """加载或构建 BM25 索引"""
        self.sparse_retriever = SparseRetriever()
        documents_json_path = self.persist_dir / "documents.json"
        bm25_index_path = self.persist_dir / "bm25_index.pkl"
        
        documents = []
        if documents_json_path.exists():
            try:
                with open(documents_json_path, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                print(f"从 JSON 加载了 {len(documents)} 个文档用于 BM25。")
            except Exception as e:
                print(f"加载 BM25 文档失败: {e}")
        
        if not documents:
            print("警告: 未找到文档，禁用 BM25。")
            self.enable_hybrid = False
            return

        if bm25_index_path.exists():
            try:
                self.sparse_retriever.load_index(bm25_index_path, documents)
                return
            except Exception:
                pass # 加载失败则重建

        print("正在构建 BM25 索引...")
        self.sparse_retriever.build_index(documents)

    def _setup_llm(self):
        """配置 LLM 连接"""
        if not self.llm_api_base:
            Settings.llm = None
            return
            
        print(f"正在初始化 LLM 连接: {self.llm_api_base}")
        try:
            # 尝试自动检测模型名称
            models_url = f"{self.llm_api_base.rstrip('/')}/models"
            with urllib.request.urlopen(models_url, timeout=3) as response:
                data = json.loads(response.read().decode())
                if "data" in data and len(data["data"]) > 0:
                    self.llm_model_name = data["data"][0]["id"]
                    print(f"检测到模型: {self.llm_model_name}")
        except Exception:
            pass

        try:
            from llama_index.llms.openai import OpenAI
            client_model_name = "gpt-3.5-turbo" if "localhost" in self.llm_api_base or "127.0.0.1" in self.llm_api_base else self.llm_model_name
            
            Settings.llm = OpenAI(
                model=client_model_name,
                api_base=self.llm_api_base,
                api_key="lm-studio",
                temperature=0.7,
                timeout=120.0,
                max_retries=1
            )
        except ImportError:
            print("警告: 缺少 llama-index-llms-openai，无法生成回答。")
            Settings.llm = None

    def run(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行完整的 RAG 流程。
        """
        print(f"正在运行 RAG 管道，查询: '{query}'，top_k={top_k}")
        
        # 1. 检索阶段
        if self.enable_hybrid and self.sparse_retriever:
            # 混合检索：向量 + BM25
            retrieval_top_k = top_k * 5 # 扩大检索池
            
            # A. 向量检索
            dense_results = self.vector_retriever.retrieve(query, retrieval_top_k, filters)
            
            # B. BM25 检索
            sparse_results = self.sparse_retriever.retrieve(query, top_k=retrieval_top_k)
            
            # C. 动态权重融合
            is_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
            query_len = len(query) if is_chinese else len(query.split())
            
            if query_len < (6 if is_chinese else 15):
                print("短查询 -> 提升 BM25 权重")
                weights = [1.0, 1.5]
                mode = "hybrid_short"
            else:
                print("长查询 -> 提升向量权重")
                weights = [2.0, 1.0]
                mode = "hybrid_long"
                
            fused_docs = reciprocal_rank_fusion([dense_results, sparse_results], weights=weights)
            
            # D. 重排序 (可选)
            if self.reranker:
                reranker_pool = fused_docs[:top_k * 4]
                final_results = self.reranker.rerank(query, reranker_pool, top_n=top_k)
                for doc in final_results:
                    doc['is_reranked'] = True
            else:
                final_results = fused_docs[:top_k]
                
        else:
            # 纯向量检索
            final_results = self.vector_retriever.retrieve(query, top_k, filters)
            mode = "vector_only"

        # 标记检索模式
        for doc in final_results:
            if 'retrieval_mode' not in doc:
                doc['retrieval_mode'] = mode
                
        return final_results

    def synthesize(self, query: str, nodes: List[Dict[str, Any]]) -> str:
        """生成回答（非流式）"""
        if not Settings.llm:
            return "未配置 LLM。请确保 LM Studio 正在运行并且已安装 'llama-index-llms-openai'。"

        print(f"正在为查询综合答案: '{query}' 使用 {len(nodes)} 个上下文节点...")
        fmt_prompt = self._prepare_prompt(query, nodes)
        
        try:
            response = Settings.llm.complete(fmt_prompt)
            return str(response)
        except Exception as e:
            print(f"调用 LLM 时出错: {e}")
            return f"生成答案时出错: {e}"

    def synthesize_stream(self, query: str, nodes: List[Dict[str, Any]]):
        """生成回答（流式）"""
        if not Settings.llm:
            yield "未配置 LLM。"
            return

        fmt_prompt = self._prepare_prompt(query, nodes)
        
        try:
            response_stream = Settings.llm.stream_complete(fmt_prompt)
            for chunk in response_stream:
                yield chunk.delta
        except Exception as e:
            print(f"调用 LLM 流式输出时出错: {e}")
            yield f"错误: {e}"

    def _prepare_prompt(self, query: str, nodes: List[Dict[str, Any]]) -> str:
        """
        生成的通用逻辑：应用优化器 -> 构造 Prompt
        """
        # 1. 准备上下文：应用句子嵌入优化
        context_texts = []
        for n in nodes:
            original_text = n.get("window", n.get("text"))
            # 构造临时节点用于优化器
            from llama_index.core.schema import TextNode
            temp_node = TextNode(text=original_text)
            
            try:
                optimized_nodes = self.optimizer.postprocess_nodes([temp_node], query_str=query)
                if optimized_nodes:
                    opt_text = optimized_nodes[0].get_content()
                    # 仅当压缩有效时使用
                    text_to_use = opt_text if opt_text and len(opt_text) < len(original_text) else original_text
                else:
                    text_to_use = original_text
            except Exception:
                text_to_use = original_text
                
            context_texts.append(f"来源: {n['source']}\n内容: {text_to_use}")

        context_str = "\n\n".join(context_texts)
        
        template_str = (
            "上下文:\n{context_str}\n\n"
            "问题: {query_str}\n"
            "请根据上下文回答:"
        )
        prompt_tmpl = PromptTemplate(template_str)
        return prompt_tmpl.format(context_str=context_str, query_str=query)