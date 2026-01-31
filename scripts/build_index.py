import os
from pathlib import Path
import shutil
import json
import sys

# Add project root to the Python path to allow absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.reader import DocumentReader
from app.components.sparse_retriever import SparseRetriever

# LlamaIndex components
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from app.components.preprocessor import LlamaIndexPreprocessor

def main():
    """
    使用 ChromaDB 构建 LlamaIndex 向量索引的主脚本。
    
    流程简化：
    1. Reader: 读取 PDF 为 Markdown 字典。
    2. Preprocessor: 清洗 -> Markdown切分 -> 句子窗口切分。
    3. Indexing: 存入 ChromaDB。
    """
    print("--- 开始构建 LlamaIndex 过程 (ChromaDB + 分层切分) ---")
    
    # 1. 设置路径
    data_dir = Path("./data/embodia/pdf")
    persist_dir = Path("./chroma_db")
    
    # 如果需要，清理之前的存储（可选，Chroma 可以追加）
    if persist_dir.exists():
        print(f"正在删除现有的存储目录: {persist_dir}")
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"警告: 无法删除 {persist_dir}: {e}")
        
    # 2. 初始化组件
    reader = DocumentReader(input_dir=data_dir, timeout=600)
    preprocessor = LlamaIndexPreprocessor(window_size=3)
    
    # 使用相同的 sentence-transformer 模型以保持一致性
    embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-mpnet-base-v2")
    Settings.embed_model = embed_model
    Settings.llm = None # 索引期间我们不使用 LLM
    
    # 3. 读取文档并创建节点
    print("--- 正在读取和处理文档 ---")
    
    llama_nodes = []
    bm25_documents = []
    
    # 初始化 ChromaDB 客户端
    print("正在初始化 ChromaDB 客户端...")
    db_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = db_client.get_or_create_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 一次性读取所有文档（如果内存足够），或者分批处理
    # 这里为了简单，我们收集所有 raw docs 然后交给 preprocessor 处理
    raw_docs = []
    doc_generator = reader.read()
    
    for doc_dict in doc_generator:
        raw_docs.append(doc_dict)
        
    if not raw_docs:
        print("未读取到任何文档。中止。")
        return

    # 4. 使用 Preprocessor 处理
    # 这步完成了 Cleaning, Markdown Splitting, Sentence Window Splitting
    llama_nodes = preprocessor.process(raw_docs)

    if not llama_nodes:
        print("处理后未创建任何节点。中止。")
        return
        
    print(f"成功转换了 {len(llama_nodes)} 个节点 (分层切分)。")

    # 5. 准备 BM25 数据
    print("正在准备 BM25 数据...")
    for node in llama_nodes:
        window_text = node.metadata.get("window", node.get_content())
        bm25_documents.append({
            "text": node.get_content(), # 索引时用单句
            "window": window_text,      # BM25 也存窗口，方便检索
            "source": node.metadata.get("source", "N/A"),
            "page_number": None 
        })

    # 6. 构建并持久化索引
    print("\n--- 正在构建并持久化 LlamaIndex VectorStoreIndex (ChromaDB) ---")
    
    index = VectorStoreIndex(
        nodes=llama_nodes,
        storage_context=storage_context
    )
    
    print(f"索引已构建并持久化到: {persist_dir}")
        
    # 7. 为 BM25 检索器保存 documents.json
    print("\n--- 正在为 BM25 检索器保存 documents.json ---")
    documents_json_path = persist_dir / "documents.json"
    
    with open(documents_json_path, "w", encoding="utf-8") as f:
        json.dump(bm25_documents, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(bm25_documents)} 个文档到 {documents_json_path}")
    
    # 8. 构建并持久化 BM25 索引
    print("\n--- 正在构建并持久化 BM25 索引 ---")
    sparse_retriever = SparseRetriever()
    sparse_retriever.build_index(bm25_documents)
    bm25_index_path = persist_dir / "bm25_index.pkl"
    sparse_retriever.save_index(bm25_index_path)

    print("\n--- 索引构建完成！---")
    print(f"LlamaIndex (ChromaDB 存储) 已持久化到: {persist_dir}")

if __name__ == "__main__":
    main()
