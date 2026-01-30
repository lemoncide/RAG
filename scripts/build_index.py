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
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

def main():
    """
    使用 ChromaDB 和递归字符切分构建 LlamaIndex 向量索引的主脚本。
    
    1.  指向 'data' 目录。
    2.  使用 DocumentReader 读取所有支持的文件（PDF 转换为 Markdown）。
    3.  使用 SentenceSplitter（递归字符切分）将它们处理成节点。
    4.  设置 ChromaDB VectorStore 和 HuggingFace 嵌入模型。
    5.  构建并将索引持久化到磁盘（ChromaDB 自动处理持久化）。
    """
    print("--- 开始构建 LlamaIndex 过程 (ChromaDB + 递归切分) ---")
    
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
    
    # 使用 SentenceSplitter 进行递归字符切分
    # 这尊重 chunk_size 并避免来自长 markdown 部分的过大节点
    # 将切分大小减小到 256，以提高本地推理速度
    node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=50)
    
    # 使用相同的 sentence-transformer 模型以保持一致性
    embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-mpnet-base-v2")
    Settings.embed_model = embed_model
    Settings.llm = None # 索引期间我们不使用 LLM
    Settings.chunk_size = 256 # 设置合理的块大小
    
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
    
    BATCH_SIZE = 50 # 每次处理 50 个文件
    current_batch_docs = []
    
    doc_generator = reader.read()
    
    for doc_dict in doc_generator:
        # 将字典转换为 LlamaIndex Document 对象
        # 注意：reader 现在以 Markdown 格式返回全文
        doc = Document(
            text=doc_dict["text"],
            metadata={"source": doc_dict["source"]}
        )
        current_batch_docs.append(doc)
        
        if len(current_batch_docs) >= BATCH_SIZE:
            print(f"正在处理 {len(current_batch_docs)} 个文档的批次...")
            
            # 使用 SentenceSplitter 解析节点
            nodes = node_parser.get_nodes_from_documents(current_batch_docs)
            llama_nodes.extend(nodes)
            
            # 准备 BM25 数据
            for node in nodes:
                bm25_documents.append({
                    "text": node.get_content(),
                    "window": node.get_content(), # 节点现在是切分后的块
                    "source": node.metadata.get("source", "N/A"),
                    "page_number": None 
                })
            
            current_batch_docs = [] # 清空批次

    # 处理剩余的文档
    if current_batch_docs:
        print(f"正在处理最后的 {len(current_batch_docs)} 个文档批次...")
        nodes = node_parser.get_nodes_from_documents(current_batch_docs)
        llama_nodes.extend(nodes)
        for node in nodes:
            bm25_documents.append({
                "text": node.get_content(),
                "window": node.get_content(),
                "source": node.metadata.get("source", "N/A"),
                "page_number": None
            })

    if not llama_nodes:
        print("处理后未创建任何节点。中止。")
        return
        
    print(f"成功使用 SentenceSplitter 转换了 {len(llama_nodes)} 个节点。")

    # 6. 构建并持久化索引
    print("\n--- 正在构建并持久化 LlamaIndex VectorStoreIndex (ChromaDB) ---")
    
    index = VectorStoreIndex(
        nodes=llama_nodes,
        storage_context=storage_context
    )
    
    # ChromaDB 自动持久化，但我们为了其他存储调用 persist 以防万一
    # index.storage_context.persist(persist_dir=persist_dir) 
    print(f"索引已构建并持久化到: {persist_dir}")
        
    # 7. 为 BM25 检索器保存 documents.json
    print("\n--- 正在为 BM25 检索器保存 documents.json ---")
    # 为了方便，将 documents.json 存储在同一目录中
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
