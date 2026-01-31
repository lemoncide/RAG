from typing import List, Dict, Any
from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import MarkdownNodeParser, SentenceWindowNodeParser
from app.components.cleaners import clean_text

class LlamaIndexPreprocessor:
    """
    预处理器：负责清洗文档并将其切分为富含元数据的节点。
    
    流程：
    1. 清洗 (Cleaning): 使用 cleaners.py 去除水印、修复格式。
    2. 结构化切分 (Markdown): 按章节切分文档。
    3. 细粒度切分 (Sentence Window): 按句子切分并附带上下文窗口。
    """
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        
        # 初始化 LlamaIndex 解析器
        self.markdown_parser = MarkdownNodeParser()
        self.sentence_window_parser = SentenceWindowNodeParser(
            window_size=self.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            # 不指定 sentence_splitter，默认使用基于标点的自然句子切分
        )

    def process(self, documents: List[Dict[str, Any]]) -> List[BaseNode]:
        """
        接收原始文档字典列表，返回处理好的 LlamaIndex 节点列表。
        """
        print(f"正在使用 LlamaIndexPreprocessor 处理 {len(documents)} 个文档...")
        
        llama_documents = []
        
        # 1. 转换格式并清洗
        for doc_dict in documents:
            raw_text = doc_dict.get("text", "")
            source = doc_dict.get("source", "N/A")
            
            # 使用 cleaners.py 进行清洗
            # 注意：Markdown 格式包含特殊字符，clean_text 需要足够健壮
            # 这里我们假设 clean_text 主要处理通用文本噪音
            cleaned_text = clean_text(raw_text)
            
            llama_doc = Document(
                text=cleaned_text,
                metadata={"source": source}
            )
            llama_documents.append(llama_doc)
            
        if not llama_documents:
            return []

        # 2. 第一层切分：Markdown 结构化
        print("  正在进行 Markdown 结构化切分...")
        markdown_nodes = self.markdown_parser.get_nodes_from_documents(llama_documents)
        print(f"  -> 生成了 {len(markdown_nodes)} 个章节节点。")
        
        # 3. 第二层切分：句子窗口
        print("  正在进行句子窗口切分...")
        final_nodes = self.sentence_window_parser.get_nodes_from_documents(markdown_nodes)
        print(f"  -> 生成了 {len(final_nodes)} 个句子节点。")
        
        return final_nodes