import os
from pathlib import Path

# Add project root to the Python path to allow absolute imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.reader import PDFReader
from app.components.preprocessor import SentenceWindowPreprocessor
from app.components.retriever import DenseRetriever
from app.components.sparse_retriever import SparseRetriever

def main():
    """
    Main script to build the vector index using the Sentence Window strategy.
    
    1. Point to the 'data' directory.
    2. Read all PDF files.
    3. Process them into sentence nodes with context windows.
    4. Embed the sentences and build a FAISS index.
    5. Save the index and the corresponding nodes for later use.
    """
    print("--- Starting Index Building Process (Sentence Window) ---")
    
    # 1. Setup paths
    data_dir = Path("./data/embodia/pdf")
    faiss_index_path = "faiss_index.bin"
    bm25_index_path = "bm25_index.pkl"
    documents_path = "documents.json"
    
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"Error: The '{data_dir}' directory is empty or does not exist.")
        print("Please add your PDF files to it and run again.")
        return

    # 2. Initialize components
    reader = PDFReader(input_dir=data_dir)
    preprocessor = SentenceWindowPreprocessor(window_size=2)
    dense_retriever = DenseRetriever(model_name='paraphrase-multilingual-mpnet-base-v2')
    sparse_retriever = SparseRetriever()
    
    # 3. Read documents and create sentence nodes
    structured_docs = reader.read()
    if not structured_docs:
        print("No processable documents found.")
        return
    nodes = preprocessor.process(structured_docs)
    
    # 4. Build and save indices
    print("\n--- Building Dense Index (FAISS) ---")
    dense_retriever.build_index(nodes)
    dense_retriever.save_index(faiss_index_path)
    
    print("\n--- Building Sparse Index (BM25) ---")
    sparse_retriever.build_index(nodes)
    sparse_retriever.save_index(bm25_index_path)
    
    # 5. Save the nodes themselves for retrieval
    import json
    with open(documents_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=4)
        
    print("\n--- Index Building Complete! ---")
    print(f"FAISS index saved to: {faiss_index_path}")
    print(f"BM25 index saved to: {bm25_index_path}")
    print(f"Sentence nodes saved to: {documents_path}")
    print("You can now start the FastAPI application.")


if __name__ == "__main__":
    main()
