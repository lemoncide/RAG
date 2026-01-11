import os
from pathlib import Path

# Add project root to the Python path to allow absolute imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.reader import PDFReader
from app.components.preprocessor import SemanticSplitter
from app.components.retriever import DenseRetriever

def main():
    """
    Main script to build the vector index.
    
    1. Point to the 'data' directory.
    2. Read all PDF files.
    3. Split them into chunks.
    4. Embed the chunks and build a FAISS index.
    5. Save the index and the corresponding documents for later use.
    """
    print("--- Starting Index Building Process ---")
    
    # 1. Setup paths
    data_dir = Path("./data/embodia/pdf")
    index_path = "faiss_index.bin"
    documents_path = "documents.json" # To store the text chunks
    
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"Error: The '{data_dir}' directory is empty or does not exist.")
        print("Please add your PDF files to it and run again.")
        return

    # 2. Initialize components
    reader = PDFReader()
    splitter = SemanticSplitter(chunk_size=500, chunk_overlap=100)
    retriever = DenseRetriever(model_name='all-MiniLM-L6-v2')
    
    # 3. Read, split, and process all PDFs in the data directory
    all_docs = []
    for pdf_path in data_dir.glob("*.pdf"):
        docs = reader.read(pdf_path)
        all_docs.extend(docs)
        
    if not all_docs:
        print("No text could be extracted from the PDF files found.")
        return
        
    chunks = splitter.split(all_docs)
    
    # 4. Build the index
    retriever.build_index(chunks)
    
    # 5. Save the index and documents
    retriever.save_index(index_path)
    
    # We also need to save the chunks themselves to retrieve their text later
    import json
    with open(documents_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
        
    print("\n--- Index Building Complete! ---")
    print(f"FAISS index saved to: {index_path}")
    print(f"Document chunks saved to: {documents_path}")
    print("You can now run the FastAPI application.")


if __name__ == "__main__":
    main()
