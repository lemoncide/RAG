import os
from pathlib import Path

# Add project root to the Python path to allow absolute imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.reader import PDFReader
from app.components.preprocessor import SentenceWindowPreprocessor
from app.components.retriever import DenseRetriever

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
    
    # 1. Setup paths (respecting user's custom path)
    data_dir = Path("./data/embodia/pdf")
    index_path = "faiss_index.bin"
    documents_path = "documents.json" # To store the sentence nodes
    
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"Error: The '{data_dir}' directory is empty or does not exist.")
        print("Please add your PDF files to it and run again.")
        return

    # 2. Initialize components
    # The reader now takes the input directory directly
    reader = PDFReader(input_dir=data_dir)
    # Using the new SentenceWindowPreprocessor
    preprocessor = SentenceWindowPreprocessor(window_size=2)
    # Use the recommended multilingual model to support cross-lingual search
    retriever = DenseRetriever(model_name='paraphrase-multilingual-mpnet-base-v2')
    
    # 3. Read all PDFs and get structured documents (including metadata)
    # The reader now returns a list of structured docs, which the preprocessor can handle directly.
    structured_docs = reader.read()
        
    if not structured_docs:
        print("No processable documents found or text could be extracted.")
        return
        
    # Process structured documents into sentence nodes, preserving metadata
    nodes = preprocessor.process(structured_docs)
    
    # 4. Build the index from the 'text' field of the nodes (which contains the sentence)
    retriever.build_index(nodes)
    
    # 5. Save the index and the nodes
    retriever.save_index(index_path)
    
    # We also need to save the nodes themselves to retrieve their text/window later
    import json
    with open(documents_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=4)
        
    print("\n--- Index Building Complete! ---")
    print(f"FAISS index saved to: {index_path}")
    print(f"Sentence nodes saved to: {documents_path}")
    print("You can now re-run the build script and then start the FastAPI application.")


if __name__ == "__main__":
    main()
