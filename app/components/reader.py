from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader

class PDFReader:
    """
    A component responsible for reading text content from PDF files.
    
    Inspired by RAGFlow's focus on quality document parsing.
    """
    def __init__(self):
        pass

    def read(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Reads a PDF file and extracts text from each page.
        
        Args:
            file_path: The path to the PDF file.
            
        Returns:
            A list of dictionaries, where each dictionary represents a page
            with its number and content.
        """
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"Reading PDF: {file_path.name}")
        reader = PdfReader(file_path)
        
        documents = []
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content: # Only add pages with text
                documents.append({
                    "page_number": i + 1,
                    "content": content,
                    "source": file_path.name
                })
        
        print(f"Extracted {len(documents)} pages with text.")
        return documents

