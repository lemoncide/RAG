import os
import re
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from app.components.cleaners import is_garbled

class PDFReader:
    """
    A class to read PDF documents, partition them into structured elements,
    and perform initial filtering.
    """
    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def read(self) -> List[Dict[str, Any]]:
        """
        Reads all PDFs from the input directory, returning a list of structured documents.
        Each document contains its source and a list of filtered, structured elements.
        """
        structured_docs = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if not file.endswith(".pdf"):
                    continue
                
                file_path = os.path.join(root, file)
                print(f"Processing document: {file_path}")
                try:
                    # Use hi_res strategy for better layout detection
                    elements = partition_pdf(
                        filename=file_path,
                        languages=["chi_sim", "eng"],
                        strategy="hi_res" 
                    )
                    
                    excluded_categories = ["Header", "Footer", "Formula", "FigureCaption", "Table", "Image"]
                    
                    filtered_elements = []
                    for element in elements:
                        # Stop processing if a "References" or "Bibliography" title is found
                        if element.category == "Title" and re.search(r'^\s*(references|bibliography)\b', str(element), re.IGNORECASE):
                            print(f"References section found in {file}. Stopping processing for this document.")
                            break
                        
                        if element.category in excluded_categories:
                            continue

                        # Per user feedback, add pre-filtering in the reader
                        element_text = str(element).strip()
                        if len(element_text) < 20 and element.category == "Text": # Filter short, isolated text snippets
                            continue
                        if is_garbled(element_text): # Pre-filter obviously garbled text
                            continue

                        filtered_elements.append({
                            "text": element_text,
                            "page_number": getattr(element.metadata, 'page_number', None),
                            "category": element.category
                        })

                    structured_docs.append({
                        "source": file,
                        "elements": filtered_elements
                    })
                    print(f"Successfully processed {file} with {len(filtered_elements)} elements.")

                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    
        return structured_docs
