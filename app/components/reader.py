import os
import re
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from app.components.cleaners import is_garbled, is_low_alphanum_ratio

class DocumentReader:
    """
    A class to read documents (PDF, TXT, MD), partition them into structured elements,
    and perform initial filtering. 
    It unifies the output structure regardless of the input file type.
    """
    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def read(self) -> List[Dict[str, Any]]:
        """
        Reads all supported files from the input directory, returning a list of structured documents.
        Each document contains its source and a list of filtered, structured elements.
        """
        structured_docs = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                elements = []
                
                if ext == ".pdf":
                    print(f"Processing PDF document: {file_path}")
                    elements = self._read_pdf(file_path)
                elif ext in [".txt", ".md"]:
                    print(f"Processing Text document: {file_path}")
                    elements = self._read_text(file_path)
                else:
                    continue
                
                if elements:
                    structured_docs.append({
                        "source": file,
                        "elements": elements
                    })
                    print(f"Successfully processed {file} with {len(elements)} elements.")
                    
        return structured_docs

    def _read_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Internal method to handle PDF parsing using unstructured."""
        filtered_elements = []
        try:
            # Use hi_res strategy for better layout detection
            elements = partition_pdf(
                filename=file_path,
                languages=["chi_sim", "eng"],
                strategy="hi_res" 
            )
            
            # Updated based on notebook analysis: Exclude UncategorizedText, but keep Table
            # UncategorizedText is handled conditionally below to rescue bullet points
            excluded_categories = ["Header", "Footer", "Formula", "FigureCaption", "Image"]
            
            for element in elements:
                # Stop processing if a "References" or "Bibliography" title is found
                if element.category == "Title" and re.search(r'^\s*(references|bibliography)\b', str(element), re.IGNORECASE):
                    print(f"References section found in {os.path.basename(file_path)}. Stopping processing for this document.")
                    break
                
                if element.category in excluded_categories:
                    continue
                
                element_text = str(element).strip()

                # Special handling for UncategorizedText: only keep if it looks like a list item
                if element.category == "UncategorizedText":
                    if not element_text.startswith(("•", "-", "*")):
                        continue

                # Apply cleaning rules from data_cleaning_notebook
                
                # Rule 1: Length filter (applied to all categories except Table, as tables can be dense but short)
                if len(element_text) < 10 and element.category != "Table":
                    continue
                
                # Rule 2: Alphanumeric ratio filter (filters out "......" or purely symbolic lines)
                # Rule 3: Garbled text filter
                if is_low_alphanum_ratio(element_text, threshold=0.6) or is_garbled(element_text):
                    continue

                filtered_elements.append({
                    "text": element_text,
                    "page_number": getattr(element.metadata, 'page_number', None),
                    "category": element.category
                })
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            
        return filtered_elements

    def _read_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Internal method to handle simple text/markdown files."""
        filtered_elements = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newlines to simulate paragraphs/elements
            paragraphs = content.split('\n\n')
            
            for p in paragraphs:
                text = p.strip()
                if not text:
                    continue
                
                # Apply similar basic filters as PDF to keep logic consistent
                if len(text) < 20: 
                    continue
                if is_garbled(text):
                    continue
                    
                filtered_elements.append({
                    "text": text,
                    "page_number": None, # Text files don't have page numbers
                    "category": "Text"
                })
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            
        return filtered_elements
