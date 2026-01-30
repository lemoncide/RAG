import os
import re
import json
import time
from typing import List, Dict, Any, Generator
from func_timeout import func_timeout, FunctionTimedOut
import pymupdf4llm

class DocumentReader:
    """
    A class to read documents (PDF, TXT, MD), partition them into structured elements,
    and perform initial filtering. 
    It unifies the output structure regardless of the input file type.
    """
    def __init__(self, input_dir: str, timeout: int = 600):
        self.input_dir = input_dir
        self.timeout = timeout
        self.failed_files = []

    def read(self) -> Generator[Dict[str, Any], None, None]:
        """
        Reads all supported files from the input directory, yielding structured documents one by one.
        Each document contains its source and a list of filtered, structured elements.
        """
        self.failed_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                content = ""
                
                if ext == ".pdf":
                    print(f"Processing PDF document: {file_path}")
                    content = self._process_with_timeout(self._read_pdf, file_path)
                elif ext in [".txt", ".md"]:
                    print(f"Processing Text document: {file_path}")
                    content = self._read_text(file_path)
                else:
                    continue
                
                if content:
                    yield {
                        "source": file,
                        "text": content
                    }
                    print(f"Successfully processed {file}.")
        
        if self.failed_files:
            self._save_failed_files()

    def _save_failed_files(self):
        """Save the list of failed files to a JSON log (Dead Letter Queue)."""
        log_path = "failed_files.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.failed_files, f, ensure_ascii=False, indent=2)
            print(f"WARNING: {len(self.failed_files)} files failed to process. Details saved to {log_path}")
        except Exception as e:
            print(f"Error saving failed files log: {e}")

    def _process_with_timeout(self, func, file_path):
        """Wrapper to execute parsing with timeout and error handling."""
        try:
            return func_timeout(self.timeout, func, args=(file_path,))
        except FunctionTimedOut:
            print(f"TIMEOUT: Processing {file_path} took longer than {self.timeout}s.")
            self.failed_files.append({"file": file_path, "error": "Timeout"})
            return ""
        except Exception as e:
            print(f"ERROR: Processing {file_path} failed: {e}")
            self.failed_files.append({"file": file_path, "error": str(e)})
            return ""

    def _read_pdf(self, file_path: str) -> str:
        """
        Convert PDF to Markdown using pymupdf4llm.
        """
        return pymupdf4llm.to_markdown(file_path, write_images=False)

    def _read_text(self, file_path: str) -> str:
        """Internal method to handle simple text/markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            return ""
