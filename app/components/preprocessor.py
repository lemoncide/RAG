import nltk
from typing import List, Dict, Any
from collections import defaultdict
from app.components.cleaners import extract_metadata_and_clean, is_formula, is_caption, is_garbled, is_reference_section

class SentenceWindowPreprocessor:
    """
    Takes structured documents from the Reader, cleans them, extracts metadata,
    tokenizes them into sentences, applies fine-grained filters, and creates
    sentence-window nodes with rich metadata.
    """
    def __init__(self, window_size: int = 2, min_sentence_length_words: int = 7):
        self.window_size = window_size
        self.min_sentence_length_words = min_sentence_length_words
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes structured documents and processes them into sentence-window nodes.
        """
        print("Processing documents with metadata-aware SentenceWindowPreprocessor...")
        all_nodes = []
        for doc in documents:
            doc_metadata = {"source": doc["source"]}
            
            # 1. First pass: Extract metadata from the first page, which often contains authors/title.
            if doc["elements"]:
                # Combine first few elements to get a good chunk of the header
                header_text = " ".join([e["text"] for e in doc["elements"][:3]])
                _, header_meta = extract_metadata_and_clean(header_text)
                doc_metadata.update(header_meta)
            
            # 2. Collect all sentences from the document's elements with their page numbers
            doc_sentences = []
            for element in doc["elements"]:
                # extract_metadata_and_clean handles basic cleaning + metadata extraction
                element_text, element_meta = extract_metadata_and_clean(element["text"])
                
                # If a DOI is found anywhere, add it to the doc-level metadata
                if 'doi' in element_meta:
                    doc_metadata['doi'] = element_meta['doi']
                
                if not element_text or is_reference_section(element_text):
                    continue
                
                try:
                    sentences = nltk.sent_tokenize(element_text)
                    for sentence in sentences:
                        doc_sentences.append({
                            "text": sentence,
                            "page_number": element["page_number"]
                        })
                except Exception as e:
                    print(f"NLTK sentence tokenization failed for an element. Skipping. Error: {e}")
                    continue

            # 3. Apply fine-grained filters to the collected sentences
            filtered_sentences = []
            for s_info in doc_sentences:
                s_text = s_info["text"].strip()
                if (
                    len(s_text.split()) <= self.min_sentence_length_words or
                    is_formula(s_text) or
                    is_caption(s_text) or
                    is_garbled(s_text)
                ):
                    continue
                
                s_info["text"] = s_text 
                filtered_sentences.append(s_info)

            # 4. Create sentence window nodes from the filtered sentences
            for i, sentence_info in enumerate(filtered_sentences):
                start_index = max(0, i - self.window_size)
                end_index = min(len(filtered_sentences), i + self.window_size + 1)
                
                window_text = " ".join([s["text"] for s in filtered_sentences[start_index:end_index]])
                
                # Consolidate all metadata into a single dictionary
                node_metadata = {
                    "page_number": sentence_info["page_number"],
                    **doc_metadata # Add doc-level metadata (source, authors, doi)
                }

                node = {
                    "text": sentence_info["text"], # The sentence to be embedded
                    "window": window_text,        # The context window
                    "metadata": node_metadata     # All other info
                }
                all_nodes.append(node)

        print(f"Created {len(all_nodes)} sentence nodes after final filtering.")
        return all_nodes