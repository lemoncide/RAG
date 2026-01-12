import nltk
from typing import List, Dict, Any
from app.components.cleaners import clean_text, is_formula, is_caption, is_garbled

class SentenceWindowPreprocessor:
    """
    Takes structured documents from the Reader, cleans and tokenizes them into
    sentences, applies fine-grained filters, and creates sentence-window nodes
    while preserving page number metadata.
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
            source_file = doc["source"]
            
            # 1. Collect all sentences from the document's elements with their page numbers
            doc_sentences = []
            for element in doc["elements"]:
                element_text = clean_text(element["text"])
                if not element_text:
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

            # 2. Apply fine-grained filters to the collected sentences
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
                
                # Add the cleaned sentence back with its metadata
                s_info["text"] = s_text 
                filtered_sentences.append(s_info)

            # 3. Create sentence window nodes from the filtered sentences
            for i, sentence_info in enumerate(filtered_sentences):
                # Determine the window boundaries
                start_index = max(0, i - self.window_size)
                end_index = min(len(filtered_sentences), i + self.window_size + 1)
                
                # Create the window of text (text only)
                window_text = " ".join([s["text"] for s in filtered_sentences[start_index:end_index]])
                
                node = {
                    "text": sentence_info["text"],
                    "window": window_text,
                    "source": source_file,
                    "page_number": sentence_info["page_number"] # Page number is now preserved!
                }
                all_nodes.append(node)

        print(f"Created {len(all_nodes)} sentence nodes after final filtering.")
        return all_nodes