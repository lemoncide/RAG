import pickle
from typing import List, Dict, Any

class SparseRetriever:
    """
    A component for creating and querying a sparse keyword-based index using BM25.
    """
    def __init__(self):
        self.index = None
        self.documents = []
        self._tokenizer = self._get_tokenizer()
        print("SparseRetriever initialized.")

    def _get_tokenizer(self):
        try:
            import jieba
            # Add custom words if necessary, e.g. from a domain-specific dictionary
            # jieba.add_word('特定词')
            print("Using 'jieba' for BM25 tokenization.")
            return lambda text: jieba.lcut_for_search(text)
        except ImportError:
            print("Warning: 'jieba' library not found. Falling back to simple whitespace tokenizer. For better Chinese tokenization, please install it: pip install jieba")
            return lambda text: text.split()

    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Creates a BM25 index from a list of document chunks.

        Args:
            documents: A list of dictionaries, each with a "text" key.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 library not found. Please install it with: pip install rank_bm25")

        self.documents = documents
        texts = [doc["text"] for doc in self.documents]
        
        print(f"Tokenizing {len(texts)} text chunks for BM25...")
        tokenized_corpus = [self._tokenizer(text) for text in texts]
        
        print("Building BM25 index...")
        self.index = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the index for the most relevant document chunks for a given query.
        Returns a list of documents, each with an added 'bm25_score'.
        """
        if self.index is None:
            raise RuntimeError("BM25 index is not built. Please call 'build_index' first.")
            
        print(f"Retrieving top {top_k} documents for query '{query}' using BM25...")
        tokenized_query = self._tokenizer(query)
        
        # get_top_n returns documents, but we need scores and indices
        doc_scores = self.index.get_scores(tokenized_query)
        
        # Get the indices of the top_k documents
        # We need to handle cases where top_k is larger than the number of documents
        num_docs = len(self.documents)
        if top_k > num_docs:
            top_k = num_docs

        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            # We must return a copy to avoid modifying the original document list
            doc = self.documents[i].copy()
            doc['bm25_score'] = doc_scores[i]
            results.append(doc)
            
        return results

    def save_index(self, file_path: str):
        """Saves the BM25 index to a file using pickle."""
        print(f"Saving BM25 index to {file_path}")
        with open(file_path, 'wb') as f:
            # We only need to save the index object itself for rank_bm25
            pickle.dump(self.index, f)

    def load_index(self, file_path: str, documents: List[Dict[str, Any]]):
        """
        Loads a BM25 index from a file and associates it with documents.
        
        Args:
            file_path: Path to the pickled BM25 index file.
            documents: The list of document dictionaries, which are not saved
                       with the BM25 index to avoid data duplication.
        """
        print(f"Loading BM25 index from {file_path}")
        with open(file_path, 'rb') as f:
            self.index = pickle.load(f)
        
        # The documents are not stored in the index file, so they must be loaded
        # separately and associated here.
        self.documents = documents
        print(f"BM25 index loaded and associated with {len(self.documents)} documents.")

