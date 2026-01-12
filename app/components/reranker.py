from typing import List, Dict, Any

class Reranker:
    """
    A component for re-ranking a list of retrieved documents based on a more
    powerful cross-encoder model.
    """
    def __init__(self, model_name: str = 'BAAI/bge-reranker-base'):
        """
        Initializes the Reranker with a cross-encoder model.
        
        Args:
            model_name: The name of the cross-encoder model to use from the
                        Hugging Face Hub.
        """
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "FlagEmbedding library not found. Please install it with: "
                "pip install FlagEmbedding"
            )
            
        print(f"Initializing Reranker with model: {model_name}")
        # The 'use_fp16=True' is recommended for faster inference if a GPU is available
        self.model = FlagReranker(model_name, use_fp16=True) 
        print("Reranker initialized.")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of documents for a given query.

        Args:
            query: The original user query.
            documents: The list of documents retrieved by the initial retriever.
            top_n: The final number of documents to return.

        Returns:
            A sorted list of the top_n documents with added 'rerank_score'.
        """
        if not documents:
            return []

        # The model expects a list of [query, document_text] pairs
        sentence_pairs = [[query, doc['text']] for doc in documents]
        
        print(f"Re-ranking {len(documents)} documents...")
        # The compute_score method returns a list of scores
        scores = self.model.compute_score(sentence_pairs)
        
        # Add the rerank_score to each document
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score
            
        # Sort the documents by the new rerank_score in descending order
        documents.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        
        print(f"Re-ranking complete. Returning top {top_n} documents.")
        
        return documents[:top_n]
