class RAGPipeline:
    """
    Implements the core RAG logic, inspired by Haystack's Pipeline design.
    This class orchestrates the components (dense/sparse retrievers, reranker)
    to answer a query based on the provided documents.
    """
    def __init__(self, reader, preprocessor, dense_retriever, sparse_retriever, reranker=None):
        self.reader = reader
        self.preprocessor = preprocessor
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker

    def run(self, query: str, top_k: int = 5):
        """
        Executes the hybrid RAG pipeline for a given query.
        
        1. Retrieves documents from both dense and sparse retrievers.
        2. Fuses the results using Reciprocal Rank Fusion (RRF).
        3. (Optional) Re-ranks the fused results for better relevance.
        
        Returns:
            A list of final document chunks.
        """
        print(f"Running hybrid pipeline for query: '{query}' with top_k={top_k}")

        # We retrieve more documents initially to give fusion and reranking a better pool to work with.
        retrieval_top_k = top_k * 5

        # 1. Retrieve from both retrievers
        print(f"Step 1: Retrieving top {retrieval_top_k} docs from dense and sparse retrievers...")
        dense_results = self.dense_retriever.retrieve(query, top_k=retrieval_top_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=retrieval_top_k)
        print(f"Retrieved {len(dense_results)} dense results and {len(sparse_results)} sparse results.")

        # 2. Fuse the results using RRF
        print("Step 2: Fusing results with Reciprocal Rank Fusion...")
        fused_docs = self._fuse_results([dense_results, sparse_results])
        
        # After fusion, we might have a lot of documents. We take a pool for the reranker.
        reranker_pool_size = top_k * 4
        fused_docs_for_reranking = fused_docs[:reranker_pool_size]
        print(f"Fused to {len(fused_docs)} documents. Taking top {len(fused_docs_for_reranking)} for reranking.")

        # 3. Re-rank if a reranker is available
        if self.reranker:
            print(f"Step 3: Re-ranking fused documents...")
            reranked_docs = self.reranker.rerank(query, fused_docs_for_reranking, top_n=top_k)
            print("Re-ranking complete.")
            # Add a flag to indicate which documents were reranked
            for doc in reranked_docs:
                doc['is_reranked'] = True
            return reranked_docs
        
        # If no reranker, return the top_k from the fused results
        return fused_docs[:top_k]

    def _fuse_results(self, results_lists: list, k: int = 60) -> list:
        """
        Fuses multiple ranked lists of documents using Reciprocal Rank Fusion (RRF).

        Args:
            results_lists: A list of lists, where each inner list contains ranked documents.
            k: A constant used in the RRF formula, defaults to 60.

        Returns:
            A single, re-ranked list of documents.
        """
        # Use a dictionary to store the RRF scores for each unique document.
        # We need a unique identifier. Using 'text' is a simple approach.
        rrf_scores = {}
        doc_map = {} # To store the full document object, avoiding duplicates

        for results in results_lists:
            for rank, doc in enumerate(results):
                # Use a combination of source and text as a more robust ID
                doc_id = (doc["source"], doc["text"]) 
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                    doc_map[doc_id] = doc

                rrf_scores[doc_id] += 1 / (k + rank + 1) # rank is 0-based

        # Sort documents based on their combined RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)
        
        # Create the final sorted list of documents
        fused_list = [doc_map[doc_id] for doc_id in sorted_doc_ids]
        
        return fused_list


