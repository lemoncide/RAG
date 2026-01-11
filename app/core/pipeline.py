class RAGPipeline:
    """
    Implements the core RAG logic, inspired by Haystack's Pipeline design.
    This class orchestrates the components (reader, preprocessor, retriever)
    to answer a query based on the provided documents.
    """
    def __init__(self, reader, preprocessor, retriever):
        self.reader = reader
        self.preprocessor = preprocessor
        self.retriever = retriever

    def run(self, query: str, top_k: int = 5):
        """
        Executes the RAG pipeline for a given query.
        
        1. Pre-processes the query.
        2. Retrieves relevant document chunks.
        3. (Future) Generates an answer based on the chunks.
        
        Returns:
            A list of retrieved document chunks.
        """
        print(f"Running pipeline for query: '{query}' with top_k={top_k}")
        
        # In a real scenario, you might preprocess the query too
        # query_embedding = self.retriever.embed_text(query)
        
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        
        print(f"Retrieved {len(retrieved_docs)} documents.")
        
        # TODO: Add a generator component to produce a final answer
        
        return retrieved_docs

