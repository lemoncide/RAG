# RAG From Scratch Project

This project aims to build a comprehensive Retrieval-Augmented Generation (RAG) system from the ground up

## Project Plan

| Module                 | Reference Project | Key Task                                                              |
| ---------------------- | ----------------- | --------------------------------------------------------------------- |
| **Document Parsing**   | RAGFlow           | Implement robust layout analysis for PDF/Table data extraction.       |
| **Architecture**       | Haystack          | Design a modular `Pipeline` class for a clean and scalable codebase.  |
| **Advanced Retrieval** | LlamaIndex        | Implement Sentence Window Retrieval or a similar hierarchical index.  |
| **Core Logic**         | rag-from-scratch  | Understand and implement embedding space calculations and similarity. |
| **Intent Routing**     | NeMo-Guardrails   | Create a "router" to decide when to engage the RAG system.            |
| **Evaluation**         | RAGAS             | Quantitatively benchmark our RAG against baselines like RAGFlow.      |

## Project Structure

```
.
├── app/                  # FastAPI application
│   ├── components/       # Core RAG components (reader, retriever, etc.)
│   ├── core/             # Pipeline logic
│   └── main.py           # API entrypoint
├── data/                 # Source documents for indexing
├── scripts/              # Helper scripts (e.g., index building)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```
