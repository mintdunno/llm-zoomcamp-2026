"""Reusable vector-based RAG pipeline.

Week: 2 | Lesson: 6–7 | Type: Core
"""

from typing import Any

from sentence_transformers import SentenceTransformer

from llm_zoomcamp_2026.week1.core.rag import RAG


class RAGVector(RAG):
    """RAG pipeline that uses vector similarity for document retrieval.

    Inherits prompt construction and LLM generation from RAG.
    Overrides only the search method to use semantic search
    instead of keyword-based search.
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        **kwargs: Any,
    ) -> None:
        """Initialize the RAG pipeline and its embedding model.

        Args:
            embedder: Model used to convert queries into vectors.
            **kwargs: Arguments forwarded to the parent RAG class.
        """
        # Initialize inherited attributes: index, client, model, course.
        super().__init__(**kwargs)

        self.embedder = embedder

    def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> list[dict]:
        """Retrieve the most relevant documents using vector search.

        Args:
            query: User question in natural language.
            num_results: Maximum number of documents to retrieve.

        Returns:
            List of matching FAQ documents for the configured course.
        """
        # Convert the query into an embedding vector.
        query_vector = self.embedder.encode(query)

        # Search precomputed document vectors and filter by course.
        return self.index.search(
            query_vector,
            filter_dict={"course": self.course},
            num_results=num_results,
        )
