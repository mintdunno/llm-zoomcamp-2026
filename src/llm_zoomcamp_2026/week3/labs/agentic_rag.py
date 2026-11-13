"""Week 3 - Lesson 6: Agentic RAG (Retrieval)."""

import os
from functools import lru_cache

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COURSE = "llm-zoomcamp"

SEARCH_SQL = """
SELECT
    document_key,
    section,
    question,
    answer,
    embedding <=> %s AS distance
FROM faq_hnsw_lab
WHERE course = %s
  AND embedding_model = %s
ORDER BY embedding <=> %s
LIMIT %s
"""


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once."""
    return SentenceTransformer(EMBEDDING_MODEL)


def search_faq(query: str, k: int = 5) -> list[dict]:
    """Search FAQ documents using pgvector cosine distance."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20.")

    # 1. Encode the query.
    query_vector = get_embedder().encode(query)

    # 2. Connect to PostgreSQL.
    with psycopg.connect(
        os.environ["DATABASE_URL"],
        connect_timeout=10,
        row_factory=dict_row,
    ) as conn:
        register_vector(conn)

        # 3. Execute vector similarity search.
        with conn.cursor() as cursor:
            cursor.execute(
                SEARCH_SQL,
                (
                    query_vector,
                    COURSE,
                    EMBEDDING_MODEL,
                    query_vector,
                    k,
                ),
                prepare=False,
            )

            return cursor.fetchall()


def main() -> None:
    load_dotenv()

    results = search_faq(
        "Can I join the course after it starts?",
        k=3,
    )

    print(f"Retrieved {len(results)} documents\n")

    for doc in results:
        print("Source:", doc["document_key"])
        print("Section:", doc["section"])
        print("Question:", doc["question"])
        print("Answer:", doc["answer"])
        print("Distance:", doc["distance"])
        print("---")


if __name__ == "__main__":
    main()
