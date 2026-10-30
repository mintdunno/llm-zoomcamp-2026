import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from llm_zoomcamp_2026.week2.pgvector_ingest import EMBEDDING_MODEL


def main() -> None:
    load_dotenv()

    question = "Can I still join the course after it started?"
    course = "llm-zoomcamp"
    top_k = 5

    # 1. Embed the user question.
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = embedder.encode(question)

    # 2. Search for nearest documents.
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)

        results = conn.execute(
            """
            SELECT
                question,
                answer,
                embedding <=> %s AS distance
            FROM documents
            WHERE course = %s
                AND embedding_model = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (
                query_vector,
                course,
                EMBEDDING_MODEL,
                query_vector,
                top_k,
            ),
        ).fetchall()

    # 3. Display top-k results.
    for rank, (doc_question, answer, distance) in enumerate(results, start=1):
        similarity = 1 - distance

        print(f"\n{rank}. {doc_question}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
