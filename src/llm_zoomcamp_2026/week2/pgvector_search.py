import os
import time

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

    # 1. Load embedding model.
    print("Loading embedding model...", flush=True)
    start = time.perf_counter()

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Model loaded: {time.perf_counter() - start:.4f}s")

    # 2. Encode the query.
    print("Encoding query...", flush=True)
    start = time.perf_counter()

    query_vector = embedder.encode(question)

    print(f"Query encoded: {time.perf_counter() - start:.4f}s")

    # 3. Connect to PostgreSQL.
    print("Connecting to PostgreSQL...", flush=True)
    start = time.perf_counter()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)

        print(f"DB connected: {time.perf_counter() - start:.4f}s")

        # 4. Execute vector search.
        print("Searching vectors...", flush=True)
        start = time.perf_counter()

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

        print(f"Vector search: {time.perf_counter() - start:.4f}s")

    # 5. Display results.
    for rank, (doc_question, answer, distance) in enumerate(results, start=1):
        similarity = 1 - distance

        print(f"\n{rank}. {doc_question}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
