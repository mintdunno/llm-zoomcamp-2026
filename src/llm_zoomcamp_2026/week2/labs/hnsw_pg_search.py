import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


def main() -> None:
    load_dotenv()

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    question = "Can I still join the course after it started?"
    course = "llm-zoomcamp"
    top_k = 5

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    v_question = embedder.encode(question)

    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as conn:
        register_vector(conn)
        with conn.cursor() as cursor:
            # Prefer HNSW for this experiment.
            cursor.execute("SET LOCAL enable_seqscan = off")

            # Control the HNSW search candidate list.
            cursor.execute("SET LOCAL hnsw.ef_search = 100")
            cursor.execute(
                """
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT
                    question,
                    answer,
                    embedding <=> %s AS distance
                FROM faq_hnsw_lab
                WHERE course = %s
                AND embedding_model = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    v_question,
                    course,
                    EMBEDDING_MODEL,
                    v_question,
                    top_k,
                ),
            )

            plan = cursor.fetchall()

    import re

    # Display the execution plan without the full embedding values.
    for (line,) in plan:
        clean_line = re.sub(
            r"'\[[^']+\]'::vector",
            "<query_vector>",
            line,
        )
        print(clean_line)

    # 5. Display results.
    # for rank, (doc_question, answer, distance) in enumerate(results, start=1):
    #     similarity = 1 - distance

    #     print(f"\n{rank}. {doc_question}")
    #     print(f"Similarity: {similarity:.4f}")
    #     print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
