"""Compare exact and HNSW approximate vector search in PostgreSQL using pgvector."""

import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


def main() -> None:
    """Run vector search benchmarks comparing exact search with HNSW at different ef_search settings."""
    load_dotenv()

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    questions = [
        "Can I join the course after it has already started?",
        "How do I register for the LLM Zoomcamp course?",
        "Is the course completely free?",
        "Can I get a certificate after completing the course?",
        "What should I do if I miss a homework deadline?",
        "How do I submit my homework assignments?",
        "Can I submit homework after the deadline?",
        "How are homework assignments graded?",
        "Do I need a powerful GPU to complete the course?",
        "What Python knowledge is required before starting?",
        "Do I need prior experience with machine learning?",
        "How do I set up the development environment?",
        "Can I use Windows instead of Linux?",
        "How do I run PostgreSQL using Docker?",
        "What should I do if Docker is not working?",
        "Can I use a different LLM provider for the homework?",
        "How can I reduce the cost of using an LLM API?",
        "Where can I find the course materials and recordings?",
        "How do I ask questions or get help from the community?",
        "Can I complete the course at my own pace?",
    ]

    course = "llm-zoomcamp"
    top_k = 5

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    v_questions = embedder.encode(questions)

    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as conn:
        register_vector(conn)
        recalls = {10: [], 20: [], 40: [], 100: []}
        for idx, v_question in enumerate(v_questions):
            with conn.cursor() as cursor:
                search_sql = """
                    SELECT id
                    FROM faq_hnsw_lab
                    WHERE course = %s
                    AND embedding_model = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """

                params = (course, EMBEDDING_MODEL, v_question, top_k)

                # 1. Exact Search: disable HNSW index scans.
                # Exact Search
                cursor.execute("SET LOCAL enable_seqscan = on")
                cursor.execute("SET LOCAL enable_indexscan = off")
                cursor.execute(search_sql, params, prepare=False)

                exact_ids = [row[0] for row in cursor.fetchall()]
                print(f"\nQuestion {idx + 1}: {questions[idx]}")
                print("Exact Top-5:", exact_ids)

                # 2. HNSW Search: enable index scans.
                cursor.execute("SET LOCAL enable_indexscan = on")
                cursor.execute("SET LOCAL enable_seqscan = off")
                cursor.execute(search_sql, params, prepare=False)

                for ef in (10, 20, 40, 100):
                    cursor.execute(
                        "SELECT set_config('hnsw.ef_search', %s, true)",
                        (str(ef),),
                    )
                    cursor.execute(
                        "EXPLAIN (ANALYZE, BUFFERS) " + search_sql,
                        params,
                    )
                    plan_lines = [line for (line,) in cursor.fetchall()]
                    if not any("Index Scan" in line for line in plan_lines):
                        print(
                            f"WARNING: ef_search={ef} did NOT use an index scan; "
                            "recall is not a valid HNSW measurement."
                        )
                    if ef == 10 and idx in (0, 1, 2):
                        print(f"\nExecution Plan — Question {idx + 1}")

                        for line in plan_lines:
                            if any(
                                keyword in line
                                for keyword in (
                                    "Scan",
                                    "Sort",
                                    "Limit",
                                    "Rows Removed",
                                    "Execution Time",
                                )
                            ):
                                print(line)

                    cursor.execute(search_sql, params)
                    hnsw_ids = [row[0] for row in cursor.fetchall()]

                    recall = recall_at_k(exact_ids, hnsw_ids)
                    recalls[ef].append(recall)

                    print(f"ef_search={ef}")
                    print("HNSW IDs:", hnsw_ids)
                    print(f"Recall@{top_k}: {recall:.0%}")

        print("\n--- Average Recall@5 ---")
        for ef, values in recalls.items():
            avg_recall = sum(values) / len(values)
            print(f"ef_search={ef}: {avg_recall:.2%}")


def recall_at_k(
    exact_ids: list[int],
    approx_ids: list[int],
) -> float:
    """Calculate Recall@K by comparing approximate result IDs against ground truth exact IDs."""
    if not exact_ids:
        return 0.0

    matches = set(exact_ids) & set(approx_ids)
    return len(matches) / len(exact_ids)


if __name__ == "__main__":
    main()
