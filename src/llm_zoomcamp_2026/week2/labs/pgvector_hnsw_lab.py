"""Ingest FAQ embeddings into PostgreSQL for the HNSW lab.

Week: 2 | Lesson: 8 extension | Type: Lab
"""

import hashlib
import json
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from llm_zoomcamp_2026.week1.core.ingest import load_documents

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
BATCH_SIZE = 50

UPSERT_SQL = """
INSERT INTO faq_hnsw_lab (
    document_key,
    course,
    section,
    question,
    answer,
    embedding,
    embedding_model
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (document_key)
DO UPDATE SET
    course = EXCLUDED.course,
    section = EXCLUDED.section,
    question = EXCLUDED.question,
    answer = EXCLUDED.answer,
    embedding = EXCLUDED.embedding,
    embedding_model = EXCLUDED.embedding_model;
"""


def make_document_key(document: dict) -> str:
    """Build a stable key from a FAQ's identifying fields."""
    identity = json.dumps(
        [document["question"], document["course"], document["section"]],
        ensure_ascii=False,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main() -> None:
    """Embed FAQ documents and upsert them into the lab table."""
    load_dotenv()

    # 1. Load the source FAQ data.
    documents = load_documents()
    if not documents:
        raise RuntimeError("No FAQ documents found.")

    # 2. Encode each question and answer as one vector.
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [f"{doc['question']} {doc['answer']}" for doc in documents]
    vectors = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # The database column expects exactly 384 dimensions per embedding.
    if vectors.shape != (len(documents), EMBEDDING_DIMENSIONS):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")

    # 3. Match each FAQ with its embedding and stable document key.
    records = [
        (
            make_document_key(doc),
            doc["course"],
            doc["section"],
            doc["question"],
            doc["answer"],
            vector,
            EMBEDDING_MODEL,
        )
        for doc, vector in zip(documents, vectors, strict=True)
    ]

    # 4. Register vector adapters and upsert in one transaction.
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cursor:
            cursor.executemany(UPSERT_SQL, records)
        # The connection context commits on success; errors trigger rollback.

    print(f"Successfully processed {len(records)} documents.")


if __name__ == "__main__":
    main()
