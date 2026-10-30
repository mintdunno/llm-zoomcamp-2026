"""Ingest FAQ embeddings into a new SQLite index.

Week: 2 | Lesson: 7 | Type: Script
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

from llm_zoomcamp_2026.week1.core.ingest import load_documents


def main() -> None:
    """Create a SQLite vector index from FAQ documents."""
    db_path = Path("faq_vectors.db")

    if db_path.exists():
        raise SystemExit("Database already exists. Skip ingestion.")

    # Load documents
    documents = load_documents()
    texts = [doc["question"] + " " + doc["answer"] for doc in documents]

    # Local embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")

    X = model.encode(
        texts,
        batch_size=50,
        show_progress_bar=True,
    )

    # Store in SQLite
    index = VectorSearchIndex(
        keyword_fields=["course"],
        mode="ivf",
        db_path=str(db_path),
    )

    index.fit(X, documents)

    print("Vector shape:", X.shape)
    print("Saved to:", db_path)

    index.close()


if __name__ == "__main__":
    main()
