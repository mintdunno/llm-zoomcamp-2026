import os

from dotenv import load_dotenv
from minsearch import VectorSearch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from llm_zoomcamp_2026.week1.ingest import load_documents
from llm_zoomcamp_2026.week2.vector_rag import RAGVector


def main() -> None:
    """Run an end-to-end RAG pipeline using vector search."""
    load_dotenv()

    # 1. Load documents and generate their embeddings.
    documents = load_documents()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["question"] + " " + doc["answer"] for doc in documents]

    X = embedder.encode(
        texts,
        batch_size=50,
        show_progress_bar=True,
    )

    # 2. Build the vector search index.
    vindex = VectorSearch(keyword_fields=["course"])
    vindex.fit(X, documents)

    # 3. Configure the LLM client.
    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"],
    )

    # 4. Initialize the vector-based RAG pipeline.
    assistant = RAGVector(
        embedder=embedder,
        index=vindex,
        client=client,
        model=os.environ["MODEL"],
    )

    # 5. Run the complete RAG pipeline.
    question = "The course has already started. Can I still join?"

    answer, usage = assistant.rag(question)

    print("Question:", question)
    print("Answer:", answer)

    if usage:
        print("Token usage:", usage.model_dump())


if __name__ == "__main__":
    main()
