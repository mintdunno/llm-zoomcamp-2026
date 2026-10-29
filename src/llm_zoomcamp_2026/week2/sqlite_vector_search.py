import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

from llm_zoomcamp_2026.week2.vector_rag import RAGVector

load_dotenv()

db_path = Path("faq_vectors.db")

if not db_path.exists():
    raise SystemExit("Database not found. Run ingestion first.")

# 1. Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Open existing SQLite database
vindex = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path=str(db_path),
)


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
    course="llm-zoomcamp",
)

# 5. Test vector search
question = "The course has already started. Can I still join?"

print("Documents in DB:", vindex.count())

results = assistant.search(question)

for i, doc in enumerate(results, start=1):
    print(f"{i}. {doc['question']}")
    print(f"Course: {doc['course']}")

# 6. Run complete RAG pipeline
try:
    answer, usage = assistant.rag(question)

    print("Question:", question)
    print("Answer:", answer)

    if usage:
        print("Token usage:", usage.model_dump())
finally:
    vindex.close()
