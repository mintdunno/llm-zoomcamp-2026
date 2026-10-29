import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sqlitesearch import VectorSearchIndex

from llm_zoomcamp_2026.week1.ingest import load_documents

load_dotenv()

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["BASE_URL"],
)

documents = load_documents()

texts = [doc["question"] + " " + doc["answer"] for doc in documents]

db_path = Path("faq_vectors2.db")

if db_path.exists():
    raise SystemExit("Database already exists. Skip ingestion.")

batch_size = 20
vectors = []

for i in range(0, len(texts), batch_size):
    batch = texts[i : i + batch_size]

    response = client.embeddings.create(
        model=os.environ["EMBEDDING_MODEL"],
        input=batch,
    )

    batch_vectors = [item.embedding for item in response.data]

    vectors.extend(batch_vectors)

vectors = np.array(vectors, dtype=np.float32)

index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db",
)

index.fit(vectors, documents)

print("Vector shape:", vectors.shape)
print("Saved to faq_vectors.db")

index.close()
