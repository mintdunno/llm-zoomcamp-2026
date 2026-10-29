import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["BASE_URL"],
)

embedding_model = os.environ["EMBEDDING_MODEL"]

response = client.embeddings.create(
    model=embedding_model,
    input=["Hello world", "How are you?"],
)

X = np.array([item.embedding for item in response.data])

print("Number of embeddings:", len(X))
print("Shape:", X.shape)
print("First embedding:", X[0][:5])
