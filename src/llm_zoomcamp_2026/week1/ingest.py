import requests
from minsearch import Index

FAQ_URL = "https://datatalks.club/faq"


def load_documents() -> list[dict]:
    response = requests.get(
        f"{FAQ_URL}/json/courses.json",
        timeout=30,
    )
    response.raise_for_status()

    documents = []

    for course in response.json():
        response = requests.get(
            f"{FAQ_URL}{course['path']}",
            timeout=30,
        )
        response.raise_for_status()

        documents.extend(response.json())

    return documents


def build_index(documents: list[dict]) -> Index:
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
    )

    index.fit(documents)
    return index
