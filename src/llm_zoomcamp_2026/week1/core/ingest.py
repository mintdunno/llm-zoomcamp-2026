"""Load FAQ documents and build a keyword index.

Week: 1 | Lesson: 4–5, 8 | Type: Core
"""

import requests
from minsearch import Index

FAQ_URL = "https://datatalks.club/faq"


def load_documents() -> list[dict]:
    """Download and combine FAQ documents from all courses."""
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
    """Build a searchable index from FAQ documents."""
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
    )

    index.fit(documents)
    return index
