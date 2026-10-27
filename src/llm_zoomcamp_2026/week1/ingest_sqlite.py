from sqlitesearch import TextSearchIndex

from llm_zoomcamp_2026.week1.ingest import load_faq_data


def main():
    documents = load_faq_data()

    docs = [doc for doc in documents if doc["course"] == "data-engineering-zoomcamp"]

    print(f"Loaded {len(docs)} documents")

    index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path="faq.db",
    )

    try:
        for doc in docs:
            index.add(doc)
    finally:
        index.close()

    print("Ingestion completed!")


if __name__ == "__main__":
    main()
