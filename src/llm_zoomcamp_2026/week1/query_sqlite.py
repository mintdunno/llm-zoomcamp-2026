from sqlitesearch import TextSearchIndex


def main():
    index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path="faq.db",
    )

    try:
        print(f"Total documents: {index.count()}")

        results = index.search(
            "Can I join the course now?",
            filter_dict={"course": "data-engineering-zoomcamp"},
            num_results=5,
        )

        for doc in results:
            print(doc["question"])
    finally:
        index.close()


if __name__ == "__main__":
    main()
