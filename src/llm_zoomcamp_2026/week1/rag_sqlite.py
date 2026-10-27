import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sqlitesearch import TextSearchIndex

from llm_zoomcamp_2026.week1.rag_helper import RAGBase


def main():
    load_dotenv()

    db_path = Path("faq.db")

    if not db_path.exists():
        raise FileNotFoundError(db_path)

    index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path=str(db_path),
    )

    try:
        client = OpenAI(
            api_key=os.environ["AIBOX_API_KEY"],
            base_url=os.environ["BASE_URL"],
        )

        assistant = RAGBase(
            index=index,
            llm_client=client,
            course="data-engineering-zoomcamp",
            model="ds/deepseek-flash",
        )

        answer, usage = assistant.rag("Can I join the course now?")

        print(answer)
        print(usage)

    finally:
        index.close()


if __name__ == "__main__":
    main()
