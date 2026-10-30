"""Demo: Run a manual agent loop with FAQ search."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from llm_zoomcamp_2026.week1.core.ingest import build_index, load_documents
from llm_zoomcamp_2026.week1.core.rag import RAG
from llm_zoomcamp_2026.week1.core.agent import agent_loop


def main() -> None:
    """Run the manual FAQ agent loop."""
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"],
    )

    model = os.environ["MODEL"]

    documents = load_documents()
    index = build_index(documents)

    instructions = """
    You are an LLM Zoomcamp teaching assistant.

    Answer only questions related to LLM Zoomcamp.

    Use the search tool to find relevant FAQ information.
    If results are insufficient, try different keywords.

    Only use facts supported by the FAQ.
    If you cannot find the answer, say you don't know.

    If the question is unrelated to the course,
    politely decline to answer.
    """.strip()

    rag = RAG(
        index=index,
        client=client,
        model=model,
    )
    question = "What's the Queen's Gambit?"

    answer = agent_loop(
        client=client,
        model=model,
        question=question,
        rag=rag,
        max_iterations=5,
        system_prompt=instructions,
    )

    print("\nFinal answer:", answer)


if __name__ == "__main__":
    main()
