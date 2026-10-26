import os

from dotenv import load_dotenv
from openai import OpenAI

from llm_zoomcamp_2026.week1.ingest import build_index, load_faq_data
from llm_zoomcamp_2026.week1.rag_helper import RAGBase


def llm_cost(
    usage,
    input_price=0.30,
    output_price=1.20,
    cache_price=0.006,
) -> float:
    if usage is None:
        return 0.0

    cached = (
        usage.prompt_tokens_details.cached_tokens or 0
        if usage.prompt_tokens_details
        else 0
    )

    uncached = usage.prompt_tokens - cached
    output = usage.completion_tokens

    return (
        uncached * input_price + cached * cache_price + output * output_price
    ) / 1_000_000


def main():
    load_dotenv()

    documents = load_faq_data()
    index = build_index(documents)

    client = OpenAI(
        api_key=os.environ["AIBOX_API_KEY"],
        base_url=os.environ["BASE_URL"],
    )

    assistant = RAGBase(
        index=index,
        llm_client=client,
    )

    answer, usage = assistant.rag("Can I join the course now?")

    print(answer)
    print(f"Estimated cost: ${llm_cost(usage):.8f}")


if __name__ == "__main__":
    main()
