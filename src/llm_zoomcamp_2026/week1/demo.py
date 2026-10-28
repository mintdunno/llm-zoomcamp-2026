import os
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage

from llm_zoomcamp_2026.week1.agent import (
    SEARCH_TOOL,
    chat_with_tools,
    create_messages,
    execute_tool_call,
)
from llm_zoomcamp_2026.week1.ingest import (
    build_index,
    load_documents,
)
from llm_zoomcamp_2026.week1.rag import RAG


def estimate_cost(
    usage: CompletionUsage,
    input_price: float,
    output_price: float,
    cached_input_price: float | None = None,
) -> float:
    """Estimate cost from token usage and prices per million tokens."""
    cached_tokens = 0

    if usage.prompt_tokens_details:
        cached_tokens = usage.prompt_tokens_details.cached_tokens or 0

    input_tokens = usage.prompt_tokens - cached_tokens
    output_tokens = usage.completion_tokens

    if cached_input_price is None:
        cached_input_price = input_price

    return (
        input_tokens * input_price
        + cached_tokens * cached_input_price
        + output_tokens * output_price
    ) / 1_000_000


def main() -> None:
    """Run the RAG example and display its answer and token usage."""
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["AIBOX_API_KEY"],
        base_url=os.environ["BASE_URL"],
    )

    documents = load_documents()
    index = build_index(documents)

    assistant = RAG(
        index=index,
        client=client,
        model=os.environ["MODEL"],
    )

    question = "Can I join the course now?"

    answer, usage = assistant.rag(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    if usage is not None:
        print(f"Input tokens: {usage.prompt_tokens}")
        print(f"Output tokens: {usage.completion_tokens}")

        input_price = os.getenv("INPUT_PRICE_PER_M")
        output_price = os.getenv("OUTPUT_PRICE_PER_M")

        if input_price and output_price:
            cached_price = os.getenv("CACHED_INPUT_PRICE_PER_M")

            cost = estimate_cost(
                usage=usage,
                input_price=float(input_price),
                output_price=float(output_price),
                cached_input_price=(float(cached_price) if cached_price else None),
            )

            print(f"Estimated cost: ${cost:.8f}")


def test_function_call() -> None:
    """Demonstrate a search tool call and the model's follow-up response."""
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["AIBOX_API_KEY"],
        base_url=os.environ["BASE_URL"],
    )

    model = os.environ["MODEL"]

    messages = create_messages("I just discovered LLM Zoomcamp. Can I join now?")

    messages = create_messages("I just discovered LLM Zoomcamp. Can I join now?")

    response = chat_with_tools(
        client=client,
        model=model,
        messages=messages,
        tools=[SEARCH_TOOL],
    )

    print("This is first messages:", messages, end="\n\n")

    message = response.choices[0].message
    tool_calls = message.tool_calls or []

    messages.append(
        cast(
            ChatCompletionMessageParam,
            message.model_dump(exclude_none=True),
        )
    )

    print("This is 2nd messages:", messages, end="\n\n")

    if not tool_calls:
        print("Model did not request any tools.")
        return

    # Prepare the real search engine
    documents = load_documents()
    index = build_index(documents)

    rag = RAG(
        index=index,
        client=client,
        model=model,
    )

    # Execute every tool call
    for call in tool_calls:
        if call.type != "function":
            raise ValueError(f"Unsupported tool type: {call.type}")

        tool_message = execute_tool_call(call, rag)
        messages.append(tool_message)

        print("Tool:", call.function.name)
        print("Arguments:", call.function.arguments, end="\n\n")

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    print(
        "This is response from llm:",
        final_response.choices[0].message.model_dump(),
    )


if __name__ == "__main__":
    main()
