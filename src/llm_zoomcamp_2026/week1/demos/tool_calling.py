"""Demo: Execute a requested search tool and return its result to the model.

Week: 1 | Lesson: 13 | Type: Demo
"""

import os
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm_zoomcamp_2026.week1.core.agent import (
    SEARCH_TOOL,
    chat_with_tools,
    create_messages,
    execute_tool_call,
)
from llm_zoomcamp_2026.week1.core.ingest import build_index, load_documents
from llm_zoomcamp_2026.week1.core.rag import RAG


def main() -> None:
    """Demonstrate a search tool call and the model's follow-up response."""
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"],
    )

    model = os.environ["MODEL"]

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
