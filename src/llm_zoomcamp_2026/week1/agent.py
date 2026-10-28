import json

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)

from llm_zoomcamp_2026.week1.rag import RAG

SEARCH_TOOL: ChatCompletionFunctionToolParam = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the LLM Zoomcamp FAQ database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the FAQ.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


# Request tool call
def request_tool_call(
    client: OpenAI,
    model: str,
    question: str,
) -> tuple[ChatCompletion, list[ChatCompletionMessageParam]]:
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a course assistant. "
                "Use the search tool for course-specific facts."
            ),
        },
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[SEARCH_TOOL],
        tool_choice="auto",
    )

    return response, messages


# Execute search tool call
def execute_tool_call(
    call: ChatCompletionMessageFunctionToolCall,
    rag: RAG,
) -> ChatCompletionToolMessageParam:
    if call.function.name != "search":
        raise ValueError(f"Unknown tool: {call.function.name}")

    args = json.loads(call.function.arguments)

    if (
        not isinstance(args, dict)
        or set(args) != {"query"}
        or not isinstance(args["query"], str)
        or not args["query"].strip()
    ):
        raise ValueError("Invalid search arguments")

    results = rag.search(args["query"])

    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(results, ensure_ascii=False),
    }
