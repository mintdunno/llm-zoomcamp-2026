import json
from typing import cast

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


SYSTEM_PROMPT = (
    "You are a course assistant. Use the search tool for course-specific facts."
)


def create_messages(
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[ChatCompletionMessageParam]:
    """Create the initial system and user messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def chat_with_tools(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionFunctionToolParam],
) -> ChatCompletion:
    """Send the conversation and let the model request tools."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )


def execute_tool_call(
    call: ChatCompletionMessageFunctionToolCall,
    rag: RAG,
) -> ChatCompletionToolMessageParam:
    """Validate and execute a search tool call for the model."""
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

    # Check rag quality by viewing logging
    print(f"Query: {args['query']}")
    print(f"Documents found: {len(results)}")

    for i, doc in enumerate(results[:3], start=1):
        print(f"{i}. {doc['question']}")

    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(results, ensure_ascii=False),
    }


def agent_loop(
    client: OpenAI,
    model: str,
    question: str,
    rag: RAG,
    max_iterations: int = 5,
) -> str:
    """Run the conversation and tools until an answer or the iteration limit."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    messages = create_messages(question)

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")

        response = chat_with_tools(
            client=client,
            model=model,
            messages=messages,
            tools=[SEARCH_TOOL],
        )

        message = response.choices[0].message

        messages.append(
            cast(
                ChatCompletionMessageParam,
                message.model_dump(exclude_none=True),
            )
        )

        tool_calls = message.tool_calls or []

        if not tool_calls:
            return message.content or ""

        if iteration == max_iterations - 1:
            break

        for call in tool_calls:
            if call.type != "function":
                raise ValueError(f"Unsupported tool type: {call.type}")

            print(f"Tool: {call.function.name}")
            print(f"Arguments: {call.function.arguments}")

            tool_result = execute_tool_call(call, rag)
            messages.append(tool_result)

    raise RuntimeError("Agent exceeded max iterations")
