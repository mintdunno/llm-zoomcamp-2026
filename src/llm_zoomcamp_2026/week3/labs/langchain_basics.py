"""LangChain fundamentals: models, messages, tools, and tool calling.

Week: 3 | Session: 02 | Type: Lab
"""

import os

from dotenv import load_dotenv
from langchain.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, ValidationError


@tool
def lookup_course_policy(topic: str) -> str:
    """Look up a course policy by its topic."""
    policies = {
        "late_enrollment": "Students can join after the course starts.",
        "certificate": "A certificate is available after completion.",
    }
    return policies.get(topic, "Policy not found.")


def main() -> None:
    load_dotenv()

    llm = ChatOpenAI(
        model=os.environ["MODEL"],
        api_key=SecretStr(os.environ["API_KEY"]),
        base_url=os.getenv("BASE_URL") or None,
    )

    # 1. Basic model invocation
    print("\n=== 1. MODEL INVOCATION ===")

    response = llm.invoke(
        [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Explain RAG in one sentence."),
        ]
    )

    print("Type:", type(response).__name__)
    print("Content:", response.content)
    print("Usage:", response.usage_metadata)

    # 2. Tool schema and direct execution
    print("\n=== 2. TOOL DEFINITION ===")

    print("Name:", lookup_course_policy.name)
    print("Description:", lookup_course_policy.description)
    print(
        "Schema:",
        lookup_course_policy.get_input_schema().model_json_schema(),
    )

    result = lookup_course_policy.invoke({"topic": "late_enrollment"})
    print("Direct result:", result)

    # 3. Bind tools to the model
    print("\n=== 3. TOOL CALLING ===")

    llm_with_tools = llm.bind_tools([lookup_course_policy])

    messages: list[BaseMessage] = [
        SystemMessage(content="Use tools for course-specific questions."),
        HumanMessage(
            content="What is the late enrollment policy? "
            "Use lookup_course_policy with topic='late_enrollment'."
        ),
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    print("Content:", response.content)
    print("Tool calls:", response.tool_calls)

    # 4. Manually execute tool calls
    print("\n=== 4. TOOL EXECUTION ===")

    for call in response.tool_calls:
        if call["name"] != lookup_course_policy.name:
            raise ValueError(f"Unknown tool: {call['name']}")

        tool_result = lookup_course_policy.invoke(call)

        if not isinstance(tool_result, ToolMessage):
            raise TypeError("Expected a ToolMessage")

        messages.append(tool_result)

        print("Tool result:", tool_result.content)
        print("Tool call ID:", tool_result.tool_call_id)

    # 5. Send tool results back to the model
    print("\n=== 5. FINAL ANSWER ===")

    if response.tool_calls:
        final_response = llm_with_tools.invoke(messages)
        print("Final answer:", final_response.content)
        print("Additional tool calls:", final_response.tool_calls)
    else:
        print("Model answered without calling a tool.")

    # 6. Input validation
    print("\n=== 6. INPUT VALIDATION ===")

    try:
        lookup_course_policy.invoke({})
    except ValidationError as exc:
        print("Validation failed:", exc)

    # 7. Model response without requiring a tool
    print("\n=== 7. NO TOOL CALL ===")

    response = llm_with_tools.invoke("Say hello in one sentence without using tools.")

    print("Content:", response.content)
    print("Tool calls:", response.tool_calls)


if __name__ == "__main__":
    main()
