from openai import OpenAI

SEARCH_TOOL = {
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


def request_tool_call(
    client: OpenAI,
    model: str,
    question: str,
):
    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a course assistant. "
                    "Use the search tool for course-specific facts."
                ),
            },
            {"role": "user", "content": question},
        ],
        tools=[SEARCH_TOOL],  # type: ignore
        tool_choice="auto",
    )  # type: ignore
