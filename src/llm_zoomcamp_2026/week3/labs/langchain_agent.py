import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from llm_zoomcamp_2026.week1.core.ingest import (
    build_index,
    load_documents,
)


def main() -> None:
    load_dotenv()

    index = build_index(load_documents())

    @tool
    def search_faq(query: str) -> str:
        """Search the LLM Zoomcamp FAQ for course information."""
        docs = index.search(
            query,
            filter_dict={"course": "llm-zoomcamp"},
            boost_dict={"question": 2.0, "answer": 0.5},
            num_results=3,
        )
        return json.dumps(docs, ensure_ascii=False)

    model = ChatOpenAI(
        model=os.environ["MODEL"],
        api_key=SecretStr(os.environ["API_KEY"]),
        base_url=os.getenv("BASE_URL") or None,
    )

    agent = create_agent(
        model=model,
        tools=[search_faq],
        system_prompt=(
            "You are an LLM Zoomcamp assistant. "
            "Use the FAQ tool for course-specific questions. "
            "Only answer with supported evidence. "
            "If evidence is insufficient, say you don't know."
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Can I join the course after it starts?",
                }
            ]
        }
    )

    for message in result["messages"]:
        print(f"\n{type(message).__name__}")
        print("Content:", str(message.content)[:400])

        if getattr(message, "tool_calls", None):
            print("Tool calls:", message.tool_calls)


if __name__ == "__main__":
    main()
