"""Demo: Run an FAQ agent with ToyAIKit.

Week: 1 | Lesson: 15 | Type: Demo
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from toyaikit.chat.interface import StdOutputInterface
from toyaikit.chat.runners import OpenAIChatCompletionsRunner
from toyaikit.llm import OpenAIChatCompletionsClient
from toyaikit.tools import Tools

from llm_zoomcamp_2026.week1.core.ingest import build_index, load_documents
from llm_zoomcamp_2026.week1.core.rag import RAG


def main() -> None:
    """Run the FAQ agent through the ToyAIKit chat interface."""
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"],
    )

    documents = load_documents()
    index = build_index(documents)

    rag = RAG(
        index=index,
        client=client,
        model=os.environ["MODEL"],
    )

    agent_tools = Tools()
    agent_tools.add_tool(rag.search)
    print(agent_tools.get_tools())

    chat_interface = StdOutputInterface()

    llm_client = OpenAIChatCompletionsClient(
        model=os.environ["MODEL"],
        client=client,
    )

    runner = OpenAIChatCompletionsRunner(
        tools=agent_tools,
        developer_prompt=(
            "You are an LLM Zoomcamp teaching assistant. "
            "Use the search tool for course-related facts. "
            "Only answer using information supported by the FAQ. "
            "If you cannot find an answer, say you don't know."
        ),
        chat_interface=chat_interface,
        llm_client=llm_client,
    )

    runner.run()


if __name__ == "__main__":
    main()
