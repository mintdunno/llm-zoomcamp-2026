import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.store.postgres import PostgresStore
from pydantic import SecretStr


@tool
def save_language(language: str, runtime: ToolRuntime) -> str:
    """Save the user's preferred programming language."""
    assert runtime.store is not None

    runtime.store.put(
        ("users", "alex"),
        "language",
        {"value": language},
    )
    return "Language saved."


@tool
def get_language(runtime: ToolRuntime) -> str:
    """Retrieve the user's saved programming language."""
    assert runtime.store is not None

    memory = runtime.store.get(
        ("users", "alex"),
        "language",
    )

    return str(memory.value) if memory else "No language saved."


def main() -> None:
    load_dotenv()

    model = ChatOpenAI(
        model=os.environ["MODEL"],
        api_key=SecretStr(os.environ["API_KEY"]),
        base_url=os.getenv("BASE_URL") or None,
    )

    with PostgresStore.from_conn_string(  # pyright: ignore[reportGeneralTypeIssues]
        os.environ["DATABASE_URL"]
    ) as store:
        store.setup()

        agent = create_agent(
            model=model,
            tools=[save_language, get_language],
            store=store,
            system_prompt=(
                "You are a helpful assistant. "
                "Use save_language when the user asks you to remember "
                "their programming language. "
                "Use get_language when they ask you to recall it."
            ),
        )

        questions = [
            "Remember that my favorite programming language is Python.",
            "What is my favorite programming language?",
        ]

        for question in questions:
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})

            for message in result["messages"]:
                if isinstance(message, AIMessage) and message.tool_calls:
                    print("Tool calls:", message.tool_calls)

                if isinstance(message, ToolMessage):
                    print("Tool executed:", message.name)
                    print("Tool result:", message.content)

            print("Question:", question)
            print("Answer:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
