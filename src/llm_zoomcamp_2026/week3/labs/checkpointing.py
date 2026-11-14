import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from llm_zoomcamp_2026.week3.labs.agentic_rag import lookup_faq


def main() -> None:
    load_dotenv()

    model = ChatOpenAI(
        model=os.environ["MODEL"],
        api_key=SecretStr(os.environ["API_KEY"]),
        base_url=os.getenv("BASE_URL") or None,
    )

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[lookup_faq],
        system_prompt=(
            "You are a helpful assistant. "
            "Use lookup_faq for course-specific factual questions."
        ),
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {"thread_id": "lesson7-A"},
        "recursion_limit": 12,
    }

    questions = [
        "Remember this code: Z7-ALPHA.",
        "What code did I tell you to remember?",
    ]

    for question in questions:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )

        print("Answer:", result["messages"][-1].content)
        print("Messages:", len(result["messages"]))

    state = agent.get_state(config)
    human_messages = [
        msg.content for msg in state.values["messages"] if isinstance(msg, HumanMessage)
    ]
    print("Human message history:", human_messages)


if __name__ == "__main__":
    main()
