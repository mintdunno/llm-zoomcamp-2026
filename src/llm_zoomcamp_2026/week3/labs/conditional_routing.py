import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr


@tool
def search_faq(query: str) -> str:
    """Search the LLM Zoomcamp FAQ for course information."""
    raise RuntimeError("Simulated database connection failure")


def main() -> None:
    load_dotenv()

    # 1. Configure the LLM and tools.
    tools = [search_faq]

    model = ChatOpenAI(
        model=os.environ["MODEL"],
        api_key=SecretStr(os.environ["API_KEY"]),
        base_url=os.getenv("BASE_URL") or None,
    )

    model_with_tools = model.bind_tools(tools)

    # 2. Define the model node.
    def call_model(state: MessagesState) -> dict:
        messages = [
            SystemMessage(
                content=(
                    "You are an LLM Zoomcamp assistant. "
                    "Use search_faq for course-specific questions. "
                    "If the tool fails, explain that the information "
                    "is unavailable. Do not retry a failed tool call "
                    "or invent course policies."
                )
            ),
            *state["messages"],
        ]

        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # 3. Build the graph.
    builder = StateGraph(MessagesState)

    builder.add_node("model", call_model)
    builder.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=(RuntimeError,)),
    )

    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    graph = builder.compile()

    # 4. Execute and inspect each node update.
    try:
        for event in graph.stream(
            {"messages": [("user", "Can I join LLM Zoomcamp after it starts?")]},
            config={"recursion_limit": 10},
            stream_mode="updates",
        ):
            for node_name, update in event.items():
                print(f"\n===== NODE: {node_name} =====")

                for message in update.get("messages", []):
                    print("Type:", message.type)
                    print("Content:", message.content)

                    if isinstance(message, AIMessage):
                        print("Tool calls:", message.tool_calls)

                    if isinstance(message, ToolMessage):
                        print("Tool status:", message.status)
                        print("Tool call ID:", message.tool_call_id)

    except GraphRecursionError:
        print("\nGraph stopped: recursion limit exceeded.")


if __name__ == "__main__":
    main()
