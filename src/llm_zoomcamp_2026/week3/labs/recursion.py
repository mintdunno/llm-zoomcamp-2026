from typing import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph


class State(TypedDict):
    count: int


def increment(state: State) -> dict:
    count = state["count"] + 1
    print("Count:", count)
    return {"count": count}


from langgraph.graph import END


def should_continue(state: State) -> str:
    if state["count"] >= 4:
        return END

    return "a"


builder = StateGraph(State)

builder.add_node("a", increment)
builder.add_node("b", increment)

builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_conditional_edges("b", should_continue)


graph = builder.compile()

try:
    graph.invoke(
        {"count": 0},
        config={"recursion_limit": 10},
    )
except GraphRecursionError:
    print("Graph stopped: recursion limit exceeded")
