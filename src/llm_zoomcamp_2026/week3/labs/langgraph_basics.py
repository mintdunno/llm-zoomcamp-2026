from operator import add
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph


class TaskState(TypedDict):
    question: str
    route: NotRequired[str]
    result: NotRequired[str]
    history: Annotated[list[str], add]


def prepare(state: TaskState) -> dict:
    print("Prepare received:", state)

    return {
        "route": "faq",
        "history": ["prepare executed"],
    }


def answer(state: TaskState) -> dict:
    print("Answer received:", state)

    return {
        "result": f"Processing: {state['question']}",
        "history": ["answer executed"],
    }


builder = StateGraph(TaskState)

builder.add_node("prepare", prepare)
builder.add_node("answer", answer)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "answer")
builder.add_edge("answer", END)

graph = builder.compile()

result = graph.invoke(
    {
        "question": "Can I join late?",
        "history": [],
    }
)

print("Final state:", result)
