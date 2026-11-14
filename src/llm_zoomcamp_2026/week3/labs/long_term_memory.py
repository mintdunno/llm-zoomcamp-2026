from langchain.tools import ToolRuntime, tool


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
