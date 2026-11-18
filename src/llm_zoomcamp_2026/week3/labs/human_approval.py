import argparse
import os
from typing import TypedDict

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt


# 1. Graph state
class ApprovalState(TypedDict):
    request_id: str
    action: str
    approved: bool | None
    status: str


# 2. Pause and request human approval
def ask_approval(state: ApprovalState) -> dict:
    print("[approval] Node started")

    approved = interrupt(
        {
            "request_id": state["request_id"],
            "action": state["action"],
            "question": "Do you approve this action?",
        }
    )

    print("[approval] Decision:", approved)
    return {"approved": approved}


# 3. Mock write action with idempotency
def execute_action(state: ApprovalState) -> dict:
    print("[execute] Running mock write")

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        row = conn.execute(
            """
            INSERT INTO hitl_mock_actions (request_id, action)
            VALUES (%s, %s)
            ON CONFLICT (request_id) DO NOTHING
            RETURNING request_id
            """,
            (state["request_id"], state["action"]),
        ).fetchone()

    status = "executed" if row else "already_executed"
    return {"status": status}


# 4. Reject action
def reject_action(state: ApprovalState) -> dict:
    print("[reject] Action rejected")
    return {"status": "rejected"}


# 5. Conditional routing
def route_approval(state: ApprovalState) -> str:
    return "execute" if state["approved"] is True else "reject"


# 6. Build LangGraph
def build_graph(checkpointer):
    builder = StateGraph(ApprovalState)

    builder.add_node("approval", ask_approval)
    builder.add_node(
        "execute",
        execute_action,
        retry_policy=RetryPolicy(
            max_attempts=3,
            retry_on=psycopg.OperationalError,
        ),
    )
    builder.add_node("reject", reject_action)

    builder.add_edge(START, "approval")

    builder.add_conditional_edges(
        "approval",
        route_approval,
        {
            "execute": "execute",
            "reject": "reject",
        },
    )

    builder.add_edge("execute", END)
    builder.add_edge("reject", END)

    return builder.compile(checkpointer=checkpointer)


# 7. Create a safe demo table
def setup_mock_table() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hitl_mock_actions (
                request_id TEXT PRIMARY KEY,
                action TEXT NOT NULL
            )
            """
        )


# 8. CLI runner
def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["start", "approve", "reject", "status"],
    )
    parser.add_argument("--thread", default="lesson8-A")
    parser.add_argument(
        "--action",
        default="Mark unused records as inactive",
    )
    args = parser.parse_args()

    setup_mock_table()

    with PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)

        config = {"configurable": {"thread_id": args.thread}}

        snapshot = graph.get_state(config)

        if args.command == "status":
            print("State:", snapshot.values)
            print("Next nodes:", snapshot.next)
            return

        if args.command == "start":
            if snapshot.values:
                print("Thread already exists. Use a new thread ID.")
                return

            result = graph.invoke(
                {
                    "request_id": args.thread,
                    "action": args.action,
                    "approved": None,
                    "status": "pending",
                },
                config=config,
            )
        else:
            if not snapshot.next:
                print("No pending execution to resume.")
                return

            result = graph.invoke(
                Command(resume=args.command == "approve"),
                config=config,
            )

        if "__interrupt__" in result:
            print("PAUSED:", result["__interrupt__"][0].value)
        else:
            print("FINAL:", result)


if __name__ == "__main__":
    main()
