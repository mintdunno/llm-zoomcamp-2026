import os

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model=os.environ["MODEL"],
    api_key=SecretStr(os.environ["API_KEY"]),
    base_url=os.getenv("BASE_URL") or None,
)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain RAG in one sentence."),
]

response = llm.invoke(messages)

print("Type:", type(response).__name__)
print("Content:", response.content)
print("Usage:", response.usage_metadata)


from langchain.tools import tool


@tool
def lookup_course_policy(topic: str) -> str:
    """Look up a course policy by its topic."""
    policies = {
        "late_enrollment": "Students can join after the course starts.",
        "certificate": "A certificate is available after completion.",
    }
    return policies.get(topic, "Policy not found.")


print("####################################")
print("Name:", lookup_course_policy.name)
print("Description:", lookup_course_policy.description)
print("Args_schema", lookup_course_policy.args_schema)
print("Schema:", lookup_course_policy.args_schema.model_json_schema())

result = lookup_course_policy.invoke({"topic": "late_enrollment"})
print("Result:", result)
