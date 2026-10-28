import os

from dotenv import load_dotenv
from openai import OpenAI
from toyaikit.chat.interface import StdOutputInterface
from toyaikit.chat.runners import (
    DisplayingRunnerCallback,
    OpenAIChatCompletionsRunner,
)
from toyaikit.llm import OpenAIChatCompletionsClient
from toyaikit.tools import Tools

from llm_zoomcamp_2026.week1.ingest import build_index, load_documents
from llm_zoomcamp_2026.week1.rag import RAG

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

callback = DisplayingRunnerCallback(chat_interface)

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

result = runner.loop(
    prompt="Can I join LLM Zoomcamp now?",
    # callback=callback,
)

print("Final answer:", result.last_message)

print("Input tokens:", result.tokens.input_tokens)
print("Output tokens:", result.tokens.output_tokens)

print("Cost:", result.cost)

print("Total messages:", len(result.all_messages))

result2 = runner.loop(
    prompt="How do I run a different model?",
    previous_messages=result.all_messages,
    callback=callback,
)

print("Final answer 2:", result2.last_message)

print("Input tokens 2:", result2.tokens.input_tokens)
print("Output tokens 2:", result2.tokens.output_tokens)

print("Cost 2:", result2.cost)

print("Total messages 2:", len(result2.all_messages))
