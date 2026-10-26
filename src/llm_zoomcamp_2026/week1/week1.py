import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["AIBOX_API_KEY"], base_url=os.environ["BASE_URL"])


def llm(
    instruction: str, user_prompt: str, model: str = "ds/deepseek-flash"
) -> tuple[str, object]:
    message_history = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        messages=message_history,  # type:ignore
        model=model,
    )

    return response.choices[0].message.content or "", response.usage


doc_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(doc_url)
courses_raw = response.json()

documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    couse_url = f"{url_prefix}{course['path']}"
    course_response = requests.get(couse_url)
    course_response.raise_for_status()
    course_data = course_response.json()
    documents.extend(course_data)

from minsearch import Index

index = Index(text_fields=["question", "section", "answer"], keyword_fields=["course"])

index.fit(documents)

question = "I just discovered the course. Can I join now?"


def search(question: str, course: str = "data-engineering-zoomcamp") -> list:
    boost_dict = {"question": 2.0, "answer": 0.5}
    filter_dict = {"course": course}

    return index.search(
        question, boost_dict=boost_dict, filter_dict=filter_dict, num_results=5
    )


search_results = search(question)

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")

    return "\n".join(lines).strip()


def build_prompt(question, search_results):
    context = build_context(search_results)
    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
    return prompt.strip()


def llm_cost(
    usage,
    input_price=0.15,
    output_price=0.60,
    cache_price=0.003,
) -> float:
    if usage is None:
        return 0.0

    cached = (
        usage.prompt_tokens_details.cached_tokens or 0
        if usage.prompt_tokens_details
        else 0
    )

    uncached = usage.prompt_tokens - cached
    output = usage.completion_tokens

    cost = (
        uncached * input_price + cached * cache_price + output * output_price
    ) / 1_000_000

    return cost


def rag(query: str, model: str = "ds/deepseek-flash") -> tuple[str, object]:
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answers, usage = llm(INSTRUCTIONS, prompt, model=model)
    return answers, usage


answer, usage = rag("Can I join the course now?")
print(answer)
cost = llm_cost(
    usage,
    input_price=0.30,
    output_price=1.20,
    cache_price=0.006,
)
print(cost)
