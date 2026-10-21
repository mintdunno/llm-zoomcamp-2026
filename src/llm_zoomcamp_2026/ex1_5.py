import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["AIBOX_API_KEY"], base_url=os.environ["BASE_URL"])


def llm(prompt: str) -> str:
    response = client.responses.create(
        input=prompt, model=os.getenv("MODEL", "ds/deepseek")
    )
    return response.output_text


doc_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(doc_url)
courses_raw = response.json()

document = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    couse_url = f"{url_prefix}{course['path']}"
    course_response = requests.get(couse_url)
    course_response.raise_for_status()
    course_data = course_response.json()
    document.extend(course_data)

import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["AIBOX_API_KEY"], base_url=os.environ["BASE_URL"])


def llm(prompt: str) -> str:
    response = client.responses.create(
        input=prompt, model=os.getenv("MODEL", "ds/deepseek")
    )
    return response.output_text


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
