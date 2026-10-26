from openai.types.completion_usage import CompletionUsage

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
""".strip()


class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="data-engineering-zoomcamp",
        model="ds/deepseek-flash",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(self, query: str, num_results: int = 5) -> list:
        return self.index.search(
            query,
            boost_dict={"question": 2.0, "answer": 0.5},
            filter_dict={"course": self.course},
            num_results=num_results,
        )

    def build_context(self, search_results: list) -> str:
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: list) -> str:
        context = self.build_context(search_results)

        return self.prompt_template.format(
            question=query,
            context=context,
        )

    def llm(self, prompt: str) -> tuple[str, CompletionUsage | None]:
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content or "", response.usage

    def rag(self, query: str) -> tuple[str, CompletionUsage | None]:
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)

        return self.llm(prompt)
