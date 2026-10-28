from minsearch import Index
from openai import OpenAI
from openai.types.completion_usage import CompletionUsage

INSTRUCTIONS = """
You are a course teaching assistant.
Answer questions using only the provided FAQ context.
If the context doesn't contain the answer, say "I don't know".
""".strip()


PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
""".strip()


class RAG:
    def __init__(
        self,
        index: Index,
        client: OpenAI,
        model: str,
        course: str = "llm-zoomcamp",
    ):
        """Configure the index, model, and course for RAG queries."""
        self.index = index
        self.client = client
        self.model = model
        self.course = course

    def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> list[dict]:
        """Search FAQs for the configured course."""
        return self.index.search(
            query,
            filter_dict={"course": self.course},
            boost_dict={
                "question": 2.0,
                "answer": 0.5,
            },
            num_results=num_results,
        )

    def build_context(
        self,
        documents: list[dict],
    ) -> str:
        """Format FAQ documents as context for the model."""
        lines = []

        for doc in documents:
            lines.append(f"Section: {doc['section']}")
            lines.append(f"Q: {doc['question']}")
            lines.append(f"A: {doc['answer']}")
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(
        self,
        question: str,
        documents: list[dict],
    ) -> str:
        """Combine the question and FAQ context into a prompt."""
        context = self.build_context(documents)

        return PROMPT_TEMPLATE.format(
            question=question,
            context=context,
        )

    def llm(
        self,
        prompt: str,
    ) -> tuple[str, CompletionUsage | None]:
        """Send a prompt to the model and return its answer and token usage."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": INSTRUCTIONS,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = response.choices[0].message.content or ""

        return answer, response.usage

    def rag(
        self,
        question: str,
    ) -> tuple[str, CompletionUsage | None]:
        """Find relevant FAQs and use them to answer the question."""
        documents = self.search(question)
        prompt = self.build_prompt(question, documents)

        return self.llm(prompt)
