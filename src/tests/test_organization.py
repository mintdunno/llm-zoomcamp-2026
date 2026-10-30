"""Offline checks for retrieval and module imports."""

import importlib
import pkgutil
import unittest
from unittest.mock import Mock, patch

import llm_zoomcamp_2026
from llm_zoomcamp_2026.week1.core.ingest import build_index
from llm_zoomcamp_2026.week1.core.rag import RAG
from llm_zoomcamp_2026.week2.core.vector_rag import RAGVector


class RetrievalTests(unittest.TestCase):
    def test_keyword_rag_filters_course_and_builds_context(self):
        documents = [
            {
                "course": "llm-zoomcamp",
                "section": "Joining",
                "question": "Can I join late?",
                "answer": "Yes.",
            },
            {
                "course": "other",
                "section": "Joining",
                "question": "Can I join late?",
                "answer": "No.",
            },
        ]
        rag = RAG(build_index(documents), client=Mock(), model="example")
        results = rag.search("join late")
        self.assertEqual([doc["course"] for doc in results], ["llm-zoomcamp"])
        self.assertIn("A: Yes.", rag.build_prompt("Can I join?", results))

    def test_vector_rag_encodes_query_and_applies_course_filter(self):
        embedder = Mock()
        embedder.encode.return_value = [0.1, 0.2]
        index = Mock()
        index.search.return_value = [{"question": "Example"}]
        rag = RAGVector(embedder=embedder, index=index, client=Mock(), model="example")
        self.assertEqual(rag.search("question"), [{"question": "Example"}])
        embedder.encode.assert_called_once_with("question")
        index.search.assert_called_once_with(
            [0.1, 0.2], filter_dict={"course": "llm-zoomcamp"}, num_results=5
        )

    def test_modules_import_without_running_external_operations(self):
        import openai
        import psycopg
        import requests
        import sentence_transformers
        import sqlitesearch

        with (
            patch.object(requests, "get", side_effect=AssertionError("network")),
            patch.object(openai, "OpenAI", side_effect=AssertionError("client")),
            patch.object(psycopg, "connect", side_effect=AssertionError("database")),
            patch.object(
                sentence_transformers,
                "SentenceTransformer",
                side_effect=AssertionError("model"),
            ),
            patch.object(
                sqlitesearch, "VectorSearchIndex", side_effect=AssertionError("index")
            ),
        ):
            for module in pkgutil.walk_packages(
                llm_zoomcamp_2026.__path__, llm_zoomcamp_2026.__name__ + "."
            ):
                with self.subTest(module=module.name):
                    importlib.import_module(module.name)


if __name__ == "__main__":
    unittest.main()
