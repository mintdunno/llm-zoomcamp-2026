import json

from llm_zoomcamp_2026.week3.labs import agentic_rag


def test_lookup_faq_empty(monkeypatch):
    monkeypatch.setattr(
        agentic_rag,
        "search_faq",
        lambda query, k=3: [],
    )

    result = agentic_rag.lookup_faq.invoke({"query": "Unknown course policy"})

    assert json.loads(result) == []
