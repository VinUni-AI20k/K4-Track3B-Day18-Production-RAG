"""Check API adapters and failure paths without sending external requests."""

import json
from unittest.mock import mock_open
from types import SimpleNamespace

import pandas as pd
import pytest

from src import m4_eval as evaluation, m5_enrichment as enrichment


def test_ragas_success_and_report(monkeypatch):
    import config
    import ragas
    from src import ai_client
    monkeypatch.delenv("LAB18_OFFLINE", raising=False)
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(ai_client, "create_evaluation_models", lambda: (object(), object()))
    scores = {"faithfulness": [0.8], "answer_relevancy": [0.7],
              "context_precision": [0.9], "context_recall": [0.6]}

    def fake_evaluate(dataset, **kwargs):
        assert dataset[0]["question"] == "Q"
        assert len(kwargs["metrics"]) == 4
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame(scores))

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    result = evaluation.evaluate_ragas(["Q"], ["A"], [["C"]], ["GT"])
    assert result["evaluation_status"] == "success"
    assert result["faithfulness"] == 0.8
    failures = evaluation.failure_analysis(result["per_question"], bottom_n=1)
    assert failures[0]["worst_metric"] == "context_recall"
    writer = mock_open()
    monkeypatch.setattr("builtins.open", writer)
    evaluation.save_report(result, failures, "report.json")
    report = json.loads("".join(call.args[0] for call in writer().write.call_args_list))
    assert report["num_questions"] == 1
    assert report["per_question"][0]["answer"] == "A"


def test_ragas_failure_is_explicit(monkeypatch):
    import config
    import ragas
    from src import ai_client
    monkeypatch.delenv("LAB18_OFFLINE", raising=False)
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(ai_client, "create_evaluation_models", lambda: (object(), object()))

    def fail(*args, **kwargs):
        raise RuntimeError("Test failure")

    monkeypatch.setattr(ragas, "evaluate", fail)
    result = evaluation.evaluate_ragas(["Q"], ["A"], [["C"]], ["GT"])
    assert result["evaluation_status"] == "failed"
    assert not result["per_question"]
    with pytest.raises(ValueError):
        evaluation.evaluate_ragas(["Q"], [], [], [])


def test_combined_one_call_and_preserves_source(monkeypatch):
    import openai
    monkeypatch.delenv("LAB18_OFFLINE", raising=False)
    monkeypatch.setattr(enrichment, "GROQ_API_KEY", "test-key")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        data = {"summary": "short", "questions": ["Question?"], "context": "Context",
                "metadata": {"source": "wrong", "parent_id": "wrong", "topic": "policy"}}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    result = enrichment.enrich_chunks([{"text": "Original", "metadata": {
        "source": "correct.md", "parent_id": "parent_0"}}])[0]
    assert len(calls) == 1
    assert result.auto_metadata["source"] == "correct.md"
    assert result.auto_metadata["parent_id"] == "parent_0"
    assert all(text in result.enriched_text for text in ("Original", "Context", "Question?", "short"))


def test_invalid_enrichment_response_falls_back(monkeypatch):
    import openai
    monkeypatch.delenv("LAB18_OFFLINE", raising=False)
    monkeypatch.setattr(enrichment, "GROQ_API_KEY", "test-key")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"summary": null}'))]))))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    result = enrichment._enrich_single_call("A full sentence.", "source.md")
    assert result["metadata"]["enrichment_mode"] == "fallback"
    assert result["summary"] == "A full sentence."


def test_pipeline_uses_parent_context(monkeypatch):
    from src import pipeline
    from src.m2_search import SearchResult
    monkeypatch.setenv("LAB18_OFFLINE", "1")
    result = SearchResult("Synthetic indexed text", 1.0,
                          {"parent_text": "Original parent context"}, "hybrid")
    search = SimpleNamespace(search=lambda query: [result, result])
    reranker = SimpleNamespace(rerank=lambda *args, **kwargs: [result, result])
    answer, contexts = pipeline.run_query("Question", search, reranker)
    assert contexts == ["Original parent context"]
    assert answer == "Original parent context"


def test_client_uses_groq_endpoint(monkeypatch):
    import openai
    from src import ai_client
    monkeypatch.setattr(ai_client, "GROQ_API_KEY", "test-groq-key")
    settings = {}

    def capture(**kwargs):
        settings.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "OpenAI", capture)
    ai_client.create_client()
    assert settings["api_key"] == "test-groq-key"
    assert settings["base_url"] == "https://api.groq.com/openai/v1"


def test_evaluation_uses_groq_and_local_embeddings(monkeypatch):
    from src import ai_client
    import langchain_openai
    import langchain_community.embeddings
    settings = {}
    monkeypatch.setattr(ai_client, "GROQ_API_KEY", "test-groq-key")

    def chat(**kwargs):
        settings["chat"] = kwargs
        return object()

    def embeddings(**kwargs):
        settings["embeddings"] = kwargs
        return object()

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", chat)
    monkeypatch.setattr(langchain_community.embeddings, "HuggingFaceEmbeddings", embeddings)
    ai_client.create_evaluation_models()
    assert settings["chat"]["api_key"] == "test-groq-key"
    assert "api.groq.com" in settings["chat"]["base_url"]
    assert settings["embeddings"]["model_name"] == "BAAI/bge-m3"


def test_ragas_executor_completes_on_current_python():
    import asyncio
    from ragas.executor import Executor
    from ragas.run_config import RunConfig
    evaluation._prepare_ragas_event_loop()
    executor = Executor(run_config=RunConfig(max_workers=1), raise_exceptions=True)

    async def score(value):
        await asyncio.sleep(0)
        return value

    executor.submit(score, 0.8)
    executor.submit(score, 0.9)
    assert executor.results() == [0.8, 0.9]


def test_flashrank_preserves_original_scores_and_metadata():
    from src.m3_rerank import FlashrankReranker
    reranker = FlashrankReranker()
    reranker._model = SimpleNamespace(rerank=lambda request: [
        {"id": 1, "score": 0.9}, {"id": 0, "score": 0.1}])
    docs = [{"text": "other", "score": 0.8, "metadata": {"source": "a"}},
            {"text": "relevant", "score": 0.7, "metadata": {"source": "b"}}]
    result = reranker.rerank("query", docs, top_k=1)
    assert result[0].text == "relevant"
    assert result[0].original_score == 0.7
    assert result[0].metadata == {"source": "b"}
    assert result[0].rank == 0
    assert reranker.rerank("query", []) == []


def test_groq_budget_waits_until_token_reset(monkeypatch):
    import httpx
    from src import ai_client
    monkeypatch.setattr(ai_client.time, "monotonic", lambda: 100.0)
    budget = ai_client._TokenBudget()
    response = httpx.Response(200, headers={
        "x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "1m2.5s"})
    budget.update(response)
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions", content=b"x" * 300)
    assert budget.remaining == 100
    assert budget.delay(request) == 63.5
    monkeypatch.setattr(ai_client.time, "monotonic", lambda: 163.0)
    assert budget.delay(request) == 0.0
