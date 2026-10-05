from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, asdict
import math
import hashlib
from pathlib import Path


def _prepare_ragas_event_loop():
    """RAGAS 0.1 creates as_completed before asyncio.run.

    Python 3.13 schedules those jobs immediately on the current loop.
    nest_asyncio makes asyncio.run reuse that loop instead of creating another.
    """
    if sys.version_info >= (3, 13):
        import nest_asyncio
        nest_asyncio.apply()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str],
                   cache_dir: str | None = None) -> dict:
    """Run RAGAS evaluation."""
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("Evaluation inputs must have matching lengths")
    fallback = {**dict.fromkeys(metrics, 0.0), "per_question": [],
                "evaluation_status": "skipped"}
    if not questions:
        return {**fallback, "evaluation_reason": "No questions"}
    from config import GROQ_API_KEY
    if os.getenv("LAB18_OFFLINE") == "1" or not GROQ_API_KEY:
        return {**fallback, "evaluation_reason": "Offline mode or missing GROQ_API_KEY; scores are placeholders"}
    try:
        from ragas import evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        from ragas.run_config import RunConfig
        from datasets import Dataset
        from src.ai_client import create_evaluation_models
        llm, embeddings = create_evaluation_models()
        _prepare_ragas_event_loop()
        metric_objects = [Faithfulness(), AnswerRelevancy(strictness=1),
                          ContextPrecision(), ContextRecall()]
        # Keep one worked example per prompt to reduce token use on the lab quota.
        from ragas.llms.prompt import Prompt
        for metric in metric_objects:
            for value in vars(metric).values():
                if isinstance(value, Prompt):
                    value.examples = value.examples[:1]
        per_question = []
        from config import GROQ_MODEL
        if cache_dir:
            input_data = {"questions": questions, "answers": answers,
                          "contexts": contexts, "ground_truths": ground_truths}
            serialized = json.dumps(input_data, ensure_ascii=False, sort_keys=True)
            input_path = Path(cache_dir) / ("inputs_" + hashlib.sha256(serialized.encode()).hexdigest() + ".json")
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_text(serialized, encoding="utf-8")
        for i, (question, answer, context, ground_truth) in enumerate(
                zip(questions, answers, contexts, ground_truths)):
            payload = {"question": [question], "answer": [answer],
                       "contexts": [context], "ground_truth": [ground_truth]}
            signature = json.dumps({"data": payload, "model": GROQ_MODEL,
                                    "metric_config": "ragas-0.1-strictness1-examples1-v1"},
                                   ensure_ascii=False, sort_keys=True)
            cache_path = (Path(cache_dir) / (hashlib.sha256(signature.encode()).hexdigest() + ".json")
                          if cache_dir else None)
            scores = None
            if cache_path and cache_path.exists():
                scores = json.loads(cache_path.read_text(encoding="utf-8"))
            if scores is None:
                print(f"RAGAS question {i + 1}/{len(questions)}", flush=True)
                result = evaluate(Dataset.from_dict(payload), metrics=metric_objects,
                                  llm=llm, embeddings=embeddings,
                                  run_config=RunConfig(timeout=360, max_retries=3, max_workers=1),
                                  raise_exceptions=True)
                rows = result.to_pandas().to_dict(orient="records")
                if len(rows) != 1:
                    raise ValueError("RAGAS returned an unexpected number of rows")
                scores = {m: float(rows[0][m]) for m in metrics}
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
                raise ValueError("RAGAS returned an invalid metric")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(scores, allow_nan=False), encoding="utf-8")
            per_question.append(EvalResult(question, answer, context, ground_truth, **scores))
        return {**{m: sum(getattr(r, m) for r in per_question) / len(per_question)
                   for m in metrics}, "per_question": per_question, "evaluation_status": "success",
                "evaluation_model": GROQ_MODEL,
                "evaluation_config": "strictness=1; one example per prompt; local bge-m3 embeddings"}
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        print(f"RAGAS evaluation failed: {type(exc).__name__}; HTTP status={status}", flush=True)
        return {**fallback, "evaluation_status": "failed",
                "evaluation_reason": f"{type(exc).__name__}: evaluation did not complete; scores are placeholders"}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
    }
    failures = []
    for result in eval_results:
        scores = {metric: getattr(result, metric) for metric in diagnostic_tree}
        worst = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree[worst]
        failures.append({"question": result.question, "answer": result.answer,
                         "contexts": result.contexts, "ground_truth": result.ground_truth,
                         "score": sum(scores.values()) / len(scores),
                         "worst_metric": worst, "diagnosis": diagnosis, "suggested_fix": fix})
    return sorted(failures, key=lambda f: f["score"])[:max(0, bottom_n)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(r) if isinstance(r, EvalResult) else r
                         for r in results.get("per_question", [])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
