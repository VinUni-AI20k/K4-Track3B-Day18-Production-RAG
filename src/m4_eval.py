from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

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
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        def _val(row, key):
            v = row.get(key, 0.0)
            return 0.0 if v is None or v != v else float(v)  # NaN → 0.0

        per_question = [
            EvalResult(question=row["question"], answer=row["answer"],
                       contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                       faithfulness=_val(row, "faithfulness"),
                       answer_relevancy=_val(row, "answer_relevancy"),
                       context_precision=_val(row, "context_precision"),
                       context_recall=_val(row, "context_recall"))
            for _, row in df.iterrows()
        ]
        n = max(len(per_question), 1)
        return {
            "faithfulness": sum(r.faithfulness for r in per_question) / n,
            "answer_relevancy": sum(r.answer_relevancy for r in per_question) / n,
            "context_precision": sum(r.context_precision for r in per_question) / n,
            "context_recall": sum(r.context_recall for r in per_question) / n,
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    metrics = list(diagnostic_tree)
    scored = []
    for r in eval_results:
        vals = {m: getattr(r, m) for m in metrics}
        worst = min(vals, key=vals.get)
        scored.append((sum(vals.values()) / len(vals), worst, vals[worst], r))
    scored.sort(key=lambda x: x[0])

    failures = []
    for avg, worst, score, r in scored[:bottom_n]:
        diagnosis, fix = diagnostic_tree[worst]
        failures.append({
            "question": r.question,
            "expected": r.ground_truth,
            "answer": r.answer,
            "worst_metric": worst,
            "score": round(score, 4),
            "avg_score": round(avg, 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
