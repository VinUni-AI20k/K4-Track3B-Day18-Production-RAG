"""Resume RAGAS from saved inputs without regenerating answers.

CMD: python resume_evaluation.py INPUT_JSON --output reports/naive_baseline_report.json
"""
import argparse
import json
from pathlib import Path

from src.m4_eval import evaluate_ragas, failure_analysis, save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.inputs.read_text(encoding="utf-8"))
    results = evaluate_ragas(**data, cache_dir=".run_cache/evaluation")
    if results["evaluation_status"] != "success":
        print("Chưa hoàn tất đánh giá. Các câu thành công vẫn được lưu để chạy tiếp.")
        return 1
    failures = failure_analysis(results["per_question"], bottom_n=5)
    save_report(results, failures, str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
