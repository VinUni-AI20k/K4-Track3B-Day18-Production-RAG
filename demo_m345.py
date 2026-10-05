"""Small demo of enrichment, retrieval, reranking and evaluation.

Use LAB18_OFFLINE=1 for a demo without external API requests.
"""

from uuid import uuid4

from src.m2_search import BM25Search, DenseSearch, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks


def main():
    question = "Nhân viên được nghỉ phép bao nhiêu ngày mỗi năm?"
    chunks = [
        {"text": "Nhân viên được nghỉ phép năm 15 ngày làm việc mỗi năm.",
         "metadata": {"source": "nghi_phep_demo.md"}},
        {"text": "Mật khẩu phải thay đổi mỗi 120 ngày và bật MFA.",
         "metadata": {"source": "bao_mat_demo.md"}},
        {"text": "Thời gian thử việc là 60 ngày.",
         "metadata": {"source": "nhan_su_demo.md"}},
    ]
    print("M5: Enrichment", flush=True)
    enriched = enrich_chunks(chunks)
    print(f"Summary: {enriched[0].summary}")
    print(f"Questions: {enriched[0].hypothesis_questions}")
    print(f"Mode: {enriched[0].auto_metadata.get('enrichment_mode')}")
    indexed = [{"text": chunk.enriched_text, "metadata": chunk.auto_metadata} for chunk in enriched]
    dense, bm25 = DenseSearch(), BM25Search()
    collection = "lab18_m345_demo_" + uuid4().hex
    try:
        bm25.index(indexed)
        dense.index(indexed, collection=collection)
        retrieved = reciprocal_rank_fusion([
            bm25.search(question), dense.search(question, collection=collection)])
        reranker = CrossEncoderReranker()
        ranked = reranker.rerank(question, [
            {"text": result.text, "score": result.score, "metadata": result.metadata}
            for result in retrieved])
        print("\nM3: Reranking")
        for result in ranked:
            print(f"  [{result.rank}] {result.rerank_score:.4f}: {result.metadata['original_text']}")
        assert ranked and ranked[0].metadata["source"] == "nghi_phep_demo.md"
        contexts = [result.metadata["original_text"] for result in ranked]
        # Known demo answer isolates evaluation from LLM answer generation.
        answer = "Nhân viên được nghỉ phép năm 15 ngày làm việc mỗi năm."
        result = evaluate_ragas([question], [answer], [contexts], [answer])
        print(f"\nM4: {result['evaluation_status']}")
        if result["evaluation_status"] != "success":
            print(result.get("evaluation_reason"))
        else:
            print({name: result[name] for name in (
                "faithfulness", "answer_relevancy", "context_precision", "context_recall")})
        save_report(result, failure_analysis(result["per_question"]), "reports/demo_m345_report.json")
        print("Demo retrieval and reranking PASSED.")
    finally:
        if dense.client.collection_exists(collection_name=collection):
            dense.client.delete_collection(collection_name=collection)
        dense.client.close()


if __name__ == "__main__":
    main()
