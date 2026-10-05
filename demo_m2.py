"""Run Module 2 on a small corpus without changing the production collection."""

from uuid import uuid4

from src.m2_search import BM25Search, DenseSearch, reciprocal_rank_fusion


def main():
    chunks = [
        {"text": "Nhân viên được nghỉ phép năm 15 ngày làm việc.",
         "metadata": {"source": "nghi_phep_demo"}},
        {"text": "Mật khẩu phải được thay đổi mỗi 120 ngày và bật MFA.",
         "metadata": {"source": "bao_mat_demo"}},
        {"text": "Thời gian thử việc là 60 ngày.",
         "metadata": {"source": "nhan_su_demo"}},
    ]
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    collection = "lab18_m2_demo_" + uuid4().hex
    bm25, dense = BM25Search(), DenseSearch()
    print(f"Backend: {dense.backend}", flush=True)
    print(f"Câu hỏi: {query}", flush=True)
    try:
        bm25.index(chunks)
        dense.index(chunks, collection=collection)
        keyword_results = bm25.search(query, top_k=3)
        dense_results = dense.search(query, top_k=3, collection=collection)
        hybrid_results = reciprocal_rank_fusion([keyword_results, dense_results], top_k=3)
        for title, results in [("BM25", keyword_results), ("Dense", dense_results),
                               ("Hybrid (RRF)", hybrid_results)]:
            print(f"\n{title}:")
            for rank, result in enumerate(results, start=1):
                print(f"  {rank}. [{result.score:.4f}] {result.text}")
        assert dense_results and dense_results[0].metadata["source"] == "nghi_phep_demo"
        assert hybrid_results and hybrid_results[0].metadata["source"] == "nghi_phep_demo"
        dense.index([], collection=collection)
        assert dense.search(query, collection=collection) == []
        print("\nDemo PASSED: đúng đoạn nghỉ phép; index rỗng trả về danh sách rỗng.")
    finally:
        if dense.client.collection_exists(collection_name=collection):
            dense.client.delete_collection(collection_name=collection)
        dense.client.close()


if __name__ == "__main__":
    main()
