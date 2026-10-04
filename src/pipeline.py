from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

# Latency breakdown (giây) cho từng bước — in ra + lưu vào reports/latency_report.json
TIMINGS: dict[str, float] = {}
QUERY_TIMINGS: dict[str, list[float]] = {"search": [], "rerank": [], "generate": []}


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)
    TIMINGS["chunk"] = time.time() - t0

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
        TIMINGS["enrich"] = time.time() - t0
    else:
        print("  ⚠️  M5 returned nothing — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)
    TIMINGS["index"] = time.time() - t0

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)
    TIMINGS["load_reranker"] = time.time() - t0

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t = time.perf_counter()
    results = search.search(query)
    QUERY_TIMINGS["search"].append(time.perf_counter() - t)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    t = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    QUERY_TIMINGS["rerank"].append(time.perf_counter() - t)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    from config import OPENAI_API_KEY
    t = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    QUERY_TIMINGS["generate"].append(time.perf_counter() - t)
    return answer, contexts


def save_latency_report(eval_seconds: float, path: str = "reports/latency_report.json") -> None:
    """In bảng latency từng bước + lưu JSON (bonus: latency breakdown)."""
    import json

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    rows = {
        "M1 chunking (total, s)": TIMINGS.get("chunk", 0.0),
        "M5 enrichment (total, s)": TIMINGS.get("enrich", 0.0),
        "M2 indexing BM25+dense (total, s)": TIMINGS.get("index", 0.0),
        "M3 load reranker (s)": TIMINGS.get("load_reranker", 0.0),
        "M2 hybrid search (avg ms/query)": avg(QUERY_TIMINGS["search"]) * 1000,
        "M3 rerank (avg ms/query)": avg(QUERY_TIMINGS["rerank"]) * 1000,
        "LLM generation (avg ms/query)": avg(QUERY_TIMINGS["generate"]) * 1000,
        "M4 RAGAS eval (total, s)": eval_seconds,
    }
    print("\n" + "=" * 60 + "\nLATENCY BREAKDOWN\n" + "=" * 60)
    for k, v in rows.items():
        print(f"  {k:<38} {v:>10.2f}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: round(v, 3) for k, v in rows.items()}, f, ensure_ascii=False, indent=2)
    print(f"Latency report saved to {path}")


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_seconds = time.time() - t0
    print(f"  ✓ RAGAS done ({eval_seconds:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    save_latency_report(eval_seconds)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
