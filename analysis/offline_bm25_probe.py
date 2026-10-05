"""Offline retrieval probe (KHÔNG phải RAGAS): BM25-only, không cần model/API.

Với mỗi câu trong test_set.json, lấy top-3 chunk bằng BM25 rồi kiểm tra có bao nhiêu "dữ kiện số"
(số ngày, số tiền, %...) của ground_truth xuất hiện trong context. Dùng để xác định lỗi retrieval
khi chưa có mạng tải bge-m3 / gọi OpenAI.

Chạy: python analysis/offline_bm25_probe.py [hierarchical|structure|basic]
"""
import os, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.m1_chunking import load_documents, chunk_hierarchical, chunk_structure_aware, chunk_basic
from src.m2_search import BM25Search
from src.m4_eval import load_test_set

strategy = sys.argv[1] if len(sys.argv) > 1 else "hierarchical"
chunks = []
for d in load_documents():
    if strategy == "hierarchical":
        cs = chunk_hierarchical(d["text"], metadata=d["metadata"])[1]
    elif strategy == "structure":
        cs = chunk_structure_aware(d["text"], metadata=d["metadata"])
    else:
        cs = chunk_basic(d["text"], metadata=d["metadata"])
    chunks += [{"text": c.text, "metadata": c.metadata} for c in cs]

bm25 = BM25Search()
bm25.index(chunks)


def facts(text: str) -> set[str]:
    return {n.replace(".", "").replace(",", "") for n in re.findall(r"\d[\d.,]*\d|\d", text)}


rows = []
for i, item in enumerate(load_test_set(), 1):
    ctx = bm25.search(item["question"], top_k=3)
    blob = " ".join(r.text for r in ctx)
    need = facts(item["ground_truth"])
    have = facts(blob)
    cov = len(need & have) / len(need) if need else 1.0
    rows.append((cov, i, item["question"], [r.metadata.get("source") for r in ctx], sorted(need - have)))

print(f"strategy={strategy} chunks={len(chunks)}  mean fact-coverage@3={sum(r[0] for r in rows)/len(rows):.3f}")
for cov, i, q, src, miss in sorted(rows)[:8]:
    print(f"Q{i:<2} cov={cov:.2f} missing={miss} src={src}\n     {q}")
