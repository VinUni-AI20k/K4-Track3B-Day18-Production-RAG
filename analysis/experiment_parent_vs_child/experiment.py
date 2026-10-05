"""So sánh biến thể generation/retrieval trên cùng index (enrichment đã cache). Không ghi vào reports/."""
import os, sys, json
sys.path.insert(0, os.getcwd())
HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "enrich_cache.json")

from openai import OpenAI
from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas
from config import RERANK_TOP_K

docs = load_documents()
parents = {}
for d in docs:
    ps, _ = chunk_hierarchical(d["text"], metadata=d["metadata"])
    for p in ps:
        parents[(d["metadata"]["source"], p.metadata["parent_id"])] = p.text

enriched = json.load(open(CACHE))
search = HybridSearch(); search.index(enriched)
rr = CrossEncoderReranker()
client = OpenAI()
test_set = load_test_set()

# Retrieval 1 lần, dùng chung cho mọi biến thể -> khác biệt chỉ đến từ biến thể.
retrieved = []
for item in test_set:
    res = search.search(item["question"])
    docs_ = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in res]
    retrieved.append(rr.rerank(item["question"], docs_, top_k=RERANK_TOP_K))


def child_ctx(top):
    return [r.text for r in top]


def parent_ctx(top):
    seen, out = set(), []
    for r in top:
        key = (r.metadata.get("source"), r.metadata.get("parent_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(parents.get(key, r.text))
    return out


def gen(q, ctxs, temperature):
    kw = {} if temperature is None else {"temperature": temperature}
    context_str = "\n\n".join(ctxs)  # giống hệt src/pipeline.py::run_query
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
        {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
        {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {q}"},
    ], **kw)
    return resp.choices[0].message.content


def run(name, ctx_fn, temperature):
    qs, ans, cts, gts = [], [], [], []
    for item, top in zip(test_set, retrieved):
        c = ctx_fn(top)
        qs.append(item["question"]); cts.append(c); gts.append(item["ground_truth"])
        ans.append(gen(item["question"], c, temperature))
    r = evaluate_ragas(qs, ans, cts, gts)
    agg = {k: round(r[k], 4) for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
    nf = sum(1 for a in ans if "không tìm thấy" in a.lower())
    zeros = sum(1 for p in r["per_question"] for k in agg if getattr(p, k) == 0.0)
    print(f"{name:<28} {agg}  'Không tìm thấy'={nf}/20  zero-cells={zeros}", flush=True)
    json.dump({"agg": agg, "answers": ans, "per_q": [vars(p) for p in r["per_question"]]},
              open(os.path.join(HERE, f"exp_{name}.json"), "w"), ensure_ascii=False, indent=1, default=str)
    return r

for name, fn, t in [("A0_child_tempdefault", child_ctx, None),
                    ("A1_child_temp0", child_ctx, 0),
                    ("B_parent_temp0", parent_ctx, 0),
                    ("B2_parent_temp0_rerun", parent_ctx, 0)]:
    run(name, fn, t)
